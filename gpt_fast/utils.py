import itertools
import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from torch import Tensor

_USE_FLASH_ATTENTION: bool = False


@contextmanager
def set_flash_attention(enable: bool):
    global _USE_FLASH_ATTENTION

    original_value = _USE_FLASH_ATTENTION
    _USE_FLASH_ATTENTION = enable

    yield

    _USE_FLASH_ATTENTION = original_value


def is_flash_attention_enabled() -> bool:
    global _USE_FLASH_ATTENTION
    return _USE_FLASH_ATTENTION


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        if torch.compiler.is_compiling():
            output = self._norm(x.float()).type_as(x)
            output = output * self.weight
        else:
            output = LigerRMSNormFunction.apply(x, self.weight, self.eps)

        return output


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        assert config.intermediate_size % tp_world_size == 0
        assert config.dim % tp_world_size == 0

        self.w1 = nn.Linear(config.dim, 2 * config.intermediate_size // tp_world_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size // tp_world_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        u, g = x.chunk(2, dim=-1)
        y = self.w2(F.silu(g) * u)
        return y


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim

        assert total_head_dim % tp_world_size == 0
        assert config.dim % tp_world_size == 0
        assert config.n_head % tp_world_size == 0
        assert config.n_local_heads % tp_world_size == 0

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim // tp_world_size, bias=False)
        self.wo = nn.Linear(config.dim // tp_world_size, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

        self.n_head = self.n_head // tp_world_size
        self.dim = self.dim // tp_world_size
        self.n_local_heads = self.n_local_heads // tp_world_size

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if is_flash_attention_enabled():
            device = q.device

            if seqlen <= 1:  # decode time
                k_cache = self.kv_cache.k_cache  # (batch_size, n_local_heads, seqlen_cache, head_dim)
                k_cache = k_cache.transpose(1, 2)
                v_cache = self.kv_cache.v_cache
                v_cache = v_cache.transpose(1, 2)
                cache_seqlens = k_cache.size(1)
                y = flash_attn_with_kvcache(
                    q,  # (batch_size, seqlen_q, n_heads, head_dim)
                    k_cache,  # (batch_size, seqlen_cache, n_local_heads, head_dim)
                    v_cache,
                    k=k,  # (batch_size, seqlen_new, n_local_heads, head_dim)
                    v=v,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=None,
                    cache_leftpad=None,
                    block_table=None,
                    rotary_cos=None,
                    rotary_sin=None,
                    softmax_scale=None,
                    causal=True,
                )
                k_cache = k_cache.transpose(1, 2)
                v_cache = v_cache.transpose(1, 2)
                self.kv_cache.k_cache = k_cache
                self.kv_cache.v_cache = v_cache
            else:
                if self.kv_cache is not None:
                    k, v = map(lambda x: x.transpose(1, 2), (k, v))
                    k, v = self.kv_cache.update(input_pos, k, v)
                    k, v = map(lambda x: x.transpose(1, 2), (k, v))
                # y = flash_attn_func(q, k, v, causal=True)
                q_var = q.reshape(-1, q.shape[-2], q.shape[-1])
                k_var = k.reshape(-1, k.shape[-2], k.shape[-1])
                v_var = v.reshape(-1, v.shape[-2], v.shape[-1])
                lens = torch.full([q.shape[0]], seqlen, dtype=torch.int32, device=device)

                cu_seqlens = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=device),
                        torch.cumsum(lens, dim=0, dtype=torch.int32),
                    ]
                ).int()
                y = flash_attn_varlen_func(
                    q_var, k_var, v_var, cu_seqlens, cu_seqlens, q.size(1), k.size(1), causal=True
                )
        else:
            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
            if self.kv_cache is not None:
                k, v = self.kv_cache.update(input_pos, k, v)

            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)

        return y


class FuseAttentionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim

        assert total_head_dim % tp_world_size == 0
        assert config.dim % tp_world_size == 0
        assert config.intermediate_size % tp_world_size == 0
        assert config.n_head % tp_world_size == 0
        assert config.n_local_heads % tp_world_size == 0

        # key, query, value projections for all heads, but in a batch
        self.wqkv1 = nn.Linear(
            config.dim, total_head_dim // tp_world_size + 2 * config.intermediate_size // tp_world_size, bias=False
        )
        self.wo = nn.Linear(config.dim // tp_world_size, config.dim, bias=False)
        self.w2 = nn.Linear(config.intermediate_size // tp_world_size, config.dim, bias=False)

        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim

        self.n_head = self.n_head // tp_world_size
        self.dim = self.dim // tp_world_size
        self.n_local_heads = self.n_local_heads // tp_world_size

        self.intermediate_size = config.intermediate_size // tp_world_size

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        # q, k, v = self.wqkv1(x).split([self.dim, kv_size, kv_size], dim=-1)
        # use fuse qkv1
        q, k, v, u, g = self.wqkv1(x).split(
            [self.dim, kv_size, kv_size, self.intermediate_size, self.intermediate_size], dim=-1
        )

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if is_flash_attention_enabled():
            device = q.device

            if seqlen <= 1:  # decode time
                k_cache = self.kv_cache.k_cache  # (batch_size, n_local_heads, seqlen_cache, head_dim)
                k_cache = k_cache.transpose(1, 2)
                v_cache = self.kv_cache.v_cache
                v_cache = v_cache.transpose(1, 2)
                cache_seqlens = k_cache.size(1)
                y = flash_attn_with_kvcache(
                    q,  # (batch_size, seqlen_q, n_heads, head_dim)
                    k_cache,  # (batch_size, seqlen_cache, n_local_heads, head_dim)
                    v_cache,
                    k=k,  # (batch_size, seqlen_new, n_local_heads, head_dim)
                    v=v,
                    cache_seqlens=cache_seqlens,
                    cache_batch_idx=None,
                    cache_leftpad=None,
                    block_table=None,
                    rotary_cos=None,
                    rotary_sin=None,
                    softmax_scale=None,
                    causal=True,
                )
                k_cache = k_cache.transpose(1, 2)
                v_cache = v_cache.transpose(1, 2)
                self.kv_cache.k_cache = k_cache
                self.kv_cache.v_cache = v_cache
            else:
                if self.kv_cache is not None:
                    k, v = map(lambda x: x.transpose(1, 2), (k, v))
                    k, v = self.kv_cache.update(input_pos, k, v)
                    k, v = map(lambda x: x.transpose(1, 2), (k, v))
                # y = flash_attn_func(q, k, v, causal=True)
                q_var = q.reshape(-1, q.shape[-2], q.shape[-1])
                k_var = k.reshape(-1, k.shape[-2], k.shape[-1])
                v_var = v.reshape(-1, v.shape[-2], v.shape[-1])
                lens = torch.full([q.shape[0]], seqlen, dtype=torch.int32, device=device)

                cu_seqlens = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int32, device=device),
                        torch.cumsum(lens, dim=0, dtype=torch.int32),
                    ]
                ).int()
                y = flash_attn_varlen_func(
                    q_var, k_var, v_var, cu_seqlens, cu_seqlens, q.size(1), k.size(1), causal=True
                )
        else:
            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
            if self.kv_cache is not None:
                k, v = self.kv_cache.update(input_pos, k, v)

            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)
        y = self.w2(F.silu(g) * u) + y

        return y


def all_reduce_func(x: torch.Tensor, clone: bool, async_op=False) -> torch.Tensor:
    if torch.compiler.is_compiling() or clone:
        x = funcol.all_reduce(x, reduceOp="sum", group=tp_group)
        handle = None
    else:
        handle = dist.all_reduce(x, async_op=async_op)

    return x, handle


def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [p.numel() * p.dtype.itemsize for p in itertools.chain(child.parameters(), child.buffers())]
            )
            params += sum([p.numel() for p in itertools.chain(child.parameters(), child.buffers())])
    return model_size, params
