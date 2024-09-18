import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import math
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
import torch.distributed as dist
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from .tp import maybe_init_dist


maybe_init_dist()
tp_rank = dist.get_rank()
tp_world_size = dist.get_world_size()
tp_group = list(range(tp_world_size))

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
    seq_len: int, n_elem: int, base: int = 10000,
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

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim

        assert total_head_dim % tp_world_size == 0
        assert config.dim % tp_world_size == 0

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

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None, compile=False) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if compile:
            if self.kv_cache is not None:
                k, v = self.kv_cache.update(input_pos, k, v)
            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            q = q.transpose(1, 2)
            q_len = q.size(-2)
            k = k[..., :q_len, :]
            v = v[..., :q_len, :]
            
            if q_len <= 10:
                k_cache = self.kv_cache.k_cache
                v_cache = self.kv_cache.v_cache
                cache_seqlens = k_cache.size(-2)

                y = flash_attn_with_kvcache(
                    q,                      # (batch_size, n_heads, seqlen_q, head_dim)
                    k_cache,                # (batch_size, seqlen_cache, n_local_heads, head_dim)
                    v_cache,
                    k=k,                    # (batch_size, seqlen_new, n_local_heads, head_dim)
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
                self.kv_cache.k_cache = k_cache
                self.kv_cache.v_cache = v_cache
            else:
                if self.kv_cache is not None:
                    k, v = self.kv_cache.update(input_pos, k, v)
                y = flash_attn_func(q, k, v, causal=True)
                
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)

        return y
