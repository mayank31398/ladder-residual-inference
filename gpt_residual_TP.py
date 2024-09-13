# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple
from torch.distributed import _functional_collectives as funcol
import torch.distributed as dist
import os

import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from tp import maybe_init_dist

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

maybe_init_dist()
tp_world_size = dist.get_world_size()
tp_group = list(range(tp_world_size))


def all_reduce_func(x: torch.Tensor) -> torch.Tensor:
    return funcol.all_reduce(x, reduceOp="sum", group=tp_group)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]
        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
        
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
}

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


class GPTResidual(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TurboTransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.tp = False
        self.world_size = 1
        

        self.comment_attention = False
        self.comment_mlp = False
        self.comment_norm = False
        self.comment_comm = False
        self.all_reduce_stream = None
        
    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype, self.config.rope_scaling)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        
    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        # print(f'=== Current two stream status is {self.all_reduce_stream is not None} ===')
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        # This holds the output of all residual modules
        residual_hidden_states = [x]
        
        for i, layer in enumerate(self.layers):
            layer(input_pos, freqs_cis, mask, past_hidden_states=residual_hidden_states,
                attn_residual_flow=self.residual_flow[i*2],
                attn_compute_flow=self.compute_flow[i*2],
                mlp_residual_flow=self.residual_flow[i*2+1],
                mlp_compute_flow=self.compute_flow[i*2+1],
                use_tp=self.tp,
                world_size=self.world_size,
                comment_attention=self.comment_attention,
                comment_mlp=self.comment_mlp,
                comment_norm=self.comment_norm,
                comment_comm=self.comment_comm,
                all_reduce_stream=self.all_reduce_stream)
        # wait for last MLP's all-reduce to finish
        if self.all_reduce_stream is not None:
            self.all_reduce_stream.synchronize()
        x = residual_hidden_states[-1] # last attention + mlp(last mlp's output) as final output as v0
        if self.comment_norm == False:
            x = self.norm(x)
        logits = self.output(x)
        return logits

    def _inital_turbo_module(self, turbo_mode, nonturbo_initial_layers, nonturbo_final_layers, additional_non_turbo_modules):
        self.turbo_mode = turbo_mode
        
        # ===================== Non Turbo Module =====================
        self.nonturbo_initial_layers = nonturbo_initial_layers
        self.nonturbo_final_layers = nonturbo_final_layers
        self.additional_non_turbo_modules = additional_non_turbo_modules
        self.nonturbo_module = []

        for i in range(nonturbo_initial_layers):
            self.nonturbo_module.append(i*2)
            self.nonturbo_module.append(i*2+1)
            
        for i in range(self.nonturbo_final_layers):
            self.nonturbo_module.append((self.config.n_layer-i)*2)
            self.nonturbo_module.append((self.config.n_layer-i)*2+1)
        
        self.nonturbo_module += self.additional_non_turbo_modules
        
        # ===================== Turbo Module =====================
        self._build_compute_flow()
        
    def _build_compute_flow(self):
        self.compute_flow = []
        self.residual_flow = []
        # build compute_flow_matrix
        for i in range(self.config.n_layer):
            self.compute_flow.append(self._get_single_compute_flow(2*i))
            self.residual_flow.append(self._get_residual_flow(2*i))
            self.compute_flow.append(self._get_single_compute_flow(2*i + 1))
            self.residual_flow.append(self._get_residual_flow(2*i+1))
    
    def _get_single_compute_flow(self, i):
        flow = [0] * (i+1)
        if self.turbo_mode == "none" or i in self.nonturbo_module:
            flow[-1] = 1
        # layer-wise parallelization
        elif self.turbo_mode == "parallel-attn-mlp":
            # attn and mlp parallelization
            if i == 0:
                flow[-1] = 1
            elif i % 2 == 0:
                flow[-1] = 1
            else:
                flow[-2] = 1
        # module-wise skip-residual, overlap all of the all-reduce
        elif self.turbo_mode == "skip-residual-v0":
            if i == 0:
                flow[-1] = 1
            # all module takes the output before the previous module
            else:
                flow[-2] = 1
        # layer-wise skip-residual; pay half of the all-reduce
        elif self.turbo_mode == "skip-residual-v1":
            # mlp takes the correct input
            if i == 0 or i % 2 == 1:
                flow[-1] = 1
            # attn takes prev-prev layer's mlp as input
            else:
                flow[-3] = 1
        # layer-wise skip-residual plus attn/mlp parallelization, essentially making attn and mlp only need one all-reduce
        elif self.turbo_mode == "skip-residual-v2":
            if i == 0:
                flow[-1] = 1
            if i == 1:
                flow[-2] = 1
            # attn takes prev-prev layer's mlp as input
            elif i % 2 == 0:
                flow[-3] = 1
            # mlp takes previous layer mlp before previous layers as input
            else:
                flow[-4] = 1
        else:
            raise ValueError("Invalid turbo mode")
        return flow
    
    def _get_residual_flow(self, i):
        flow = [0] * (i+1)
        # Normal module has its previous output as the residual stream (same as input)
        if i in self.nonturbo_module:
            flow[-1] = 1
        else:
            flow[-1] = 1
        
        return flow
    
    @classmethod
    def from_name(cls, name: str, hidden_size: int = -1, vocab_size: int = -1, intermediate_size: int = -1):
        config = ModelArgs.from_name(name)
        if hidden_size != -1:
            config.dim = hidden_size
        if vocab_size != -1:
            config.vocab_size = vocab_size
        config.__post_init__()
        if intermediate_size != -1:
            config.intermediate_size = intermediate_size
        return cls(config)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class TurboTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor, 
        past_hidden_states: Optional[Tuple[torch.Tensor]] = (),
        attn_residual_flow = None,
        attn_compute_flow = None,
        mlp_residual_flow = None,
        mlp_compute_flow = None,
        use_tp = False,
        world_size = 1,
        comment_attention = False,
        comment_mlp = False,
        comment_norm = False,
        comment_comm = False,
        all_reduce_stream=None) -> Tensor:
        
        # ====================== Attention ======================
        # ========== compute connection ==========
        
        attn_inputs = []
        for weight, hid in zip(attn_compute_flow, past_hidden_states):
            if hid is None:
                # if weight > 0:
                #     print("=== warnings: Weight > 0 but hidden state is not stored")
                continue
            else:
                if weight > 0:
                    attn_inputs.append(weight * hid) 
        if len(attn_inputs) > 1:
            attn_input = torch.stack(attn_inputs, dim=0).sum(dim=0)
        else:
            attn_input = attn_inputs[0]
        # last layer mlp'communication happens here (last module) ==> overlap calculation and communication
        if comment_attention == True:
            # print("we do w/o attention")
            attn_out = attn_input
        elif comment_norm == True:
            # print("we do attention w/o norm")
            attn_out = self.attention(attn_input, freqs_cis, mask, input_pos)
        else:
            # print("we do full attention")
            attn_out = self.attention(self.attention_norm(attn_input), freqs_cis, mask, input_pos) 
        
        
        if all_reduce_stream is not None:
            all_reduce_stream.synchronize() # before calculation of residual we need mlp's all-reduce to finish
            #dist.barrier()
            
        if use_tp and comment_comm == False:
            if all_reduce_stream is not None:
                with torch.cuda.stream(all_reduce_stream):
                    attn_out[0] = all_reduce_func(attn_out[0])
            else:
                attn_out[0] = all_reduce_func(attn_out[0])
                # handle = dist.all_reduce(mlp_out, op=dist.ReduceOp.SUM, async_op=True)
                # handle.wait()
        
        # ========== residual connection ==========
        attn_residuals = []
        for weight, hid in zip(attn_residual_flow, past_hidden_states):
            if hid is None:
                # if weight > 0:
                #     print("=== warnings: Weight > 0 but hidden state is not stored===")
                continue
            else:
                if weight > 0:
                    attn_residuals.append(weight * hid) 
        
        if len(attn_residuals) > 1:
            residual = torch.stack(attn_residuals, dim=0).sum(dim=0)
        else:
            residual = attn_residuals[0]
        
        # this shall be done when the all-reduce is done
        if all_reduce_stream is not None:
            with torch.cuda.stream(all_reduce_stream):
                attn_out = residual + attn_out
                past_hidden_states.append(attn_out) # => needed by the mlp residual
        else:
            attn_out = residual + attn_out
            past_hidden_states.append(attn_out)
        
        # ===================== MLP =====================
        # ========== compute connection ==========
        mlp_inputs = []
        for weight, hid in zip(mlp_compute_flow, past_hidden_states):
            if hid is None:
                # if weight > 0:
                #     print("=== warnings: Weight > 0 but hidden state is not stored")
                continue
            else:
                if weight > 0:
                    mlp_inputs.append(weight * hid) 
        if len(mlp_inputs) > 1:
            mlp_input = torch.stack(mlp_inputs, dim=0).sum(dim=0)
        else:
            mlp_input = mlp_inputs[0]
        if comment_mlp == True:
            # print("we do w/o mlp")
            mlp_out = mlp_input
        elif comment_norm == True:
            # print("we do mlp w/o norm")
            mlp_out = self.feed_forward(mlp_input)
        else:
            # print("we do full mlp")
            mlp_out = self.feed_forward(self.ffn_norm(mlp_input)) 
        
        if all_reduce_stream is not None:
            all_reduce_stream.synchronize() # before calculation of residual we need attention's all-reduce to finish
            #dist.barrier()
        
        if use_tp and comment_comm == False:
            if all_reduce_stream is not None:
                with torch.cuda.stream(all_reduce_stream):
                    mlp_out = all_reduce_func(mlp_out)
            else:
                mlp_out = all_reduce_func(mlp_out)
                # handle = dist.all_reduce(mlp_out, op=dist.ReduceOp.SUM, async_op=True)
                # handle.wait()
        
        # ========== residual connection ==========
        mlp_residuals = []
        for weight, hid in zip(mlp_residual_flow, past_hidden_states):
            if hid is None:
                if weight > 0:
                    print("=== warnings: Weight > 0 but hidden state is not stored")
                continue
            else:
                if weight > 0:
                    mlp_residuals.append(weight * hid) 
        if len(mlp_residuals) > 1:
            residual = torch.stack(mlp_residuals, dim=0).sum(dim=0)
        else:
            residual = mlp_residuals[0]
        
        
        if all_reduce_stream is not None:
            with torch.cuda.stream(all_reduce_stream):
                mlp_out = residual + mlp_out
                past_hidden_states.append(mlp_out)
        else:
            mlp_out = residual + mlp_out
            past_hidden_states.append(mlp_out)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
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

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)

        return y



class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        print('our tp world size is', tp_world_size)
        assert config.intermediate_size % tp_world_size == 0
        assert config.dim % tp_world_size == 0

        self.w1 = nn.Linear(config.dim, 2 * config.intermediate_size // tp_world_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size // tp_world_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        u, g = x.chunk(2, dim=-1)
        y = self.w2(F.silu(g) * u)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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