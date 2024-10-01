# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

import torch.distributed as dist
from .tp import maybe_init_dist

from .utils import RMSNorm, precompute_freqs_cis, KVCache, Attention, FeedForward, all_reduce_func


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

maybe_init_dist()
tp_rank = dist.get_rank()
tp_world_size = dist.get_world_size()
tp_group = list(range(tp_world_size))


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
    semi_compiled_model: bool = False
    reduce_pattern: Optional[dict] = None
    force_disable_last_all_reduce: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

        assert self.dim % tp_world_size == 0
        assert self.intermediate_size % tp_world_size == 0

        if self.reduce_pattern is None:
            self.reduce_pattern = [{"attention": False, "mlp": True} for i in range(self.n_layer)]

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "70B-infinite": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672, reduce_pattern=[{"attention": False, "mlp": False} for _ in range(80)]),
    "70B-semi-compiled": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672, semi_compiled_model=True),
    "70B-infinite-semi-compiled": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672, semi_compiled_model=True, reduce_pattern=[{"attention": False, "mlp": False} for _ in range(80)]),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-8b-upper-bound": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000, reduce_pattern=[{"attention": False, "mlp": False} for _ in range(32)], force_disable_last_all_reduce=True),
    "llama-3-8b-infinite": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000, reduce_pattern=[{"attention": False, "mlp": False} for _ in range(32)]),
    
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "llama-3-70b-upper-bound": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000, reduce_pattern=[{"attention": False, "mlp": False} for _ in range(80)], force_disable_last_all_reduce=True),
    "llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
}

class GPTEnsemble(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(EnsembleTransformerBlock(config, layer_idx=i) for i in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

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
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads // tp_world_size, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype, self.config.rope_scaling)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class EnsembleTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

        self.do_attention_all_reduce = config.reduce_pattern[layer_idx]["attention"]
        self.do_mlp_all_reduce = layer_idx == config.n_layer - 1 or config.reduce_pattern[layer_idx]["mlp"]

        if layer_idx == config.n_layer - 1 and config.force_disable_last_all_reduce:
            self.do_mlp_all_reduce = False

        def _attn(x, freqs_cis, mask, input_pos):
            y = self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)

            if self.do_attention_all_reduce:
                y = y + x / tp_world_size
            else:
                y = y + x

            return y

        def _ffn(x):
            y = self.feed_forward(self.ffn_norm(x))

            if self.do_mlp_all_reduce:
                y = y + x / tp_world_size
            else:
                y = y + x

            return y

        self._attn = torch.compile(_attn)
        self._ffn = torch.compile(_ffn)

        self.semi_compiled_model = config.semi_compiled_model

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        if self.semi_compiled_model:
            x = self._attn(x, freqs_cis, mask, input_pos)

            if self.do_attention_all_reduce:
                x = all_reduce_func(x, clone=True)[0]
            else:
                x = x.clone()

            x = self._ffn(x)

            if self.do_mlp_all_reduce:
                x = all_reduce_func(x, clone=True)[0]
            else:
                x = x.clone()
        else:
            residual = x
            x = self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)

            if self.do_attention_all_reduce:
                x = x + residual / tp_world_size
                x = all_reduce_func(x, clone=False)[0]
            else:
                x = x + residual

            residual = x
            x = self.feed_forward(self.ffn_norm(x))

            if self.do_mlp_all_reduce:
                x = x + residual / tp_world_size
                x = all_reduce_func(x, clone=False)[0]
            else:
                x = x + residual

        return x

    def extra_repr(self) -> str:
        return f"semi_compiled = {self.semi_compiled_model}"
