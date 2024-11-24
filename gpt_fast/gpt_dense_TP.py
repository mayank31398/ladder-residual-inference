# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .parallel import ProcessGroupManager
from .utils import Attention, FeedForward, KVCache, RMSNorm, all_reduce_func, precompute_freqs_cis


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


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

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        assert self.dim % tp_world_size == 0
        assert self.intermediate_size % tp_world_size == 0

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000),
    "1b": dict(
        block_size=2048,
        n_layer=40,
        n_head=24,
        n_local_heads=24,
        dim=1536,
        intermediate_size=4096,
        vocab_size=49152,
        rope_base=10000,
    ),
    "3b": dict(
        block_size=2048,
        n_layer=40,
        n_head=36,
        n_local_heads=36,
        dim=2304,
        intermediate_size=9216,
        vocab_size=49152,
        rope_base=10000,
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "70B-semi-compiled": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672, semi_compiled_model=True
    ),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "llama-3-8b-4layers": dict(
        block_size=8192,
        n_layer=4,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3-8b": dict(
        block_size=8192,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3-70b": dict(
        block_size=8192,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3.1-405b": dict(
        block_size=131072,
        n_layer=126,
        n_head=128,
        n_local_heads=8,
        dim=16384,
        intermediate_size=53248,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192
        ),
    ),
}


class GPTDense(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        if ProcessGroupManager.get_pipeline_parallel_rank() == 0:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        assert config.n_layer % ProcessGroupManager.get_pipeline_parallel_world_size() == 0

        self.layers = nn.ModuleList(
            DenseTransformerBlock(config)
            for _ in range(config.n_layer // ProcessGroupManager.get_pipeline_parallel_world_size())
        )

        if (
            ProcessGroupManager.get_pipeline_parallel_rank()
            == ProcessGroupManager.get_pipeline_parallel_world_size() - 1
        ):
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
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads // ProcessGroupManager.get_tensor_parallel_world_size(),
                head_dim,
                dtype,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            dtype,
            self.config.rope_scaling,
        )
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]

        pp_rank = ProcessGroupManager.get_pipeline_parallel_rank()

        if pp_rank == 0:
            x = self.tok_embeddings(x)

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)

        if pp_rank == ProcessGroupManager.get_pipeline_parallel_world_size() - 1:
            x = self.norm(x)
            x = self.output(x)

        return x

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class DenseTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

        is_tp_first_rank = ProcessGroupManager.is_tensor_parallel_first_rank()

        def _attn(x, residual, freqs_cis, mask, input_pos):
            x = self.attention_norm(x)
            x = self.attention(x, freqs_cis, mask, input_pos)

            if is_tp_first_rank:
                x = x + residual

            return x

        def _ffn(x, residual):
            x = self.ffn_norm(x)
            x = self.feed_forward(x)

            if is_tp_first_rank:
                x = x + residual

            return x

        self._attn = torch.compile(_attn)
        self._ffn = torch.compile(_ffn)

        self.semi_compiled_model = config.semi_compiled_model

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        if self.semi_compiled_model:
            x = self._attn(x, x, freqs_cis, mask, input_pos)
            x = all_reduce_func(x, clone=True)[0]
            x = self._ffn(x, x)
            x = all_reduce_func(x, clone=True)[0]
        else:
            x = x + all_reduce_func(self.attention(self.attention_norm(x), freqs_cis, mask, input_pos), clone=False)[0]
            x = x + all_reduce_func(self.feed_forward(self.ffn_norm(x)), clone=False)[0]

        return x

    def extra_repr(self) -> str:
        return f"semi_compiled = {self.semi_compiled_model}"
