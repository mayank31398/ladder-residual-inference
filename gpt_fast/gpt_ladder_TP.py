# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor
from torch.distributed._functional_collectives import AsyncCollectiveTensor

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
    "llama-3-8b-semi-compiled": dict(
        block_size=8192,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
        semi_compiled_model=True,
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
    "llama-3-70b-semi-compiled": dict(
        block_size=8192,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
        semi_compiled_model=True,
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


class GPTLadder(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        if ProcessGroupManager.get_pipeline_parallel_rank() == 0:
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        assert config.n_layer % ProcessGroupManager.get_pipeline_parallel_world_size() == 0

        self.layers = nn.ModuleList(
            LadderTransformerBlock(config)
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

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

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

    def forward(
        self,
        x: Tensor,
        previous_attention_out: Optional[Tensor] = None,
        previous_mlp_out: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> tuple[Tensor]:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]

        pp_rank = ProcessGroupManager.get_pipeline_parallel_rank()

        if pp_rank == 0:
            x = self.tok_embeddings(x)

            previous_attention_out = torch.zeros_like(x)
            previous_mlp_out = torch.zeros_like(x)

        attention_handle = None
        mlp_handle = None

        for layer in self.layers:
            previous_attention_out, previous_mlp_out, x, attention_handle, mlp_handle = layer(
                previous_attention_out,
                previous_mlp_out,
                x,
                attention_handle,
                mlp_handle,
                input_pos,
                freqs_cis,
                mask,
            )

        if attention_handle is not None:
            attention_handle.wait()

        if mlp_handle is not None:
            mlp_handle.wait()

        if pp_rank == ProcessGroupManager.get_pipeline_parallel_world_size() - 1:
            x = x + previous_attention_out + previous_mlp_out
            x = self.norm(x)
            x = self.output(x)
            return x
        else:
            if isinstance(previous_attention_out, AsyncCollectiveTensor):
                previous_attention_out.wait()

            if isinstance(previous_mlp_out, AsyncCollectiveTensor):
                previous_mlp_out.wait()

            return x, previous_attention_out, previous_attention_out

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class LadderTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

        def _attn(x, freqs_cis, mask, input_pos):
            current_attention_out = self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
            return current_attention_out

        def _ffn(x):
            current_mlp_out = self.feed_forward(self.ffn_norm(x))
            return current_mlp_out

        self.semi_compiled_model = config.semi_compiled_model
        if self.semi_compiled_model:
            self._attn = torch.compile(_attn)
            self._ffn = torch.compile(_ffn)

    def forward(
        self,
        previous_attention_out: Tensor,
        previous_mlp_out: Tensor,
        residual: Tensor,
        attention_handle,
        mlp_handle,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if attention_handle is not None:
            attention_handle.wait()

        numel = residual.numel()
        grid = (triton.cdiv(numel, 1024),)

        output = torch.empty_like(residual)
        with torch.device(residual.device):
            add_tensor_forward_triton_kernel[grid](residual, previous_attention_out, output, numel, 1024)
        residual = output

        if self.semi_compiled_model:
            current_attention_out = self._attn(residual, freqs_cis, mask, input_pos)
        else:
            current_attention_out = self.attention(self.attention_norm(residual), freqs_cis, mask, input_pos)

        current_attention_out, attention_handle = all_reduce_func(
            current_attention_out, clone=self.semi_compiled_model, async_op=True
        )

        if mlp_handle is not None:
            mlp_handle.wait()

        output = torch.empty_like(residual)
        with torch.device(residual.device):
            add_tensor_forward_triton_kernel[grid](residual, previous_mlp_out, output, numel, 1024)
        residual = output

        if self.semi_compiled_model:
            current_mlp_out = self._ffn(residual)
        else:
            current_mlp_out = self.feed_forward(self.ffn_norm(residual))

        current_mlp_out, mlp_handle = all_reduce_func(current_mlp_out, clone=self.semi_compiled_model, async_op=True)

        return current_attention_out, current_mlp_out, residual, attention_handle, mlp_handle

    def extra_repr(self) -> str:
        return f"semi_compiled = {self.semi_compiled_model}"


@triton.jit
def add_tensor_forward_triton_kernel(x_ptr, y_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements

    x = tl.load(x_ptr + indices, mask=mask)
    y = tl.load(y_ptr + indices, mask=mask)

    output = x + y

    tl.store(output_ptr + indices, output, mask=mask)
