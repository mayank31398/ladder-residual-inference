# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.distributed as dist
from gpt_fast.utils import set_flash_attention, _get_model_size

def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

import argparse

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
# torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from gpt_fast import GPTDense, GPTEnsemble, GPTParallel, GPTLadder


_MODELS = {
    "gpt_dense": GPTDense,
    "gpt_ensemble": GPTEnsemble,
    "gpt_parallel": GPTParallel,
    "gpt_ladder": GPTLadder,
}


def device_sync(device):
    device = str(device)
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print_rank_0(f"device={device} is not yet suppported")


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

@torch.no_grad()
def prefill(model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)
    
@torch.no_grad()
def decode_n_tokens(
    model: torch.nn.Module,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    use_flash_attention: bool,
    callback=lambda _: _,
    **sampling_kwargs
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        # Actually better for Inductor to codegen attention here
        with (
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True),
            set_flash_attention(use_flash_attention),
        ):
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    return new_tokens, new_probs

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    empty: torch.Tensor,
    *,
    use_flash_attention,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    T = prompt.size(-1)
    device = prompt.device
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    input_pos = torch.arange(0, T, device=device)

    device_sync(device)
    prefill_start = time.perf_counter()
    with (
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True),
            set_flash_attention(use_flash_attention),
        ):
        next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    device_sync(device)
    prefill_latency = time.perf_counter() - prefill_start
    print_rank_0(f"Prefill latency: {prefill_latency} sec")

    next_token = next_token.clone()
    empty[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    device_sync(device)
    decode_start = time.perf_counter()
    generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, use_flash_attention=use_flash_attention, callback=callback, **sampling_kwargs)
    device_sync(device)
    decode_latency = time.perf_counter() - decode_start
    print_rank_0(f"Decode latency: {decode_latency} sec")

    empty[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    return empty, decode_latency, prefill_latency


@torch.no_grad()
def generate_using_cuda_graphs(
    prefill_graph,
    static_x: torch.Tensor,
    static_input_pos: torch.Tensor,
    static_next_token_prefill: torch.Tensor,
    decode_graph,
    static_cur_token: torch.Tensor,
    static_decode_input_pos: torch.Tensor,
    static_next_token_decode: torch.Tensor,
    prompt: torch.Tensor,
    batch_size: int,
    empty: torch.Tensor,
    num_new_tokens: int,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    T = prompt.size(-1)
    device = prompt.device

    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt

    static_x.copy_(prompt)
    static_input_pos.copy_(torch.arange(0, T, device=device))

    device_sync(device)
    prefill_start = time.perf_counter()
    prefill_graph.replay()
    torch.cuda.synchronize()
    prefill_latency = time.perf_counter() - prefill_start
    print_rank_0(f"Prefill latency: {prefill_latency} sec")

    empty[:, T] = static_next_token_prefill.squeeze()

    device_sync(device)
    decode_start = time.perf_counter()

    static_cur_token.copy_(static_next_token_prefill)
    static_decode_input_pos.copy_(torch.tensor([T], device=device, dtype=torch.int))

    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens - 1):
        decode_graph.replay()
        static_decode_input_pos += 1

        new_tokens.append(static_next_token_decode.clone())
        static_cur_token.copy_(static_next_token_decode.clone())

    torch.cuda.synchronize()
    decode_latency = time.perf_counter() - decode_start
    print_rank_0(f"Decode latency: {decode_latency} sec")

    empty[:, T + 1:] = torch.cat(new_tokens, dim=-1)

    return empty, decode_latency, prefill_latency


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(model_name, device, precision):
    with torch.device('meta'):
        model = _MODELS[model_name.split(":")[0]].from_name(model_name.split(":")[1])

    model = model.to(dtype=precision)
    model = model.to_empty(device=device)

    for p in model.parameters():
        torch.nn.init.normal_(p, mean=0, std=0.02)

    print_rank_0(model)

    return model.eval()

B_INST, E_INST = "[INST]", "[/INST]"

@torch.no_grad()
def get_cuda_graphs_for_prefill(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    batch_size: int,
    **sampling_kwargs
):
    T = prompt.size(-1)
    device = prompt.device

    # We are just making the same prompt for every batch
    static_x = prompt.view(1, -1).repeat(batch_size, 1)
    static_input_pos = torch.arange(0, T, device=device)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        for _ in range(3):
            static_next_token = prefill(model, static_x.view(batch_size, -1), static_input_pos, **sampling_kwargs)

    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_next_token = prefill(model, static_x.view(batch_size, -1), static_input_pos, **sampling_kwargs)

    return g, static_x, static_input_pos, static_next_token


@torch.no_grad()
def get_cuda_graphs_for_decode(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    batch_size: int,
    max_new_tokens: int,
    cur_token: torch.Tensor,
    use_flash_attention: bool,
    **sampling_kwargs
):
    T = prompt.size(-1)
    device = prompt.device

    static_cur_token = cur_token.clone()
    static_input_pos = torch.tensor([T], device=device, dtype=torch.int)

    # Warm up
    for _ in range(3):
        static_input_pos.copy_(torch.tensor([T], device=device, dtype=torch.int))

        decode_one_token(model, static_cur_token, static_input_pos, **sampling_kwargs)

        # decode_n_tokens(
        #     model,
        #     static_cur_token.view(batch_size, -1),
        #     static_input_pos,
        #     max_new_tokens - 1,
        #     use_flash_attention=use_flash_attention,
        #     **sampling_kwargs
        # )

    static_input_pos.copy_(torch.tensor([T], device=device, dtype=torch.int))

    # Capture CUDA graph
    g_decode = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_decode):
        static_next_token, _ = decode_one_token(model, static_cur_token, static_input_pos, **sampling_kwargs)

        # static_generated_tokens, _ = decode_n_tokens(
        #     model,
        #     static_cur_token.view(batch_size, -1),
        #     static_input_pos,
        #     max_new_tokens - 1,
        #     use_flash_attention=use_flash_attention,
        #     **sampling_kwargs
        # )

    return g_decode, static_cur_token, static_input_pos, static_next_token


def main(
    model_name: str,
    prompt_length: int = 1,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    device=default_device,
    use_cuda_graphs: bool = False,
    use_flash_attention: bool = False,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """

    print_rank_0(f"flash_kv_decode is set to {use_flash_attention}")

    from gpt_fast import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None

    print_rank_0(f"Using device={device}")
    precision = torch.float16

    print_rank_0("Loading model ...")
    t0 = time.time()
    model = _load_model(model_name, device, precision)

    device_sync(device=device) # MKG
    print_rank_0(f"Time to load model: {time.time() - t0:.02f} seconds")

    # generate a fully synthetic prompt
    encoded = torch.randint(0, 1024, (prompt_length,), device=device, dtype=torch.int64)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)

    T_new = encoded.size(-1) + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    if compile:
        global decode_one_token, decode_multi_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
    
        if compile_prefill:
            dynamic = False
            print_rank_0(f"Compiling prefill with dynamic={dynamic}")
            prefill = torch.compile(prefill, fullgraph=True, dynamic=dynamic)
            
    elif use_cuda_graphs:
        prefill_graph, static_x, static_input_pos, static_next_token_prefill = get_cuda_graphs_for_prefill(
            model,
            prompt=encoded,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
        )

        decode_graph, static_cur_token, static_decode_input_pos, static_next_token_decode = get_cuda_graphs_for_decode(
            model,
            prompt=encoded,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            cur_token=static_next_token_prefill,
            use_flash_attention=use_flash_attention,
            temperature=temperature,
            top_k=top_k,
        )

    aggregate_metrics = {'tokens_per_sec': [], 'decode_latency': [], 'prefill_latency': []}
    start = -5

    for i in range(start, num_samples):
        device_sync(device=device) # MKG

        callback = lambda x : x
        t0 = time.perf_counter()

        import contextlib

        if not profile or (use_tp and rank != 0) or i != num_samples - 1:
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile),
                record_shapes=True,
            )

        empty = torch.empty(batch_size, T_new, dtype=encoded.dtype, device=device)

        with prof:
            if use_cuda_graphs:
                # NOTE we need to reset the static variable pointers for CUDA graph on each geenration here
                # however, for benchmarking throughput, it doesn't matter
                y, decode_latency, prefill_latency = generate_using_cuda_graphs(
                    prefill_graph,
                    static_x,
                    static_input_pos,
                    static_next_token_prefill,
                    decode_graph,
                    static_cur_token,
                    static_decode_input_pos,
                    static_next_token_decode,
                    encoded,
                    batch_size=batch_size,
                    empty=empty,
                    num_new_tokens=max_new_tokens,
                )
            else:
                y, decode_latency, prefill_latency = generate(
                    model,
                    encoded,
                    max_new_tokens,
                    batch_size=batch_size,
                    empty=empty,
                    use_flash_attention=use_flash_attention,
                    callback=callback,
                    temperature=temperature,
                    top_k=top_k,
                )

        if i == -5:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

        if i < 0:
            continue

        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        tokens_generated = (y.size(-1) - prompt_length)*y.size(0)
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        aggregate_metrics['decode_latency'].append(decode_latency)
        aggregate_metrics['prefill_latency'].append(prefill_latency)
        print_rank_0(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print_rank_0(f"Decode latency: {decode_latency:.02f} sec")
        print_rank_0(f"Prefill latency: {prefill_latency:.02f} sec")
        print_rank_0(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        total_tokens_sec = y.numel() / t
        print_rank_0(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print_rank_0()

    print_rank_0("==========")

    print_rank_0(f"Batch Size: {batch_size}")
    print_rank_0(f"Prompt Length: {prompt_length}")
    print_rank_0(f"Generated tokens: {max_new_tokens}")
    print_rank_0(f"Average decode latency: {torch.mean(torch.tensor(aggregate_metrics['decode_latency'])).item():.04f} sec")
    print_rank_0(f"Average prefill latency: {torch.mean(torch.tensor(aggregate_metrics['prefill_latency'])).item():.04f} sec")
    print_rank_0(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print_rank_0(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    dist.barrier()
    dist.destroy_process_group()
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_name', type=str, required=True, help="model name")
    parser.add_argument('--prompt_length', type=int, required=True, help="Input prompt length")
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--cuda_graph', action='store_true', help='Whether to use cuda graphs the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--use_flash_attention', action='store_true', help='Whether to flash decode with kv cache in attn (not compile generated one)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')

    args = parser.parse_args()

    if args.cuda_graph:
        assert not args.compile

    main(
        args.model_name, args.prompt_length, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.temperature, args.compile, args.compile_prefill, args.profile, args.device, args.cuda_graph, args.use_flash_attention
    )
