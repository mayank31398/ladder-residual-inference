# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

import argparse

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from gpt_dolomite_TP import Transformer


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


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


def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
    seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    return seq


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(model_name, device, precision):
    with torch.device('meta'):
        model = Transformer.from_name(model_name)

    model = model.to(dtype=precision)
    model = model.to_empty(device=device)

    for p in model.parameters():
        torch.nn.init.normal_(p, mean=0, std=0.02)

    print(model)

    return model.eval()


def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

B_INST, E_INST = "[INST]", "[/INST]"

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
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """

    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(model_name, device, precision)

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # generate a fully synthetic prompt
    encoded = torch.randint(0, 1024, (prompt_length,), device=device, dtype=torch.int64)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)
    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {'tokens_per_sec': []}
    start = -5

    for i in range(start, num_samples):
        device_sync(device=device) # MKG

        callback = lambda x : x
        t0 = time.perf_counter()

        import contextlib

        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()

        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )

        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue

        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")

        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        total_tokens_sec = y.numel() / t
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print()

    print("==========")

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


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
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')

    args = parser.parse_args()
    main(
        args.model_name, args.prompt_length, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.temperature, args.compile, args.compile_prefill, args.profile, args.device
    )
