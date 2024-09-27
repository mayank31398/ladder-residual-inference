import torch
import torch.distributed as dist
import time
from typing import Optional
import os

def print_rank_0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()
        
    except KeyError:
        # not run via torchrun, no-op
        return None

    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank

""" Initialize distributed environment """
rank = maybe_init_dist()

""" Test NCCL all_reduce latency """
# Create a tensor filled with ones
tensor_size = 1024 * 1024  # Size of the tensor (1M elements)
tensor = torch.ones(tensor_size).cuda(rank)

# Warm up
for _ in range(10):
    dist.all_reduce(tensor)

# Timing the latency
num_iters = 100
start_time = time.time()

for _ in range(num_iters):
    dist.all_reduce(tensor)

end_time = time.time()
avg_latency = (end_time - start_time) / num_iters

if rank == 0:
    print_rank_0(f"Average all_reduce latency over {num_iters} iterations: {avg_latency * 1000:.4f} ms")


