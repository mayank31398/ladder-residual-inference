# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def __get_global_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def _get_global_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_local():
    return _get_rank() == 0


def local_break():
    if is_local():
        breakpoint()
    dist.barrier()


def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        global_rank = __get_global_rank()
        world_size = _get_world_size()
        global_world_size = _get_global_world_size()
        print(
            f"rank: {rank}, global_rank: {global_rank}, world_size: {world_size}, global_world_size: {global_world_size}"
        )
    except KeyError:
        # not run via torchrun, no-op
        return None

    if not dist.is_initialized():
        torch.cuda.set_device(rank)
        if global_world_size > 1:
            dist.init_process_group(
                backend="nccl", rank=global_rank, world_size=global_world_size, timeout=timedelta(seconds=60)
            )
        else:
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))

    return rank
