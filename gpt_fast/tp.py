# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

if os.uname().sysname != "Darwin":
    from torch.distributed import _functional_collectives as funcol
else:
    # Distributed is not supported on MacOS
    funcol = None

from datetime import timedelta

from model import Attention, FeedForward, Transformer
from quantize import WeightOnlyInt4Linear


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_local():
    return _get_rank() == 0


def local_break():
    if is_local():
        breakpoint()
    dist.barrier()


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
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))

    return rank
