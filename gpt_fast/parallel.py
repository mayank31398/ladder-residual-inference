import os
from datetime import timedelta

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


# general
_MESH: DeviceMesh | None = None
_GLOBAL_RANK: int | None = None
_LOCAL_RANK: int | None = None
_WORLD_SIZE: int | None = None

# tensor parallel
_TENSOR_PARALLEL_MESH: DeviceMesh | None = None
_TENSOR_PARALLEL_GROUP: ProcessGroup | None = None
_TENSOR_PARALLEL_RANK: int | None = None
_TENSOR_PARALLEL_WORLD_SIZE: int | None = None
_TENSOR_PARALLEL_FIRST_RANK: int | None = None

# pipeline parallel
_PIPELINE_PARALLEL_MESH: DeviceMesh | None = None
_PIPELINE_PARALLEL_GROUP: ProcessGroup | None = None
_PIPELINE_PARALLEL_RANK: int | None = None
_PIPELINE_PARALLEL_WORLD_SIZE: int | None = None


class ProcessGroupManager:
    def __init__(
        self,
        tensor_parallel_world_size: int = 1,
        pipeline_parallel_world_size: int = 1,
        timeout_minutes: int | None = None,
        use_async_tensor_parallel: bool = False,
    ) -> None:
        if timeout_minutes is not None:
            timeout_minutes = timedelta(timeout_minutes)

        torch.distributed.init_process_group(
            rank=ProcessGroupManager.get_global_rank(),
            world_size=ProcessGroupManager.get_world_size(),
            timeout=timeout_minutes,
        )

        total_gpus = int(os.getenv("WORLD_SIZE", 1))
        assert tensor_parallel_world_size * pipeline_parallel_world_size == total_gpus

        global _MESH
        _MESH = init_device_mesh("cuda", (pipeline_parallel_world_size, tensor_parallel_world_size), mesh_dim_names=("pp", "tp"))

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        if use_async_tensor_parallel:
            enable_symm_mem_for_group(ProcessGroupManager.get_tensor_parallel_group().group_name)
            torch._inductor.config._micro_pipeline_tp = True

    @staticmethod
    def is_initialized() -> bool:
        return torch.distributed.is_initialized()

    @staticmethod
    def get_mesh() -> DeviceMesh:
        global _MESH
        return _MESH

    @staticmethod
    def get_global_rank() -> int:
        global _GLOBAL_RANK

        if _GLOBAL_RANK is None:
            _GLOBAL_RANK = int(os.getenv("RANK", 0))
        return _GLOBAL_RANK

    @staticmethod
    def get_local_rank() -> int:
        global _LOCAL_RANK

        if _LOCAL_RANK is None:
            _LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
        return _LOCAL_RANK

    @staticmethod
    def get_world_size() -> int:
        global _WORLD_SIZE

        if _WORLD_SIZE is None:
            _WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
        return _WORLD_SIZE

    # tensor parallel
    @staticmethod
    def get_tensor_parallel_mesh() -> DeviceMesh:
        global _TENSOR_PARALLEL_MESH

        if _TENSOR_PARALLEL_MESH is None:
            _TENSOR_PARALLEL_MESH = ProcessGroupManager.get_mesh()["tp"]
        return _TENSOR_PARALLEL_MESH

    @staticmethod
    def get_tensor_parallel_group() -> ProcessGroup:
        global _TENSOR_PARALLEL_GROUP

        if _TENSOR_PARALLEL_GROUP is None:
            _TENSOR_PARALLEL_GROUP = ProcessGroupManager.get_tensor_parallel_mesh().get_group()
        return _TENSOR_PARALLEL_GROUP

    @staticmethod
    def get_tensor_parallel_rank() -> int:
        global _TENSOR_PARALLEL_RANK

        if _TENSOR_PARALLEL_RANK is None:
            _TENSOR_PARALLEL_RANK = ProcessGroupManager.get_tensor_parallel_mesh().get_local_rank()
        return _TENSOR_PARALLEL_RANK

    @staticmethod
    def get_tensor_parallel_world_size() -> int:
        global _TENSOR_PARALLEL_WORLD_SIZE

        if _TENSOR_PARALLEL_WORLD_SIZE is None:
            _TENSOR_PARALLEL_WORLD_SIZE = ProcessGroupManager.get_tensor_parallel_mesh().size()
        return _TENSOR_PARALLEL_WORLD_SIZE

    @staticmethod
    def get_tensor_parallel_first_rank() -> int:
        global _TENSOR_PARALLEL_FIRST_RANK

        if _TENSOR_PARALLEL_FIRST_RANK is None:
            group = ProcessGroupManager.get_tensor_parallel_group()
            ranks = torch.distributed.get_process_group_ranks(group)
            _TENSOR_PARALLEL_FIRST_RANK = ranks[0]
        return _TENSOR_PARALLEL_FIRST_RANK

    @staticmethod
    def is_tensor_parallel_enabled() -> bool:
        return ProcessGroupManager.get_tensor_parallel_world_size() > 1

    @staticmethod
    def is_tensor_parallel_first_rank() -> bool:
        return ProcessGroupManager.get_tensor_parallel_rank() == 0

    # pipeline parallel
    @staticmethod
    def get_pipeline_parallel_mesh() -> DeviceMesh:
        global _PIPELINE_PARALLEL_MESH

        if _PIPELINE_PARALLEL_MESH is None:
            _PIPELINE_PARALLEL_MESH = ProcessGroupManager.get_mesh()["pp"]
        return _PIPELINE_PARALLEL_MESH

    @staticmethod
    def get_pipeline_parallel_group() -> ProcessGroup:
        global _PIPELINE_PARALLEL_GROUP

        if _PIPELINE_PARALLEL_GROUP is None:
            _PIPELINE_PARALLEL_GROUP = ProcessGroupManager.get_pipeline_parallel_mesh().get_group()
        return _PIPELINE_PARALLEL_GROUP

    @staticmethod
    def get_pipeline_parallel_rank() -> int:
        global _PIPELINE_PARALLEL_RANK

        if _PIPELINE_PARALLEL_RANK is None:
            _PIPELINE_PARALLEL_RANK = ProcessGroupManager.get_pipeline_parallel_mesh().get_local_rank()
        return _PIPELINE_PARALLEL_RANK

    @staticmethod
    def get_pipeline_parallel_world_size() -> int:
        global _PIPELINE_PARALLEL_WORLD_SIZE

        if _PIPELINE_PARALLEL_WORLD_SIZE is None:
            _PIPELINE_PARALLEL_WORLD_SIZE = ProcessGroupManager.get_pipeline_parallel_mesh().size()
        return _PIPELINE_PARALLEL_WORLD_SIZE

    def __str__(self) -> str:
        return str(self.get_mesh())

    @staticmethod
    def destroy_process_groups() -> None:
        if ProcessGroupManager.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()


def is_tracking_rank() -> bool:
    return (
        ProcessGroupManager.is_tensor_parallel_first_rank()
        and ProcessGroupManager.get_pipeline_parallel_rank()
        == ProcessGroupManager.get_pipeline_parallel_world_size() - 1
    )
