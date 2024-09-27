export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
NCCL_P2P_DISABLE=0 NCCL_DEBUG=INFO torchrun --nproc_per_node=8 comm_benchmark.py > enable.log