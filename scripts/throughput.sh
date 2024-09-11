# gpt_dense:70B
# gpt_ensemble:70B
# gpt_ensemble:70B-infinite


for tpsize in 8
do
    #echo "Running with dim=${h}"
    echo "Running with tp=${tpsize}"
    NCCL_SOCKET_IFNAME=eth0 TORCHDYNAMO_VERBOSE=1 CUDA_LAUNCH_BLOCKING=1 ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                    --model_name gpt_residual:8B \
                                    --profile /home/charlie/gpt-fast/profiles/different_tp_without_comm/without_layer_norm/tpsize \
                                    --num_samples 16 \
                                    --batch_size 4 \
                                    --prompt_length 1024 \
                                    --max_new_tokens 1024 \
                                    --device cuda \
                                    --compile \
                                    --compile_prefill \
                                    --turbo_mode skip-residual-v0 \
                                    --nonturbo_initial_layers 1 \
                                    --nonturbo_final_layers 0 > tmp.log
    #echo "Finished running with dim=${h}"
    echo "Finished running with tp=${tpsize}"
done

