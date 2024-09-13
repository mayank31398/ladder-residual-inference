# gpt_dense:70B
# gpt_ensemble:70B
# gpt_ensemble:70B-infinite
# gpt_residual:70B

model_name=gpt_residual:llama-3-8b
turbo_mode=skip-residual-v0
nonturbo_initial_layers=1
nonturbo_final_layers=0
mkdir -p logs
for bssize in 1
do
    for tpsize in 1
    do
        echo "Running with bs=${bssize} tp=${tpsize}"
        NCCL_SOCKET_IFNAME=eth0 TORCHDYNAMO_VERBOSE=1 CUDA_LAUNCH_BLOCKING=1 ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                        --model_name ${model_name} \
                                        --profile /home/charlie/gpt-fast/profiles/different_tp_without_comm/without_layer_norm/tpsize \
                                        --num_samples 16 \
                                        --batch_size ${bssize} \
                                        --prompt_length 1024 \
                                        --max_new_tokens 1024 \
                                        --device cuda \
                                        --compile \
                                        --compile_prefill \
                                        --turbo_mode ${turbo_mode} \
                                        --nonturbo_initial_layers ${nonturbo_initial_layers} \
                                        --nonturbo_final_layers ${nonturbo_final_layers} > logs/${model_name}_tpsize${tpsize}_bssize${bssize}_turbo${turbo_mode}_nonturbobefore${nonturbo_initial_layers}-after${nonturbo_final_layers}.log 2>&1
        echo "Finished running with bs=${bssize} tp=${tpsize}"
    done
done

