# gpt_dense:70B
# gpt_ensemble:70B
# gpt_ensemble:70B-infinite
# gpt_residual:70B

# --comment_comm
# --turbo_mode None

# ${folder}/${model_name}_tpsize${tpsize}_bssize${bssize}_turbo${turbo_mode}_nonturbobefore${nonturbo_initial_layers}-after${nonturbo_final_layers}.log 2>&1

model_name=gpt_residual:llama-3-8b
turbo_mode=none
nonturbo_initial_layers=0
nonturbo_final_layers=0
folder=logs/8b/dist_comm/${turbo_mode}
mkdir -p ${folder}
for bssize in 1 4 16 32
do
    for tpsize in 1 2 4 8
    do
        echo "Running with bs=${bssize} tp=${tpsize}"
        TORCHDYNAMO_VERBOSE=1 CUDA_LAUNCH_BLOCKING=1 ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                        --model_name ${model_name} \
                                        --num_samples 10 \
                                        --batch_size ${bssize} \
                                        --prompt_length 1024 \
                                        --max_new_tokens 1024 \
                                        --device cuda \
                                        --compile \
                                        --compile_prefill \
                                        --turbo_mode ${turbo_mode} \
                                        --nonturbo_initial_layers ${nonturbo_initial_layers} \
                                        --nonturbo_final_layers ${nonturbo_final_layers} > ${folder}/${model_name}_tpsize${tpsize}_bssize${bssize}_turbo${turbo_mode}_nonturbobefore${nonturbo_initial_layers}-after${nonturbo_final_layers}.log 2>&1
        echo "Finished running with bs=${bssize} tp=${tpsize}"
    done
done    