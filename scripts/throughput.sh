# gpt_dense:70B
# gpt_ensemble:70B
# gpt_ensemble:70B-infinite
# gpt_residual:70B

# ${folder}/${model_name}_tpsize${tpsize}_bssize${bssize}_turbo${turbo_mode}_nonturbobefore${nonturbo_initial_layers}-after${nonturbo_final_layers}.log 2>&1
# turbo_mode=none
# nonturbo_initial_layers=0
# nonturbo_final_layers=0
# --turbo_mode ${turbo_mode} \
# --nonturbo_initial_layers ${nonturbo_initial_layers} \
# --nonturbo_final_layers ${nonturbo_final_layers} > ${folder}/${model_name}_tpsize${tpsize}_bssize${bssize}_turbo${turbo_mode}_nonturbobefore${nonturbo_initial_layers}-after${nonturbo_final_layers}.log 2>&1

model_name=gpt_ladder:llama-3-8b
folder=./
mkdir -p ${folder}
for bssize in 16
do
    for tpsize in 2
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
                                        --compile_prefill  > tmp.log 2>&1
        echo "Finished running with bs=${bssize} tp=${tpsize}"
    done
done