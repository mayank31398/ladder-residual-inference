

# \
#                                         --compile \
#                                         --compile_prefill

# --use_flash_attention \
# -semi-compiled

model_name="gpt_dense:llama-3-8b"
folder=./logs/tmp/
mkdir -p ${folder}
for bssize in 4
do
    for tpsize in 4
    do
        echo "Running with bs=${bssize} tp=${tpsize}"
        CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 TORCHDYNAMO_VERBOSE=1 ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                        --model_name ${model_name} \
                                        --num_samples 10 \
                                        --batch_size ${bssize} \
                                        --prompt_length 1024 \
                                        --max_new_tokens 512 \
                                        --use_flash_attention \
                                        --device cuda > ${folder}/model_name_bssize${bssize}_tpsize${tpsize}_91.log 2>&1
        echo "Finished running with bs=${bssize} tp=${tpsize}" 
    done
done
