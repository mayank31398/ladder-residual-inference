

# \
#                                         --compile \
#                                         --compile_prefill

# --use_flash_attention \
# -semi-compiled

model_name="gpt_ladder:llama-3-8b"
folder=./logs/tmp/
mkdir -p ${folder}
for bssize in 4
do
    for tpsize in 4
    do
        echo "Running with bs=${bssize} tp=${tpsize}"
        ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                        --model_name ${model_name} \
                                        --num_samples 10 \
                                        --batch_size ${bssize} \
                                        --prompt_length 1024 \
                                        --max_new_tokens 512 \
                                        --cuda_graph \
                                        --use_flash_attention \
                                        --device cuda > tmp.log
        echo "Finished running with bs=${bssize} tp=${tpsize}" 
    done
done
