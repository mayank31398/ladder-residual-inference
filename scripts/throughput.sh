

# \
#                                         --compile \
#                                         --compile_prefill

# --use_flash_attention \
# -semi-compiled

model_name="gpt_ladder:llama-3-8b"
folder=./logs/09_20/2/model_name
mkdir -p ${folder}
for bssize in 4
do
    for tpsize in 1 2 4 8
    do
        echo "Running with bs=${bssize} tp=${tpsize}"
        ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                        --model_name ${model_name} \
                                        --num_samples 10 \
                                        --batch_size ${bssize} \
                                        --prompt_length 1024 \
                                        --max_new_tokens 256 \
                                        --
                                        --device cuda > tmp.log
        echo "Finished running with bs=${bssize} tp=${tpsize}" 
    done
done
