

# \
#                                         --compile \
#                                         --compile_prefill
# > tmp.log 2>&1
model_name=gpt_ladder:llama-3-8b-semi-compiled
folder=./
mkdir -p ${folder}
for bssize in 1
do
    for tpsize in 2
    do
        echo "Running with bs=${bssize} tp=${tpsize}"
        TORCHDYNAMO_VERBOSE=1 ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                        --model_name ${model_name} \
                                        --num_samples 5 \
                                        --batch_size ${bssize} \
                                        --prompt_length 13 \
                                        --max_new_tokens 13 \
                                        --device cuda
        echo "Finished running with bs=${bssize} tp=${tpsize}"
    done
done
