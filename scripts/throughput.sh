# gpt_dense:70B
# gpt_ensemble:70B
# gpt_ensemble:70B-infinite
# gpt_ladder:70B
# gpt_ladder:70B-semi-compiled
# gpt_parallel:70B





# if fully compile:
#     flash decode should be False
# elif semi compile:
#     flash decode to True / False
# else:
#     flash decode to True


# NOTE flash decode doesn't compile even with semi-compile



model_name=gpt_ladder:llama-3-8b
folder=./
mkdir -p ${folder}
for bssize in 16
do
    for tpsize in 2
    do
        echo "Running with bs=${bssize} tp=${tpsize}"
        TORCHDYNAMO_VERBOSE=1 ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                        --model_name ${model_name} \
                                        --num_samples 10 \
                                        --batch_size ${bssize} \
                                        --prompt_length 1024 \
                                        --max_new_tokens 256 \
                                        --device cuda \
                                        --compile \
                                        --compile_prefill  > tmp.log 2>&1
        echo "Finished running with bs=${bssize} tp=${tpsize}"
    done
done
