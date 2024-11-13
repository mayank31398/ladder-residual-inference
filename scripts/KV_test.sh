mode=compile_cmpile_prefill
for model_name in "gpt_dense:llama-3-8b"
do
    folder=./logs/kv_test
    mkdir -p ${folder}
    for bssize in 1 4 8 16 64
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
                                            --compile \
                                            --compile_prefill \
                                            --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
            echo "Finished running with bs=${bssize} tp=${tpsize}" 
        done
    done
done