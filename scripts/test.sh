mode=cuda_graph_use_flash_attention
for model_name in "gpt_dense:llama-3-8b"
do
    folder=./logs/09_20_float16/${mode}/${model_name}
    mkdir -p ${folder}
    for bssize in 4 8
    do
        for tpsize in 4
        do
            echo "Running with bs=${bssize} tp=${tpsize}"
            NCCL_DEBUG=INFO ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                            --model_name ${model_name} \
                                            --num_samples 2 \
                                            --batch_size ${bssize} \
                                            --prompt_length 128 \
                                            --max_new_tokens 64 \
                                            --cuda_graph \
                                            --use_flash_attention \
                                            --device cuda 2>&1 | tee tmp.log
            echo "Finished running with bs=${bssize} tp=${tpsize}" 
        done
    done
done