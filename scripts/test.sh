mode=cuda_graph_use_flash_attention
for model_name in "gpt_parallel:llama-3-8b"
do
    for bssize in 4 8
    do
        for tpsize in 4
        do
            echo "Running with bs=${bssize} tp=${tpsize}"
            torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
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