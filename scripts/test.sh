for model_name in "gpt_ladder:llama-3-8b"
do
    for bssize in 4
    do
        for tpsize in 4
        do
            echo "Running with bs=${bssize} tp=${tpsize}"
            NCCL_BLOCKING_WAIT=0 ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
                                            --model_name ${model_name} \
                                            --num_samples 10 \
                                            --batch_size ${bssize} \
                                            --prompt_length 256 \
                                            --max_new_tokens 32 \
                                            --cuda_graph \
                                            --use_flash_attention \
                                            --profile ./profiles/use_flash_attention_v6 \
                                            --device cuda 2>&1 | tee ./tmp.log
            echo "Finished running with bs=${bssize} tp=${tpsize}" 
        done
    done
done