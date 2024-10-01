mode=cuda_graph_use_flash_attention
for P2P_DISABLE in 0 1
do
    export NCCL_P2P_DISABLE=${P2P_DISABLE}
    for model_name in "gpt_ladder:3b" "gpt_dense:3b" "gpt_ensemble:3b-upper-bound" "gpt_parallel:3b" "gpt_ensemble:3b"
    do
        folder=./logs/09_20_float16_disable${P2P_DISABLE}/${mode}/${model_name}
        mkdir -p ${folder}
        for bssize in 1 4 8 16 64
        do
            for tpsize in 1 2 4 8
            do
                echo "Running with bs=${bssize} tp=${tpsize}"
                NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} torchrun --standalone --nproc_per_node=${tpsize} --master_port=15328 benchmark.py \
                                                --model_name ${model_name} \
                                                --num_samples 10 \
                                                --batch_size ${bssize} \
                                                --prompt_length 1024 \
                                                --max_new_tokens 256 \
                                                --cuda_graph \
                                                --use_flash_attention \
                                                --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
                echo "Finished running with bs=${bssize} tp=${tpsize}" 
            done
        done
    done
done