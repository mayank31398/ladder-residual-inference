mode=compile
nodenum=1
prompt_length=1024
max_new_tokens=512
# master_address=192.169.71.2
# --master_addr=${master_address} --node_rank=$1
for P2P_DISABLE in 0
do
    export NCCL_P2P_DISABLE=${P2P_DISABLE}
    for model_name in "gpt_dense:llama-3-70b" # "gpt_dense:llama-3-70b" "gpt_ladder:llama-3-70b" "gpt_ensemble:llama-3-70b-upper-bound"  "gpt_parallel:llama-3-70b"
    do
        folder=./logs/11_30/prompt_length_${prompt_length}_max_new_${max_new_tokens}/p2p_disable${P2P_DISABLE}/${mode}/${model_name}
        mkdir -p ${folder}
        for bssize in 1
        do
            # combinations=(
            #     "4 2"
            #     "8 1"
            #     "8 2"
            #     "16 1"
            # )
            combinations=(
                "4 2"
            )
            for combo in "${combinations[@]}"
            do
                # Split the combination into tpsize and ppsize
                tpsize=$(echo $combo | awk '{print $1}')
                ppsize=$(echo $combo | awk '{print $2}')
                echo "Running with bs=${bssize} tp=${tpsize} pp=${ppsize}"
                # NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} 
                torchrun --standalone --nproc_per_node=$((tpsize*ppsize/nodenum)) --nnodes=${nodenum} --master_port=15328 benchmark.py \
                                                --model_name ${model_name} \
                                                --num_samples 10 \
                                                --batch_size ${bssize} \
                                                --prompt_length ${prompt_length} \
                                                --max_new_tokens ${max_new_tokens} \
                                                --compile \
                                                --compile_prefill \
                                                --tensor_parallel_world_size ${tpsize} \
                                                --pipeline_parallel_world_size ${ppsize} \
                                                --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}_pp_${ppsize}.log
                echo "Finished running with bs=${bssize} tp=${tpsize}" 
            done
        done
    done
done