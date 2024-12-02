mode=compile
nodenum=2
prompt_length=1024
max_new_tokens=512
master_address=172.27.36.158

for P2P_DISABLE in 0 1
do
    export NCCL_P2P_DISABLE=${P2P_DISABLE}
    export NCCL_NVLS_ENABLE=1
    for model_name in "gpt_dense:llama-3.1-405b" "gpt_ladder:llama-3.1-405b" "gpt_ensemble:llama-3.1-405b-upper-bound" "gpt_parallel:llama-3.1-405b"
    do
        folder=./logs/final/prompt_length_${prompt_length}_max_new_${max_new_tokens}/p2p_disable${P2P_DISABLE}/${mode}/${model_name}
        mkdir -p ${folder}
        for bssize in 1 4 16 64
        do
            # combinations=(
            #     "4 2"
            #     "8 1"
            #     "8 2"
            #     "16 1"
            # )
            combinations=(
                "8 2"
            )
            for combo in "${combinations[@]}"
            do
                # Split the combination into tpsize and ppsize
                tpsize=$(echo $combo | awk '{print $1}')
                ppsize=$(echo $combo | awk '{print $2}')
                echo "Running with bs=${bssize} tp=${tpsize} pp=${ppsize}"
                # NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} 
                NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} torchrun --nproc_per_node=8 --nnodes=${nodenum} --master_addr=172.27.35.50 --master_port=15318 --node_rank=$1 benchmark.py \
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
                echo "Finished running with bs=${bssize} tp=${tpsize} pp=${ppsize}"
            done
        done
    done
done