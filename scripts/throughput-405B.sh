mode=compile
nodenum=2
prompt_length=1024
max_new_tokens=512
# "gpt_ladder:llama-3.1-405b" "gpt_parallel:llama-3.1-405b"
for P2P_DISABLE in 1
do
    export NCCL_P2P_DISABLE=${P2P_DISABLE}
    for model_name in "gpt_dense:llama-3.1-405b"
    do
        folder=./logs/11_14/prompt_length_${prompt_length}_max_new_${max_new_tokens}/p2p_disable${P2P_DISABLE}/${mode}/${model_name}
        mkdir -p ${folder}
        for bssize in 1 4 8 16
        do
            for tpsize in 16
            do
                nproc_per_node=$((tpsize/nodenum))
                echo "Running with bs=${bssize} tp=${tpsize} nproc_per_node=${nproc_per_node} node_num=${nodenum}"
                NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} torchrun --nproc_per_node=${nproc_per_node} --nnodes=${nodenum} --master_addr=172.27.39.142 --master_port=15328 --node_rank=$1 benchmark.py \
                                                --model_name ${model_name} \
                                                --num_samples 10 \
                                                --batch_size ${bssize} \
                                                --prompt_length ${prompt_length} \
                                                --max_new_tokens ${max_new_tokens} \
                                                --compile \
                                                --compile_prefill \
                                                --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
                echo "Finished running with bs=${bssize} tp=${tpsize}" 
            done
        done
    done
done