mode=compile
nodenum=1
prompt_length=1024
max_new_tokens=512

for P2P_DISABLE in 0
do
    export NCCL_P2P_DISABLE=${P2P_DISABLE}
    for model_name in "gpt_dense:llama-3-70b" "gpt_ladder:llama-3-70b" "gpt_desync:llama-3-70b-upper-bound" "gpt_parallel:llama-3-70b"
    do
        folder=./logs/prompt_length_${prompt_length}_max_new_${max_new_tokens}/p2p_disable${P2P_DISABLE}/${mode}/${model_name}
        mkdir -p ${folder}
        for bssize in 1 4 16 64
        do
            for tpsize in 2
            do
                echo "Running with P2P_DISABLE=${P2P_DISABLE} bs=${bssize} tp=${tpsize}"
                ENABLE_INTRA_NODE_COMM=1 NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} torchrun --standalone --nproc_per_node=${tpsize} --nnodes=${nodenum} --master_port=15328 benchmark.py \
                                                --model_name ${model_name} \
                                                --num_samples 10  \
                                                --batch_size ${bssize} \
                                                --prompt_length ${prompt_length} \
                                                --max_new_tokens ${max_new_tokens} \
                                                --compile \
                                                --compile_prefill \
                                                --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
                echo "Finished running with P2P_DISABLE=${P2P_DISABLE} bs=${bssize} tp=${tpsize}"
            done
        done
    done
done