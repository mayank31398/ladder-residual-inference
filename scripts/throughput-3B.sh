mode=compile
nodenum=1
prompt_length=1024
max_new_tokens=512
# --master_addr=104.171.200.62
# --node_rank=1
# "gpt_ensemble:llama-3-8b-upper-bound" "gpt_parallel:llama-3-8b"
for P2P_DISABLE in 0 1
do
    export NCCL_P2P_DISABLE=${P2P_DISABLE}
    export NCCL_NVLS_ENABLE=1
    export TRITON_CACHE_DIR="/home/charlie/tmp"
    export TORCHINDUCTOR_CACHE_DIR="/home/charlie/tmp"
    for model_name in "gpt_dense:3b" "gpt_ladder:3b"
    do
        folder=./logs/arxiv/prompt_length_${prompt_length}_max_new_${max_new_tokens}/p2p_disable${P2P_DISABLE}/${mode}/${model_name}
        mkdir -p ${folder}
        for bssize in 4
        do
            for tpsize in 8
            do
                for ppsize in 1
                do
                    echo "Running with bs=${bssize} tp=${tpsize} pp=${ppsize}"
                    NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} torchrun --standalone --nproc_per_node=$((tpsize*ppsize/nodenum)) --nnodes=${nodenum} --master_port=15328 benchmark.py \
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
done