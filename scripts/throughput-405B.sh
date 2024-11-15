mode=cuda_graph_use_flash_attention
nodenum=1
prompt_length=1024
max_new_tokens=512
# --master_addr=104.171.200.62
# --node_rank=1
for P2P_DISABLE in 1
do
    export NCCL_P2P_DISABLE=${P2P_DISABLE}
    for model_name in "gpt_dense:llama-3.1-405b" "gpt_ladder:llama-3.1-405b" "gpt_ensemble:llama-3.1-405b" "gpt_ensemble:llama-3.1-405b-upper-bound" "gpt_parallel:llama-3.1-405b" 
    do
        folder=./logs/11_14/prompt_length_${prompt_length}_max_new_${max_new_tokens}/p2p_disable${P2P_DISABLE}/${mode}/${model_name}
        mkdir -p ${folder}
        for bssize in 1 4 16 64
        do
            for tpsize in 1 2 4 8
            do
                echo "Running with bs=${bssize} tp=${tpsize}"
                NCCL_NVLS_ENABLE=1 NCCL_P2P_DISABLE=${P2P_DISABLE} torchrun --standalone --nproc_per_node=${tpsize} --nnodes=${nodenum} --master_port=15328 benchmark.py \
                                                --model_name ${model_name} \
                                                --num_samples 10 \
                                                --batch_size ${bssize} \
                                                --prompt_length ${prompt_length} \
                                                --max_new_tokens ${max_new_tokens} \
                                                --cuda_graph \
                                                --use_flash_attention \
                                                --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
                echo "Finished running with bs=${bssize} tp=${tpsize}" 
            done
        done
    done
done

"llama-3-70b-upper-bound": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000, reduce_pattern=[{"attention": False, "mlp": False} for _ in range(80)], force_disable_last_all_reduce=True),
"llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
    rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
),
"llama-3.1-405b-upper-bound": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000, reduce_pattern=[{"attention": False, "mlp": False} for _ in range(126)], force_disable_last_all_reduce=True,
    rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
),