# export NCCL_P2P_DISABLE=1
# NCCL_P2P_DISABLE=1

# === 8B ===
# mode=cuda_graph_use_flash_attention
# for model_name in "gpt_ladder:llama-3-8b" "gpt_dense:llama-3-8b" "gpt_ensemble:llama-3-8b-upper-bound" "gpt_parallel:llama-3-8b" "gpt_ensemble:llama-3-8b"
# do
#     folder=./logs/09_20_float16/${mode}/${model_name}
#     mkdir -p ${folder}
#     for bssize in 1 4 8 16 64
#     do
#         for tpsize in 1 2 4 8
#         do
#             echo "Running with bs=${bssize} tp=${tpsize}"
#             ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
#                                             --model_name ${model_name} \
#                                             --num_samples 10 \
#                                             --batch_size ${bssize} \
#                                             --prompt_length 1024 \
#                                             --max_new_tokens 256 \
#                                             --cuda_graph \
#                                             --use_flash_attention \
#                                             --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
#             echo "Finished running with bs=${bssize} tp=${tpsize}" 
#         done
#     done
# done

# mode=use_flash_attention
# for model_name in "gpt_ladder:llama-3-8b" "gpt_dense:llama-3-8b" "gpt_ensemble:llama-3-8b-upper-bound" "gpt_parallel:llama-3-8b" "gpt_ensemble:llama-3-8b"
# do
#     folder=./logs/09_20_float16/${mode}/${model_name}
#     mkdir -p ${folder}
#     for bssize in 1 4 8 16 64
#     do
#         for tpsize in 1 2 4 8
#         do
#             echo "Running with bs=${bssize} tp=${tpsize}"
#             ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
#                                             --model_name ${model_name} \
#                                             --num_samples 10 \
#                                             --batch_size ${bssize} \
#                                             --prompt_length 1024 \
#                                             --max_new_tokens 256 \
#                                             --use_flash_attention \
#                                             --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
#             echo "Finished running with bs=${bssize} tp=${tpsize}" 
#         done
#     done
# done

# mode=compile_cmpile_prefill
# for model_name in "gpt_ladder:llama-3-8b" "gpt_dense:llama-3-8b" "gpt_ensemble:llama-3-8b-upper-bound" "gpt_parallel:llama-3-8b" "gpt_ensemble:llama-3-8b"
# do
#     folder=./logs/09_20_float16/${mode}/${model_name}
#     mkdir -p ${folder}
#     for bssize in 1 4 8 16 64
#     do
#         for tpsize in 1 2 4 8
#         do
#             echo "Running with bs=${bssize} tp=${tpsize}"
#             ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
#                                             --model_name ${model_name} \
#                                             --num_samples 10 \
#                                             --batch_size ${bssize} \
#                                             --prompt_length 1024 \
#                                             --max_new_tokens 256 \
#                                             --compile \
#                                             --compile_prefill \
#                                             --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
#             echo "Finished running with bs=${bssize} tp=${tpsize}" 
#         done
#     done
# done

# === 70B ===
# mode=cuda_graph_use_flash_attention
# for model_name in "gpt_ladder:llama-3-70b" "gpt_dense:llama-3-70b" "gpt_ensemble:llama-3-70b-upper-bound" "gpt_parallel:llama-3-70b" "gpt_ensemble:llama-3-70b"
# do
#     folder=./logs/09_20_float16/${mode}/${model_name}
#     mkdir -p ${folder}
#     for bssize in 1 4 8 16 64
#     do
#         for tpsize in 1 2 4 8
#         do
#             echo "Running with bs=${bssize} tp=${tpsize}"
#             ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
#                                             --model_name ${model_name} \
#                                             --num_samples 10 \
#                                             --batch_size ${bssize} \
#                                             --prompt_length 1024 \
#                                             --max_new_tokens 256 \
#                                             --cuda_graph \
#                                             --use_flash_attention \
#                                             --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
#             echo "Finished running with bs=${bssize} tp=${tpsize}" 
#         done
#     done
# done

# mode=use_flash_attention
# for model_name in "gpt_ladder:llama-3-70b" "gpt_dense:llama-3-70b" "gpt_ensemble:llama-3-70b-upper-bound" "gpt_parallel:llama-3-70b" "gpt_ensemble:llama-3-70b"
# do
#     folder=./logs/09_20_float16/${mode}/${model_name}
#     mkdir -p ${folder}
#     for bssize in 1 4 8 16 64
#     do
#         for tpsize in 1 2 4 8
#         do
#             echo "Running with bs=${bssize} tp=${tpsize}"
#             ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
#                                             --model_name ${model_name} \
#                                             --num_samples 10 \
#                                             --batch_size ${bssize} \
#                                             --prompt_length 1024 \
#                                             --max_new_tokens 256 \
#                                             --use_flash_attention \
#                                             --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
#             echo "Finished running with bs=${bssize} tp=${tpsize}" 
#         done
#     done
# done

# mode=compile_cmpile_prefill
# for model_name in "gpt_ladder:llama-3-70b" "gpt_dense:llama-3-70b" "gpt_ensemble:llama-3-70b-upper-bound" "gpt_parallel:llama-3-70b" "gpt_ensemble:llama-3-70b"
# do
#     folder=./logs/09_20_float16/${mode}/${model_name}
#     mkdir -p ${folder}
#     for bssize in 1 4 8 16 64
#     do
#         for tpsize in 1 2 4 8
#         do
#             echo "Running with bs=${bssize} tp=${tpsize}"
#             ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=${tpsize} benchmark.py \
#                                             --model_name ${model_name} \
#                                             --num_samples 10 \
#                                             --batch_size ${bssize} \
#                                             --prompt_length 1024 \
#                                             --max_new_tokens 256 \
#                                             --compile \
#                                             --compile_prefill \
#                                             --device cuda 2>&1 | tee ${folder}/bs_${bssize}_tp_${tpsize}.log
#             echo "Finished running with bs=${bssize} tp=${tpsize}" 
#         done
#     done
# done
