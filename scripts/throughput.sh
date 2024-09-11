# gpt_dense:70B
# gpt_ensemble:70B
# gpt_ensemble:70B-infinite


torchrun --nproc_per_node 4 benchmark.py \
    --model_name gpt_dense:70B \
    --prompt_length 1024 \
    --num_samples 1 \
    --max_new_tokens 256 \
    --batch_size 1 \
    --device cuda \
    --compile \
    --compile_prefill
