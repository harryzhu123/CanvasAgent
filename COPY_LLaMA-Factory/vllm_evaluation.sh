#!/bin/bash

# 使用 vLLM 进行推理评估
# 参考 evaluation.sh 的配置

CUDA_VISIBLE_DEVICES=2,3 python scripts/vllm_infer.py \
    --model_name_or_path /data/zhuhairui/LLaMA-Factory/saves/qwen2_5vl-7b/full/smartagent-10epochs/checkpoint-40 \
    --adapter_name_or_path None \
    --dataset smartagent-val \
    --dataset_dir data \
    --template qwen2_vl \
    --cutoff_len 8192 \
    --max_samples 100000 \
    --save_name saves/Qwen2.5-VL-7B-Instruct-40steps/freeze/eval-easy/vllm_generated_predictions.jsonl \
    --temperature 0.95 \
    --top_p 0.7 \
    --top_k 50 \
    --max_new_tokens 2048 \
    --repetition_penalty 1.0 \
    --skip_special_tokens True \
    --enable_thinking True \
    --pipeline_parallel_size 1 \
    --image_max_pixels 589824 \
    --image_min_pixels 1024 \
    --batch_size 32

echo "vLLM inference completed!"
echo "Results saved to: saves/Qwen2.5-VL-7B-Instruct-40steps/freeze/eval-easy/vllm_generated_predictions.jsonl"
