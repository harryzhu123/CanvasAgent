# CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration

[Chinese version](README-ZN.md)

## Qwen3-VL SFT + RL Training Pipeline

This document describes the training pipeline of CanvasAgent, from **LLaMA-Factory SFT** to **verl + Ray + GRPO RL**. The project trains a multi-turn visual tool-use agent that observes images, plans image operations, invokes visual tools, and produces a final result according to the user goal.

## Project Goal

The base model is `Qwen3-VL-8B-Instruct`. Training has two stages:

1. **SFT**: Teach the model the multi-turn tool-call format, tool argument conventions, image ID references, and basic task trajectories.
2. **RL**: Roll out multiple trajectories in an interactive environment, then use GRPO to further optimize tool selection, tool-call order, and final output quality from trajectory scores.

High-level flow:

```text
ShareGPT multimodal data
  -> LLaMA-Factory full SFT
  -> SFT checkpoint
  -> verl multi-turn rollout
  -> GRPO policy update
  -> RL checkpoint
```

## Key Directories

| Purpose | Path |
| --- | --- |
| LLaMA-Factory code and SFT config | `/LLaMA-Factory` |
| SFT config | `/LLaMA-Factory/qwen3-vl.yaml` |
| SFT dataset registry | `/LLaMA-Factory/data/dataset_info.json` |
| SFT data | `/zhuhairui/data/smartagentV2/for-cluster` |
| verl multi-turn RL code | `/verl/examples/qwen3vl_multiturn` |
| RL parquet data | `/RL10kV2/data/verl_parquet` |
| RL checkpoint output | `/verl/checkpoints` |
| RL logs | `/verl/examples/qwen3vl_multiturn/log` |

Historical scripts may use `/jiangwenhao/zhuhairui`; it is another runtime path for the same project data.

## Environment

Enter the training environment:

```bash
cd /verl
source /env/llama/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If training curves should be logged, inject the WandB key through the environment instead of hardcoding it in scripts:

```bash
export WANDB_API_KEY="<your-wandb-key>"
```

Security note: some historical run scripts used to contain hardcoded WandB or API keys. Before sharing or submitting code, replace those values with environment-variable reads and rotate any exposed keys.

## Stage 1: SFT

SFT is run with LLaMA-Factory. The core config is:

```yaml
model_name_or_path: /jiangwenhao/Qwen3-VL-8B-Instruct
output_dir: saves/qwen3-vl-8b/full/smartagentV3
dataset: smartagentV2-multiturn-reason
eval_dataset: smartagentV2-val-reason
stage: sft
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
template: qwen3_vl
cutoff_len: 16384
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
deepspeed: examples/deepspeed/ds_z3_offload_config.json
bf16: true
```

The data is registered in `dataset_info.json` as ShareGPT-style multimodal data:

```json
"smartagentV2-multiturn-reason": {
  "file_name": "/jiangwenhao/zhuhairui/zhuhairui/data/smartagentV2/for-cluster/train_reason_imglist.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  }
}
```

Launch SFT:

```bash
cd /LLaMA-Factory
source /env/llama/bin/activate
llamafactory-cli train qwen3-vl.yaml
```

Available SFT checkpoints include:

```text
/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV3/checkpoint-2142
/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV2-multiturn/checkpoint-1358
```

`smartagentV3/checkpoint-2142` is commonly used as the initialization checkpoint for RL.

## What SFT Teaches

SFT mainly teaches the model how to follow the tool-use format:

- Read the tool schemas in the system prompt.
- Produce valid JSON tool calls.
- Reference image IDs such as `img_1` and `ImageEdit_0` across turns.
- Learn basic usage patterns for Crop, Grounding, SAM, SR, OCR, ImageGeneration, ImageEdit, and related tools.
- Develop initial reasoning and termination behavior.

SFT only imitates offline trajectories. It does not guarantee optimal tool-chain selection or directly optimize final image quality, so a second RL stage is used.

## Stage 2: RL

RL is run with verl. For each prompt, the model samples multiple multi-turn trajectories, interacts with the environment, and updates the policy with GRPO from trajectory scores.

Main scripts:

| Script | Purpose |
| --- | --- |
| `run_qwen3vl-8b_rl10kV2.sh` | Main GRPO training script |
| `run_qwen3vl-8b_rl10k_test.sh` | Test or quick-run script |

Launch GRPO:

```bash
cd /verl/examples/qwen3vl_multiturn
source /env/llama/bin/activate

bash run_qwen3vl-8b_rl10kV2.sh \
  actor_rollout_ref.model.path=/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV3/checkpoint-2142 \
  data.train_files=/RL10kV2/data/verl_parquet/train.parquet \
  data.val_files=/RL10kV2/data/verl_parquet/test.parquet
```

Common RL settings:

| Parameter | Value |
| --- | --- |
| Algorithm | GRPO |
| Rollout backend | `sglang` async |
| `train_batch_size` | 6 prompts |
| `rollout.n` | 2 to 8, commonly 8 for production |
| Real batch | `train_batch_size * rollout.n` |
| `ppo_mini_batch_size` | 6 |
| `ppo_micro_batch_size_per_gpu` | 1 |
| `n_gpus_per_node` | 6 |
| `rollout.agent.num_workers` | 4 |
| `max_assistant_turns` | 6 to 12 |
| `max_single_turn_length` | 1536 |
| Learning rate | `1e-6` |
| KL loss | `use_kl_loss=True`, `kl_loss_coef=0.001` |

The real batch must be divisible by the number of training GPUs. For example, with `train_batch_size=6` and `rollout.n=8`, the real batch is 48, and `48 / 6 = 8`.

## Recommended Run Order

1. Check the SFT data:

```bash
cd /LLaMA-Factory
rg -n "smartagentV2-multiturn-reason|smartagentV2-val-reason" data/dataset_info.json
```

2. Train or select an SFT checkpoint:

```bash
llamafactory-cli train qwen3-vl.yaml
```

3. Start GRPO RL:

```bash
cd /verl/examples/qwen3vl_multiturn
bash run_qwen3vl-8b_rl10kV2.sh \
  actor_rollout_ref.model.path=/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV3/checkpoint-2142
```

4. After training, inspect the generated run directory and checkpoint under `/verl/checkpoints`.
