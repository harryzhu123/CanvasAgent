# CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration

## Qwen3-VL SFT + RL 训练流程

本文档记录本项目从 **LLaMA-Factory SFT** 到 **verl + Ray + GRPO RL** 的训练链路。当前任务是训练一个多轮视觉工具调用 Agent：模型根据用户目标观察图片、规划操作、调用图像工具，并输出最终结果。

## 项目目标

基础模型使用 `Qwen3-VL-8B-Instruct`。训练分两段：

1. **SFT**：让模型学会多轮工具调用格式、工具参数写法、图像 ID 引用方式和基本任务轨迹。
2. **RL**：在真实交互环境中 rollout 多条轨迹，根据分数信号用 GRPO 继续优化模型的工具选择、调用顺序和最终输出质量。

简化流程如下：

```text
ShareGPT 多模态数据
  -> LLaMA-Factory full SFT
  -> SFT checkpoint
  -> verl 多轮 rollout
  -> GRPO 更新策略模型
  -> RL checkpoint
```

## 关键目录

| 用途 | 路径 |
| --- | --- |
| LLaMA-Factory 代码与 SFT 配置 | `/COPY_LLaMA-Factory` |
| SFT 配置 | `/COPY_LLaMA-Factory/qwen3-vl.yaml` |
| SFT 数据注册 | `/COPY_LLaMA-Factory/data/dataset_info.json` |
| SFT 数据 | `/zhuhairui/data/smartagentV2/for-cluster` |
| verl 多轮 RL 目录 | `/verl/examples/qwen3vl_multiturn` |
| RL parquet 数据 | `/RL10kV2/data/verl_parquet` |
| RL checkpoint 输出 | `/verl/checkpoints` |
| RL 日志 | `/verl/examples/qwen3vl_multiturn/log` |

说明：历史脚本里常见的 `/jiangwenhao/zhuhairui` 是另一种运行路径写法，和当前项目目录对应同一套数据。

## 环境准备

进入训练环境：

```bash
cd /verl
source /env/llama/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

如果需要记录训练曲线，可以通过环境变量注入 WandB key，不要把 key 写入脚本：

```bash
export WANDB_API_KEY="<your-wandb-key>"
```

安全注意：部分历史运行脚本里曾经直接写入 WandB 或 API key。对外分享或提交前，建议先把这些 key 改为读取环境变量，并轮换已经暴露过的旧 key。

## 阶段一：SFT

SFT 在 LLaMA-Factory 中完成，核心配置是：

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

数据在 `dataset_info.json` 中注册为 ShareGPT 多模态格式：

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

启动 SFT：

```bash
cd /COPY_LLaMA-Factory
source /env/llama/bin/activate
llamafactory-cli train qwen3-vl.yaml
```

当前可用的 SFT checkpoint 示例：

```text
/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV3/checkpoint-2142
/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV2-multiturn/checkpoint-1358
```

其中 `smartagentV3/checkpoint-2142` 是后续 RL 脚本里常用的初始化模型。

## SFT 学到什么

SFT 主要解决“会不会按格式做事”的问题：

- 识别系统 prompt 里的工具 schema。
- 生成合法的工具调用 JSON。
- 在多轮对话中引用 `img_1`、`ImageEdit_0` 等图像 ID。
- 学习 Crop、Grounding、SAM、SR、OCR、ImageGeneration、ImageEdit 等工具的基本使用顺序。
- 形成初步 reasoning 和 terminate 行为。

SFT 本身只是模仿离线轨迹，不能保证模型真的选择最优工具链，也不能直接优化最终图像质量。因此需要第二阶段 RL。

## 阶段二：RL

RL 在 verl 中运行。核心思想是：每个 prompt 采样多条多轮轨迹，轨迹中真实调用环境能力，最后根据轨迹分数用 GRPO 更新模型。

常用主脚本：

| 脚本 | 用途 |
| --- | --- |
| `run_qwen3vl-8b_rl10kV2.sh` | GRPO 训练主脚本 |
| `run_qwen3vl-8b_rl10k_test.sh` | 测试或快速运行脚本 |

GRPO 启动示例：

```bash
cd /verl/examples/qwen3vl_multiturn
source /env/llama/bin/activate

bash run_qwen3vl-8b_rl10kV2.sh \
  actor_rollout_ref.model.path=/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV3/checkpoint-2142 \
  data.train_files=/RL10kV2/data/verl_parquet/train.parquet \
  data.val_files=/RL10kV2/data/verl_parquet/test.parquet
```

当前 RL 训练的常用配置：

| 参数 | 值 |
| --- | --- |
| 算法 | GRPO |
| rollout backend | `sglang` async |
| `train_batch_size` | 6 prompts |
| `rollout.n` | 2 到 8，生产常用 8 |
| real batch | `train_batch_size * rollout.n` |
| `ppo_mini_batch_size` | 6 |
| `ppo_micro_batch_size_per_gpu` | 1 |
| `n_gpus_per_node` | 6 |
| `rollout.agent.num_workers` | 4 |
| `max_assistant_turns` | 6 到 12 |
| `max_single_turn_length` | 1536 |
| learning rate | `1e-6` |
| KL loss | `use_kl_loss=True`, `kl_loss_coef=0.001` |

注意：real batch 必须能被训练 GPU 数整除。例如 `train_batch_size=6` 且 `rollout.n=8` 时，real batch 为 48，`48 / 6 = 8`，可以整除。

## 推荐运行顺序

1. 检查 SFT 数据：

```bash
cd /COPY_LLaMA-Factory
rg -n "smartagentV2-multiturn-reason|smartagentV2-val-reason" data/dataset_info.json
```

2. 训练或选择 SFT checkpoint：

```bash
llamafactory-cli train qwen3-vl.yaml
```

3. 启动 GRPO RL：

```bash
cd /verl/examples/qwen3vl_multiturn
bash run_qwen3vl-8b_rl10kV2.sh \
  actor_rollout_ref.model.path=/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV3/checkpoint-2142
```

4. 训练完成后，检查 `/verl/checkpoints` 下生成的 run 目录和 checkpoint。
