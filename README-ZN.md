# CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration

[English](README.md)

本文档仅记录 CanvasAgent 的环境配置、SFT 启动和 RL 启动操作。

## 仓库目录

| 路径 | 用途 |
| --- | --- |
| `environment-llama.yml` | 本项目使用的 Conda 环境导出文件 |
| `LLaMA-Factory/qwen3-vl.yaml` | SFT 配置 |
| `LLaMA-Factory/data/dataset_info.json` | SFT 数据集注册文件 |
| `verl/examples/qwen3vl_multiturn/run_qwen3vl-8b_rl10k.sh` | RL 训练入口 |
| `verl/examples/qwen3vl_multiturn/config/tool_config/` | 视觉工具配置 |

## 1. 创建环境

导出的环境使用 Python 3.10、PyTorch 2.7.1 和 CUDA 12.6 相关依赖。

```bash
git clone https://github.com/harryzhu123/CanvasAgent.git
cd CanvasAgent

conda env create -f environment-llama.yml
conda activate llama

export CANVAS_AGENT_ROOT="$(pwd)"
export PYTHONPATH="$CANVAS_AGENT_ROOT/LLaMA-Factory/src:$CANVAS_AGENT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
```

配置 CUDA：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

检查环境：

```bash
python -c "import torch, ray, sglang, llamafactory, verl; print(torch.__version__, torch.version.cuda, ray.__version__, sglang.__version__)"
nvcc --version
```

## 2. 配置路径

启动任务前，将以下文件中的机器绝对路径替换为实际路径：

| 文件 | 需要修改的内容 |
| --- | --- |
| `LLaMA-Factory/qwen3-vl.yaml` | `model_name_or_path`、`output_dir` |
| `LLaMA-Factory/data/dataset_info.json` | `smartagentV2-multiturn-reason` 和 `smartagentV2-val-reason` 的 `file_name` |
| `verl/examples/qwen3vl_multiturn/run_qwen3vl-8b_rl10k.sh` | 环境激活路径、`PROJECT_DIR`、模型路径、训练 parquet、验证 parquet |
| `verl/examples/qwen3vl_multiturn/config/tool_config/image_tools_config_matched.yaml` | 工具模型路径和 GPU 编号 |

配置日志和奖励评分服务所需的凭据：

```bash
export WANDB_API_KEY="<wandb-api-key>"
export REWARD_API_KEY="<openai-compatible-api-key>"
export REWARD_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export REWARD_MODEL_NAME="qwen3.5-plus"
```

## 3. 启动 SFT

```bash
conda activate llama
cd "$CANVAS_AGENT_ROOT/LLaMA-Factory"
python -m llamafactory.cli train qwen3-vl.yaml
```

SFT 输出位置由 `qwen3-vl.yaml` 中的 `output_dir` 控制。

## 4. 启动 RL

```bash
conda activate llama
cd "$CANVAS_AGENT_ROOT/verl/examples/qwen3vl_multiturn"

bash run_qwen3vl-8b_rl10k.sh \
  actor_rollout_ref.model.path=/path/to/sft/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/validation.parquet
```

脚本将标准输出和 rollout trace 写入
`$CANVAS_AGENT_ROOT/verl/examples/qwen3vl_multiturn/log/`。RL checkpoint
写入 verl 配置指定的 checkpoint 目录。
