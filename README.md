# CanvasAgent: Enabling Complex Image Creation and Editing via Visual Tool Orchestration

[中文说明](README-ZN.md)

This manual covers environment setup and the commands used to run the
CanvasAgent SFT and RL pipelines.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `environment-llama.yml` | Export of the Conda environment used by this project |
| `LLaMA-Factory/qwen3-vl.yaml` | SFT configuration |
| `LLaMA-Factory/data/dataset_info.json` | SFT dataset registry |
| `verl/examples/qwen3vl_multiturn/run_qwen3vl-8b_rl10k.sh` | RL training entry point |
| `verl/examples/qwen3vl_multiturn/config/tool_config/` | Visual tool configurations |

## 1. Create the Environment

The exported environment uses Python 3.10, PyTorch 2.7.1, and CUDA 12.6
packages.

```bash
git clone https://github.com/harryzhu123/CanvasAgent.git
cd CanvasAgent

conda env create -f environment-llama.yml
conda activate llama

export CANVAS_AGENT_ROOT="$(pwd)"
export PYTHONPATH="$CANVAS_AGENT_ROOT/LLaMA-Factory/src:$CANVAS_AGENT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
```

Configure CUDA:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Verify the environment:

```bash
python -c "import torch, ray, sglang, llamafactory, verl; print(torch.__version__, torch.version.cuda, ray.__version__, sglang.__version__)"
nvcc --version
```

## 2. Configure Paths

Replace the machine-specific absolute paths in the following files before
launching a job:

| File | Fields to update |
| --- | --- |
| `LLaMA-Factory/qwen3-vl.yaml` | `model_name_or_path`, `output_dir` |
| `LLaMA-Factory/data/dataset_info.json` | `file_name` for `smartagentV2-multiturn-reason` and `smartagentV2-val-reason` |
| `verl/examples/qwen3vl_multiturn/run_qwen3vl-8b_rl10k.sh` | environment activation, `PROJECT_DIR`, model path, training parquet, validation parquet |
| `verl/examples/qwen3vl_multiturn/config/tool_config/image_tools_config_matched.yaml` | tool model paths and GPU IDs |

Set the service credentials used by logging and reward scoring:

```bash
export WANDB_API_KEY="<wandb-api-key>"
export REWARD_API_KEY="<openai-compatible-api-key>"
export REWARD_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export REWARD_MODEL_NAME="qwen3.5-plus"
```

## 3. Run SFT

```bash
conda activate llama
cd "$CANVAS_AGENT_ROOT/LLaMA-Factory"
python -m llamafactory.cli train qwen3-vl.yaml
```

The output directory is controlled by `output_dir` in `qwen3-vl.yaml`.

## 4. Run RL

```bash
conda activate llama
cd "$CANVAS_AGENT_ROOT/verl/examples/qwen3vl_multiturn"

bash run_qwen3vl-8b_rl10k.sh \
  actor_rollout_ref.model.path=/path/to/sft/model \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/validation.parquet
```

The script writes stdout and rollout traces to
`$CANVAS_AGENT_ROOT/verl/examples/qwen3vl_multiturn/log/`. RL checkpoints are
written under the checkpoint directory configured by verl.
