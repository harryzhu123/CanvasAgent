#!/bin/bash
# Qwen3VL-8B RL-10K evaluation script
# This script runs validation only on the test parquet, saves:
# 1. Sample-level generations
# 2. Tool-agent traces and intermediate images
# 3. Parsed validation metrics
# 4. Full stdout logs

set -x

ulimit -n 65535

# Clean up old Ray session files to prevent disk from filling up
echo "Cleaning up old Ray session files in /tmp/ray ..."
rm -rf /tmp/ray
cd /nfsdata4/zhuhairui/verl
source /nfsdata4/zhuhairui/env/llama/bin/activate
# pip install -U wandb
# pip install protobuf==6.32.1
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
PROJECT_DIR="/nfsdata4/zhuhairui/verl"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
# Pure evaluation mode on this server (8 GPUs available):
#   physical GPU 0,1,2,3,4,5 -> main Qwen3-VL model (HF device_map=auto)
#   physical GPU 6 -> ImageGeneration + SR
#   physical GPU 7 -> ImageEdit + Grounding + SAM
MODEL_CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-$PROJECT_DIR/examples/qwen3vl_multiturn/config/tool_config/image_tools_config_matched_8gpu_eval.yaml}
echo $CUDA_HOME
nvcc --version
# Use /tmp for Ray object store instead of /dev/shm (avoids SIGBUS when /dev/shm is small)
# 32GB for larger batches (train_batch_size=6, rollout.n=8 → 48 trajectories)
export RAY_OBJECT_STORE_MEMORY=32000000000
ray stop --force || true
# Set wandb environment variables
export WANDB_PROJECT="qwen3vl-rl10kV2"
export WANDB_ENTITY=""

# Judge model: Qwen3-VL-Plus via DashScope OpenAI-compatible API
export REWARD_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export REWARD_TIMEOUT="180"
export REWARD_MODEL_NAME="qwen3.5-plus"
# Suppress modelscope verbose warnings (allow_remote, preprocessor, task schema, etc.)
export MODELSCOPE_LOG_LEVEL=40  # logging.ERROR=40; modelscope expects an integer, not a string like "ERROR"
# Generate timestamp for experiment name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Evaluation configuration
MODEL_PATH=${MODEL_PATH:-/nfsdata4/LLaVA-OneVision-7B}
RESUME_CKPT=${RESUME_CKPT:-}
VAL_FILE=${VAL_FILE:-/nfsdata4/zhuhairui/RL10kV2/data/verl_parquet/test_nfsdata4.parquet}
TRAIN_FILE=${TRAIN_FILE:-$VAL_FILE}
MODEL_PARENT_TAG=$(basename "$(dirname "$MODEL_PATH")")
MODEL_CKPT_TAG=$(basename "$MODEL_PATH")
RUN_TAG="${MODEL_PARENT_TAG}-${MODEL_CKPT_TAG}"
export WANDB_RUN_NAME="eval-rl10k-${RUN_TAG}-${TIMESTAMP}"

# Save all eval artifacts together
EVAL_ROOT="/nfsdata4/zhuhairui/verl/evals"
EVAL_DIR="$EVAL_ROOT/${WANDB_PROJECT}/${WANDB_RUN_NAME}"
VAL_GEN_DIR="$EVAL_DIR/val_generations"
VAL_TRACE_DIR="$EVAL_DIR/val_traces"
STDOUT_LOG="$EVAL_DIR/stdout_${TIMESTAMP}.log"
METRICS_JSON="$EVAL_DIR/val_metrics_step0.json"
mkdir -p "$EVAL_DIR" "$VAL_GEN_DIR" "$VAL_TRACE_DIR"
export VERL_TRACE_LOG_DIR="$EVAL_DIR"
echo "Model path: $MODEL_PATH"
echo "Resume checkpoint: ${RESUME_CKPT:-<disabled>}"
echo "Validation file: $VAL_FILE"
echo "Eval dir: $EVAL_DIR"
echo "Validation generations dir: $VAL_GEN_DIR"
echo "Validation trace dir: $VAL_TRACE_DIR"
echo "Stdout log: $STDOUT_LOG"
echo "Metrics json: $METRICS_JSON"
echo "Trace log dir: $VERL_TRACE_LOG_DIR"
echo "Model CUDA_VISIBLE_DEVICES: $MODEL_CUDA_VISIBLE_DEVICES"
echo "Tool config path: $TOOL_CONFIG_PATH"

if [ -n "$RESUME_CKPT" ]; then
    echo "Resume checkpoint is ignored in standalone eval mode: $RESUME_CKPT"
fi

CUDA_VISIBLE_DEVICES="$MODEL_CUDA_VISIBLE_DEVICES" python3 "$PROJECT_DIR/examples/qwen3vl_multiturn/standalone_eval_qwen3vl_rl10k.py" \
    --model-path "$MODEL_PATH" \
    --val-file "$VAL_FILE" \
    --eval-dir "$EVAL_DIR" \
    --val-generations-dir "$VAL_GEN_DIR" \
    --val-trace-dir "$VAL_TRACE_DIR" \
    --metrics-json "$METRICS_JSON" \
    --tool-config-path "$TOOL_CONFIG_PATH" \
    --reward-fn-path "$PROJECT_DIR/examples/qwen3vl_multiturn/multiturn_reward.py" \
    --reward-fn-name "compute_score" \
    --max-prompt-length 50000 \
    --max-response-length 12288 \
    --max-single-turn-length 1536 \
    --max-assistant-turns 12 \
    "$@" 2>&1 | tee "$STDOUT_LOG"

CMD_EXIT=${PIPESTATUS[0]}

exit "$CMD_EXIT"
