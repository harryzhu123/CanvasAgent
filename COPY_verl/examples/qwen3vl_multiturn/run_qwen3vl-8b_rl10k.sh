#!/bin/bash
# Qwen3VL-8B RL Training TEST with RL-10K Dataset
# This is a TESTING script to verify:
# 1. Data loading works correctly
# 2. Tool classes can be initialized and deployed with Ray
# 3. ToolAgentLoop can parse tool calls and invoke tools
# 4. Basic training loop runs without errors
#
# Uses simplified reward (all = 1.0) for testing purposes

set -x

ulimit -n 65535

# Clean up old Ray session files to prevent disk from filling up
echo "Cleaning up old Ray session files in /tmp/ray ..."
rm -rf /tmp/ray
cd /nfsdata4/zhuhairui/verl
source /nfsdata4/zhuhairui/env/llama/bin/activate
pip install -U wandb
pip install protobuf==6.32.1
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo $CUDA_HOME
nvcc --version
# Use /tmp for Ray object store instead of /dev/shm (avoids SIGBUS when /dev/shm is small)
# 32GB for larger batches (train_batch_size=6, rollout.n=8 → 48 trajectories)
export RAY_OBJECT_STORE_MEMORY=32000000000
PROJECT_DIR="/nfsdata4/zhuhairui/verl"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
ray stop --force
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
export WANDB_RUN_NAME="test-rl10k-${TIMESTAMP}"

# Log directory setup
LOG_DIR="$PROJECT_DIR/examples/qwen3vl_multiturn/log"
mkdir -p "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/stdout_${TIMESTAMP}.log"
export VERL_TRACE_LOG_DIR="$LOG_DIR"
echo "Stdout log: $STDOUT_LOG"
echo "Trace log dir: $LOG_DIR"

# Training Configuration with RL-10K dataset
# GPU allocation: 8 GPUs total (6 training + 2 tools)
#   Training: 6 GPUs for FSDP actor/ref (GPU 0-5)
#   Tools: GPU 6 (ImageGen 0.5 + SR 0.5), GPU 7 (ImageEdit 0.5 + Grounding 0.25 + SAM 0.25)
#   Tools pinned via CUDA_VISIBLE_DEVICES in code, won't interfere with training
#
# Batch size hierarchy (4 GPUs, GRPO):
# real_train_batch_size = train_batch_size × rollout.n = 6 × 2 = 12 (test) / 6 × 8 = 48 (prod)
# ppo_mini_batch_size (6) = ppo_micro_batch_size_per_gpu (1) × n_gpus (6) × grad_accum (1)
#
# IMPORTANT: real_train_batch_size MUST be divisible by n_gpus (12/6=2 ✓)

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=6 \
    data.max_prompt_length=50000 \
    data.max_response_length=12288 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/nfsdata4/zhuhairui/zhuhairui/LLaMA-Factory/saves/qwen3-vl-8b/full/smartagentV3/checkpoint-2142 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.trace.backend=wandb \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/qwen3vl_multiturn/config/tool_config/image_tools_config_matched.yaml" \
    actor_rollout_ref.rollout.multi_turn.max_single_turn_length=1536 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=12 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    custom_reward_function.path="$PROJECT_DIR/examples/qwen3vl_multiturn/multiturn_reward.py" \
    custom_reward_function.name="compute_score" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='qwen3vl-rl10k-test' \
    trainer.experiment_name="test-rl10k-${TIMESTAMP}" \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    +trainer.validation_data_dir="$LOG_DIR/val_grpo_generations_${TIMESTAMP}" \
    +trainer.validation_trace_dir="$LOG_DIR/val_grpo_traces_${TIMESTAMP}" \
    +ray_kwargs.ray_init._plasma_directory=/tmp \
    data.train_files=/nfsdata4/zhuhairui/RL10kV2/data/verl_parquet/train_nfsdata4.parquet \
    data.val_files=/nfsdata4/zhuhairui/RL10kV2/data/verl_parquet/test_nfsdata4.parquet \
    trainer.total_epochs=1 $@ 2>&1 | tee "$STDOUT_LOG"




