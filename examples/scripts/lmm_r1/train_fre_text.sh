# Download the datasets

#!/bin/bash
# =================== User Configuration ===================
# Please modify these variables according to your environment
# =========================================================

# Here are the key parameters that can effectively control running speed and memory
# usage:
# 
# Memory Usage Control
# 
# GPU Memory:
# - --vllm_gpu_memory_utilization 0.4 (line 83) - Controls vLLM GPU memory usage (currently 40%)
# - --zero_stage 3 (line 100) - DeepSpeed ZeRO stage for memory optimization
# - --gradient_checkpointing (line 111) - Trade computation for memory
# 
# Batch Sizes:
# - --micro_train_batch_size 2 (line 88) - Smaller = less memory
# - --micro_rollout_batch_size 2 (line 90) - Smaller = less memory
# - --train_batch_size 64 (line 89) - Total training batch size
# - --rollout_batch_size 64 (line 91) - Total rollout batch size
# 
# Speed Control
# 
# Parallelization:
# - --vllm_num_engines 8 (line 79) - Number of vLLM engines
# - --vllm_tensor_parallel_size 1 (line 80) - Tensor parallelism degree
# - Ray GPU allocation: --ref_num_gpus_per_node 8, --actor_num_gpus_per_node 8, --critic_num_gpus_per_node 8
# 
# Generation Parameters:
# - --n_samples_per_prompt 8 (line 93) - Samples per prompt (fewer = faster)
# - --generate_max_len 4096 (line 98) - Max generation length
# - --prompt_max_len 4096 (line 96) - Max prompt length
# 
# Training Scale:
# - --max_samples 10000 (line 97) - Total samples to process
# - --num_episodes 2 (line 95) - Number of training episodes
# 
# Key suggestions:
# - Reduce vllm_gpu_memory_utilization to 0.3 or lower if OOM
# - Decrease batch sizes if memory issues
# - Reduce n_samples_per_prompt for faster iteration
# - Adjust max_samples and num_episodes for shorter runs

# Base paths - MODIFY THESE
export WORKSPACE_DIR="$(pwd)"                      # Path to project root directory
export DATASET_PATH="$(pwd)/data/deepscaler/deepscaler_message.jsonl"  # Path to your dataset
export PRETRAIN_MODEL_PATH="/opt/local/llm_models/huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"  # Path to pretrained model
export SAVE_PATH="${WORKSPACE_DIR}/checkpoints"    # Absolute path to save checkpoints

# Model configuration
export MODEL_NAME="lmm-r1-fre-text"              # Name for this training run

# Wandb configuration (optional)
# export WANDB_DIR="${WORKSPACE_DIR}"                # Directory for wandb files
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"          # Your wandb API key (if online)
#    --use_wandb ${WANDB_API_KEY} \
#    --wandb_run_name ${MODEL_NAME} \
#    --wandb_group "lmm-r1-training" \

# =================== Script Execution ===================
# You shouldn't need to modify anything below this line
# ======================================================

# Get script PID and setup directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Stop any existing ray processes
ray stop

# Create necessary directories
mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${SAVE_PATH}/${MODEL_NAME}/ckpt"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"

# Print help information
echo "================================================================"
echo "LMM-R1 FRE-Text Training"
echo "================================================================"
echo "Model name: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
echo "Save path: ${SAVE_PATH}/${MODEL_NAME}"
echo "Checkpoint path: ${SAVE_PATH}/${MODEL_NAME}/ckpt"
echo "Logs will be saved to: ${CUR_LOG_DIR}"
echo
echo "Path verification:"
ls -la "${SAVE_PATH}" && echo "✓ Save directory exists" || echo "✗ Save directory missing"
ls -la "${SAVE_PATH}/${MODEL_NAME}" && echo "✓ Model directory exists" || echo "✗ Model directory missing"
ls -la "${SAVE_PATH}/${MODEL_NAME}/ckpt" && echo "✓ Checkpoint directory exists" || echo "✗ Checkpoint directory missing"
echo
echo "To monitor logs:"
echo "  tail -f ${CUR_LOG_DIR}/train.log"
echo
echo "================================================================"

# Start ray
echo "Starting ray..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir ~/.cache/ray

# Start remote reward model server
echo "Starting remote reward model server..."
python -m openrlhf.models.remote_rm.math_verifier \
    --input_key message \
    --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
REMOTE_RM_PID=$!


# Start training
echo "Starting training..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\",\"env_vars\":{\"VLLM_USE_V1\":\"1\",\"VLLM_ENABLE_V1_MULTIPROCESSING\":\"0\"}}" \
   -- python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.4 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain ${PRETRAIN_MODEL_PATH} \
   --save_path ${SAVE_PATH}/${MODEL_NAME} \
   --micro_train_batch_size 2 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 64 \
   --temperature 1.0 \
   --n_samples_per_prompt 4 \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 2048 \
   --max_samples 10000 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce_baseline \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 4e-7 \
   --init_kl_coef 0.001 \
   --prompt_data ${DATASET_PATH} \
   --input_key message \
   --label_key "answer" \
   --normalize_reward \
   --flash_attn \
   --lambd 1 \
   --gamma 1 \
   --gradient_checkpointing \
   --save_steps 1 \
   --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
   --max_ckpt_num 1 \
   --save_hf_ckpt \
   --load_checkpoint \
   --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &

TRAIN_PID=$!

# Record process IDs
echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

# Wait for training to complete
echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"

# Uncomment to wait for training to complete before exiting
# wait $TRAIN_PID

# Cleanup instructions
echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "ray stop"
echo "All logs are available in ${CUR_LOG_DIR}"