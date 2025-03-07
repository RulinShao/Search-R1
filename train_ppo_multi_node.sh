#!/bin/bash
#SBATCH --job-name=ppo_array
#SBATCH --nodes=1                    # One node per task
#SBATCH --ntasks-per-node=1         
#SBATCH --hint=nomultithread   
#SBATCH --account comem
#SBATCH --qos comem_high
#SBATCH --mem 1000G
#SBATCH --gres=gpu:8           
#SBATCH --time 120:00:00      
#SBATCH --requeue
#SBATCH --chdir=/fsx-comem/rulin/Search-R1
#SBATCH --output=/fsx-comem/rulin/Search-R1/outputs/slurm_cache/slurm-%A_%a.out
#SBATCH --array=0-1              # Two array tasks: 0 for head, 1 for worker

export DATA_DIR='data/nq_search'
export VLLM_ATTENTION_BACKEND=XFORMERS

WAND_PROJECT='Search-R1'
export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-it-em-multi

# Initialize Python environment
cd /fsx-comem/rulin/Search-R1
source /home/rulin/miniconda3/bin/activate
conda activate searchr1

# Shared files for coordination
COORDINATOR_DIR="/fsx-comem/rulin/Search-R1/coordination/${SLURM_ARRAY_JOB_ID}"
HEAD_IP_FILE="${COORDINATOR_DIR}/ray_head_ip"
WORKER_READY_FILE="${COORDINATOR_DIR}/worker_ready"
HEAD_READY_FILE="${COORDINATOR_DIR}/head_ready"
TRAINING_STARTED_FILE="${COORDINATOR_DIR}/training_started"

# Create coordination directory
mkdir -p ${COORDINATOR_DIR}

# If this is the head node (array task 0)
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    echo "Running as HEAD node (array task 0)"
    
    # Start Ray head node
    ip_head=$(hostname -i)
    echo "Head node IP: $ip_head"
    echo "$ip_head" > $HEAD_IP_FILE
    
    # Start Ray on head node with CPU resources
    ray start --head --port=6379 --num-cpus=4 --block &
    RAY_PID=$!
    
    # Wait for Ray to initialize
    sleep 20
    
    # Check Ray status
    echo "Checking Ray cluster status..."
    ray status || echo "Warning: Ray status check failed, but continuing..."
    
    # Signal that head node is ready
    touch $HEAD_READY_FILE
    
    # Wait for worker node to connect
    echo "Waiting for worker node to connect..."
    MAX_WAIT=300  # 5 minutes timeout
    WAIT_COUNT=0
    
    while [ ! -f $WORKER_READY_FILE ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        sleep 10
        WAIT_COUNT=$((WAIT_COUNT+10))
        echo "Waiting for worker node... $WAIT_COUNT seconds elapsed"
    done
    
    if [ ! -f $WORKER_READY_FILE ]; then
        echo "ERROR: Worker node did not connect within timeout. Exiting."
        kill $RAY_PID
        exit 1
    fi
    
    echo "Worker node connected! Starting training process."
    touch $TRAINING_STARTED_FILE
    
    # Start the training process
    echo "Starting training on head node..."
    export RAY_ADDRESS="auto"  # Ensure Python connects to the existing Ray cluster
    
    python3 -u -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=1024 \
        data.val_batch_size=512 \
        data.max_prompt_length=4096 \
        data.max_response_length=500 \
        data.max_start_length=2048 \
        data.max_obs_length=500 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=gae \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
        actor_rollout_ref.actor.ppo_mini_batch_size=512 \
        actor_rollout_ref.actor.ppo_micro_batch_size=128 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.grad_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=256 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=256 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.n_agent=2 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.actor.state_masking=true \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.optim.lr_warmup_steps_ratio=0.05 \
        critic.model.path=$BASE_MODEL \
        critic.model.enable_gradient_checkpointing=true \
        critic.ppo_micro_batch_size=16 \
        critic.model.fsdp_config.param_offload=true \
        critic.model.fsdp_config.grad_offload=true \
        critic.model.fsdp_config.optimizer_offload=true \
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.no_think_rl=false \
        trainer.critic_warmup=0 \
        trainer.logger=['wandb'] \
        +trainer.val_only=false \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=100 \
        trainer.test_freq=100 \
        trainer.project_name=$WAND_PROJECT \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=15 \
        trainer.total_training_steps=300 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
        max_turns=2 \
        retriever.url="http://rulin@a100-st-p4de24xlarge-946:38649/search" \
        retriever.topk=3 \
        2>&1 | tee ${COORDINATOR_DIR}/${EXPERIMENT_NAME}.log
        
    # Clean up
    echo "Training completed, cleaning up Ray processes"
    kill $RAY_PID
    
else
    # This is the worker node (array task 1)
    echo "Running as WORKER node (array task 1)"
    
    # Wait for head node to be ready and provide its IP
    echo "Waiting for head node to initialize Ray..."
    MAX_WAIT=300  # 5 minutes timeout
    WAIT_COUNT=0
    
    while [ ! -f $HEAD_READY_FILE ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        sleep 10
        WAIT_COUNT=$((WAIT_COUNT+10))
        echo "Waiting for head node... $WAIT_COUNT seconds elapsed"
        # Check if the IP file exists even if the ready file doesn't
        if [ -f $HEAD_IP_FILE ]; then
            echo "Found head IP file but no ready signal yet"
        fi
    done
    
    if [ ! -f $HEAD_READY_FILE ]; then
        echo "ERROR: Head node did not initialize within timeout. Exiting."
        exit 1
    fi
    
    # Get the head node IP
    ip_head=$(cat $HEAD_IP_FILE)
    echo "Found head node IP: $ip_head"
    
    # Connect to the Ray cluster
    echo "Connecting to Ray head node at $ip_head:6379"
    ray start --address=$ip_head:6379 --num-cpus=4 --block &
    RAY_PID=$!
    
    # Wait to ensure connection is established
    sleep 10
    
    # Verify connection to Ray cluster
    if ray status > /dev/null 2>&1; then
        echo "Successfully connected to Ray cluster"
    else
        echo "WARNING: Could not verify Ray connection, but continuing..."
    fi
    
    # Signal to the head node that we're ready
    touch $WORKER_READY_FILE
    echo "Worker node connected and ready - signaled to head node"
    
    # Wait for training to start
    echo "Waiting for training to start..."
    while [ ! -f $TRAINING_STARTED_FILE ]; do
        sleep 5
    done
    echo "Training has started. Worker node is active."
    
    # Keep worker alive and monitor Ray connection
    while true; do
        # Check if training is still running (head node still exists)
        if ! squeue -j ${SLURM_ARRAY_JOB_ID}_0 > /dev/null 2>&1; then
            echo "Head node job no longer exists. Exiting worker."
            kill $RAY_PID
            exit 0
        fi
        
        # Log that we're still alive
        echo "Worker node active at $(date)"
        sleep 300  # Check every 5 minutes
    done
fi