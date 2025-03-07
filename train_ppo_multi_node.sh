#!/bin/bash
#SBATCH --job-name=ppo_multi
#SBATCH --nodes=2                    # Changed to 2 nodes
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
#SBATCH --array=0



cd /fsx-comem/rulin/Search-R1
source /home/rulin/miniconda3/bin/activate
conda activate searchr1


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='data/nq_search'

WAND_PROJECT='Search-R1'

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em
export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# Ray multi-node setup
# Note: SLURM_NODEID is 0 for the first node and 1 for the second node
if [ "$SLURM_NODEID" == "0" ]; then
    # Head node setup
    ip_head=$(hostname -i)
    port=6379
    export ip_head=$ip_head:$port
    echo "Head node IP: $ip_head"
    
    # Start Ray head node
    ray start --head --port=$port --num-cpus=0 --block
    
    # Save head node info for worker nodes
    echo $ip_head > /fsx-comem/rulin/Search-R1/ray_head_ip
    
    # Wait for worker node to connect
    sleep 30
else
    # Worker node setup
    # Wait for head node to be ready
    sleep 10
    
    # Get the head node IP
    ip_head=$(cat /fsx-comem/rulin/Search-R1/ray_head_ip)
    echo "Connecting to head node: $ip_head"
    
    # Connect to the Ray cluster
    ray start --address=$ip_head --num-cpus=0 --block
fi

# Make sure all nodes are set up before proceeding
sleep 5

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
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
    trainer.experiment_name=${EXPERIMENT_NAME}_multi \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/${EXPERIMENT_NAME}_multi \
    max_turns=2 \
    retriever.url="http://rulin@a100-st-p4de24xlarge-946:38649/search" \
    retriever.topk=3 \
    2>&1 | tee ${EXPERIMENT_NAME}_multi.log