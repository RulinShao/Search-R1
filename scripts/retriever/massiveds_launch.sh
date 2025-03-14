#!/bin/bash
#SBATCH --job-name=massive-retriever
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         
#SBATCH --hint=nomultithread   
#SBATCH --account comem
#SBATCH --qos comem_high
#SBATCH --mem 1000G
#SBATCH --gres=gpu:1           
#SBATCH --time 120:00:00      
#SBATCH --requeue
#SBATCH --chdir=/fsx-comem/rulin/Search-R1
#SBATCH --output=/fsx-comem/rulin/Search-R1/outputs/slurm_cache/slurm-%A_%a.out
#SBATCH --array=0


cd /fsx-comem/rulin/Search-R1
source /home/rulin/miniconda3/bin/activate
conda activate searchr1



python /fsx-comem/rulin/Search-R1/search_r1/search/massiveds_main_node_server.py
