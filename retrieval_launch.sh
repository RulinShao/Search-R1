#!/bin/bash
#SBATCH --job-name=retriever
#SBATCH --nodes=1
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
conda activate retriever



file_path=/fsx-comem/rulin/Search-R1/index
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever
