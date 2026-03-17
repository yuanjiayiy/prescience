#!/usr/bin/env bash
#SBATCH --job-name=compute-embedding
#SBATCH --partition=gpu-a100                     
#SBATCH --nodes=1                          
#SBATCH --ntasks-per-node=1                
#SBATCH --account=xlab
#SBATCH --cpus-per-task=32                 # CPU cores per task (adjusted for 2 GPUs)
#SBATCH --gres=gpu:4                        # Number of GPUs per node
#SBATCH --mem=200G                          # Memory per node (adjusted for 2 GPUs)
#SBATCH --time=4-00:00:00                  # 3 days - adjust batch sizes/epochs to fit
#SBATCH --output=/mmfs1/gscratch/socialrl/jiayiy9/prescience/logs/slurm-%x-%j.out  # Output file in logs directory
#SBATCH --error=/mmfs1/gscratch/socialrl/jiayiy9/prescience/logs/slurm-%x-%j.err   # Error file in logs directory

SPLIT=test
EMBEDDING_TYPE=grit
OUTPUT_DIR=data/corpus/$SPLIT
python3 -m dataset.embeddings.compute_paper_embeddings --split $SPLIT --embedding_type $EMBEDDING_TYPE --output_dir $OUTPUT_DIR
