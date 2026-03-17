#!/usr/bin/env bash
#SBATCH --job-name=prescience-historical
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=xlab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/mmfs1/gscratch/socialrl/jiayiy9/prescience/logs/slurm-historical-%j.out
#SBATCH --error=/mmfs1/gscratch/socialrl/jiayiy9/prescience/logs/slurm-historical-%j.err

# ── Environment ───────────────────────────────────────────────────────────────
source /mmfs1/gscratch/socialrl/jiayiy9/anaconda3/etc/profile.d/conda.sh
conda activate sci4sci

# S2 API key — set this before submitting, e.g.:
#   export S2_API_KEY=<your_key>
#   sbatch run_pipeline_historical_slurm.sh
if [[ -z "$S2_API_KEY" ]]; then
    echo "ERROR: S2_API_KEY not set. Submit with: S2_API_KEY=<key> sbatch ..."
    exit 1
fi

cd /mmfs1/gscratch/socialrl/jiayiy9/sci4sci/prescience
mkdir -p /mmfs1/gscratch/socialrl/jiayiy9/prescience/logs

python3 -m dataset.analysis.compute_disruption_index --hf_repo_id allenai/prescience --split train --window_months 24 --output_path data/di/train/

echo "All done."
