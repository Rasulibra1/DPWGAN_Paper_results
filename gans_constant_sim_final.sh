#!/bin/bash
#SBATCH --job-name=gan_high_order_constant
#SBATCH --partition=standard          # <-- change to your CPU partition name
#SBATCH --account=AKENNEY1_LAB                # <-- your real account
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # 1 task
#SBATCH --cpus-per-task=4            # e.g. 4 CPU cores (adjust as needed)
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ibragimr@uci.edu

module purge
module load python/3.10.2
#cd /data/homezvol2/ibragimr/paper-results-vinecops
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install statsmodels
python dpwgan_constant.py

