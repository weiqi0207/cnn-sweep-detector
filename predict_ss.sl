#!/bin/bash
#SBATCH -J cnn-predict
#SBATCH -p volta-gpu
#SBATCH --qos=gpu_access
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

# ---------- conda ----------
source /nas/longleaf/home/wang0207/miniconda3/etc/profile.d/conda.sh
conda activate cnn

# ---------- paths ----------
CKPT="cnn_ms_sweep_SWA_xxxxxx.best.pt"   # replace with your trained checkpoint
META="cnn_ms_sweep_SWA_xxxxxx.meta.json" # matching meta
IMG_DIR="/work/users/w/a/wang0207/GHIST/Data Files and Submission Templates/ss_by_CNN/vcf_png"
OUT="vcf_png_predictions.csv"

# ---------- run prediction ----------
srun python predict_ss.py \
  --imgs "${IMG_DIR}" \
  --ckpt "${CKPT}" \
  --meta "${META}" \
  --out "${OUT}"