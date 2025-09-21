#!/bin/bash
#SBATCH -J te-cnn-swa
#SBATCH -p volta-gpu
#SBATCH --qos=gpu_access
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

export CONDA_NO_PLUGINS=1
export CONDA_SOLVER=classic
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
export CUDNN_BENCHMARK=1
unset LD_PRELOAD || true

export NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}
export SEED=${SEED:-42}

source /nas/longleaf/home/wang0207/miniconda3/etc/profile.d/conda.sh
conda activate cnn

cd "$SLURM_SUBMIT_DIR"

echo "Host: $(hostname)"
nvidia-smi -L || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY

IMG_ROOT="/work/users/w/a/wang0207/GHIST/Data Files and Submission Templates/ss_by_CNN/ms_png"

echo "=== START SWA TRAINING ==="
srun --cpu-bind=cores python cnn_train_ss.py \
  --root "${IMG_ROOT}" \
  --val-frac 0.10 \
  --epochs 12 \
  --batch-size 32 \
  --lr 1e-3 \
  --swa-start-frac 0.7 \
  --swa-lr 5e-4 \
  --resize 50x512
echo "=== DONE ==="