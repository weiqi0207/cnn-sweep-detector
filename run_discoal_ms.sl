#!/bin/bash
#SBATCH --job-name=discoal_ms
#SBATCH --output=discoal_ms.%A_%a.out
#SBATCH --error=discoal_ms.%A_%a.err
#SBATCH --array=0-363
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --signal=TERM@120

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CSV="GHIST_2025_singlesweep.21.w100k.minimal.csv"   # columns: SNP,thetaW,rho
OUT_ROOT="/work/users/w/a/wang0207/GHIST/Data Files and Submission Templates/ss_by_CNN"
OUT_DIR="${OUT_ROOT}/ms_out"
N_CHROM=50
REPS_SWEEP=10
REPS_OTHER=10
BASE_SEED=2025

DISCOAL="/work/users/w/a/wang0207/GHIST/Data Files and Submission Templates/discoal/discoal"
if [[ ! -x "$DISCOAL" ]]; then
  echo "ERROR: discoal is not executable at: $DISCOAL" >&2
  ls -l "$DISCOAL" || true
  exit 1
fi
if [[ ! -f "$CSV" ]]; then
  echo "ERROR: CSV not found: $CSV" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

module purge
module load gcc
source /nas/longleaf/home/wang0207/miniconda3/etc/profile.d/conda.sh
conda activate demoinf2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Accessible region and winsize for coordinate reconstruction
CHROM="21"
REGION_START=10326675
REGION_END=46709983
WINSIZE=100000

IDX=${SLURM_ARRAY_TASK_ID}
echo "[$(date)] START idx=${IDX} on $(hostname)"

srun -u python discoal_ms_from_windows.py \
  --csv "${CSV}" \
  --idx "${IDX}" \
  --discoal "${DISCOAL}" \
  --out_dir "${OUT_DIR}" \
  --n_chrom "${N_CHROM}" \
  --reps_sweep "${REPS_SWEEP}" \
  --reps_other "${REPS_OTHER}" \
  --seed "$((BASE_SEED + IDX))" \
  --chrom "${CHROM}" \
  --region-start "${REGION_START}" \
  --region-end "${REGION_END}" \
  --winsize "${WINSIZE}"

echo "[$(date)] DONE idx=${IDX} -> ${OUT_DIR}"