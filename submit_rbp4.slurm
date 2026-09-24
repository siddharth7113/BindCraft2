#!/bin/bash
#SBATCH --job-name=bc2_rbp4
#SBATCH --partition=main
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=0                 # 0 = no limit; partition 'main' has MaxTime=UNLIMITED
#SBATCH --output=bindcraft_%j.out
#
# Site submit script for AIMecoSPECS. Use INSTEAD of the shipped bindcraft.slurm, which
# requests 4 cores + 24 GB *per GPU* (16 cores / 96 GB for 4 GPUs) and 8 cores / 48 GB as its
# #SBATCH fallback. This node has 4 cores and 31 GB in total, so every one of those figures is
# unschedulable and the job would pend indefinitely rather than fail.
#
#   sbatch submit_rbp4.slurm campaigns/rbp4_peptide.json
#   sbatch submit_rbp4.slurm campaigns/rbp4_peptide.json number_of_final_designs=20
#   sbatch submit_rbp4.slurm campaigns/rbp4_peptide.json --set 'binder_lengths=[12,25]'
#
# No wall limit is imposed (--time=0). A peptide campaign at 1200 trajectories runs ~25 h on the
# three 8 GB cards, and campaigns resume by default, so a limit would only mean resubmitting.
#
# Why --gres=gpu:4 when only three cards are used: gres.conf declares the GPUs without a Type=
# field, so Slurm cannot be asked for "a 3070" and --gres=gpu:3 may hand over the 6 GB GTX 1660 Ti.
# Taking all four and selecting the >=8 GB cards in env.sh is the only reliable way to keep the
# 1660 Ti out. The node has 4 cores total, so this is a whole-node job either way.
set -euo pipefail

cd "${BINDCRAFT_HOME:-${SLURM_SUBMIT_DIR:-$PWD}}"
source ./env.sh                       # venv + in-repo weights/compile cache/TMPDIR + GPU selection

echo "bindcraft: $(hostname), job ${SLURM_JOB_ID:-none}, $(pwd)"
echo "bindcraft: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, using GPUs ${BINDCRAFT_GPU_IDS:-all}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# bindcraft.py (not the console script) rewrites a bare KEY=VALUE into --set KEY=VALUE
# (bindcraft.py:21-25), so both spellings work here. The console script accepts only --set.
python bindcraft.py "$@"
