#!/bin/bash
#SBATCH --job-name=ptse2_v25_postflight
#SBATCH --account=hmt03
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v25_job513422_postflight/RECOMPUTE.json
#SBATCH --error=/data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v25_job513422_postflight/RECOMPUTE.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname >&2
/data/home/ziyuzhu/miniconda3/envs/moirekp/bin/python3.11 /data/home/ziyuzhu/Mean_Field/reviews/ptse2_physical_three_subcell_shell6_v25_job513422_postflight/recompute_matched_r6.py
