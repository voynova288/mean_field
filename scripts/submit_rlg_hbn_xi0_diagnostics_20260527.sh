#!/bin/bash
# Submit two RLG/hBN xi=0 Fig. 6 diagnostics:
#   1. zero only the literal q=0 Fock matrix element, keep q=0 Hartree/screening;
#   2. active-window robustness test with a (4+4) projected basis.
# This script submits Slurm jobs only; it does not run numerical work itself.

set -euo pipefail

REPO_ROOT="/data/home/ziyuzhu/Mean_Field"
cd "${REPO_ROOT}"
mkdir -p logs results/RnG_hBN

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-/data/home/ziyuzhu/miniconda3/bin/python3}"
CACHE_FIG6="${CACHE_FIG6:-${REPO_ROOT}/results/RnG_hBN/cache_fig6_periodic_gauge_v2_20260520_periodic_gauge_v2_bugfix}"
CACHE_4P4="${CACHE_4P4:-${REPO_ROOT}/results/RnG_hBN/cache_fig6_active4p4_${STAMP}}"
Q0_OUT="${Q0_OUT:-${REPO_ROOT}/results/RnG_hBN/diag_xi0_q0fock_zero_${STAMP}}"
A4_OUT="${A4_OUT:-${REPO_ROOT}/results/RnG_hBN/diag_xi0_active4p4_${STAMP}}"
MANIFEST="${REPO_ROOT}/results/RnG_hBN/rlg_xi0_diagnostics_submission_${STAMP}.tsv"

printf 'diagnostic\tjob_id\toutput_dir\tcache_dir\tflags\n' > "${MANIFEST}"

Q0_JOB=$(sbatch --parsable \
  --account=hmt03 \
  -J rlg_xi0_q0fock0 \
  -p regular6430 \
  -N 1 \
  --ntasks=1 \
  --cpus-per-task=64 \
  --mem=0 \
  -t 1-12:00:00 \
  -o logs/rlg_xi0_q0fock0_%j.out \
  -e logs/rlg_xi0_q0fock0_%j.err \
  --export=ALL,MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1,MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS=16,MEAN_FIELD_RLG_HBN_USE_NUMBA=1,MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1 \
  scripts/submit_mean_field.sbatch \
  "${PYTHON}" -m mean_field.devtools.run_rlg_hbn_paper_hf \
    --paper-target fig6 \
    --output-dir "${Q0_OUT}" \
    --cache-dir "${CACHE_FIG6}" \
    --cache-policy reuse \
    --screening-solver grid \
    --screening-u-min-mev -100 \
    --screening-u-max-mev 200 \
    --screening-u-grid-points 121 \
    --skip-screening-check \
    --xi-values 0 \
    --v-values-mev 64 \
    --run-specs flavor:1 \
    --max-iter 80 \
    --precision 1e-4 \
    --checkpoint-interval 5)
printf 'q0_fock_zero\t%s\t%s\t%s\tMEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1;active=3+3\n' "${Q0_JOB}" "${Q0_OUT}" "${CACHE_FIG6}" >> "${MANIFEST}"

A4_JOB=$(sbatch --parsable \
  --account=hmt03 \
  -J rlg_xi0_act4p4 \
  -p regular6430 \
  -N 1 \
  --ntasks=1 \
  --cpus-per-task=64 \
  --mem=0 \
  -t 2-00:00:00 \
  -o logs/rlg_xi0_act4p4_%j.out \
  -e logs/rlg_xi0_act4p4_%j.err \
  --export=ALL,MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=0,MEAN_FIELD_RLG_HBN_REMOTE_CHUNK_BANDS=16,MEAN_FIELD_RLG_HBN_USE_NUMBA=1,MEAN_FIELD_RLG_HBN_REQUIRE_NUMBA=1 \
  scripts/submit_mean_field.sbatch \
  "${PYTHON}" -m mean_field.devtools.run_rlg_hbn_paper_hf \
    --paper-target fig6 \
    --output-dir "${A4_OUT}" \
    --cache-dir "${CACHE_4P4}" \
    --cache-policy reuse \
    --screening-solver grid \
    --screening-u-min-mev -100 \
    --screening-u-max-mev 200 \
    --screening-u-grid-points 121 \
    --skip-screening-check \
    --xi-values 0 \
    --v-values-mev 64 \
    --active-valence-bands 4 \
    --active-conduction-bands 4 \
    --k-mesh-size 18 \
    --run-specs flavor:1 \
    --max-iter 80 \
    --precision 1e-4 \
    --checkpoint-interval 5)
printf 'active4p4\t%s\t%s\t%s\tactive=4+4;zero_q0_fock=0\n' "${A4_JOB}" "${A4_OUT}" "${CACHE_4P4}" >> "${MANIFEST}"

echo "[submitted] manifest=${MANIFEST}"
cat "${MANIFEST}"
