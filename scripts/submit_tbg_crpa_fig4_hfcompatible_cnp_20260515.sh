#!/bin/bash
# Submit the Zhang Appendix Fig. 4 HF-compatible cRPA + HF chain after the
# cnp_index occupation fix.

set -euo pipefail

export CRPA_HF_COMPATIBLE="${CRPA_HF_COMPATIBLE:-1}"
export CRPA_CONVENTION_TAG="${CRPA_CONVENTION_TAG:-hf_compatible_cnpindex_20260515}"
export CRPA_RUN_TAG_SUFFIX="${CRPA_RUN_TAG_SUFFIX:-crpa_hfcompat_cnp_lk24_q11_20260515_gamma_m_k_gamma_kprime}"
export ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
export CHUNKS_PER_NODE="${CHUNKS_PER_NODE:-2}"
export HF_ARRAY_CONCURRENCY="${HF_ARRAY_CONCURRENCY:-1}"
export Q_LG="${Q_LG:-11}"
export LK="${LK:-24}"
export LG="${LG:-9}"
export OVERLAP_LG="${OVERLAP_LG:-9}"
export MAX_ITER="${MAX_ITER:-3000}"
export FOCK_INTERPOLATION="${FOCK_INTERPOLATION:-matrix_diagonal}"

exec bash scripts/submit_tbg_crpa_fig4_pipeline.sh "${1:-submit-all}"
