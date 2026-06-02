#!/bin/bash
set -euo pipefail

ROOT="results/TBG_HF_cRPA/crpa_gap_smoke_20260523_fockshellfix_lk24"

python -m pytest -q tests/test_crpa_core.py \
  -k 'remote_reference_eq19_direct_terms_match_half_reference_delta or crpa_fock_screening_preserves_active_bare_fock_shell or crpa_split_with_identity_screening_matches_bare_delta_hf'

python scripts/plot_tbg_hf_scf_grid_path_from_state.py --root "${ROOT}"

find "${ROOT}" -maxdepth 3 -name '*scf_grid_band_plot_summary.json' -type f -print | sort
