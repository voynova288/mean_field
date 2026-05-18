#!/bin/bash
# Verify the required cached Fig. 6 HF band outputs under one result directory.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-dir>" >&2
  exit 2
fi

OUT="$1"

required=(
  "paper_fig6_hf_bands.png"
  "paper_fig6_hf_bands.pdf"
  "hf_band_plot_summary.json"
  "cache_manifest.json"
  "xi0_V064meV/hf_ground_state.npz"
  "xi1_V064meV/hf_ground_state.npz"
  "xi0_V064meV/hf_bands_path.npz"
  "xi1_V064meV/hf_bands_path.npz"
  "xi0_V064meV/hf_convergence.json"
  "xi1_V064meV/hf_convergence.json"
)

for relpath in "${required[@]}"; do
  test -f "${OUT}/${relpath}"
  echo "[ok] ${OUT}/${relpath}"
done

echo "[done] Fig. 6 required outputs verified under ${OUT}"
