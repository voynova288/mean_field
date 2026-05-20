#!/usr/bin/env bash
set -euo pipefail

if [[ -d "${PWD}/src/mean_field" ]]; then
  REPO_ROOT="${PWD}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/RnG_hBN/fig6_hf_bands_parallel_20260516_161014}"
TAG="${TAG:-periodic_gauge_validation_$(date +%Y%m%d_%H%M%S)}"
CACHE="${CACHE:-${REPO_ROOT}/results/RnG_hBN/cache_fig6_${TAG}}"
SNAPSHOT_ROOT="${OUT_ROOT}/screened_base_band_snapshots"
MESH_OUT="${MESH_OUT:-${SNAPSHOT_ROOT}/${TAG}_mesh}"
CONTINUOUS_OUT="${CONTINUOUS_OUT:-${SNAPSHOT_ROOT}/${TAG}_continuous}"
PANELS="${PANELS:-xi0_V064meV,xi1_V064meV}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-48}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
DPI="${DPI:-180}"
YLIM_MEV="${YLIM_MEV:--120,120}"

mkdir -p "${CACHE}" "${MESH_OUT}" "${CONTINUOUS_OUT}"

echo "[validation] out_root=${OUT_ROOT}"
echo "[validation] cache=${CACHE}"
echo "[validation] mesh_out=${MESH_OUT}"
echo "[validation] continuous_out=${CONTINUOUS_OUT}"
echo "[validation] panels=${PANELS}"
echo "[validation] hostname=$(hostname)"
echo "[validation] cpus=${SLURM_CPUS_PER_TASK:-}"

python scripts/plot_rlg_hbn_screened_base_bands.py \
  --output-root "${OUT_ROOT}" \
  --output-dir "${MESH_OUT}" \
  --panels "${PANELS}" \
  --cache-dir "${CACHE}" \
  --cache-policy refresh \
  --path-mode mesh \
  --chunk-size "${CHUNK_SIZE}" \
  --ylim-mev="${YLIM_MEV}" \
  --dpi "${DPI}"

python scripts/plot_rlg_hbn_screened_base_bands.py \
  --output-root "${OUT_ROOT}" \
  --output-dir "${CONTINUOUS_OUT}" \
  --panels "${PANELS}" \
  --cache-dir "${CACHE}" \
  --cache-policy reuse \
  --path-mode continuous \
  --points-per-segment "${POINTS_PER_SEGMENT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --ylim-mev="${YLIM_MEV}" \
  --dpi "${DPI}"

python - "${MESH_OUT}" "${CONTINUOUS_OUT}" "${TAG}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


def _load_panel(path: Path, panel: str) -> dict[str, object]:
    archive = np.load(path / panel / "screened_base_bands_path.npz")
    return {name: np.asarray(archive[name]) for name in archive.files}


def _gamma_mprime_indices(kvec_pairs: np.ndarray) -> np.ndarray:
    kvec = np.asarray(kvec_pairs, dtype=float)
    # This path is Gamma-K-Kprime-Gamma-Mprime-M-Gamma.  The target segment is
    # the long contiguous block after the second Gamma with x-like coordinate
    # near zero in the stored complex-pair representation.
    x = kvec[:, 0]
    y = kvec[:, 1]
    candidates = np.flatnonzero((np.abs(x) <= 1.0e-10) & (y >= -1.0e-10))
    blocks: list[np.ndarray] = []
    current: list[int] = []
    previous = None
    for idx in candidates:
        if previous is None or idx == previous + 1:
            current.append(int(idx))
        else:
            if current:
                blocks.append(np.asarray(current, dtype=int))
            current = [int(idx)]
        previous = int(idx)
    if current:
        blocks.append(np.asarray(current, dtype=int))
    useful = [block for block in blocks if block.size >= 3]
    if not useful:
        return candidates
    return max(useful, key=lambda block: (block.size, block[0]))


def _roughness(values: np.ndarray, indices: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)[:, indices]
    if data.shape[1] < 3:
        return 0.0
    second = data[:, 2:] - 2.0 * data[:, 1:-1] + data[:, :-2]
    return float(np.max(np.abs(second)))


def _panel_metrics(path: Path, panel: str) -> dict[str, object]:
    payload = _load_panel(path, panel)
    indices = _gamma_mprime_indices(payload["kvec_nm_inv"])
    metrics: dict[str, object] = {
        "panel": panel,
        "points": int(payload["kdist"].size),
        "gamma_mprime_index_start": int(indices[0]) if indices.size else None,
        "gamma_mprime_index_stop": int(indices[-1]) if indices.size else None,
        "gamma_mprime_points": int(indices.size),
    }
    for name in ("spin_up_K_energies_mev", "spin_up_Kprime_energies_mev"):
        metrics[name.replace("_energies_mev", "_roughness_mev")] = _roughness(payload[name], indices)
    return metrics


mesh_out = Path(sys.argv[1])
continuous_out = Path(sys.argv[2])
tag = sys.argv[3]
panels = ["xi0_V064meV", "xi1_V064meV"]
report = {
    "tag": tag,
    "mesh_output_dir": str(mesh_out),
    "continuous_output_dir": str(continuous_out),
    "metric": "max absolute second finite difference on detected Gamma-to-Mprime segment",
    "mesh": [_panel_metrics(mesh_out, panel) for panel in panels],
    "continuous": [_panel_metrics(continuous_out, panel) for panel in panels],
}
report_path = mesh_out.parent / f"{tag}_validation_report.json"
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[validation] report={report_path}")
print(json.dumps(report, indent=2, sort_keys=True))
PY
