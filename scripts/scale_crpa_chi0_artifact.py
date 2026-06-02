#!/usr/bin/env python3
"""Create a diagnostic cRPA artifact with chi0 scaled by a scalar factor."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from mean_field.crpa.workflow import load_crpa_result, write_crpa_outputs


def _bare_v_from_artifact(epsilon: np.ndarray, screened_v: np.ndarray) -> np.ndarray:
    """Recover the diagonal bare V from screened_v = diag(V) @ epsilon_inv."""

    recovered = np.asarray(screened_v, dtype=np.complex128) @ np.asarray(epsilon, dtype=np.complex128)
    diag = np.diagonal(recovered, axis1=1, axis2=2)
    offdiag = recovered.copy()
    q_count, q_shift_count, _ = offdiag.shape
    idx = np.arange(q_shift_count)
    offdiag[:, idx, idx] = 0.0
    max_offdiag = float(np.max(np.abs(offdiag))) if offdiag.size else 0.0
    if max_offdiag > 1.0e-7:
        raise ValueError(f"Recovered bare V is not diagonal enough: max offdiag={max_offdiag:.3e}")
    return np.real(diag)


def _scaled_dielectric(chi0: np.ndarray, bare_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chi = np.asarray(chi0, dtype=np.complex128)
    v = np.asarray(bare_v, dtype=float)
    if chi.ndim != 3 or chi.shape[0] != v.shape[0] or chi.shape[1] != chi.shape[2] or chi.shape[1] != v.shape[1]:
        raise ValueError(f"Unexpected chi0/bare_v shapes: {chi.shape} vs {v.shape}")
    q_count, q_shift_count, _ = chi.shape
    eye = np.eye(q_shift_count, dtype=np.complex128)
    epsilon = np.empty_like(chi)
    epsilon_inv = np.empty_like(chi)
    screened_v = np.empty_like(chi)
    effective = np.empty((q_count, q_shift_count), dtype=float)
    for iq in range(q_count):
        v_diag = np.diag(v[iq].astype(np.complex128))
        eps = eye + chi[iq] @ v_diag
        eps_inv = np.linalg.inv(eps)
        epsilon[iq] = eps
        epsilon_inv[iq] = eps_inv
        screened_v[iq] = v_diag @ eps_inv
        effective[iq] = np.real(np.diag(eps))
    return epsilon, epsilon_inv, screened_v, effective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source cRPA artifact directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output diagnostic artifact directory.")
    parser.add_argument("--scale", type=float, required=True, help="Scalar multiplier applied to chi0.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    scale = float(args.scale)
    if scale <= 0.0:
        raise SystemExit("--scale must be positive")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}")

    result = load_crpa_result(source)
    bare_v = _bare_v_from_artifact(result.dielectric_matrix, result.screened_v)
    scaled_chi0 = np.asarray(result.chi0, dtype=np.complex128) * scale
    epsilon, epsilon_inv, screened_v, effective = _scaled_dielectric(scaled_chi0, bare_v)

    metadata = dict(result.metadata)
    metadata.update(
        {
            "diagnostic_chi0_scale": scale,
            "diagnostic_source_crpa_dir": str(source),
            "diagnostic_note": "chi0 was scaled after artifact generation; this is not a production cRPA calculation.",
        }
    )
    if "spin_degeneracy" in metadata:
        metadata["diagnostic_effective_spin_degeneracy_if_linear"] = float(metadata["spin_degeneracy"]) * scale

    scaled = replace(
        result,
        chi0=scaled_chi0,
        dielectric_matrix=epsilon,
        epsilon_inv=epsilon_inv,
        screened_v=screened_v,
        effective_epsilon=effective,
        metadata=metadata,
    )
    write_crpa_outputs(scaled, output)

    report = {
        "source": str(source),
        "output": str(output),
        "scale": scale,
        "effective_epsilon_min": float(np.min(effective)),
        "effective_epsilon_mean": float(np.mean(effective)),
        "effective_epsilon_max": float(np.max(effective)),
        "epsilon_times_bn_min": float(np.min(effective) * float(result.coulomb_params.epsilon_bn)),
        "epsilon_times_bn_mean": float(np.mean(effective) * float(result.coulomb_params.epsilon_bn)),
        "epsilon_times_bn_max": float(np.max(effective) * float(result.coulomb_params.epsilon_bn)),
    }
    (output / "diagnostic_chi0_scale_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
