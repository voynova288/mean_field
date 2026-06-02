#!/usr/bin/env python3
"""Create a plotting/legacy diagnostic artifact using diag(screened_v).

Current production Fock lookup uses ``epsilon_inv`` directly and does not need
this rewritten artifact.  This script is retained only to build legacy plotting
or comparison artifacts whose ``effective_epsilon`` field represents the
diagonal screened-interaction scalar

    epsilon_eff = V_bare / Re diag(screened_v),

instead of ``Re diag(epsilon)``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from mean_field.crpa.workflow import load_crpa_result, write_crpa_outputs


def _bare_v_from_artifact(epsilon: np.ndarray, screened_v: np.ndarray) -> tuple[np.ndarray, float]:
    """Recover diagonal bare V from screened_v = diag(V) @ epsilon_inv."""

    recovered = np.asarray(screened_v, dtype=np.complex128) @ np.asarray(epsilon, dtype=np.complex128)
    diag = np.diagonal(recovered, axis1=1, axis2=2)
    offdiag = recovered.copy()
    q_count, q_shift_count, _ = offdiag.shape
    idx = np.arange(q_shift_count)
    offdiag[:, idx, idx] = 0.0
    max_offdiag = float(np.max(np.abs(offdiag))) if q_count and q_shift_count else 0.0
    if max_offdiag > 1.0e-7:
        raise ValueError(f"Recovered bare V is not diagonal enough: max offdiag={max_offdiag:.3e}")
    return np.real(diag), max_offdiag


def _fock_effective_from_screened_diag(
    *,
    bare_v: np.ndarray,
    screened_v: np.ndarray,
    min_screened_diag: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    screened_diag = np.diagonal(np.asarray(screened_v, dtype=np.complex128), axis1=1, axis2=2)
    screened_diag_real = np.real(screened_diag)
    screened_diag_imag_max = float(np.max(np.abs(np.imag(screened_diag)))) if screened_diag.size else 0.0

    valid = np.isfinite(screened_diag_real) & (np.abs(screened_diag_real) > float(min_screened_diag))
    valid &= np.isfinite(bare_v)
    invalid_count = int(np.size(valid) - np.count_nonzero(valid))
    if invalid_count:
        min_abs = float(np.min(np.abs(screened_diag_real))) if screened_diag_real.size else float("nan")
        raise ValueError(
            "Cannot build Fock effective epsilon from diag(screened_v): "
            f"{invalid_count} invalid entries, min |diag(screened_v)|={min_abs:.3e}"
        )

    effective = np.asarray(bare_v, dtype=float) / screened_diag_real
    if not np.all(np.isfinite(effective)):
        raise ValueError("Derived Fock effective epsilon contains non-finite values.")

    report = {
        "screened_v_diag_imag_max": screened_diag_imag_max,
        "screened_v_diag_real_min": float(np.min(screened_diag_real)),
        "screened_v_diag_real_mean": float(np.mean(screened_diag_real)),
        "screened_v_diag_real_max": float(np.max(screened_diag_real)),
        "invalid_screened_diag_count": invalid_count,
    }
    return effective, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source cRPA artifact directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output diagnostic artifact directory.")
    parser.add_argument(
        "--min-screened-diag",
        type=float,
        default=1.0e-14,
        help="Minimum allowed absolute value for Re diag(screened_v).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}")

    result = load_crpa_result(source)
    bare_v, bare_v_offdiag_max = _bare_v_from_artifact(result.dielectric_matrix, result.screened_v)
    old_effective = np.asarray(result.effective_epsilon, dtype=float)
    new_effective, diag_report = _fock_effective_from_screened_diag(
        bare_v=bare_v,
        screened_v=result.screened_v,
        min_screened_diag=float(args.min_screened_diag),
    )
    strength_ratio = old_effective / new_effective

    metadata = dict(result.metadata)
    metadata.update(
        {
            "diagnostic_fock_effective_epsilon": "bare_v_over_real_diag_screened_v",
            "diagnostic_source_crpa_dir": str(source),
            "diagnostic_note": (
                "Only the legacy/plotting effective_epsilon field was replaced; current production "
                "Fock lookup reads epsilon_inv directly. Full Hartree screened_v, dielectric_matrix, "
                "epsilon_inv, and chi0 were kept from the source artifact."
            ),
        }
    )

    modified = replace(result, effective_epsilon=new_effective, metadata=metadata)
    write_crpa_outputs(modified, output)

    report: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "bare_v_recovery_offdiag_max_abs": bare_v_offdiag_max,
        "old_effective_epsilon_min": float(np.min(old_effective)),
        "old_effective_epsilon_mean": float(np.mean(old_effective)),
        "old_effective_epsilon_max": float(np.max(old_effective)),
        "new_effective_epsilon_min": float(np.min(new_effective)),
        "new_effective_epsilon_mean": float(np.mean(new_effective)),
        "new_effective_epsilon_max": float(np.max(new_effective)),
        "fock_screened_strength_ratio_old_scalar_to_wdiag_min": float(np.min(strength_ratio)),
        "fock_screened_strength_ratio_old_scalar_to_wdiag_mean": float(np.mean(strength_ratio)),
        "fock_screened_strength_ratio_old_scalar_to_wdiag_median": float(np.median(strength_ratio)),
        "fock_screened_strength_ratio_old_scalar_to_wdiag_max": float(np.max(strength_ratio)),
        "epsilon_times_bn_old_max": float(np.max(old_effective) * float(result.coulomb_params.epsilon_bn)),
        "epsilon_times_bn_new_max": float(np.max(new_effective) * float(result.coulomb_params.epsilon_bn)),
    }
    report.update(diag_report)
    (output / "diagnostic_fock_wdiag_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
