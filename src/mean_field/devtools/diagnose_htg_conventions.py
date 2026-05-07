from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import numpy as np
from scipy.linalg import eigh

from mean_field.systems.htg import HTGParams, build_chiral_lattice_from_alpha, build_standard_kpath
from mean_field.systems.htg.hamiltonian import _orbital_slice, dirac_block, layer_rotation_angle
from mean_field.systems.htg.lattice import _complex_key, dot_2d


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-resolution HTG convention sweep for debugging.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.377)
    parser.add_argument("--n-shells", type=int, default=4)
    parser.add_argument("--points-per-segment", type=int, default=18)
    return parser.parse_args()


def _ensure_not_login() -> None:
    if os.environ.get("SLURM_JOB_ID"):
        return
    hostname = socket.gethostname().lower()
    if hostname.startswith("login001") or hostname.startswith("login002"):
        raise SystemExit("Refusing to run HTG convention diagnostics on a login node.")


def _phase(qvec: complex, displacement: complex, sign: int) -> complex:
    value = float(sign) * dot_2d(qvec, displacement)
    return complex(np.cos(value), np.sin(value))


def _coupling_matrix(channel: int, params: HTGParams, *, valley: int, matrix_phase_sign: int):
    angle = 2.0 * np.pi * valley * matrix_phase_sign * channel / 3.0
    phase = complex(np.cos(angle), np.sin(angle))
    return params.w_ev * np.asarray(
        [[params.kappa, phase.conjugate()], [phase, params.kappa]],
        dtype=np.complex128,
    )


def _coupling_entries(g_vectors, q_vectors, *, shift_sign: int, valley: int = 1):
    mapping = {_complex_key(complex(gvec)): idx for idx, gvec in enumerate(g_vectors)}
    q0 = complex(q_vectors[0])
    entries = []
    for middle_index, g_middle in enumerate(g_vectors):
        for channel in (0, 1, 2):
            shift = complex(valley * (q_vectors[channel] - q0))
            outer_index = mapping.get(_complex_key(complex(g_middle + shift_sign * shift)))
            if outer_index is not None:
                entries.append((channel, middle_index, outer_index))
    return entries


def _offsets(lattice, mode: str, valley: int):
    if mode == "folded":
        return (valley * lattice.q0, 0.0 + 0.0j, -valley * lattice.q0)
    if mode == "workdoc":
        kappa_prime_workdoc = (2.0 * lattice.b_m1 - lattice.b_m2) / 3.0
        return (-valley * lattice.kappa_m, 0.0 + 0.0j, -valley * kappa_prime_workdoc)
    if mode == "doc":
        return (-valley * lattice.q0, 0.0 + 0.0j, valley * lattice.q0)
    if mode == "none":
        return (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
    raise ValueError(mode)


def _displacements(lattice, mode: str):
    if mode == "h":
        return (-lattice.delta, lattice.delta)
    if mode == "hbar":
        return (lattice.delta, -lattice.delta)
    if mode == "none":
        return (0.0 + 0.0j, 0.0 + 0.0j)
    raise ValueError(mode)


def _build_hamiltonian_variant(
    k_tilde,
    lattice,
    params,
    *,
    offset_mode: str,
    shift_sign: int,
    bottom_shift_sign: int,
    displacement_mode: str,
    phase_sign: int,
    matrix_phase_sign: int,
    valley: int = 1,
):
    dim = lattice.matrix_dim
    hmat = np.zeros((dim, dim), dtype=np.complex128)
    offsets = _offsets(lattice, offset_mode, valley)
    for ig, gvec in enumerate(lattice.g_vectors):
        for layer in (1, 2, 3):
            sl = _orbital_slice(ig, layer)
            hmat[sl, sl] = dirac_block(
                complex(k_tilde + gvec + offsets[layer - 1]),
                layer_rotation_angle(lattice, params, layer),
                params.vf_ev_nm,
                valley,
            )

    d_top, d_bot = _displacements(lattice, displacement_mode)
    top_entries = _coupling_entries(
        lattice.g_vectors,
        lattice.q_vectors,
        shift_sign=shift_sign,
        valley=valley,
    )
    bottom_entries = _coupling_entries(
        lattice.g_vectors,
        lattice.q_vectors,
        shift_sign=bottom_shift_sign,
        valley=valley,
    )
    for channel, middle_index, outer_index in top_entries:
        top_slice = _orbital_slice(outer_index, 1)
        middle_slice = _orbital_slice(middle_index, 2)
        base = _coupling_matrix(channel, params, valley=valley, matrix_phase_sign=matrix_phase_sign)
        q_channel = complex(lattice.q_vectors[channel])
        top = _phase(q_channel, d_top, phase_sign * valley) * base
        hmat[top_slice, middle_slice] += top
        hmat[middle_slice, top_slice] += top.conjugate().T
    for channel, middle_index, outer_index in bottom_entries:
        middle_slice = _orbital_slice(middle_index, 2)
        bottom_slice = _orbital_slice(outer_index, 3)
        base = _coupling_matrix(channel, params, valley=valley, matrix_phase_sign=matrix_phase_sign)
        q_channel = complex(lattice.q_vectors[channel])
        bottom = _phase(q_channel, d_bot, phase_sign * valley) * base
        hmat[middle_slice, bottom_slice] += bottom
        hmat[bottom_slice, middle_slice] += bottom.conjugate().T
    return hmat


def _central_width_for_variant(lattice, params, path, **variant) -> float:
    center = lattice.matrix_dim // 2
    subset = (center - 1, center)
    values = []
    for kval in path.kvec:
        evals = eigh(
            _build_hamiltonian_variant(kval, lattice, params, **variant),
            eigvals_only=True,
            subset_by_index=subset,
            driver="evr",
        )
        values.append(evals)
    arr = np.asarray(values, dtype=float)
    return float(np.max(arr) - np.min(arr))


def main() -> None:
    args = _parse_args()
    _ensure_not_login()
    params = HTGParams.chiral(zeta_rad=0.0)
    lattice = build_chiral_lattice_from_alpha(args.alpha, n_shells=args.n_shells, params=params)
    path = build_standard_kpath(lattice, points_per_segment=args.points_per_segment)
    rows = []
    for offset_mode in ("workdoc", "folded", "doc", "none"):
        for shift_sign in (-1, 1):
            for bottom_shift_sign in (-1, 1):
                for displacement_mode in ("h", "hbar", "none"):
                    for phase_sign in (-1, 1):
                        for matrix_phase_sign in (-1, 1):
                            variant = {
                                "offset_mode": offset_mode,
                                "shift_sign": shift_sign,
                                "bottom_shift_sign": bottom_shift_sign,
                                "displacement_mode": displacement_mode,
                                "phase_sign": phase_sign,
                                "matrix_phase_sign": matrix_phase_sign,
                            }
                            width = _central_width_for_variant(lattice, params, path, **variant)
                            rows.append({**variant, "width_over_vk": width / params.vk_theta_ev(lattice.k_theta)})
    rows.sort(key=lambda item: item["width_over_vk"])
    payload = {
        "alpha": args.alpha,
        "n_shells": args.n_shells,
        "points_per_segment": args.points_per_segment,
        "theta_deg": lattice.theta_deg,
        "results": rows,
    }
    output = args.output
    if output is None:
        job_id = os.environ.get("SLURM_JOB_ID", "manual")
        output = REPO_ROOT / "results" / "HTG" / f"htg_convention_diagnostics_{job_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] output={output}")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
