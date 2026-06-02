from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from .htg_adapter import (
    MaoHTGConfig,
    _orbital_slice,
    analytic_dhdk,
    build_mao_hamiltonian,
    make_mao_model,
    stacking_displacements,
)


def parse_complex_pair(text: str) -> complex:
    parts = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("k point must be 'kx,ky' in nm^-1")
    return complex(parts[0], parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "hTTG symmetry audit for candidate coordinate transforms and the exact antiunitary "
            "layer-swap/conjugation symmetry of the implemented ABA Hamiltonian."
        )
    )
    parser.add_argument("--theta-deg", type=float, default=1.95)
    parser.add_argument("--stacking", choices=("ABA", "AAA"), default="ABA")
    parser.add_argument("--domain", choices=("h", "hbar"), default="h")
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument("--n-shells", type=int, default=3)
    parser.add_argument("--mass-mev", type=float, default=30.0)
    parser.add_argument("--k-point", action="append", type=parse_complex_pair, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/shift_current_htg_symmetry_spectrum_audit/summary.json"))
    return parser.parse_args()


def build_layer_swap_conjugation_unitary(model) -> np.ndarray:
    """Return U for H(k*) = U H(k)^* U^dagger in the implemented basis.

    The map is exact for the current ABA construction:

    - reciprocal index `(n1,n2) -> (-n2,-n1)`, because `conj(b1)=-b2` and
      `conj(b2)=-b1`;
    - layer `1 <-> 3`, layer `2 -> 2`;
    - no sublattice exchange.

    The antiunitary operation is therefore `U K` combined with `k -> conj(k)`.
    """

    lattice = model.lattice
    index_by_pair = {tuple(map(int, pair)): int(index) for index, pair in enumerate(lattice.g_indices)}
    unitary = np.zeros((model.matrix_dim, model.matrix_dim), dtype=np.complex128)
    layer_map = {1: 3, 2: 2, 3: 1}
    for old_g, (n1, n2) in enumerate(lattice.g_indices):
        new_g = index_by_pair.get((-int(n2), -int(n1)))
        if new_g is None:
            raise RuntimeError(f"Conjugated reciprocal index {(-int(n2), -int(n1))} missing from cutoff basis")
        for old_layer, new_layer in layer_map.items():
            old_slice = _orbital_slice(old_g, old_layer)
            new_slice = _orbital_slice(new_g, new_layer)
            unitary[new_slice, old_slice] = np.eye(2, dtype=np.complex128)
    return unitary


def main() -> None:
    args = parse_args()
    k_points = tuple(args.k_point) if args.k_point is not None else (0.01 + 0.02j, 0.03 - 0.015j, -0.04 + 0.007j)
    config = MaoHTGConfig(
        theta_deg=float(args.theta_deg),
        stacking=str(args.stacking),
        corrugation_r=float(args.r),
        n_shells=int(args.n_shells),
        mass_ev=float(args.mass_mev) * 1.0e-3,
        domain=str(args.domain),
        zeta_rad=0.0,
    )
    model = make_mao_model(config)
    d_top, d_bot = stacking_displacements(model.lattice, config.stacking, valley=config.valley, domain=config.domain)

    transforms = {
        "identity": lambda k: k,
        "mirror_y_k_to_conj_k": lambda k: np.conj(k),
        "mirror_x_k_to_minus_conj_k": lambda k: -np.conj(k),
        "c2_k_to_minus_k": lambda k: -k,
        "c3_plus_120": lambda k: k * complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0)),
        "c3_minus_120": lambda k: k * complex(math.cos(-2.0 * math.pi / 3.0), math.sin(-2.0 * math.pi / 3.0)),
    }
    base_eigs = []
    for k in k_points:
        base_eigs.append(eigvalsh(build_mao_hamiltonian(k, model, config, d_top=d_top, d_bot=d_bot)))

    results: dict[str, object] = {}
    for name, transform in transforms.items():
        diffs = []
        for k, e0 in zip(k_points, base_eigs, strict=True):
            e1 = eigvalsh(build_mao_hamiltonian(complex(transform(k)), model, config, d_top=d_top, d_bot=d_bot))
            diffs.append(float(np.max(np.abs(e0 - e1))))
        results[name] = {
            "max_abs_eigenvalue_diff_ev_by_k": diffs,
            "max_over_test_points_ev": float(max(diffs) if diffs else 0.0),
        }

    antiunitary = build_layer_swap_conjugation_unitary(model)
    dhdk_x, dhdk_y = analytic_dhdk(model, config)
    antiunitary_errors = {
        "max_unitarity_error": float(np.max(np.abs(antiunitary.conj().T @ antiunitary - np.eye(model.matrix_dim)))),
        "hamiltonian_max_abs_error_ev_by_k": [],
        "dhdk_x_error_ev_nm": float(np.max(np.abs(dhdk_x - antiunitary @ dhdk_x.conj() @ antiunitary.conj().T))),
        "dhdk_y_error_ev_nm": float(np.max(np.abs(dhdk_y + antiunitary @ dhdk_y.conj() @ antiunitary.conj().T))),
    }
    for k in k_points:
        h_left = build_mao_hamiltonian(np.conj(k), model, config, d_top=d_top, d_bot=d_bot)
        h_right = antiunitary @ build_mao_hamiltonian(k, model, config, d_top=d_top, d_bot=d_bot).conj() @ antiunitary.conj().T
        antiunitary_errors["hamiltonian_max_abs_error_ev_by_k"].append(float(np.max(np.abs(h_left - h_right))))
    antiunitary_errors["hamiltonian_max_over_test_points_ev"] = float(
        max(antiunitary_errors["hamiltonian_max_abs_error_ev_by_k"])
    )

    summary = {
        "warning": "The antiunitary unitary-matrix check is exact for this finite cutoff. Coordinate-transform eigenvalue checks alone are only diagnostics.",
        "config": {
            "theta_deg": config.theta_deg,
            "stacking": config.stacking,
            "domain": config.domain,
            "corrugation_r": config.corrugation_r,
            "n_shells": config.n_shells,
            "mass_ev": config.mass_ev,
            "matrix_dim": int(model.matrix_dim),
        },
        "k_points_nm_inv": [[float(k.real), float(k.imag)] for k in k_points],
        "transforms": results,
        "exact_antiunitary_layer_swap_conjugation": {
            "operation": "H(k_x,-k_y) = U H(k_x,k_y)^* U^dagger",
            "reciprocal_index_map": "(n1,n2)->(-n2,-n1)",
            "layer_map": "1<->3, 2->2",
            "sublattice_map": "identity",
            "velocity_transform": "dH/dkx -> + U(dH/dkx)^*U^dagger, dH/dky -> - U(dH/dky)^*U^dagger",
            **antiunitary_errors,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
