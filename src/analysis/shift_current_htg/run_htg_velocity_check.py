from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .htg_adapter import (
    MaoHTGConfig,
    build_mao_hamiltonian,
    central_band_indices,
    make_mao_model,
    stacking_displacements,
    validate_analytic_dhdk,
)


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate analytic dH/dk for the Mao hTTG shift-current adapter."
    )
    parser.add_argument("--theta-deg", type=float, default=1.95)
    parser.add_argument("--stacking", choices=("ABA", "AAA"), default="ABA")
    parser.add_argument("--domain", choices=("h", "hbar"), default="h")
    parser.add_argument("--r", type=float, default=0.8, help="corrugation w_AA/w_AB")
    parser.add_argument("--n-shells", type=int, default=1, help="small default for a fast smoke check")
    parser.add_argument("--mass-mev", type=float, default=30.0)
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--kx", type=float, default=0.0, help="k_x in nm^-1")
    parser.add_argument("--ky", type=float, default=0.0, help="k_y in nm^-1")
    parser.add_argument("--finite-step", type=float, default=1.0e-6, help="finite-difference step in nm^-1")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MaoHTGConfig(
        theta_deg=float(args.theta_deg),
        stacking=str(args.stacking),
        corrugation_r=float(args.r),
        n_shells=int(args.n_shells),
        mass_ev=float(args.mass_mev) * 1.0e-3,
        valley=int(args.valley),
        domain=str(args.domain),
    )
    model = make_mao_model(config)
    d_top, d_bot = stacking_displacements(
        model.lattice,
        config.stacking,
        valley=config.valley,
        domain=config.domain,
    )
    k_tilde = complex(float(args.kx), float(args.ky))
    result = validate_analytic_dhdk(
        k_tilde,
        model,
        config,
        step_nm_inv=float(args.finite_step),
        d_top=d_top,
        d_bot=d_bot,
    )
    hmat = build_mao_hamiltonian(k_tilde, model, config, d_top=d_top, d_bot=d_bot)
    evals = np.linalg.eigvalsh(hmat)
    central = central_band_indices(model.matrix_dim, count=min(8, model.matrix_dim))
    payload = {
        "config": {
            "theta_deg": config.theta_deg,
            "stacking": config.stacking,
            "domain": config.domain,
            "corrugation_r": config.corrugation_r,
            "n_shells": config.n_shells,
            "mass_ev": config.mass_ev,
            "w1_ev": config.w1_ev,
            "vf_ev_nm": config.vf_ev_nm,
            "valley": config.valley,
        },
        "lattice": model.lattice_summary(),
        "d_top_nm": _complex_pair(d_top),
        "d_bot_nm": _complex_pair(d_bot),
        "k_tilde_nm_inv": _complex_pair(k_tilde),
        "dhdk_validation": {
            "max_abs_x_ev_nm": result.max_abs_x_ev_nm,
            "max_abs_y_ev_nm": result.max_abs_y_ev_nm,
            "max_abs_ev_nm": result.max_abs_ev_nm,
            "finite_step_nm_inv": result.finite_step_nm_inv,
            "passes_1e_minus_7": result.max_abs_ev_nm < 1.0e-7,
        },
        "central_eigenvalues_ev": [float(evals[i]) for i in central],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
