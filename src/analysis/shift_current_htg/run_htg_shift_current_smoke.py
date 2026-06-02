from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from mean_field.systems.htg.lattice import build_moire_k_grid

from .constants import eta_mev_to_ev
from .htg_adapter import (
    MaoHTGConfig,
    analytic_dhdk,
    build_mao_hamiltonian,
    make_mao_model,
    stacking_displacements,
    validate_analytic_dhdk,
)
from .response import (
    add_transitions_to_integral,
    parse_component,
    positive_transition_terms,
    precompute_response_tensors,
    sigma_from_integral,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny hTTG shift-current smoke run; not a paper-grade convergence calculation."
    )
    parser.add_argument("--theta-deg", type=float, default=1.95)
    parser.add_argument("--stacking", choices=("ABA", "AAA"), default="ABA")
    parser.add_argument("--domain", choices=("h", "hbar"), default="h")
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument("--n-shells", type=int, default=1)
    parser.add_argument("--mesh-size", type=int, default=3)
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.12)
    parser.add_argument("--n-energy", type=int, default=121)
    parser.add_argument("--mass-mev", type=float, default=30.0)
    parser.add_argument("--component", action="append", default=["x;yy", "y;xx"])
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_current_htg_smoke"))
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MaoHTGConfig(
        theta_deg=float(args.theta_deg),
        stacking=str(args.stacking),
        corrugation_r=float(args.r),
        n_shells=int(args.n_shells),
        mass_ev=float(args.mass_mev) * 1.0e-3,
        domain=str(args.domain),
    )
    model = make_mao_model(config)
    d_top, d_bot = stacking_displacements(
        model.lattice,
        config.stacking,
        valley=config.valley,
        domain=config.domain,
    )
    dhdk = analytic_dhdk(model, config)
    validation = validate_analytic_dhdk(0.0 + 0.0j, model, config, d_top=d_top, d_bot=d_bot)

    photon_energies = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    components = {label: parse_component(label) for label in args.component}
    integrals = {label: np.zeros_like(photon_energies, dtype=np.complex128) for label in components}

    _, k_grid = build_moire_k_grid(model.lattice, int(args.mesh_size), endpoint=False, frac_shift=(0.5, 0.5))
    k_points = np.asarray(k_grid, dtype=np.complex128).reshape(-1)
    k_weight = float(model.lattice.mbz_area) / float(k_points.size)
    skipped_total = 0
    transition_count = 0
    for k_tilde in k_points:
        hmat = build_mao_hamiltonian(k_tilde, model, config, d_top=d_top, d_bot=d_bot)
        evals, evecs = eigh(hmat)
        tensors = precompute_response_tensors(
            evals,
            evecs,
            dhdk,
            denominator_cutoff_ev=float(args.denominator_cutoff_ev),
        )
        skipped_total += tensors.skipped_small_denominators
        for label, component in components.items():
            transitions, weights = positive_transition_terms(tensors, component)
            transition_count += int(transitions.size)
            add_transitions_to_integral(
                integrals[label],
                photon_energies,
                transitions,
                weights,
                k_weight_nm_inv_sq=k_weight,
                eta_ev=eta_mev_to_ev(float(args.eta_mev)),
            )
    spectra = {label: sigma_from_integral(integral) for label, integral in integrals.items()}
    peaks = {
        label: {
            "max_abs_uA_nm_per_V2": float(np.max(np.abs(values))),
            "energy_at_max_abs_ev": float(photon_energies[int(np.argmax(np.abs(values)))]),
        }
        for label, values in spectra.items()
    }
    summary = {
        "warning": "Smoke run only: tiny mesh/cutoff, no convergence claim.",
        "config": {
            "theta_deg": config.theta_deg,
            "stacking": config.stacking,
            "domain": config.domain,
            "corrugation_r": config.corrugation_r,
            "n_shells": config.n_shells,
            "mesh_size": int(args.mesh_size),
            "matrix_dim": model.matrix_dim,
            "eta_mev": float(args.eta_mev),
            "mass_ev": config.mass_ev,
            "denominator_cutoff_ev": float(args.denominator_cutoff_ev),
        },
        "dhdk_validation_at_gamma": {
            "max_abs_ev_nm": validation.max_abs_ev_nm,
            "passes_1e_minus_7": validation.max_abs_ev_nm < 1.0e-7,
        },
        "k_point_count": int(k_points.size),
        "transition_count_with_component_repeats": int(transition_count),
        "skipped_small_denominators": int(skipped_total),
        "peaks": peaks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_dir / "htg_shift_current_smoke.npz",
            photon_energies_ev=photon_energies,
            **{f"sigma_{name.replace(';', '_')}": values for name, values in spectra.items()},
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
