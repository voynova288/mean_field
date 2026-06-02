from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constants import eta_mev_to_ev
from .slg_toy import GappedSLGParams, c3_tensor_relation_errors, compute_slg_shift_current

DEFAULT_COMPONENTS = ("x;yy", "x;xx", "y;yx", "y;xy", "y;xx", "y;yy", "x;xy", "x;yx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Appendix-A gapped-SLG shift-current toy benchmark on the full hexagonal BZ."
    )
    parser.add_argument("--mesh-size", type=int, default=24, help="rectangular midpoint resolution before hexagonal clipping")
    parser.add_argument("--eta-mev", type=float, default=50.0, help="Lorentzian broadening in meV for the toy model")
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=8.0)
    parser.add_argument("--n-energy", type=int, default=161)
    parser.add_argument("--mass-ev", type=float, default=1.5)
    parser.add_argument("--hopping-ev", type=float, default=2.73)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_current_slg_toy_smoke"))
    parser.add_argument("--no-c3-symmetrize-grid", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    photon_energies = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=float(args.mass_ev))
    spectra = compute_slg_shift_current(
        photon_energies,
        components=DEFAULT_COMPONENTS,
        mesh_size=int(args.mesh_size),
        eta_ev=eta_mev_to_ev(float(args.eta_mev)),
        params=params,
        c3_symmetrize_grid=not bool(args.no_c3_symmetrize_grid),
    )
    errors = c3_tensor_relation_errors(spectra)
    peaks = {
        name: {
            "max_abs_uA_nm_per_V2": float(np.max(np.abs(values))),
            "energy_at_max_abs_ev": float(photon_energies[int(np.argmax(np.abs(values)))]),
        }
        for name, values in spectra.items()
    }
    summary = {
        "params": {
            "mesh_size": int(args.mesh_size),
            "eta_mev": float(args.eta_mev),
            "mass_ev": params.mass_ev,
            "hopping_ev": params.hopping_ev,
            "bond_nm": params.bond_nm,
            "c3_symmetrize_grid": not bool(args.no_c3_symmetrize_grid),
        },
        "c3_relation_max_abs_errors_uA_nm_per_V2": errors,
        "peaks": peaks,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_dir / "slg_toy_shift_current.npz",
            photon_energies_ev=photon_energies,
            **{f"sigma_{name.replace(';', '_')}": values for name, values in spectra.items()},
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
