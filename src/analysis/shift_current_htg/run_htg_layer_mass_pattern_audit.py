from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from mean_field.systems.htg.lattice import build_moire_k_grid

from .constants import eta_mev_to_ev
from .htg_adapter import (
    MaoHTGConfig,
    _orbital_slice,
    analytic_dhdk,
    build_mao_hamiltonian,
    make_mao_model,
    stacking_displacements,
)
from .response import (
    component_label,
    component_transition_weight_from_D,
    fermi_occupation,
    lorentzian_delta,
    parse_component,
    sigma_from_integral,
    velocity_matrices,
)

DEFAULT_COMPONENTS = ("x;xx", "x;xy", "x;yx", "x;yy", "y;xx", "y;xy", "y;yx", "y;yy")


def _safe_key(text: str) -> str:
    return str(text).replace(";", "_").replace(",", "_").replace(":", "_").replace("|", "_").replace("-", "m")


def parse_pattern(text: str) -> tuple[str, tuple[float, float, float]]:
    if ":" in str(text):
        name, body = str(text).split(":", 1)
    else:
        name, body = "pattern", str(text)
    vals = tuple(float(part.strip()) for part in body.split(",") if part.strip())
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("mass pattern must be name:m1,m2,m3 in meV")
    return name.strip(), vals


def rotate_complex(value: complex, angle_rad: float) -> complex:
    return complex(value) * complex(math.cos(angle_rad), math.sin(angle_rad))


def c3_points(base: np.ndarray) -> np.ndarray:
    return np.concatenate([base, np.asarray([rotate_complex(k, 2.0 * math.pi / 3.0) for k in base]), np.asarray([rotate_complex(k, 4.0 * math.pi / 3.0) for k in base])])


def add_layer_masses(hmat: np.ndarray, model, masses_ev: tuple[float, float, float]) -> np.ndarray:
    out = np.array(hmat, dtype=np.complex128, copy=True)
    for ig in range(model.lattice.n_g):
        for layer, mass in enumerate(masses_ev, start=1):
            sl = _orbital_slice(ig, layer)
            out[sl.start, sl.start] += float(mass)
            out[sl.start + 1, sl.start + 1] -= float(mass)
    return out


def c3_errors(spectra: dict[str, np.ndarray]) -> dict[str, float]:
    arr = {k: np.asarray(v, dtype=float) for k, v in spectra.items()}
    group1 = np.stack([arr["x;yy"], -arr["x;xx"], arr["y;yx"], arr["y;xy"]])
    group2 = np.stack([arr["y;xx"], -arr["y;yy"], arr["x;xy"], arr["x;yx"]])
    g1 = np.mean(group1, axis=0)
    g2 = np.mean(group2, axis=0)
    return {
        "group1_peak_abs": float(np.max(np.abs(g1))),
        "group2_peak_abs": float(np.max(np.abs(g2))),
        "group1_over_group2": float(np.max(np.abs(g1)) / np.max(np.abs(g2))) if np.max(np.abs(g2)) > 0 else float("nan"),
        "group1_max_deviation": float(np.max(np.abs(group1 - g1[None, :]))),
        "group2_max_deviation": float(np.max(np.abs(group2 - g2[None, :]))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic only: test how layer-dependent sublattice mass profiles break the implemented "
            "antiunitary layer-swap/conjugation symmetry and activate the second C3 tensor coefficient."
        )
    )
    parser.add_argument("--theta-deg", type=float, default=1.95)
    parser.add_argument("--stacking", choices=("ABA", "AAA"), default="ABA")
    parser.add_argument("--domain", choices=("h", "hbar"), default="h")
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument("--n-shells", type=int, default=2)
    parser.add_argument("--mesh-size", type=int, default=8)
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.08)
    parser.add_argument("--n-energy", type=int, default=201)
    parser.add_argument("--pattern", action="append", type=parse_pattern, default=None, help="name:m_top,m_mid,m_bot in meV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_current_htg_layer_mass_pattern_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = tuple(args.pattern) if args.pattern is not None else (
        ("all_equal", (30.0, 30.0, 30.0)),
        ("bottom_only", (0.0, 0.0, 30.0)),
        ("top_only", (30.0, 0.0, 0.0)),
        ("outer_same", (30.0, 0.0, 30.0)),
        ("outer_opposite", (30.0, 0.0, -30.0)),
    )
    config = MaoHTGConfig(
        theta_deg=float(args.theta_deg),
        stacking=str(args.stacking),
        corrugation_r=float(args.r),
        n_shells=int(args.n_shells),
        mass_ev=0.0,
        domain=str(args.domain),
        zeta_rad=0.0,
    )
    model = make_mao_model(config)
    d_top, d_bot = stacking_displacements(model.lattice, config.stacking, valley=config.valley, domain=config.domain)
    dhdk = analytic_dhdk(model, config)
    _, k_grid = build_moire_k_grid(model.lattice, int(args.mesh_size), endpoint=False, frac_shift=(0.5, 0.5))
    k_points = c3_points(np.asarray(k_grid, dtype=np.complex128).reshape(-1))
    event_prefactor = float(model.lattice.mbz_area) / float(k_points.size) / (2.0 * np.pi) ** 2
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    eta_ev = eta_mev_to_ev(float(args.eta_mev))
    components = {name: parse_component(name) for name in DEFAULT_COMPONENTS}
    center = model.matrix_dim // 2
    n_abs, m_abs = center - 1, center

    output_arrays: dict[str, np.ndarray] = {"photon_energies_ev": photon}
    summary: dict[str, object] = {
        "warning": "Diagnostic only. Mao Eq. (13) uses equal layer mass; unequal layer masses are a symmetry-breaking hypothesis test, not a reproduction claim.",
        "config": {
            "theta_deg": config.theta_deg,
            "stacking": config.stacking,
            "domain": config.domain,
            "corrugation_r": config.corrugation_r,
            "n_shells": config.n_shells,
            "mesh_size": int(args.mesh_size),
            "matrix_dim": int(model.matrix_dim),
            "eta_mev": float(args.eta_mev),
            "central_pair_absolute_indices": [int(n_abs), int(m_abs)],
        },
        "patterns": {},
    }

    for name, masses_mev in patterns:
        masses_ev = tuple(float(v) * 1.0e-3 for v in masses_mev)
        integrals = {component: np.zeros_like(photon, dtype=np.complex128) for component in components}
        transition_min = float("inf")
        transition_max = float("-inf")
        for k_tilde in k_points:
            h0 = build_mao_hamiltonian(k_tilde, model, config, d_top=d_top, d_bot=d_bot)
            evals, evecs = eigh(add_layer_masses(h0, model, masses_ev))
            occupations = fermi_occupation(evals)
            D = velocity_matrices(evecs, dhdk)
            transition_ev = float(evals[m_abs] - evals[n_abs])
            transition_min = min(transition_min, transition_ev)
            transition_max = max(transition_max, transition_ev)
            if transition_ev <= 0.0 or abs(float(occupations[n_abs] - occupations[m_abs])) < 1.0e-14:
                continue
            for component_name, component in components.items():
                weight, _ = component_transition_weight_from_D(
                    D,
                    evals,
                    occupations,
                    n_abs,
                    m_abs,
                    component,
                    denominator_cutoff_ev=1.0e-8,
                )
                integrals[component_name] += event_prefactor * weight * lorentzian_delta(photon, transition_ev, eta_ev)
        spectra = {component: sigma_from_integral(integral) for component, integral in integrals.items()}
        peaks = {}
        for component, values in spectra.items():
            idx = int(np.argmax(np.abs(values)))
            peaks[component] = {
                "energy_at_max_abs_ev": float(photon[idx]),
                "signed_value_at_max_abs_uA_nm_per_V2": float(values[idx]),
                "max_abs_uA_nm_per_V2": float(np.max(np.abs(values))),
            }
            output_arrays[f"sigma_{_safe_key(name)}_{_safe_key(component)}"] = np.asarray(values, dtype=float)
        summary["patterns"][name] = {
            "layer_masses_mev_top_mid_bot": [float(v) for v in masses_mev],
            "transition_range_ev": [float(transition_min), float(transition_max)],
            "peaks": peaks,
            "c3_group_diagnostics": c3_errors(spectra),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "layer_mass_pattern_audit.npz", **output_arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
