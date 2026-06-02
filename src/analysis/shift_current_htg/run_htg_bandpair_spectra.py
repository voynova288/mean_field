from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    Component,
    component_label,
    component_transition_weight_from_D,
    fermi_occupation,
    lorentzian_delta,
    parse_component,
    sigma_from_integral,
    velocity_matrices,
)


@dataclass(frozen=True)
class PairSpec:
    name: str
    n_offset: int
    m_offset: int


@dataclass(frozen=True)
class PairWindowSpec:
    name: str
    min_offset: int
    max_offset: int

    @property
    def pair_offsets(self) -> tuple[tuple[int, int], ...]:
        """Occupied-to-empty central-window offsets.

        Negative offsets are interpreted as occupied bands below the central
        gap; non-negative offsets are empty bands above it.  For example,
        `-2,1` gives pairs (-2,0), (-2,1), (-1,0), (-1,1).
        """

        return tuple(
            (n_offset, m_offset)
            for n_offset in range(int(self.min_offset), 0)
            for m_offset in range(0, int(self.max_offset) + 1)
        )


def _safe_key(text: str) -> str:
    return (
        str(text)
        .replace(";", "_")
        .replace(",", "_")
        .replace(":", "_")
        .replace("-", "m")
        .replace("+", "p")
        .replace("|", "_")
    )


def _parse_float_csv(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def parse_pair_spec(text: str) -> PairSpec:
    """Parse `name:n_offset,m_offset` relative to the central gap.

    The central valence/conduction pair is `-1,0`, where absolute indices are
    `matrix_dim//2 - 1` and `matrix_dim//2`.
    """

    raw = str(text)
    if ":" in raw:
        name, body = raw.split(":", 1)
        name = name.strip()
    else:
        body = raw
        name = "pair_" + _safe_key(raw)
    parts = [part.strip() for part in body.split(",")]
    if len(parts) != 2 or not name:
        raise argparse.ArgumentTypeError("Pair spec must look like 'central_flat:-1,0'")
    try:
        n_offset, m_offset = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Band offsets must be integers, e.g. '-1,0'") from exc
    return PairSpec(name=name, n_offset=n_offset, m_offset=m_offset)


def parse_pair_window_spec(text: str) -> PairWindowSpec:
    """Parse `name:min_offset,max_offset` into an aggregated pair window."""

    raw = str(text)
    if ":" in raw:
        name, body = raw.split(":", 1)
        name = name.strip()
    else:
        body = raw
        name = "window_" + _safe_key(raw)
    parts = [part.strip() for part in body.split(",")]
    if len(parts) != 2 or not name:
        raise argparse.ArgumentTypeError("Window spec must look like 'central_window:-4,3'")
    try:
        min_offset, max_offset = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Window offsets must be integers, e.g. '-4,3'") from exc
    if min_offset >= 0 or max_offset < 0:
        raise argparse.ArgumentTypeError("Window must cross the central gap: min_offset < 0 <= max_offset")
    return PairWindowSpec(name=name, min_offset=min_offset, max_offset=max_offset)


def resolve_relative_index(matrix_dim: int, offset: int) -> int:
    center = int(matrix_dim) // 2
    index = center + int(offset)
    if index < 0 or index >= int(matrix_dim):
        raise ValueError(f"Relative band offset {offset} resolves to {index}, outside matrix_dim={matrix_dim}")
    return index


def rotate_complex(value: complex, angle_rad: float) -> complex:
    return complex(value) * complex(np.cos(float(angle_rad)), np.sin(float(angle_rad)))


def c3_symmetrize_k_points(k_points: np.ndarray) -> np.ndarray:
    """Average a primitive-cell grid over three C3-related primitive cells.

    The continuum Hamiltonian can be evaluated outside the first parallelogram;
    averaging the original, C3-rotated, and C3^2-rotated grids greatly reduces
    finite-mesh violations of the C3 tensor identities without using an
    irreducible wedge.
    """

    flat = np.asarray(k_points, dtype=np.complex128).reshape(-1)
    angle = 2.0 * np.pi / 3.0
    return np.concatenate(
        [
            flat,
            np.asarray([rotate_complex(k, angle) for k in flat], dtype=np.complex128),
            np.asarray([rotate_complex(k, 2.0 * angle) for k in flat], dtype=np.complex128),
        ]
    )


def spectrum_from_weighted_events(
    photon_energies_ev: np.ndarray,
    transition_energies_ev: np.ndarray,
    weighted_transition_weights: np.ndarray,
    *,
    eta_ev: float,
) -> np.ndarray:
    """Build sigma from already-BZ-weighted transition events."""

    integral = np.zeros_like(photon_energies_ev, dtype=np.complex128)
    for transition_ev, weight in zip(transition_energies_ev, weighted_transition_weights, strict=True):
        integral += complex(weight) * lorentzian_delta(photon_energies_ev, float(transition_ev), eta_ev)
    return sigma_from_integral(integral)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream selected hTTG band-pair transition weights and produce spectra. "
            "This is the first Fig. 2-style decomposition workflow."
        )
    )
    parser.add_argument("--theta-deg", type=float, default=1.95)
    parser.add_argument("--stacking", choices=("ABA", "AAA"), default="ABA")
    parser.add_argument("--domain", choices=("h", "hbar"), default="h")
    parser.add_argument("--r", type=float, default=0.8)
    parser.add_argument("--n-shells", type=int, default=1)
    parser.add_argument("--mesh-size", type=int, default=6)
    parser.add_argument("--frac-shift", type=_parse_float_csv, default=(0.5, 0.5), help="fractional grid shift, e.g. 0.5,0.5")
    parser.add_argument(
        "--c3-symmetrize-grid",
        action="store_true",
        help="Average the full primitive-cell mesh over C3-rotated copies to reduce finite-mesh tensor-symmetry error.",
    )
    parser.add_argument("--eta-mev", type=_parse_float_csv, default=(1.0,), help="comma-separated broadenings in meV")
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.12)
    parser.add_argument("--n-energy", type=int, default=241)
    parser.add_argument("--mass-mev", type=float, default=30.0)
    parser.add_argument("--mu-ev", type=float, default=0.0)
    parser.add_argument("--temperature-k", type=float, default=0.0)
    parser.add_argument("--component", action="append", default=None, help="component like x;yy; repeatable")
    parser.add_argument("--pair", action="append", type=parse_pair_spec, default=None, help="single band pair like central_flat:-1,0; repeatable")
    parser.add_argument(
        "--pair-window",
        action="append",
        type=parse_pair_window_spec,
        default=None,
        help="aggregate occupied-to-empty central-window pairs like central_window:-4,3; repeatable",
    )
    parser.add_argument("--denominator-cutoff-ev", type=float, default=1.0e-8)
    parser.add_argument("--finite-step", type=float, default=1.0e-6)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_current_htg_bandpairs"))
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components_text = tuple(args.component) if args.component is not None else ("x;yy", "y;xx")
    components: dict[str, Component] = {text: parse_component(text) for text in components_text}
    pair_specs = tuple(args.pair) if args.pair is not None else tuple()
    window_specs = tuple(args.pair_window) if args.pair_window is not None else tuple()
    if not pair_specs and not window_specs:
        pair_specs = (PairSpec("central_flat", -1, 0),)
    pair_groups: dict[str, tuple[tuple[int, int], ...]] = {
        pair.name: ((int(pair.n_offset), int(pair.m_offset)),) for pair in pair_specs
    }
    for window in window_specs:
        pair_groups[window.name] = window.pair_offsets
    if len(args.frac_shift) != 2:
        raise ValueError("--frac-shift must contain exactly two comma-separated values")

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
    validation = validate_analytic_dhdk(
        0.0 + 0.0j,
        model,
        config,
        step_nm_inv=float(args.finite_step),
        d_top=d_top,
        d_bot=d_bot,
    )

    photon_energies = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    _, k_grid = build_moire_k_grid(
        model.lattice,
        int(args.mesh_size),
        endpoint=False,
        frac_shift=(float(args.frac_shift[0]), float(args.frac_shift[1])),
    )
    base_k_points = np.asarray(k_grid, dtype=np.complex128).reshape(-1)
    k_points = c3_symmetrize_k_points(base_k_points) if bool(args.c3_symmetrize_grid) else base_k_points
    k_weight = float(model.lattice.mbz_area) / float(k_points.size)
    event_prefactor = k_weight / (2.0 * np.pi) ** 2

    events: dict[tuple[str, str], dict[str, list[complex] | list[float]]] = {}
    for group_name in pair_groups:
        for component_name in components:
            events[(group_name, component_name)] = {"transition_ev": [], "weighted_weight": []}

    pair_absolute_indices = {
        group_name: tuple(
            (
                int(n_offset),
                int(m_offset),
                resolve_relative_index(model.matrix_dim, n_offset),
                resolve_relative_index(model.matrix_dim, m_offset),
            )
            for n_offset, m_offset in offsets
        )
        for group_name, offsets in pair_groups.items()
    }

    skipped_total = 0
    attempted_events = 0
    kept_events = 0
    transition_min = float("inf")
    transition_max = float("-inf")
    for k_tilde in k_points:
        hmat = build_mao_hamiltonian(k_tilde, model, config, d_top=d_top, d_bot=d_bot)
        evals, evecs = eigh(hmat)
        occupations = fermi_occupation(evals, mu_ev=float(args.mu_ev), temperature_k=float(args.temperature_k))
        D = velocity_matrices(evecs, dhdk)
        for group_name, group_pairs in pair_absolute_indices.items():
            for _n_offset, _m_offset, n_abs, m_abs in group_pairs:
                transition_ev = float(evals[m_abs] - evals[n_abs])
                transition_min = min(transition_min, transition_ev)
                transition_max = max(transition_max, transition_ev)
                if transition_ev <= 0.0:
                    continue
                if abs(float(occupations[n_abs] - occupations[m_abs])) < 1.0e-14:
                    continue
                for component_name, component in components.items():
                    attempted_events += 1
                    weight, skipped = component_transition_weight_from_D(
                        D,
                        evals,
                        occupations,
                        n_abs,
                        m_abs,
                        component,
                        denominator_cutoff_ev=float(args.denominator_cutoff_ev),
                    )
                    skipped_total += int(skipped)
                    if np.isfinite(weight.real) and np.isfinite(weight.imag):
                        events[(group_name, component_name)]["transition_ev"].append(float(transition_ev))
                        events[(group_name, component_name)]["weighted_weight"].append(event_prefactor * complex(weight))
                        kept_events += 1

    spectra: dict[str, np.ndarray] = {}
    peak_summary: dict[str, dict[str, float]] = {}
    for eta_mev in args.eta_mev:
        eta_ev = eta_mev_to_ev(float(eta_mev))
        eta_tag = f"eta_{float(eta_mev):g}meV".replace(".", "p")
        for (pair_name, component_name), table in events.items():
            transition_array = np.asarray(table["transition_ev"], dtype=float)
            weight_array = np.asarray(table["weighted_weight"], dtype=np.complex128)
            key = f"{pair_name}|{component_name}|{eta_tag}"
            sigma = spectrum_from_weighted_events(photon_energies, transition_array, weight_array, eta_ev=eta_ev)
            spectra[key] = sigma
            if sigma.size:
                peak_index = int(np.argmax(np.abs(sigma)))
                peak_summary[key] = {
                    "max_abs_uA_nm_per_V2": float(np.max(np.abs(sigma))),
                    "energy_at_max_abs_ev": float(photon_energies[peak_index]),
                    "signed_value_at_max_abs_uA_nm_per_V2": float(sigma[peak_index]),
                }

    transition_range = (
        [float(transition_min), float(transition_max)]
        if np.isfinite(transition_min) and np.isfinite(transition_max)
        else None
    )
    summary = {
        "warning": "Selected-band-pair workflow. Full Fig. 1 requires additional band-pair sums and convergence.",
        "config": {
            "theta_deg": config.theta_deg,
            "stacking": config.stacking,
            "domain": config.domain,
            "corrugation_r": config.corrugation_r,
            "n_shells": config.n_shells,
            "mesh_size": int(args.mesh_size),
            "frac_shift": [float(args.frac_shift[0]), float(args.frac_shift[1])],
            "c3_symmetrize_grid": bool(args.c3_symmetrize_grid),
            "base_k_point_count": int(base_k_points.size),
            "matrix_dim": int(model.matrix_dim),
            "eta_mev": [float(value) for value in args.eta_mev],
            "mass_ev": config.mass_ev,
            "mu_ev": float(args.mu_ev),
            "temperature_k": float(args.temperature_k),
            "denominator_cutoff_ev": float(args.denominator_cutoff_ev),
        },
        "dhdk_validation_at_gamma": {
            "max_abs_ev_nm": validation.max_abs_ev_nm,
            "passes_1e_minus_7": validation.max_abs_ev_nm < 1.0e-7,
        },
        "pair_groups": {
            group_name: {
                "pair_count": int(len(group_pairs)),
                "relative_offsets": [[int(n_offset), int(m_offset)] for n_offset, m_offset, _n_abs, _m_abs in group_pairs],
                "absolute_indices": [[int(n_abs), int(m_abs)] for _n_offset, _m_offset, n_abs, m_abs in group_pairs],
            }
            for group_name, group_pairs in pair_absolute_indices.items()
        },
        "components": {name: component_label(component) for name, component in components.items()},
        "k_point_count": int(k_points.size),
        "event_counts": {
            "attempted": int(attempted_events),
            "kept": int(kept_events),
            "skipped_small_denominators": int(skipped_total),
        },
        "selected_transition_energy_range_ev": transition_range,
        "peaks": peak_summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"photon_energies_ev": photon_energies}
        for (pair_name, component_name), table in events.items():
            prefix = _safe_key(f"{pair_name}_{component_name}")
            arrays[f"events_{prefix}_transition_ev"] = np.asarray(table["transition_ev"], dtype=float)
            arrays[f"events_{prefix}_weighted_weight"] = np.asarray(table["weighted_weight"], dtype=np.complex128)
        for key, sigma in spectra.items():
            arrays[f"sigma_{_safe_key(key)}"] = np.asarray(sigma, dtype=float)
        np.savez(args.output_dir / "htg_bandpair_spectra.npz", **arrays)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
