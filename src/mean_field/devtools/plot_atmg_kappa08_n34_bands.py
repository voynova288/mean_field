#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mean_field.runtime import collect_runtime_environment, current_timestamp
from mean_field.systems.atmg import ATMGModel, ATMGParameters
from mean_field.systems.atmg.bilayer_map import build_atmg_via_tbg_sum


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "ATMG"


@dataclass(frozen=True)
class ATMGCaseSpec:
    name: str
    n_layers: int
    alpha: float


@dataclass(frozen=True)
class ATMGCaseResult:
    spec: ATMGCaseSpec
    theta_deg: float
    kappa: float
    w_ab_ev: float
    w_aa_ev: float
    moire_energy_scale_ev: float
    labels: tuple[str, ...]
    kdist: np.ndarray
    tick_positions: list[float]
    tick_labels: list[str]
    combined_energies_ev: np.ndarray
    subspace_energies_ev: tuple[np.ndarray, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ATMG n=3,4 bands for kappa = w_AA / w_AB = 0.8.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-shells", type=int, default=5)
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--kappa", type=float, default=0.8)
    parser.add_argument("--w-ab-ev", type=float, default=0.110)
    parser.add_argument("--native-window-ev", type=float, default=0.12)
    parser.add_argument("--scaled-window", type=float, default=1.0)
    parser.add_argument("--energy-unit", choices=("native", "vfkdtheta", "both"), default="both")
    return parser.parse_args()


def _default_output_dir() -> Path:
    timestamp = current_timestamp().replace(":", "").replace("-", "")
    return DEFAULT_OUTPUT_ROOT / f"kappa08_n34_bands_{timestamp}"


def _requested_units(selection: str) -> tuple[str, ...]:
    if selection == "both":
        return ("native", "vfkdtheta")
    return (selection,)


def _theta_deg_from_alpha(alpha: float, *, w_ab_ev: float, vf: float, graphene_a_nm: float) -> float:
    graphene_k_mag = 4.0 * math.pi / (3.0 * float(graphene_a_nm))
    theta_rad = float(w_ab_ev) / (float(alpha) * float(vf) * graphene_k_mag)
    return float(theta_rad * 180.0 / math.pi)


def _select_window_bands(energies: np.ndarray, *, emin: float, emax: float, fallback_each_side: int) -> np.ndarray:
    energies = np.asarray(energies, dtype=float)
    band_min = np.min(energies, axis=0)
    band_max = np.max(energies, axis=0)
    indices = np.nonzero((band_max >= emin) & (band_min <= emax))[0]
    if indices.size > 0:
        return indices
    center = energies.shape[1] // 2
    lower = max(0, center - fallback_each_side)
    upper = min(energies.shape[1], center + fallback_each_side)
    return np.arange(lower, upper, dtype=int)


def _plot_case_grid(
    cases: list[ATMGCaseResult],
    output_path: Path,
    *,
    energy_unit: str,
    native_window_ev: float,
    scaled_window: float,
) -> None:
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.2, 7.8), sharex=False)
    axes_array = np.atleast_1d(axes)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")

    if energy_unit == "native":
        ylabel = "Energy (eV)"
        window = (-float(native_window_ev), float(native_window_ev))
        title_suffix = ""
    else:
        ylabel = r"$E / (v_F k_D \theta)$"
        window = (-float(scaled_window), float(scaled_window))
        title_suffix = r", unit $E/(v_F k_D \theta)$"

    for axis, case in zip(axes_array, cases, strict=True):
        scale = 1.0 if energy_unit == "native" else float(case.moire_energy_scale_ev)
        for xpos in case.tick_positions[1:-1]:
            axis.axvline(xpos, color="0.87", linewidth=0.8, zorder=0)

        for block_index, (label, energies) in enumerate(zip(case.labels, case.subspace_energies_ev, strict=True)):
            color = colors[min(block_index, len(colors) - 1)]
            energies_plot = np.asarray(energies, dtype=float) / scale
            band_indices = _select_window_bands(
                energies_plot,
                emin=float(window[0]),
                emax=float(window[1]),
                fallback_each_side=8,
            )
            for band in band_indices:
                axis.plot(case.kdist, energies_plot[:, band], color=color, linewidth=0.8, alpha=0.9)

        axis.set_title(
            f"ATMG n={case.spec.n_layers}, alpha={case.spec.alpha:.6f}, "
            f"kappa={case.kappa:.3f}{title_suffix}"
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks(case.tick_positions, case.tick_labels)
        axis.set_xlim(float(case.kdist[0]), float(case.kdist[-1]))
        axis.set_ylim(*window)
        axis.grid(True, alpha=0.25)
        axis.legend(
            handles=[
                Line2D([0], [0], color=colors[min(index, len(colors) - 1)], linewidth=1.2, label=label)
                for index, label in enumerate(case.labels)
            ],
            frameon=False,
            loc="upper right",
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_case_npz(output_path: Path, case: ATMGCaseResult) -> None:
    payload: dict[str, object] = {
        "kdist": np.asarray(case.kdist, dtype=float),
        "combined_energies_ev": np.asarray(case.combined_energies_ev, dtype=float),
        "theta_deg": np.asarray(case.theta_deg, dtype=float),
        "alpha_dimless": np.asarray(case.spec.alpha, dtype=float),
        "kappa": np.asarray(case.kappa, dtype=float),
        "w_ab_ev": np.asarray(case.w_ab_ev, dtype=float),
        "w_aa_ev": np.asarray(case.w_aa_ev, dtype=float),
        "moire_energy_scale_ev": np.asarray(case.moire_energy_scale_ev, dtype=float),
        "tick_positions": np.asarray(case.tick_positions, dtype=float),
        "tick_labels": np.asarray(case.tick_labels, dtype=object),
        "labels": np.asarray(case.labels, dtype=object),
    }
    for index, energies in enumerate(case.subspace_energies_ev, start=1):
        payload[f"subspace_{index}_energies_ev"] = np.asarray(energies, dtype=float)
    np.savez(output_path, **payload)


def _write_summary(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_original_paper_path(model: ATMGModel, points_per_segment: int):
    kprime = -complex(model.lattice.q0)
    kpoint = 0.0 + 0.0j
    gamma = -complex(model.lattice.q0) - complex(model.lattice.q_plus)
    gamma_prime = gamma + complex(model.lattice.g_m1)
    kprime_translated = kprime + complex(model.lattice.g_m1)
    return model.build_kpath(
        (
            kprime,
            kpoint,
            gamma,
            gamma_prime,
            kprime_translated,
        ),
        ("K'", "K", r"$\Gamma$", r"$\Gamma'$", "K'"),
        points_per_segment=points_per_segment,
    )


def _build_case(
    spec: ATMGCaseSpec,
    *,
    kappa: float,
    w_ab_ev: float,
    n_shells: int,
    points_per_segment: int,
) -> ATMGCaseResult:
    prototype = ATMGParameters.realistic(2, 1.0)
    theta_deg = _theta_deg_from_alpha(
        spec.alpha,
        w_ab_ev=w_ab_ev,
        vf=prototype.vf,
        graphene_a_nm=prototype.graphene_lattice_constant_nm,
    )
    params = ATMGParameters.realistic(
        spec.n_layers,
        theta_deg,
        w_ab=w_ab_ev,
        kappa=kappa,
    )
    model = ATMGModel.from_config(spec.n_layers, theta_deg, n_shells=n_shells, params=params)
    path = _build_original_paper_path(model, points_per_segment)
    first_mapped = build_atmg_via_tbg_sum(complex(path.kvec[0]), model.lattice, model.params, valley=1)
    combined_energies = np.zeros((path.kvec.size, first_mapped.combined_energies.size), dtype=float)
    subspace_energies = [np.zeros((path.kvec.size, block.size), dtype=float) for block in first_mapped.subspace_energies]

    for ik, kval in enumerate(path.kvec):
        mapped = build_atmg_via_tbg_sum(complex(kval), model.lattice, model.params, valley=1)
        combined_energies[ik, :] = np.asarray(mapped.combined_energies, dtype=float)
        for block_index, block in enumerate(mapped.subspace_energies):
            subspace_energies[block_index][ik, :] = np.asarray(block, dtype=float)

    tick_positions = [float(path.kdist[index - 1]) for index in path.node_indices]
    tick_labels = [str(label) for label in path.labels]

    return ATMGCaseResult(
        spec=spec,
        theta_deg=float(theta_deg),
        kappa=float(kappa),
        w_ab_ev=float(params.w_ab),
        w_aa_ev=float(params.kappa * params.w_ab),
        moire_energy_scale_ev=float(params.moire_energy_scale),
        labels=tuple(first_mapped.labels),
        kdist=np.asarray(path.kdist, dtype=float),
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        combined_energies_ev=np.asarray(combined_energies, dtype=float),
        subspace_energies_ev=tuple(np.asarray(item, dtype=float) for item in subspace_energies),
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phi = (math.sqrt(5.0) + 1.0) / 2.0
    specs = [
        ATMGCaseSpec(name="n3_alpha1", n_layers=3, alpha=0.586 / math.sqrt(2.0)),
        ATMGCaseSpec(name="n4_alpha1", n_layers=4, alpha=0.586 / phi),
    ]
    cases = [
        _build_case(
            spec,
            kappa=float(args.kappa),
            w_ab_ev=float(args.w_ab_ev),
            n_shells=int(args.n_shells),
            points_per_segment=int(args.points_per_segment),
        )
        for spec in specs
    ]

    figure_paths: dict[str, str] = {}
    for energy_unit in _requested_units(args.energy_unit):
        suffix = ".png" if energy_unit == "native" else "_vfkdtheta.png"
        plot_path = output_dir / f"atmg_kappa08_n3_n4_bands{suffix}"
        _plot_case_grid(
            cases,
            plot_path,
            energy_unit=energy_unit,
            native_window_ev=float(args.native_window_ev),
            scaled_window=float(args.scaled_window),
        )
        figure_paths[energy_unit] = str(plot_path)

    for case in cases:
        _save_case_npz(output_dir / f"{case.spec.name}_bands.npz", case)

    summary_payload = {
        "generated_at": current_timestamp(),
        "output_dir": str(output_dir),
        "runtime_environment": asdict(collect_runtime_environment()),
        "plot_settings": {
            "energy_unit": args.energy_unit,
            "native_window_ev": args.native_window_ev,
            "scaled_window": args.scaled_window,
            "n_shells": args.n_shells,
            "points_per_segment": args.points_per_segment,
        },
        "figures": figure_paths,
        "cases": {
            case.spec.name: {
                "n_layers": case.spec.n_layers,
                "alpha_dimless": case.spec.alpha,
                "theta_deg": case.theta_deg,
                "kappa": case.kappa,
                "w_ab_ev": case.w_ab_ev,
                "w_aa_ev": case.w_aa_ev,
                "moire_energy_scale_ev": case.moire_energy_scale_ev,
                "path_labels": list(case.tick_labels),
                "labels": list(case.labels),
                "npz_path": str(output_dir / f"{case.spec.name}_bands.npz"),
            }
            for case in cases
        },
    }
    _write_summary(output_dir / "summary.json", summary_payload)

    print(f"output_dir={output_dir}")
    for energy_unit, path in figure_paths.items():
        print(f"figure_{energy_unit}={path}")
    print(f"summary_json={output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
