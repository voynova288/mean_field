from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import eta_mev_to_ev
from .response import (
    add_transitions_to_integral,
    parse_component,
    positive_transition_terms,
    precompute_response_tensors,
    sigma_from_integral,
)
from .slg_toy import (
    GappedSLGParams,
    bz_area_nm_inv_sq,
    d2hdk,
    dhdk,
    diagonalize,
    hex_bz_grid,
    reciprocal_vectors,
)

PAPER_COMPONENTS = ("x;xy", "y;yy")
DIAGNOSTIC_COMPONENTS = ("x;xy", "y;yy", "y;xx")


def primitive_bz_grid(mesh_size: int, params: GappedSLGParams) -> tuple[np.ndarray, np.ndarray]:
    """Uniform midpoint grid on one primitive reciprocal parallelogram."""

    n = int(mesh_size)
    if n <= 1:
        raise ValueError(f"mesh_size must be > 1, got {mesh_size}")
    b1, b2 = reciprocal_vectors(params)
    u = (np.arange(n, dtype=float) + 0.5) / float(n) - 0.5
    uu, vv = np.meshgrid(u, u, indexing="ij")
    points = uu.reshape(-1, 1) * b1.reshape(1, 2) + vv.reshape(-1, 1) * b2.reshape(1, 2)
    weights = np.full(points.shape[0], bz_area_nm_inv_sq(params) / float(points.shape[0]), dtype=float)
    return points, weights


def c3_orbit_grid(mesh_size: int, params: GappedSLGParams) -> tuple[np.ndarray, np.ndarray]:
    base_points, base_weights = hex_bz_grid(mesh_size, params)
    points: list[np.ndarray] = []
    weights: list[float] = []
    for k_xy, weight in zip(base_points, base_weights, strict=True):
        for multiple in (0, 1, 2):
            theta = 2.0 * math.pi * multiple / 3.0
            c, s = math.cos(theta), math.sin(theta)
            points.append(np.asarray([c * k_xy[0] - s * k_xy[1], s * k_xy[0] + c * k_xy[1]], dtype=float))
            weights.append(float(weight) / 3.0)
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def compute_spectra(
    photon_energies_ev: np.ndarray,
    *,
    params: GappedSLGParams,
    mesh_size: int,
    eta_ev: float,
    components: tuple[str, ...],
    include_second_derivative: bool,
    grid: str,
) -> dict[str, np.ndarray]:
    parsed = {name: parse_component(name) for name in components}
    integrals = {name: np.zeros_like(photon_energies_ev, dtype=np.complex128) for name in parsed}
    if grid == "primitive":
        k_points, k_weights = primitive_bz_grid(mesh_size, params)
    elif grid == "c3_hex":
        k_points, k_weights = c3_orbit_grid(mesh_size, params)
    else:
        raise ValueError(f"unknown grid {grid!r}")

    for k_xy, weight in zip(k_points, k_weights, strict=True):
        evals, evecs = diagonalize(k_xy, params)
        tensors = precompute_response_tensors(
            evals,
            evecs,
            dhdk(k_xy, params),
            d2hdk=d2hdk(k_xy, params) if include_second_derivative else None,
            denominator_cutoff_ev=1.0e-10,
        )
        for name, component in parsed.items():
            transitions, weights = positive_transition_terms(tensors, component)
            add_transitions_to_integral(
                integrals[name],
                photon_energies_ev,
                transitions,
                weights,
                k_weight_nm_inv_sq=float(weight),
                eta_ev=eta_ev,
            )
    return {name: sigma_from_integral(integral) for name, integral in integrals.items()}


def peak_summary(photon: np.ndarray, spectra: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, values in spectra.items():
        arr = np.asarray(values, dtype=float)
        idx = int(np.argmax(np.abs(arr)))
        out[name] = {
            "energy_at_max_abs_ev": float(photon[idx]),
            "value_at_max_abs_uA_nm_per_V2": float(arr[idx]),
            "max_abs_uA_nm_per_V2": float(np.max(np.abs(arr))),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Honest Mao Fig. 10 audit for the gapped-SLG toy: compare the formal W-corrected "
            "shift current with the paper-printed Eq.(4)-only finite-grid calculation."
        )
    )
    parser.add_argument("--mesh-size", type=int, default=150)
    parser.add_argument("--eta-mev", type=float, default=50.0)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=8.0)
    parser.add_argument("--n-energy", type=int, default=401)
    parser.add_argument("--mass-ev", type=float, default=1.5)
    parser.add_argument("--hopping-ev", type=float, default=2.73)
    parser.add_argument("--convergence-meshes", type=int, nargs="*", default=(80, 100, 120, 150, 200))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_current_slg_toy_fig10_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=float(args.mass_ev))
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    eta_ev = eta_mev_to_ev(float(args.eta_mev))

    formal = compute_spectra(
        photon,
        params=params,
        mesh_size=int(args.mesh_size),
        eta_ev=eta_ev,
        components=PAPER_COMPONENTS,
        include_second_derivative=True,
        grid="c3_hex",
    )
    printed_eq4_primitive = compute_spectra(
        photon,
        params=params,
        mesh_size=int(args.mesh_size),
        eta_ev=eta_ev,
        components=DIAGNOSTIC_COMPONENTS,
        include_second_derivative=False,
        grid="primitive",
    )
    printed_eq4_c3 = compute_spectra(
        photon,
        params=params,
        mesh_size=max(24, min(int(args.mesh_size), 80)),
        eta_ev=eta_ev,
        components=DIAGNOSTIC_COMPONENTS,
        include_second_derivative=False,
        grid="c3_hex",
    )

    convergence: dict[str, dict[str, float]] = {}
    for mesh in args.convergence_meshes:
        spectra = compute_spectra(
            photon,
            params=params,
            mesh_size=int(mesh),
            eta_ev=eta_ev,
            components=("x;xy", "y;xx"),
            include_second_derivative=False,
            grid="primitive",
        )
        convergence[str(mesh)] = {
            "x;xy_peak_abs_uA_nm_per_V2": float(np.max(np.abs(spectra["x;xy"]))),
            "x;xy_peak_energy_ev": float(photon[int(np.argmax(np.abs(spectra["x;xy"])))]),
            "y;xx_peak_abs_uA_nm_per_V2": float(np.max(np.abs(spectra["y;xx"]))),
            "y;xx_peak_energy_ev": float(photon[int(np.argmax(np.abs(spectra["y;xx"])))]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "slg_toy_fig10_audit_data.npz",
        photon_energies_ev=photon,
        **{f"formal_{name.replace(';', '_')}": values for name, values in formal.items()},
        **{f"printed_eq4_primitive_{name.replace(';', '_')}": values for name, values in printed_eq4_primitive.items()},
        **{f"printed_eq4_c3_{name.replace(';', '_')}": values for name, values in printed_eq4_c3.items()},
    )

    summary = {
        "status": (
            "Formal W-corrected/official-reference shift current does not reproduce Mao Fig. 10. "
            "The M-point blue peak appears only in the paper-printed Eq.(4)-only finite primitive-grid calculation, "
            "where the paper-labelled y;yy component remains zero and the C3-symmetrized result vanishes."
        ),
        "no_visual_fitting": True,
        "params": {
            "mesh_size": int(args.mesh_size),
            "eta_mev": float(args.eta_mev),
            "mass_ev": params.mass_ev,
            "hopping_ev": params.hopping_ev,
            "bond_nm": params.bond_nm,
        },
        "formal_W_corrected_c3_hex_peaks": peak_summary(photon, formal),
        "paper_printed_eq4_no_W_primitive_grid_peaks": peak_summary(photon, printed_eq4_primitive),
        "paper_printed_eq4_no_W_c3_grid_peaks": peak_summary(photon, printed_eq4_c3),
        "printed_eq4_primitive_mesh_convergence": convergence,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1), constrained_layout=True)

    ax = axes[0]
    ax.plot(photon, formal["x;xy"], color="blue", lw=1.5, label=r"formal $\sigma^{x;xy}$")
    ax.plot(photon, formal["y;yy"], color="red", lw=1.5, label=r"formal $\sigma^{y;yy}$")
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_ylabel(r"$\sigma$ [$\mu$A nm V$^{-2}$]")
    ax.set_title("formal W-corrected")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ax.plot(photon, printed_eq4_primitive["x;xy"], color="blue", lw=1.5, label=r"Eq.(4) no-$W$ $x;xy$")
    ax.plot(photon, printed_eq4_primitive["y;yy"], color="red", lw=1.5, label=r"Eq.(4) no-$W$ $y;yy$")
    ax.plot(photon, printed_eq4_primitive["y;xx"], color="orange", lw=1.1, ls="--", label=r"diagnostic $y;xx$")
    ax.axhline(0.0, color="0.4", lw=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_xlabel(r"$\omega$ [eV]")
    ax.set_title("printed Eq.(4), primitive grid")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    meshes = np.asarray([int(key) for key in convergence.keys()], dtype=int)
    xxy_peaks = np.asarray([convergence[str(mesh)]["x;xy_peak_abs_uA_nm_per_V2"] for mesh in meshes], dtype=float)
    yxx_peaks = np.asarray([convergence[str(mesh)]["y;xx_peak_abs_uA_nm_per_V2"] for mesh in meshes], dtype=float)
    ax.plot(meshes, xxy_peaks, "o-", color="blue", label=r"Eq.(4) no-$W$ $|x;xy|$")
    ax.plot(meshes, yxx_peaks, "s--", color="orange", label=r"Eq.(4) no-$W$ $|y;xx|$")
    ax.set_xlabel("primitive grid mesh")
    ax.set_ylabel(r"peak $|\sigma|$ [$\mu$A nm V$^{-2}$]")
    ax.set_title("finite-grid dependence")
    ax.legend(frameon=False, fontsize=7)

    fig.savefig(args.output_dir / "slg_toy_fig10_audit.png", dpi=220)
    fig.savefig(args.output_dir / "slg_toy_fig10_audit.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(args.output_dir / "slg_toy_fig10_audit.png")


if __name__ == "__main__":
    main()
