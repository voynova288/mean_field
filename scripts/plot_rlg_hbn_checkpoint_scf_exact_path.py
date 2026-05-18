#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.RnG_hBN import RLGhBNModel, rlg_hbn_occupied_state_count


PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


@dataclass(frozen=True)
class ExactPath:
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]
    frac_extended: np.ndarray
    frac_lookup: np.ndarray
    kvec_extended: np.ndarray
    kvec_lookup: np.ndarray
    kdist: np.ndarray
    mesh_indices: np.ndarray


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _complex_from_pairs(values: np.ndarray) -> np.ndarray:
    pairs = np.asarray(values, dtype=float)
    if pairs.shape[-1] != 2:
        raise ValueError(f"Expected complex pairs on final axis, got {pairs.shape}")
    return np.asarray(pairs[..., 0] + 1j * pairs[..., 1], dtype=np.complex128)


def _parse_panel(panel: str) -> tuple[int, float]:
    match = PANEL_RE.match(panel)
    if match is None:
        raise ValueError(f"Cannot parse panel name {panel!r}")
    return int(match.group("xi")), float(match.group("v_mev"))


def _path_nodes(path_kind: str) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    # Coordinates are fractional in the sampled [0, 1) g1 + [0, 1) g2 cell,
    # represented with common denominator 6.  This follows the TBG diagnostic
    # path convention that uses in-cell high-symmetry representatives:
    # Gamma=(0,0), M=(1/2,1/2), K=(2/3,1/3), Kprime=(1/3,2/3).
    # For the RLG/hBN Fig. 5/6 diagnostic path, use the paper geometry:
    # Gamma -> K -> K' -> Gamma is a 120-degree-apex isosceles triangle,
    # Gamma -> M' -> M -> Gamma is an equilateral triangle, and the two
    # perpendicular bisectors are collinear.  The two M nodes are chosen in the
    # sampled cell: one edge midpoint and the cell-center midpoint.
    gamma = (0, 0)
    m_point = (3, 3)
    k_point = (4, 2)
    kprime_point = (2, 4)
    if path_kind == "tbg-gamma-m-k-gamma-kprime":
        return (gamma, m_point, k_point, gamma, kprime_point), (
            r"$\Gamma_M$",
            r"$M_M$",
            r"$K_M$",
            r"$\Gamma_M$",
            r"$K'_M$",
        )
    if path_kind == "rng-fig2-gamma-k-m-gamma-kprime":
        return (gamma, k_point, m_point, gamma, kprime_point), (
            r"$\Gamma_M$",
            r"$K_M$",
            r"$M_M$",
            r"$\Gamma_M$",
            r"$K'_M$",
        )
    if path_kind == "paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma":
        # Representatives:
        #   K=(2/3,1/3), K'=(-1/3,1/3)
        #   M'=(0,1/2), M=(1/2,1/2)
        # This matches the TBG-style exact-hit diagnostic: use exact grid
        # nodes on the chosen high-symmetry path and no nearest substitution.
        kprime_fig6 = (-2, 2)
        mprime_fig6 = (0, 3)
        m_fig6 = (3, 3)
        return (gamma, k_point, kprime_fig6, gamma, mprime_fig6, m_fig6, gamma), (
            r"$\Gamma_M$",
            r"$K_M$",
            r"$K'_M$",
            r"$\Gamma_M$",
            r"$M'_M$",
            r"$M_M$",
            r"$\Gamma_M$",
        )
    raise ValueError(f"Unknown path_kind {path_kind!r}")


def _grid_lookup(frac_grid: np.ndarray, mesh_size: int) -> dict[tuple[int, int], int]:
    frac = np.asarray(frac_grid, dtype=float).reshape((-1, 2))
    lookup: dict[tuple[int, int], int] = {}
    for idx, value in enumerate(frac):
        ij = tuple(int(round(float(coord) * int(mesh_size))) % int(mesh_size) for coord in value)
        lookup[ij] = int(idx)
    if len(lookup) != frac.shape[0]:
        raise ValueError(f"Fractional grid lookup has duplicates: {len(lookup)} unique for {frac.shape[0]} points")
    return lookup


def _build_exact_path(
    *,
    path_kind: str,
    mesh_size: int,
    g1: complex,
    g2: complex,
    lookup: dict[tuple[int, int], int],
) -> ExactPath:
    node_sixths, labels = _path_nodes(path_kind)
    node_grid: list[tuple[int, int]] = []
    for frac6 in node_sixths:
        raw = (int(frac6[0]) * int(mesh_size), int(frac6[1]) * int(mesh_size))
        if raw[0] % 6 != 0 or raw[1] % 6 != 0:
            raise ValueError(f"mesh_size={mesh_size} does not hit high-symmetry node {frac6}/6 exactly")
        node_grid.append((raw[0] // 6, raw[1] // 6))

    grid_points_extended: list[tuple[int, int]] = [node_grid[0]]
    node_indices: list[int] = [1]
    for start, end in zip(node_grid[:-1], node_grid[1:], strict=True):
        delta = (int(end[0]) - int(start[0]), int(end[1]) - int(start[1]))
        steps = math.gcd(abs(delta[0]), abs(delta[1]))
        if steps == 0:
            continue
        step_delta = (delta[0] // steps, delta[1] // steps)
        for step in range(1, steps + 1):
            grid_points_extended.append((int(start[0]) + step * step_delta[0], int(start[1]) + step * step_delta[1]))
        node_indices.append(len(grid_points_extended))

    mesh_indices = []
    frac_extended_values = []
    frac_lookup_values = []
    kvec_extended_values = []
    kvec_lookup_values = []
    for ij in grid_points_extended:
        extended_frac = np.asarray([int(ij[0]) / float(mesh_size), int(ij[1]) / float(mesh_size)], dtype=float)
        key = (int(ij[0]) % int(mesh_size), int(ij[1]) % int(mesh_size))
        if key not in lookup:
            raise KeyError(f"Grid point {key} is absent from checkpoint mesh")
        mesh_indices.append(lookup[key])
        lookup_frac = np.asarray([key[0] / float(mesh_size), key[1] / float(mesh_size)], dtype=float)
        frac_extended_values.append(extended_frac)
        frac_lookup_values.append(lookup_frac)
        kvec_extended_values.append(complex(extended_frac[0] * complex(g1) + extended_frac[1] * complex(g2)))
        kvec_lookup_values.append(complex(lookup_frac[0] * complex(g1) + lookup_frac[1] * complex(g2)))

    kvec_extended = np.asarray(kvec_extended_values, dtype=np.complex128)
    kvec_lookup = np.asarray(kvec_lookup_values, dtype=np.complex128)
    kdist = np.zeros(kvec_extended.size, dtype=float)
    if kvec_extended.size > 1:
        kdist[1:] = np.cumsum(np.abs(kvec_extended[1:] - kvec_extended[:-1]))
    return ExactPath(
        labels=labels,
        node_indices=tuple(int(value) for value in node_indices),
        frac_extended=np.asarray(frac_extended_values, dtype=float),
        frac_lookup=np.asarray(frac_lookup_values, dtype=float),
        kvec_extended=kvec_extended,
        kvec_lookup=kvec_lookup,
        kdist=kdist,
        mesh_indices=np.asarray(mesh_indices, dtype=int),
    )


def _sector_indices(*, n_spin: int, n_eta: int, n_band: int, spin: int, eta: int) -> np.ndarray:
    idx = np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )
    return np.asarray(idx[int(spin), int(eta), :], dtype=int)


def _sector_energies(
    hamiltonian: np.ndarray,
    mesh_indices: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
    spin: int,
    eta: int,
) -> np.ndarray:
    block_indices = _sector_indices(n_spin=n_spin, n_eta=n_eta, n_band=n_band, spin=spin, eta=eta)
    out = np.zeros((int(n_band), int(mesh_indices.size)), dtype=float)
    for ipath, ik in enumerate(np.asarray(mesh_indices, dtype=int)):
        block = hamiltonian[:, :, int(ik)][np.ix_(block_indices, block_indices)]
        out[:, ipath] = np.linalg.eigvalsh(block)
    return out


def _mu_from_checkpoint(energies: np.ndarray, config: dict[str, object]) -> float:
    total_occupied = rlg_hbn_occupied_state_count(
        float(config["nu"]),
        int(energies.shape[0]),
        int(energies.shape[1]),
        active_valence_bands=int(config["active_valence_bands"]),
        n_spin=2,
        n_eta=2,
    )
    values = np.sort(np.asarray(energies, dtype=float).reshape(-1))
    if total_occupied <= 0 or total_occupied >= values.size:
        return 0.0
    return float(0.5 * (values[total_occupied - 1] + values[total_occupied]))


def _parse_ylim(text: str) -> tuple[float, float]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("--ylim-mev must be lower,upper")
    return float(pieces[0]), float(pieces[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a checkpoint using exact SCF grid points on a TBG-style path.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--path-kind",
        choices=(
            "tbg-gamma-m-k-gamma-kprime",
            "rng-fig2-gamma-k-m-gamma-kprime",
            "paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma",
        ),
        default="paper-fig6-gamma-k-kprime-gamma-mprime-m-gamma",
    )
    parser.add_argument("--spin-index", type=int, default=0)
    parser.add_argument("--ylim-mev", type=str, default="-90,90")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    ensure_not_running_compute_on_login_node("RLG/hBN checkpoint exact-SCF-path band plotting")
    config = _read_json(args.config)
    xi, v_mev = _parse_panel(args.panel)
    archive = np.load(args.checkpoint)
    hamiltonian = np.asarray(archive["hamiltonian"], dtype=np.complex128)
    energies = np.asarray(archive["energies_mev"], dtype=float)
    frac_grid = np.asarray(archive["k_grid_frac"], dtype=float)
    source_kvec = _complex_from_pairs(archive["kvec_nm_inv"]).reshape(-1)
    mesh_size = int(config["k_mesh_size"])

    model = RLGhBNModel.from_config(
        layer_count=int(config["layer_count"]),
        xi=int(xi),
        theta_deg=float(config["theta_deg"]),
        displacement_field_mev=float(v_mev),
        shell_count=int(config["shell_count"]),
    )
    lookup = _grid_lookup(frac_grid, mesh_size)
    path = _build_exact_path(
        path_kind=str(args.path_kind),
        mesh_size=mesh_size,
        g1=model.lattice.g_m1,
        g2=model.lattice.g_m2,
        lookup=lookup,
    )
    if np.max(np.abs(source_kvec[path.mesh_indices] - path.kvec_lookup)) > 1.0e-10:
        raise ValueError("Checkpoint kvec and exact path lookup kvec disagree")

    n_spin = 2
    n_eta = 2
    n_band = int(hamiltonian.shape[0]) // (n_spin * n_eta)
    k_energies = _sector_energies(
        hamiltonian,
        path.mesh_indices,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        spin=int(args.spin_index),
        eta=0,
    )
    kp_energies = _sector_energies(
        hamiltonian,
        path.mesh_indices,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        spin=int(args.spin_index),
        eta=1,
    )
    mu_mev = _mu_from_checkpoint(energies, config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"current_checkpoint_scf_exact_{args.path_kind}_bands"
    output_npz = output_dir / f"{stem}.npz"
    output_png = output_dir / f"{stem}.png"
    output_pdf = output_dir / f"{stem}.pdf"
    output_summary = output_dir / f"{stem}_summary.json"

    np.savez_compressed(
        output_npz,
        kdist=path.kdist,
        kvec_extended_nm_inv=np.stack([path.kvec_extended.real, path.kvec_extended.imag], axis=-1),
        kvec_lookup_nm_inv=np.stack([path.kvec_lookup.real, path.kvec_lookup.imag], axis=-1),
        frac_extended=path.frac_extended,
        frac_lookup=path.frac_lookup,
        mesh_indices=path.mesh_indices,
        node_indices=np.asarray(path.node_indices, dtype=int),
        labels=np.asarray(path.labels, dtype=object),
        spin_up_K_energies_mev=k_energies,
        spin_up_Kprime_energies_mev=kp_energies,
        energy_zero_mev=np.asarray(mu_mev),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ylim = _parse_ylim(args.ylim_mev)
    fig, ax = plt.subplots(figsize=(4.3, 3.3), constrained_layout=True)
    for iband in range(n_band):
        ax.plot(path.kdist, k_energies[iband] - mu_mev, color="black", linewidth=0.9, marker=".", markersize=2.2)
        ax.plot(path.kdist, kp_energies[iband] - mu_mev, color="#c62828", linewidth=0.9, marker=".", markersize=2.2)
    node_positions = path.kdist[np.asarray(path.node_indices, dtype=int) - 1]
    for xpos in node_positions:
        ax.axvline(float(xpos), color="0.78", linewidth=0.6)
    ax.axhline(0.0, color="0.35", linewidth=0.55, linestyle="--")
    ax.set_xticks(node_positions)
    ax.set_xticklabels(path.labels, fontsize=8)
    ax.set_xlim(float(path.kdist[0]), float(path.kdist[-1]))
    ax.set_ylim(*ylim)
    ax.set_ylabel("$E-E_F$ (meV)", fontsize=9)
    ax.set_title(f"$\\xi={xi}$, $V={v_mev:.0f}$ meV SCF exact path", fontsize=10)
    ax.plot([], [], color="black", linewidth=1.0, label="$K$")
    ax.plot([], [], color="#c62828", linewidth=1.0, label="$K'$")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(output_png, dpi=int(args.dpi))
    fig.savefig(output_pdf)
    plt.close(fig)

    summary = {
        "approximation": "diagnostic-only SCF mesh plot; exact grid points on the selected high-symmetry path using the TBG exact-hit convention; no nearest mapping and no path reconstruction",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "energy_zero_mev": float(mu_mev),
        "iteration": int(np.asarray(archive["iteration"], dtype=int).reshape(-1)[-1]) if "iteration" in archive.files else None,
        "mesh_size": int(mesh_size),
        "output_npz": str(output_npz),
        "output_pdf": str(output_pdf),
        "output_png": str(output_png),
        "panel": str(args.panel),
        "path_kind": str(args.path_kind),
        "path_labels": list(path.labels),
        "points": int(path.kdist.size),
        "unique_mesh_points": int(np.unique(path.mesh_indices).size),
    }
    _write_json(output_summary, summary)
    print(f"[done] output_png={output_png}", flush=True)


if __name__ == "__main__":
    main()
