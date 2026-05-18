#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.devtools.run_rlg_hbn_paper_hf import PAPER_CONFIGS
from mean_field.systems.RnG_hBN import RLGhBNModel, build_kpath_from_nodes, rlg_hbn_occupied_state_count


PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _complex_from_pairs(values: np.ndarray) -> np.ndarray:
    pairs = np.asarray(values, dtype=float)
    if pairs.shape[-1] != 2:
        raise ValueError(f"Expected final axis of length 2, got {pairs.shape}")
    return np.asarray(pairs[..., 0] + 1j * pairs[..., 1], dtype=np.complex128)


def _parse_panel(panel: str) -> tuple[int, float]:
    match = PANEL_RE.match(panel)
    if match is None:
        raise ValueError(f"Cannot parse panel name {panel!r}")
    return int(match.group("xi")), float(match.group("v_mev"))


def _paper_hf_path(model: RLGhBNModel, points_per_segment: int):
    lattice = model.lattice
    nodes = (
        lattice.gamma_m,
        lattice.k_m,
        lattice.kprime_m,
        lattice.gamma_m,
        -lattice.m_m,
        lattice.m_m,
        lattice.gamma_m,
    )
    labels = ("$\\Gamma_M$", "$K_M$", "$K'_M$", "$\\Gamma_M$", "$M'_M$", "$M_M$", "$\\Gamma_M$")
    return build_kpath_from_nodes(
        nodes,
        labels,
        tuple(int(points_per_segment) for _ in range(len(nodes) - 1)),
    )


def _sector_indices(*, n_spin: int, n_eta: int, n_band: int, spin: int, eta: int) -> np.ndarray:
    idx = np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )
    return np.asarray(idx[int(spin), int(eta), :], dtype=int)


def _sector_energies_from_mesh(
    hamiltonian: np.ndarray,
    nearest_indices: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
    spin: int,
    eta: int,
) -> np.ndarray:
    block_indices = _sector_indices(n_spin=n_spin, n_eta=n_eta, n_band=n_band, spin=spin, eta=eta)
    energies = np.zeros((int(n_band), nearest_indices.size), dtype=float)
    for ipath, ik in enumerate(np.asarray(nearest_indices, dtype=int)):
        block = hamiltonian[:, :, int(ik)][np.ix_(block_indices, block_indices)]
        energies[:, ipath] = np.linalg.eigvalsh(block)
    return energies


def _source_mu_mev(energies: np.ndarray, config: dict[str, object]) -> float:
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


def _fold_kvec_to_source_cell(path_kvec: np.ndarray, *, g1: complex, g2: complex) -> np.ndarray:
    reciprocal = np.asarray([[float(g1.real), float(g2.real)], [float(g1.imag), float(g2.imag)]], dtype=float)
    xy = np.stack([np.asarray(path_kvec, dtype=np.complex128).real, np.asarray(path_kvec, dtype=np.complex128).imag], axis=0)
    frac = np.linalg.solve(reciprocal, xy).T
    frac = np.mod(frac, 1.0)
    return np.asarray(frac[:, 0] * complex(g1) + frac[:, 1] * complex(g2), dtype=np.complex128)


def _nearest_path_indices(source_kvec: np.ndarray, path_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.abs(np.asarray(source_kvec, dtype=np.complex128)[:, None] - np.asarray(path_kvec, dtype=np.complex128)[None, :])
    nearest = np.argmin(distances, axis=0)
    nearest_distance = distances[nearest, np.arange(path_kvec.size)]
    return np.asarray(nearest, dtype=int), np.asarray(nearest_distance, dtype=float)


def _parse_ylim(text: str) -> tuple[float, float]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("--ylim-mev must be lower,upper")
    return float(pieces[0]), float(pieces[1])


def _plot(args: argparse.Namespace) -> dict[str, object]:
    ensure_not_running_compute_on_login_node("RLG/hBN checkpoint mesh-nearest band plotting")
    config = _read_json(args.config)
    if str(args.paper_target) not in PAPER_CONFIGS:
        raise ValueError(f"Unsupported paper target {args.paper_target!r}")
    xi, v_mev = _parse_panel(str(args.panel))
    archive = np.load(args.checkpoint)
    source_kvec = _complex_from_pairs(archive["kvec_nm_inv"])
    hamiltonian = np.asarray(archive["hamiltonian"], dtype=np.complex128)
    energies = np.asarray(archive["energies_mev"], dtype=float)
    if hamiltonian.ndim != 3:
        raise ValueError(f"Expected hamiltonian shape (nt, nt, nk), got {hamiltonian.shape}")
    n_spin = 2
    n_eta = 2
    n_band = int(hamiltonian.shape[0]) // (n_spin * n_eta)
    if hamiltonian.shape[0] != n_spin * n_eta * n_band:
        raise ValueError(f"Cannot factor hamiltonian dimension {hamiltonian.shape[0]}")

    model = RLGhBNModel.from_config(
        layer_count=int(config["layer_count"]),
        xi=int(xi),
        theta_deg=float(config["theta_deg"]),
        displacement_field_mev=float(v_mev),
        shell_count=int(config["shell_count"]),
    )
    path = _paper_hf_path(model, int(args.points_per_segment))
    path_lookup_kvec = _fold_kvec_to_source_cell(
        np.asarray(path.kvec, dtype=np.complex128),
        g1=model.lattice.g_m1,
        g2=model.lattice.g_m2,
    )
    nearest, nearest_distance = _nearest_path_indices(source_kvec, path_lookup_kvec)
    k_energies = _sector_energies_from_mesh(
        hamiltonian,
        nearest,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        spin=int(args.spin_index),
        eta=0,
    )
    kp_energies = _sector_energies_from_mesh(
        hamiltonian,
        nearest,
        n_spin=n_spin,
        n_eta=n_eta,
        n_band=n_band,
        spin=int(args.spin_index),
        eta=1,
    )
    mu_mev = _source_mu_mev(energies, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = args.output_dir / "current_checkpoint_mesh_folded_nearest_bands.npz"
    output_png = args.output_dir / "current_checkpoint_mesh_folded_nearest_bands.png"
    output_pdf = args.output_dir / "current_checkpoint_mesh_folded_nearest_bands.pdf"
    output_summary = args.output_dir / "current_checkpoint_mesh_folded_nearest_bands_summary.json"

    np.savez_compressed(
        output_npz,
        kdist=np.asarray(path.kdist, dtype=float),
        path_kvec_nm_inv=np.stack([np.asarray(path.kvec).real, np.asarray(path.kvec).imag], axis=-1),
        path_lookup_kvec_nm_inv=np.stack([np.asarray(path_lookup_kvec).real, np.asarray(path_lookup_kvec).imag], axis=-1),
        nearest_source_indices=nearest,
        nearest_distance_nm_inv=nearest_distance,
        spin_up_K_energies_mev=k_energies,
        spin_up_Kprime_energies_mev=kp_energies,
        energy_zero_mev=np.asarray(mu_mev),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kdist = np.asarray(path.kdist, dtype=float)
    ylim = _parse_ylim(str(args.ylim_mev))
    fig, ax = plt.subplots(figsize=(4.3, 3.3), constrained_layout=True)
    for iband in range(n_band):
        ax.plot(kdist, k_energies[iband] - mu_mev, color="black", linewidth=0.9)
        ax.plot(kdist, kp_energies[iband] - mu_mev, color="#c62828", linewidth=0.9)
    node_indices = np.asarray(path.node_indices, dtype=int) - 1
    node_positions = np.asarray(path.kdist, dtype=float)[node_indices]
    for xpos in node_positions:
        ax.axvline(float(xpos), color="0.78", linewidth=0.6)
    ax.axhline(0.0, color="0.35", linewidth=0.55, linestyle="--")
    ax.set_xticks(node_positions)
    ax.set_xticklabels(path.labels, fontsize=8)
    ax.set_xlim(float(kdist[0]), float(kdist[-1]))
    ax.set_ylim(*ylim)
    ax.set_ylabel("$E-E_F$ (meV)", fontsize=9)
    ax.set_title(f"$\\xi={xi}$, $V={v_mev:.0f}$ meV checkpoint mesh", fontsize=10)
    ax.plot([], [], color="black", linewidth=1.0, label="$K$")
    ax.plot([], [], color="#c62828", linewidth=1.0, label="$K'$")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(output_png, dpi=int(args.dpi))
    fig.savefig(output_pdf)
    plt.close(fig)

    iteration = int(np.asarray(archive["iteration"], dtype=int).reshape(-1)[-1]) if "iteration" in archive.files else None
    summary = {
        "approximation": "nearest available SCF mesh k point after folding the requested high-symmetry path into the sampled reciprocal unit cell; no path self-energy recomputation",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "energy_zero_mev": float(mu_mev),
        "iteration": iteration,
        "max_nearest_distance_nm_inv": float(np.max(nearest_distance)),
        "mean_nearest_distance_nm_inv": float(np.mean(nearest_distance)),
        "output_npz": str(output_npz),
        "output_pdf": str(output_pdf),
        "output_png": str(output_png),
        "panel": str(args.panel),
        "path_labels": list(path.labels),
        "points": int(np.asarray(path.kvec).size),
        "source_mesh_points_used": int(np.unique(nearest).size),
        "source_mesh_total_points": int(source_kvec.size),
    }
    _write_json(output_summary, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot current RLG/hBN HF checkpoint bands from nearest SCF mesh points.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=str, required=True)
    parser.add_argument("--paper-target", choices=tuple(PAPER_CONFIGS), default="fig6")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points-per-segment", type=int, default=24)
    parser.add_argument("--spin-index", type=int, default=0)
    parser.add_argument("--ylim-mev", type=str, default="-90,90")
    parser.add_argument("--dpi", type=int, default=180)
    summary = _plot(parser.parse_args())
    print(f"[done] output_png={summary['output_png']}", flush=True)


if __name__ == "__main__":
    main()
