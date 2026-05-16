from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import socket
from time import perf_counter

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.devtools.run_rlg_hbn_paper_hf import PAPER_CONFIGS
from mean_field.systems.RnG_hBN import (
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNInteractionParams,
    RLGhBNModel,
    build_kpath_from_nodes,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis_for_kvec,
    evaluate_rlg_hbn_hf_path,
    rlg_hbn_occupied_state_count,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-style R5G/hBN HF band structures from saved paper HF source states."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--paper-target", choices=tuple(PAPER_CONFIGS), default=None)
    parser.add_argument("--points-per-segment", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--spin-index", type=int, default=0)
    parser.add_argument("--ylim-mev", type=str, default=None, help="Comma-separated lower,upper y limits in meV.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _complex_from_pairs(values: np.ndarray) -> np.ndarray:
    pairs = np.asarray(values, dtype=float)
    if pairs.shape[-1] != 2:
        raise ValueError(f"Expected final axis of length 2 for complex pairs, got {pairs.shape}")
    return np.asarray(pairs[..., 0] + 1j * pairs[..., 1], dtype=np.complex128)


def _panel_values(panel_dir: Path) -> tuple[int, float]:
    match = PANEL_RE.match(panel_dir.name)
    if match is None:
        raise ValueError(f"Cannot parse panel directory name {panel_dir.name!r}")
    return int(match.group("xi")), float(match.group("v_mev"))


def _build_interaction(config: dict[str, object]) -> RLGhBNInteractionParams:
    return RLGhBNInteractionParams(
        epsilon_r=float(config["epsilon_r"]),
        gate_distance_nm=float(config["gate_distance_nm"]),
        scheme=str(config["scheme"]),
        active_valence_bands=int(config["active_valence_bands"]),
        active_conduction_bands=int(config["active_conduction_bands"]),
        k_mesh_size=int(config["k_mesh_size"]),
        interaction_cutoff_q1=float(config["interaction_cutoff_q1"]),
        use_screened_basis=True,
    )


def _screened_u_from_convergence(convergence: dict[str, object], fallback_v_mev: float) -> float:
    screening = convergence.get("screening")
    if isinstance(screening, dict) and screening.get("screened_u_mev") is not None:
        return float(screening["screened_u_mev"])
    return float(fallback_v_mev)


def _reconstruct_run(panel_dir: Path, config: dict[str, object]) -> RLGhBNHartreeFockRun:
    xi, v_mev = _panel_values(panel_dir)
    state_path = panel_dir / "hf_ground_state.npz"
    convergence_path = panel_dir / "hf_convergence.json"
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    if not convergence_path.exists():
        raise FileNotFoundError(convergence_path)

    convergence = _read_json(convergence_path)
    screened_u_mev = _screened_u_from_convergence(convergence, v_mev)
    physical_model = RLGhBNModel.from_config(
        layer_count=int(config["layer_count"]),
        xi=int(xi),
        theta_deg=float(config["theta_deg"]),
        displacement_field_mev=float(v_mev),
        shell_count=int(config["shell_count"]),
    )
    basis_model = RLGhBNModel.from_config(
        layer_count=int(config["layer_count"]),
        xi=int(xi),
        theta_deg=float(config["theta_deg"]),
        displacement_field_mev=float(screened_u_mev),
        shell_count=int(config["shell_count"]),
    )
    interaction = _build_interaction(config)
    archive = np.load(state_path)
    basis_data = build_rlg_hbn_projected_basis_for_kvec(
        basis_model,
        interaction,
        _complex_from_pairs(archive["kvec_nm_inv"]),
        physical_model=physical_model,
        active_band_indices=tuple(int(value) for value in np.asarray(archive["active_band_indices"], dtype=int)),
    )
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=float(config["nu"]),
        precision=float(config.get("precision", 1.0e-6)),
    )
    state.density[:, :, :] = np.asarray(archive["density"], dtype=np.complex128)
    state.hamiltonian[:, :, :] = np.asarray(archive["hamiltonian"], dtype=np.complex128)
    state.h0[:, :, :] = np.asarray(archive["h0"], dtype=np.complex128)
    state.energies[:, :] = np.asarray(archive["energies_mev"], dtype=float)
    overlap_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data)
    best = convergence.get("best", {})
    if not isinstance(best, dict):
        best = {}
    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=np.asarray(archive["iter_energy_mev"], dtype=float),
        iter_err=np.asarray(archive["iter_err"], dtype=float),
        iter_oda=np.asarray(archive["iter_oda"], dtype=float),
        init_mode=str(best.get("init_mode", "loaded")),
        seed=int(best.get("seed", 0)),
        converged=bool(best.get("converged", False)),
        exit_reason=str(best.get("exit_reason", "loaded")),
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
    )


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


def _sector_energies(path_hamiltonian: np.ndarray, *, n_spin: int, n_eta: int, n_band: int, spin: int, eta: int) -> np.ndarray:
    hamiltonian = np.asarray(path_hamiltonian, dtype=np.complex128)
    idx = np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )
    block_indices = np.asarray(idx[int(spin), int(eta), :], dtype=int)
    energies = np.zeros((int(n_band), hamiltonian.shape[2]), dtype=float)
    for ik in range(hamiltonian.shape[2]):
        block = hamiltonian[:, :, ik][np.ix_(block_indices, block_indices)]
        energies[:, ik] = np.linalg.eigvalsh(block)
    return energies


def _source_mu_mev(run: RLGhBNHartreeFockRun) -> float:
    total_occupied = rlg_hbn_occupied_state_count(
        run.state.nu,
        run.state.nt,
        run.state.nk,
        active_valence_bands=run.state.active_valence_bands,
        n_spin=run.state.n_spin,
        n_eta=run.state.n_eta,
    )
    values = np.sort(np.asarray(run.state.energies, dtype=float).reshape(-1))
    if total_occupied <= 0 or total_occupied >= values.size:
        return 0.0
    return float(0.5 * (values[total_occupied - 1] + values[total_occupied]))


def _parse_ylim(text: str | None, paper_target: str) -> tuple[float, float]:
    if text is None:
        if paper_target == "fig5":
            return (-110.0, 80.0)
        return (-90.0, 90.0)
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("--ylim-mev must be formatted as lower,upper")
    return float(pieces[0]), float(pieces[1])


def _plot_panel(ax, panel_result: dict[str, object], *, ylim_mev: tuple[float, float], show_ylabel: bool) -> None:
    path = panel_result["path"]
    assert hasattr(path, "kdist")
    kdist = np.asarray(path.kdist, dtype=float)
    mu = float(panel_result["mu_mev"])
    k_energies = np.asarray(panel_result["k_energies_mev"], dtype=float) - mu
    kp_energies = np.asarray(panel_result["kprime_energies_mev"], dtype=float) - mu
    for iband in range(k_energies.shape[0]):
        ax.plot(kdist, k_energies[iband], color="black", linewidth=0.9)
        ax.plot(kdist, kp_energies[iband], color="#c62828", linewidth=0.9)

    node_indices = np.asarray(path.node_indices, dtype=int) - 1
    node_positions = np.asarray(path.kdist, dtype=float)[node_indices]
    for xpos in node_positions:
        ax.axvline(float(xpos), color="0.78", linewidth=0.6)
    ax.axhline(0.0, color="0.35", linewidth=0.55, linestyle="--")
    ax.set_xticks(node_positions)
    ax.set_xticklabels(path.labels, fontsize=8)
    ax.set_xlim(float(kdist[0]), float(kdist[-1]))
    ax.set_ylim(*ylim_mev)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(str(panel_result["title"]), fontsize=10)
    if show_ylabel:
        ax.set_ylabel("$E-E_F$ (meV)", fontsize=9)
    else:
        ax.set_yticklabels([])


def _write_panel_outputs(panel_dir: Path, panel_result: dict[str, object], *, dpi: int, ylim_mev: tuple[float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = panel_result["path"]
    np.savez_compressed(
        panel_dir / "hf_bands_path.npz",
        kdist=np.asarray(path.kdist, dtype=float),
        kvec_nm_inv=np.stack([np.asarray(path.kvec).real, np.asarray(path.kvec).imag], axis=-1),
        all_energies_mev=np.asarray(panel_result["all_energies_mev"], dtype=float),
        spin_up_K_energies_mev=np.asarray(panel_result["k_energies_mev"], dtype=float),
        spin_up_Kprime_energies_mev=np.asarray(panel_result["kprime_energies_mev"], dtype=float),
        energy_zero_mev=float(panel_result["mu_mev"]),
    )
    write_json(
        panel_dir / "hf_bands_path_summary.json",
        {
            "panel": panel_dir.name,
            "title": str(panel_result["title"]),
            "path_labels": list(path.labels),
            "points": int(np.asarray(path.kvec).size),
            "energy_zero_mev": float(panel_result["mu_mev"]),
            "output_png": str(panel_dir / "hf_bands_path.png"),
            "output_pdf": str(panel_dir / "hf_bands_path.pdf"),
        },
    )

    fig, ax = plt.subplots(figsize=(3.3, 3.1), constrained_layout=True)
    _plot_panel(ax, panel_result, ylim_mev=ylim_mev, show_ylabel=True)
    fig.savefig(panel_dir / "hf_bands_path.png", dpi=int(dpi))
    fig.savefig(panel_dir / "hf_bands_path.pdf")
    plt.close(fig)


def _write_combined_figure(
    output_dir: Path,
    paper_target: str,
    panel_results: list[dict[str, object]],
    *,
    dpi: int,
    ylim_mev: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(panel_results)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(3.2 * ncols, 3.4), 3.2),
        sharey=True,
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]
    for idx, (ax, result) in enumerate(zip(axes, panel_results, strict=True)):
        _plot_panel(ax, result, ylim_mev=ylim_mev, show_ylabel=idx == 0)
    axes[0].plot([], [], color="black", linewidth=1.0, label="$K$")
    axes[0].plot([], [], color="#c62828", linewidth=1.0, label="$K'$")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(output_dir / f"paper_{paper_target}_hf_bands.png", dpi=int(dpi))
    fig.savefig(output_dir / f"paper_{paper_target}_hf_bands.pdf")
    plt.close(fig)


def main() -> None:
    start = perf_counter()
    args = _parse_args()
    source_dir = Path(args.source_dir).resolve()
    config_path = source_dir / "paper_hf_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = _read_json(config_path)
    paper_target = str(args.paper_target or config.get("paper_target") or "fig5")
    if paper_target not in PAPER_CONFIGS:
        raise ValueError(f"Unsupported paper target {paper_target!r}")
    if not args.dry_run:
        ensure_not_running_compute_on_login_node(f"RLG/hBN {paper_target} HF band plotting")

    panel_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir() and (path / "hf_ground_state.npz").exists())
    if not panel_dirs:
        raise FileNotFoundError(f"No panel hf_ground_state.npz files found under {source_dir}")
    ylim_mev = _parse_ylim(args.ylim_mev, paper_target)

    write_json(
        source_dir / "hf_band_plot_config.json",
        {
            "source_dir": str(source_dir),
            "paper_target": paper_target,
            "points_per_segment": int(args.points_per_segment),
            "chunk_size": int(args.chunk_size),
            "spin_index": int(args.spin_index),
            "ylim_mev": list(ylim_mev),
            "hostname": socket.gethostname(),
        },
    )
    if args.dry_run:
        print(f"[dry-run] source_dir={source_dir}")
        print(f"[dry-run] panels={[path.name for path in panel_dirs]}")
        return

    panel_results: list[dict[str, object]] = []
    for panel_dir in panel_dirs:
        panel_start = perf_counter()
        xi, v_mev = _panel_values(panel_dir)
        print(f"[panel] plot start {panel_dir.name}", flush=True)
        run = _reconstruct_run(panel_dir, config)
        path = _paper_hf_path(run.basis_data.basis_model, int(args.points_per_segment))
        path_result = evaluate_rlg_hbn_hf_path(
            run,
            path,
            beta=float(config.get("beta", 1.0)),
            chunk_size=int(args.chunk_size),
        )
        k_energies = _sector_energies(
            path_result.hamiltonian,
            n_spin=run.state.n_spin,
            n_eta=run.state.n_eta,
            n_band=run.state.n_band,
            spin=int(args.spin_index),
            eta=0,
        )
        kprime_energies = _sector_energies(
            path_result.hamiltonian,
            n_spin=run.state.n_spin,
            n_eta=run.state.n_eta,
            n_band=run.state.n_band,
            spin=int(args.spin_index),
            eta=1,
        )
        panel_result = {
            "panel": panel_dir.name,
            "title": f"$\\xi={xi}$, $V={v_mev:.0f}$ meV",
            "path": path_result.path,
            "all_energies_mev": path_result.energies,
            "k_energies_mev": k_energies,
            "kprime_energies_mev": kprime_energies,
            "mu_mev": _source_mu_mev(run),
            "elapsed_sec": float(perf_counter() - panel_start),
        }
        _write_panel_outputs(panel_dir, panel_result, dpi=int(args.dpi), ylim_mev=ylim_mev)
        panel_results.append(panel_result)
        print(f"[panel] plot done {panel_dir.name} elapsed_sec={panel_result['elapsed_sec']:.3f}", flush=True)

    _write_combined_figure(source_dir, paper_target, panel_results, dpi=int(args.dpi), ylim_mev=ylim_mev)
    elapsed = perf_counter() - start
    write_json(
        source_dir / "hf_band_plot_summary.json",
        {
            "source_dir": str(source_dir),
            "paper_target": paper_target,
            "elapsed_sec": float(elapsed),
            "combined_png": str(source_dir / f"paper_{paper_target}_hf_bands.png"),
            "combined_pdf": str(source_dir / f"paper_{paper_target}_hf_bands.pdf"),
            "panels": [
                {
                    "panel": str(result["panel"]),
                    "elapsed_sec": float(result["elapsed_sec"]),
                    "energy_zero_mev": float(result["mu_mev"]),
                }
                for result in panel_results
            ],
        },
    )
    print(f"[done] combined_png={source_dir / f'paper_{paper_target}_hf_bands.png'}", flush=True)


if __name__ == "__main__":
    main()
