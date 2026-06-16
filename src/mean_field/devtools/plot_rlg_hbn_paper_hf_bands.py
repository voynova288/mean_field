from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import socket
from time import perf_counter

import numpy as np

from mean_field.api.artifacts import update_artifact_manifest
from mean_field.core.lattice import KPath
from mean_field.devtools._runtime import (
    complex_from_pairs as _complex_from_pairs,
    ensure_not_running_compute_on_login_node,
    write_json,
)
from mean_field.devtools.run_rlg_hbn_paper_hf import PAPER_CONFIGS
from mean_field.systems.RnG_hBN import (
    RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNInteractionParams,
    RLGhBNModel,
    build_kpath_from_nodes,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis_for_kvec,
    evaluate_rlg_hbn_hf_path,
    load_layer_overlap_blocks_cache,
    load_path_band_cache,
    load_projected_basis_cache,
    path_cache_key,
    rlg_hbn_occupied_state_count,
    save_path_band_cache,
    update_cache_manifest_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _update_band_plot_manifest(
    source_dir: Path,
    *,
    paper_target: str,
    panel_names: list[str],
    status: str,
) -> Path:
    files: dict[str, object] = {
        "hf_band_plot_config": "hf_band_plot_config.json",
        "hf_band_plot_panels": panel_names,
    }
    if status != "dry_run":
        files.update(
            {
                "hf_band_plot_summary": "hf_band_plot_summary.json",
                "hf_band_plot_combined_png": f"paper_{paper_target}_hf_bands.png",
                "hf_band_plot_combined_pdf": f"paper_{paper_target}_hf_bands.pdf",
            }
        )
    return update_artifact_manifest(
        source_dir,
        files=files,
        metadata={
            "band_plot": {
                "workflow": "rlg_hbn.paper_hf_bands",
                "status": str(status),
                "paper_target": str(paper_target),
                "panel_count": int(len(panel_names)),
            }
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-style R5G/hBN HF band structures from saved paper HF source states."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--paper-target", choices=tuple(PAPER_CONFIGS), default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-policy", choices=("reuse", "refresh", "off"), default="reuse")
    parser.add_argument("--points-per-segment", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--spin-index", type=int, default=0)
    parser.add_argument(
        "--spectrum-mode",
        choices=("auto", "sector", "full", "both"),
        default="auto",
        help=(
            "Plot 'sector' K/K' spin blocks, 'full' Hamiltonian eigenvalues, "
            "'both', or 'auto' sector only when flavor off-block terms are negligible."
        ),
    )
    parser.add_argument("--block-offdiag-rtol", type=float, default=1.0e-8)
    parser.add_argument("--ylim-mev", type=str, default=None, help="Comma-separated lower,upper y limits in meV.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()



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
        use_screened_basis=bool(config.get("use_screened_basis", True)),
    )


def _screened_u_from_convergence(convergence: dict[str, object], fallback_v_mev: float) -> float:
    screening = convergence.get("screening")
    if isinstance(screening, dict) and screening.get("screened_u_mev") is not None:
        return float(screening["screened_u_mev"])
    return float(fallback_v_mev)


def _string_from_archive(archive: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in archive.files:
        return ""
    value = archive[key]
    try:
        return str(value.item())
    except Exception:
        return str(value)


def _reconstruct_run(
    panel_dir: Path,
    config: dict[str, object],
    *,
    cache_dir: Path | None = None,
    cache_policy: str = "reuse",
) -> RLGhBNHartreeFockRun:
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
    archive_gauge = _string_from_archive(archive, "basis_periodic_gauge")
    if archive_gauge != RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION:
        raise ValueError(
            f"{state_path} was generated with basis_periodic_gauge={archive_gauge!r}; "
            f"expected {RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION!r}. Rerun the HF source state "
            "before plotting bands with the periodic-gauge corrected basis."
        )
    basis_data = None
    overlap_blocks = None
    if cache_dir is not None and cache_policy != "off":
        basis_key = str(convergence.get("basis_cache_key") or _string_from_archive(archive, "cache_key_basis"))
        overlap_key = str(convergence.get("overlap_cache_key") or _string_from_archive(archive, "cache_key_overlap"))
        if basis_key:
            try:
                basis_data = load_projected_basis_cache(cache_dir, basis_key)
                print(f"[cache-hit] source basis {basis_key}", flush=True)
            except Exception as exc:
                print(f"[cache-miss] source basis {basis_key}: {exc}", flush=True)
                basis_data = None
        if overlap_key:
            try:
                overlap_blocks = load_layer_overlap_blocks_cache(cache_dir, overlap_key)
                print(f"[cache-hit] source overlap {overlap_key}", flush=True)
            except Exception as exc:
                print(f"[cache-miss] source overlap {overlap_key}: {exc}", flush=True)
                overlap_blocks = None
    if basis_data is None:
        basis_data = build_rlg_hbn_projected_basis_for_kvec(
            basis_model,
            interaction,
            _complex_from_pairs(archive["kvec_nm_inv"]),
            physical_model=physical_model,
            active_band_indices=tuple(int(value) for value in np.asarray(archive["active_band_indices"], dtype=int)),
        )
        print("[cache-miss] source basis fallback rebuilt from archive k grid", flush=True)
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=float(config["nu"]),
        precision=float(config.get("precision", 1.0e-6)),
    )
    state.density[:, :, :] = np.asarray(archive["density"], dtype=np.complex128)
    state.hamiltonian[:, :, :] = np.asarray(archive["hamiltonian"], dtype=np.complex128)
    state.h0[:, :, :] = np.asarray(archive["h0"], dtype=np.complex128)
    state.energies[:, :] = np.asarray(archive["energies_mev"], dtype=float)
    if overlap_blocks is None:
        overlap_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data)
        print("[cache-miss] source overlap fallback rebuilt from source basis", flush=True)
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
    g2 = lattice.g_m2
    # Keep the Fig. 6 M' representative explicit.  The projected-basis builder
    # folds each target k for diagonalization and then relabels plane-wave
    # coefficients back into this raw path convention before form factors are
    # evaluated.
    mprime_fig6 = g2 / 2.0
    m_fig6 = lattice.m_m
    nodes = (
        lattice.gamma_m,
        lattice.k_m,
        lattice.kprime_m,
        lattice.gamma_m,
        mprime_fig6,
        m_fig6,
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


def _full_energies(path_hamiltonian: np.ndarray) -> np.ndarray:
    hamiltonian = np.asarray(path_hamiltonian, dtype=np.complex128)
    energies = np.zeros((hamiltonian.shape[0], hamiltonian.shape[2]), dtype=float)
    for ik in range(hamiltonian.shape[2]):
        energies[:, ik] = np.linalg.eigvalsh(hamiltonian[:, :, ik])
    return energies


def _flavor_block_offdiag_metrics(
    path_hamiltonian: np.ndarray,
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
) -> dict[str, float]:
    hamiltonian = np.asarray(path_hamiltonian, dtype=np.complex128)
    idx = np.arange(int(n_spin) * int(n_eta) * int(n_band), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(n_band)),
        order="F",
    )
    blocks = [
        np.asarray(idx[ispin, ieta, :], dtype=int)
        for ispin in range(int(n_spin))
        for ieta in range(int(n_eta))
    ]
    max_diag = 0.0
    max_offdiag = 0.0
    for ik in range(hamiltonian.shape[2]):
        for left_idx, left in enumerate(blocks):
            diag_block = hamiltonian[:, :, ik][np.ix_(left, left)]
            max_diag = max(max_diag, float(np.max(np.abs(diag_block))))
            for right_idx, right in enumerate(blocks):
                if left_idx == right_idx:
                    continue
                off_block = hamiltonian[:, :, ik][np.ix_(left, right)]
                max_offdiag = max(max_offdiag, float(np.max(np.abs(off_block))))
    denom = max(max_diag, 1.0e-30)
    return {
        "flavor_block_diag_max_abs_mev": float(max_diag),
        "flavor_block_offdiag_max_abs_mev": float(max_offdiag),
        "flavor_block_offdiag_relative": float(max_offdiag / denom),
    }


def _resolved_spectrum_mode(requested: str, metrics: dict[str, float], *, rtol: float) -> str:
    if requested != "auto":
        return str(requested)
    return "sector" if float(metrics["flavor_block_offdiag_relative"]) <= float(rtol) else "full"


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


def _source_mu_from_archive(path: Path, config: dict[str, object]) -> float:
    archive = np.load(path)
    energies = np.asarray(archive["energies_mev"], dtype=float)
    total_occupied = rlg_hbn_occupied_state_count(
        float(config["nu"]),
        energies.shape[0],
        energies.shape[1],
        active_valence_bands=int(config["active_valence_bands"]),
        n_spin=2,
        n_eta=2,
    )
    values = np.sort(energies.reshape(-1))
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


def _path_cache_payload(path, *, points_per_segment: int) -> dict[str, object]:
    return {
        "labels": list(path.labels),
        "node_indices": [int(value) for value in path.node_indices],
        "points_per_segment": int(points_per_segment),
        "kvec_nm_inv": [
            [float(complex(value).real), float(complex(value).imag)]
            for value in np.asarray(path.kvec, dtype=np.complex128)
        ],
    }


def _panel_bands_payload(panel_result: dict[str, object]) -> dict[str, object]:
    path = panel_result["path"]
    return {
        "kdist": np.asarray(path.kdist, dtype=float),
        "kvec_nm_inv": np.stack([np.asarray(path.kvec).real, np.asarray(path.kvec).imag], axis=-1),
        "all_energies_mev": np.asarray(panel_result["all_energies_mev"], dtype=float),
        "spin_up_K_energies_mev": np.asarray(panel_result["k_energies_mev"], dtype=float),
        "spin_up_Kprime_energies_mev": np.asarray(panel_result["kprime_energies_mev"], dtype=float),
        "full_energies_mev": np.asarray(panel_result["full_energies_mev"], dtype=float),
        "energy_zero_mev": np.asarray(float(panel_result["mu_mev"])),
    }


def _plot_panel(ax, panel_result: dict[str, object], *, ylim_mev: tuple[float, float], show_ylabel: bool) -> None:
    path = panel_result["path"]
    assert hasattr(path, "kdist")
    kdist = np.asarray(path.kdist, dtype=float)
    mu = float(panel_result["mu_mev"])
    mode = str(panel_result.get("resolved_spectrum_mode", "sector"))
    if mode in ("full", "both"):
        full_energies = np.asarray(panel_result["full_energies_mev"], dtype=float) - mu
        for iband in range(full_energies.shape[0]):
            ax.plot(kdist, full_energies[iband], color="0.25", linewidth=0.62, alpha=0.72)
    k_energies = np.asarray(panel_result["k_energies_mev"], dtype=float) - mu
    kp_energies = np.asarray(panel_result["kprime_energies_mev"], dtype=float) - mu
    if mode in ("sector", "both"):
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
    np.savez_compressed(panel_dir / "hf_bands_path.npz", **_panel_bands_payload(panel_result))
    write_json(
        panel_dir / "hf_bands_path_summary.json",
        {
            "panel": panel_dir.name,
            "title": str(panel_result["title"]),
            "path_labels": list(path.labels),
            "points": int(np.asarray(path.kvec).size),
            "energy_zero_mev": float(panel_result["mu_mev"]),
            "requested_spectrum_mode": str(panel_result["requested_spectrum_mode"]),
            "resolved_spectrum_mode": str(panel_result["resolved_spectrum_mode"]),
            "flavor_block_offdiag": panel_result["flavor_block_offdiag"],
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
    modes = {str(result.get("resolved_spectrum_mode", "sector")) for result in panel_results}
    if modes & {"sector", "both"}:
        axes[0].plot([], [], color="black", linewidth=1.0, label="$K$")
        axes[0].plot([], [], color="#c62828", linewidth=1.0, label="$K'$")
    if modes & {"full", "both"}:
        axes[0].plot([], [], color="0.25", linewidth=0.8, label="full eig.")
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
    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir is not None
        else Path(str(config.get("cache_dir", source_dir / "cache"))).resolve()
    )

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
            "spectrum_mode": str(args.spectrum_mode),
            "block_offdiag_rtol": float(args.block_offdiag_rtol),
            "ylim_mev": list(ylim_mev),
            "hostname": socket.gethostname(),
            "cache_dir": str(cache_dir),
            "cache_policy": str(args.cache_policy),
        },
    )
    if args.dry_run:
        _update_band_plot_manifest(
            source_dir,
            paper_target=paper_target,
            panel_names=[path.name for path in panel_dirs],
            status="dry_run",
        )
        print(f"[dry-run] source_dir={source_dir}")
        print(f"[dry-run] panels={[path.name for path in panel_dirs]}")
        return

    panel_results: list[dict[str, object]] = []
    for panel_dir in panel_dirs:
        panel_start = perf_counter()
        xi, v_mev = _panel_values(panel_dir)
        print(f"[panel] plot start {panel_dir.name}", flush=True)
        source_archive = panel_dir / "hf_ground_state.npz"
        convergence = _read_json(panel_dir / "hf_convergence.json")
        screened_u_mev = _screened_u_from_convergence(convergence, v_mev)
        physical_model_for_key = RLGhBNModel.from_config(
            layer_count=int(config["layer_count"]),
            xi=int(xi),
            theta_deg=float(config["theta_deg"]),
            displacement_field_mev=float(v_mev),
            shell_count=int(config["shell_count"]),
        )
        basis_model_for_key = RLGhBNModel.from_config(
            layer_count=int(config["layer_count"]),
            xi=int(xi),
            theta_deg=float(config["theta_deg"]),
            displacement_field_mev=float(screened_u_mev),
            shell_count=int(config["shell_count"]),
        )
        interaction_for_key = _build_interaction(config)
        path = _paper_hf_path(basis_model_for_key, int(args.points_per_segment))
        path_payload = _path_cache_payload(path, points_per_segment=int(args.points_per_segment))
        cache_key, cache_manifest = path_cache_key(
            physical_model_for_key,
            interaction_for_key,
            source_archive=source_archive,
            path_payload=path_payload,
            chunk_size=int(args.chunk_size),
            beta=float(config.get("beta", 1.0)),
            spin_index=int(args.spin_index),
            panel=panel_dir.name,
        )
        path_cache_hit = False
        cached_path = None
        if args.cache_policy == "reuse":
            try:
                cached_path = load_path_band_cache(cache_dir, cache_key)
                path_cache_hit = True
                print(f"[cache-hit] path_bands {cache_key}", flush=True)
            except Exception as exc:
                print(f"[cache-miss] path_bands {cache_key}: {exc}", flush=True)
        if cached_path is not None:
            labels_payload = cached_path["labels"]
            path = KPath(
                kvec=np.asarray(cached_path["kvec"], dtype=np.complex128),
                kdist=np.asarray(cached_path["kdist"], dtype=float),
                labels=tuple(str(value) for value in labels_payload["labels"]),
                node_indices=tuple(int(value) for value in labels_payload["node_indices"]),
            )
            path_hamiltonian = np.asarray(cached_path["hamiltonian"], dtype=np.complex128)
            path_energies = np.asarray(cached_path["energies"], dtype=float)
            mu_mev = _source_mu_from_archive(source_archive, config)
            n_spin = 2
            n_eta = 2
            n_band = int(path_hamiltonian.shape[0]) // (n_spin * n_eta)
        else:
            run = _reconstruct_run(panel_dir, config, cache_dir=cache_dir, cache_policy=str(args.cache_policy))
            path_result = evaluate_rlg_hbn_hf_path(
                run,
                path,
                beta=float(config.get("beta", 1.0)),
                chunk_size=int(args.chunk_size),
            )
            path_hamiltonian = path_result.hamiltonian
            path_energies = path_result.energies
            mu_mev = _source_mu_mev(run)
            n_spin = run.state.n_spin
            n_eta = run.state.n_eta
            n_band = run.state.n_band
        k_energies = _sector_energies(
            path_hamiltonian,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
            spin=int(args.spin_index),
            eta=0,
        )
        kprime_energies = _sector_energies(
            path_hamiltonian,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
            spin=int(args.spin_index),
            eta=1,
        )
        full_energies = _full_energies(path_hamiltonian)
        offdiag_metrics = _flavor_block_offdiag_metrics(
            path_hamiltonian,
            n_spin=n_spin,
            n_eta=n_eta,
            n_band=n_band,
        )
        resolved_mode = _resolved_spectrum_mode(
            str(args.spectrum_mode),
            offdiag_metrics,
            rtol=float(args.block_offdiag_rtol),
        )
        if resolved_mode == "full":
            print(
                "[warn] flavor/spin off-block terms are non-negligible; "
                f"plotting full eigenvalues for {panel_dir.name}. "
                f"offdiag_rel={offdiag_metrics['flavor_block_offdiag_relative']:.3e}",
                flush=True,
            )
        panel_result = {
            "panel": panel_dir.name,
            "title": f"$\\xi={xi}$, $V={v_mev:.0f}$ meV",
            "path": path,
            "all_energies_mev": path_energies,
            "k_energies_mev": k_energies,
            "kprime_energies_mev": kprime_energies,
            "full_energies_mev": full_energies,
            "mu_mev": mu_mev,
            "elapsed_sec": float(perf_counter() - panel_start),
            "path_cache_key": cache_key,
            "path_cache_hit": bool(path_cache_hit),
            "requested_spectrum_mode": str(args.spectrum_mode),
            "resolved_spectrum_mode": resolved_mode,
            "flavor_block_offdiag": offdiag_metrics,
        }
        if cached_path is None and args.cache_policy != "off":
            save_path_band_cache(
                cache_dir,
                cache_key,
                cache_manifest,
                path_hamiltonian=path_hamiltonian,
                path_energies=path_energies,
                path_kvec=np.asarray(path.kvec, dtype=np.complex128),
                kdist=np.asarray(path.kdist, dtype=float),
                labels_payload={
                    "labels": list(path.labels),
                    "node_indices": [int(value) for value in path.node_indices],
                },
                hf_bands_payload=_panel_bands_payload(panel_result),
            )
            print(f"[cache-miss] path_bands saved {cache_key}", flush=True)
        update_cache_manifest_file(
            source_dir / "cache_manifest.json",
            cache_dir=cache_dir,
            kind="path_bands",
            key=cache_key,
            hit=path_cache_hit,
            path=None if args.cache_policy == "off" else cache_dir / "path_bands" / cache_key,
            panel=panel_dir.name,
        )
        _write_panel_outputs(panel_dir, panel_result, dpi=int(args.dpi), ylim_mev=ylim_mev)
        panel_results.append(panel_result)
        print(f"[panel] plot done {panel_dir.name} elapsed_sec={panel_result['elapsed_sec']:.3f}", flush=True)

    _write_combined_figure(source_dir, paper_target, panel_results, dpi=int(args.dpi), ylim_mev=ylim_mev)
    elapsed = perf_counter() - start
    summary_payload = {
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
                "path_cache_key": str(result["path_cache_key"]),
                "path_cache_hit": bool(result["path_cache_hit"]),
                "requested_spectrum_mode": str(result["requested_spectrum_mode"]),
                "resolved_spectrum_mode": str(result["resolved_spectrum_mode"]),
                "flavor_block_offdiag": result["flavor_block_offdiag"],
            }
            for result in panel_results
        ],
    }
    write_json(source_dir / "hf_band_plot_summary.json", summary_payload)
    _update_band_plot_manifest(
        source_dir,
        paper_target=paper_target,
        panel_names=[str(result["panel"]) for result in panel_results],
        status="complete",
    )
    print(f"[done] combined_png={source_dir / f'paper_{paper_target}_hf_bands.png'}", flush=True)


if __name__ == "__main__":
    main()
