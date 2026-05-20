#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import socket
from time import perf_counter

import numpy as np

from mean_field.core.lattice import KPath
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node
from mean_field.devtools.plot_rlg_hbn_paper_hf_bands import (
    _build_interaction,
    _paper_hf_path,
    _parse_ylim,
    _plot_panel,
    _sector_energies,
)
from mean_field.systems.RnG_hBN import (
    RLGhBNModel,
    build_rlg_hbn_projected_basis_for_kvec,
    build_rlg_hbn_remote_average_hamiltonian,
    load_or_build_projected_basis,
    rlg_hbn_occupied_state_count,
    screening_result_from_dict,
    screening_result_to_dict,
)


PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


@dataclass(frozen=True)
class PanelSource:
    panel: str
    panel_dir: Path
    task_dir: Path
    config_path: Path
    screening_path: Path | None


@dataclass(frozen=True)
class ExactMeshPath:
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]
    kvec: np.ndarray
    kdist: np.ndarray
    mesh_indices: np.ndarray


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_panel(panel: str) -> tuple[int, float]:
    match = PANEL_RE.match(panel)
    if match is None:
        raise ValueError(f"Cannot parse panel name {panel!r}")
    return int(match.group("xi")), float(match.group("v_mev"))


def _discover_panel_sources(output_root: Path, requested_panels: set[str] | None) -> list[PanelSource]:
    tasks_root = output_root / "tasks"
    if not tasks_root.exists():
        raise FileNotFoundError(tasks_root)
    by_panel: dict[str, PanelSource] = {}
    for panel_dir in sorted(tasks_root.glob("task_*/*_V*meV")):
        if not panel_dir.is_dir() or PANEL_RE.match(panel_dir.name) is None:
            continue
        panel = panel_dir.name
        if requested_panels is not None and panel not in requested_panels:
            continue
        task_dir = panel_dir.parent
        config_path = task_dir / "paper_hf_config.json"
        if not config_path.exists():
            continue
        screening_path = panel_dir / "screening_result.json"
        source = PanelSource(
            panel=panel,
            panel_dir=panel_dir,
            task_dir=task_dir,
            config_path=config_path,
            screening_path=screening_path if screening_path.exists() else None,
        )
        by_panel.setdefault(panel, source)
    missing = sorted(requested_panels - set(by_panel)) if requested_panels is not None else []
    if missing:
        raise FileNotFoundError(f"No panel sources found for: {missing}")
    if not by_panel:
        raise FileNotFoundError(f"No panel sources found under {tasks_root}")
    return [by_panel[key] for key in sorted(by_panel)]


def _diagonalize_blocks(hamiltonian: np.ndarray) -> np.ndarray:
    values = np.asarray(hamiltonian, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected Hamiltonian shape (nt, nt, nk), got {values.shape}")
    out = np.zeros((values.shape[0], values.shape[2]), dtype=float)
    for ik in range(values.shape[2]):
        out[:, ik] = np.linalg.eigvalsh(values[:, :, ik])
    return out


def _path_nodes() -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    gamma = (0, 0)
    k_point = (4, 2)
    # Canonical primitive-cell representatives for the finite-G basis:
    # K=(2/3,1/3), K'=(1/3,2/3).  The neighboring-zone representative
    # (-1/3,1/3) is equivalent only after a reciprocal-gauge relabel; modulo
    # mesh lookup alone maps it back onto K.
    kprime_fig6 = (2, 4)
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


def _grid_lookup(frac_grid: np.ndarray, mesh_size: int) -> dict[tuple[int, int], int]:
    frac = np.asarray(frac_grid, dtype=float).reshape((-1, 2))
    lookup: dict[tuple[int, int], int] = {}
    for idx, value in enumerate(frac):
        ij = tuple(int(round(float(coord) * int(mesh_size))) % int(mesh_size) for coord in value)
        lookup[ij] = int(idx)
    if len(lookup) != frac.shape[0]:
        raise ValueError(f"Fractional grid lookup has duplicates: {len(lookup)} unique for {frac.shape[0]} points")
    return lookup


def _build_exact_mesh_path(source_basis_data) -> ExactMeshPath:
    mesh_size = int(source_basis_data.mesh_size)
    if mesh_size <= 0:
        raise ValueError(f"Source basis has invalid mesh_size={mesh_size}")
    node_sixths, labels = _path_nodes()
    lookup = _grid_lookup(source_basis_data.k_grid_frac, mesh_size)
    node_grid: list[tuple[int, int]] = []
    for frac6 in node_sixths:
        raw = (int(frac6[0]) * mesh_size, int(frac6[1]) * mesh_size)
        if raw[0] % 6 != 0 or raw[1] % 6 != 0:
            raise ValueError(f"mesh_size={mesh_size} does not hit high-symmetry node {frac6}/6 exactly")
        node_grid.append((raw[0] // 6, raw[1] // 6))

    grid_points: list[tuple[int, int]] = [node_grid[0]]
    node_indices: list[int] = [1]
    for start, end in zip(node_grid[:-1], node_grid[1:], strict=True):
        delta = (int(end[0]) - int(start[0]), int(end[1]) - int(start[1]))
        steps = math.gcd(abs(delta[0]), abs(delta[1]))
        if steps == 0:
            continue
        step_delta = (delta[0] // steps, delta[1] // steps)
        for step in range(1, steps + 1):
            grid_points.append((int(start[0]) + step * step_delta[0], int(start[1]) + step * step_delta[1]))
        node_indices.append(len(grid_points))

    mesh_indices: list[int] = []
    kvec: list[complex] = []
    g1 = source_basis_data.model.lattice.g_m1
    g2 = source_basis_data.model.lattice.g_m2
    for ij in grid_points:
        key = (int(ij[0]) % mesh_size, int(ij[1]) % mesh_size)
        if key not in lookup:
            raise KeyError(f"Grid point {key} is absent from source mesh")
        mesh_indices.append(lookup[key])
        extended_frac = np.asarray([int(ij[0]) / float(mesh_size), int(ij[1]) / float(mesh_size)], dtype=float)
        kvec.append(complex(extended_frac[0] * complex(g1) + extended_frac[1] * complex(g2)))
    kvec_array = np.asarray(kvec, dtype=np.complex128)
    kdist = np.zeros(kvec_array.size, dtype=float)
    if kvec_array.size > 1:
        kdist[1:] = np.cumsum(np.abs(kvec_array[1:] - kvec_array[:-1]))
    return ExactMeshPath(
        labels=labels,
        node_indices=tuple(int(value) for value in node_indices),
        kvec=kvec_array,
        kdist=kdist,
        mesh_indices=np.asarray(mesh_indices, dtype=int),
    )


def _mu_from_mesh_energies(energies: np.ndarray, config: dict[str, object]) -> float:
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


def _base_path_hamiltonian(
    source_basis_data,
    path,
    *,
    chunk_size: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian_chunks: list[np.ndarray] = []
    energy_chunks: list[np.ndarray] = []
    kvec = np.asarray(path.kvec, dtype=np.complex128)
    for start in range(0, kvec.size, int(chunk_size)):
        stop = min(start + int(chunk_size), kvec.size)
        target_basis_data = build_rlg_hbn_projected_basis_for_kvec(
            source_basis_data.basis_model,
            source_basis_data.interaction,
            kvec[start:stop],
            physical_model=source_basis_data.model,
            active_band_indices=source_basis_data.active_band_indices,
            valleys=source_basis_data.valleys,
        )
        fixed_remote = build_rlg_hbn_remote_average_hamiltonian(
            target_basis_data,
            source_basis_data=source_basis_data,
            beta=float(beta),
        )
        if target_basis_data.physical_h0 is None:
            raise AssertionError("Path basis is missing physical_h0")
        h0 = np.asarray(target_basis_data.physical_h0, dtype=np.complex128) + fixed_remote
        hamiltonian_chunks.append(h0)
        energy_chunks.append(_diagonalize_blocks(h0))
    return np.concatenate(hamiltonian_chunks, axis=2), np.concatenate(energy_chunks, axis=1)


def _save_panel_outputs(panel_dir: Path, panel_result: dict[str, object], *, dpi: int, ylim_mev: tuple[float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = panel_result["path"]
    panel_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_dir / "screened_base_bands_path.npz",
        kdist=np.asarray(path.kdist, dtype=float),
        kvec_nm_inv=np.stack([np.asarray(path.kvec).real, np.asarray(path.kvec).imag], axis=-1),
        all_energies_mev=np.asarray(panel_result["all_energies_mev"], dtype=float),
        spin_up_K_energies_mev=np.asarray(panel_result["k_energies_mev"], dtype=float),
        spin_up_Kprime_energies_mev=np.asarray(panel_result["kprime_energies_mev"], dtype=float),
        energy_zero_mev=np.asarray(float(panel_result["mu_mev"])),
    )
    _write_json(
        panel_dir / "screened_base_bands_path_summary.json",
        {
            "panel": str(panel_result["panel"]),
            "title": str(panel_result["title"]),
            "path_labels": list(path.labels),
            "points": int(np.asarray(path.kvec).size),
            "energy_zero_mev": float(panel_result["mu_mev"]),
            "screened_u_mev": float(panel_result["screened_u_mev"]),
            "basis_cache_key": str(panel_result["basis_cache_key"]),
            "basis_cache_hit": bool(panel_result["basis_cache_hit"]),
            "hamiltonian": "pre-HF base h0 = projected physical H_sp(V) + fixed average-scheme remote-band contribution",
            "output_png": str(panel_dir / "screened_base_bands_path.png"),
            "output_pdf": str(panel_dir / "screened_base_bands_path.pdf"),
        },
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    _plot_panel(ax, panel_result, ylim_mev=ylim_mev, show_ylabel=True)
    ax.set_title(str(panel_result["title"]), fontsize=9)
    fig.savefig(panel_dir / "screened_base_bands_path.png", dpi=int(dpi))
    fig.savefig(panel_dir / "screened_base_bands_path.pdf")
    plt.close(fig)


def _save_combined(output_dir: Path, panel_results: list[dict[str, object]], *, dpi: int, ylim_mev: tuple[float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(panel_results)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(3.4 * ncols, 3.6), 3.35),
        sharey=True,
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]
    for idx, (ax, result) in enumerate(zip(axes, panel_results, strict=True)):
        _plot_panel(ax, result, ylim_mev=ylim_mev, show_ylabel=idx == 0)
        ax.set_title(str(result["title"]), fontsize=9)
    axes[0].plot([], [], color="black", linewidth=1.0, label="$K$")
    axes[0].plot([], [], color="#c62828", linewidth=1.0, label="$K'$")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(output_dir / "screened_base_bands.png", dpi=int(dpi))
    fig.savefig(output_dir / "screened_base_bands.pdf")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pre-HF screened-base RLG/hBN Fig. 6 bands.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--panels", type=str, default="", help="Comma-separated panel names; default is all discovered panels.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-policy", choices=("reuse", "refresh", "off"), default="reuse")
    parser.add_argument("--path-mode", choices=("mesh", "continuous"), default="mesh")
    parser.add_argument("--points-per-segment", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--spin-index", type=int, default=0)
    parser.add_argument("--ylim-mev", type=str, default="-120,120")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    start = perf_counter()
    args = _parse_args()
    ensure_not_running_compute_on_login_node("RLG/hBN screened-base pre-HF band plotting")

    output_root = args.output_root.resolve()
    requested_panels = {value.strip() for value in args.panels.split(",") if value.strip()} or None
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else output_root / "screened_base_band_snapshots" / "current_screened_base"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_sources = _discover_panel_sources(output_root, requested_panels)
    ylim_mev = _parse_ylim(args.ylim_mev, "fig6")

    panel_results: list[dict[str, object]] = []
    for source in panel_sources:
        panel_start = perf_counter()
        config = _read_json(source.config_path)
        xi, v_mev = _parse_panel(source.panel)
        physical_model = RLGhBNModel.from_config(
            layer_count=int(config["layer_count"]),
            xi=int(xi),
            theta_deg=float(config["theta_deg"]),
            displacement_field_mev=float(v_mev),
            shell_count=int(config["shell_count"]),
        )
        interaction = _build_interaction(config)
        cache_dir = (
            args.cache_dir.resolve()
            if args.cache_dir is not None
            else Path(str(config.get("cache_dir", output_root / "cache"))).resolve()
        )
        screening = None
        if source.screening_path is not None:
            screening = screening_result_from_dict(_read_json(source.screening_path))
        basis_cache = load_or_build_projected_basis(
            physical_model,
            interaction,
            cache_dir=cache_dir,
            cache_policy=str(args.cache_policy),
            mesh_size=int(config["k_mesh_size"]),
            screening=screening,
            screening_solver=str(config.get("screening_solver", "grid")),
            screening_mesh_size=int(config.get("screening_mesh_size", config["k_mesh_size"])),
            screening_u_min_mev=float(config.get("screening_u_min_mev", -100.0)),
            screening_u_max_mev=float(config.get("screening_u_max_mev", 200.0)),
            screening_u_grid_points=int(config.get("screening_u_grid_points", 121)),
        )
        source_basis = basis_cache.value
        source_energies = _diagonalize_blocks(source_basis.h0)
        mu_mev = _mu_from_mesh_energies(source_energies, config)
        if args.path_mode == "mesh":
            exact_path = _build_exact_mesh_path(source_basis)
            path = KPath(
                kvec=exact_path.kvec,
                kdist=exact_path.kdist,
                labels=exact_path.labels,
                node_indices=exact_path.node_indices,
            )
            path_hamiltonian = np.asarray(source_basis.h0[:, :, exact_path.mesh_indices], dtype=np.complex128)
            path_energies = _diagonalize_blocks(path_hamiltonian)
        else:
            path = _paper_hf_path(physical_model, int(args.points_per_segment))
            path_hamiltonian, path_energies = _base_path_hamiltonian(
                source_basis,
                path,
                chunk_size=int(args.chunk_size),
                beta=float(config.get("beta", 1.0)),
            )
        n_spin = 2
        n_eta = 2
        n_band = int(path_hamiltonian.shape[0]) // (n_spin * n_eta)
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
        screened_u = float(source_basis.screened_u_mev)
        result = {
            "panel": source.panel,
            "title": f"$\\xi={xi}$, $V={v_mev:.0f}$ meV base, $U={screened_u:.2f}$ meV",
            "path": path,
            "all_energies_mev": path_energies,
            "k_energies_mev": k_energies,
            "kprime_energies_mev": kprime_energies,
            "mu_mev": mu_mev,
            "screened_u_mev": screened_u,
            "basis_cache_key": str(basis_cache.key),
            "basis_cache_hit": bool(basis_cache.hit),
            "elapsed_sec": float(perf_counter() - panel_start),
        }
        panel_results.append(result)
        panel_out = output_dir / source.panel
        _save_panel_outputs(panel_out, result, dpi=int(args.dpi), ylim_mev=ylim_mev)
        print(
            f"[panel] {source.panel} U={screened_u:.6f} mu={mu_mev:.6f} "
            f"basis_hit={basis_cache.hit} elapsed={result['elapsed_sec']:.1f}s",
            flush=True,
        )

    _save_combined(output_dir, panel_results, dpi=int(args.dpi), ylim_mev=ylim_mev)
    _write_json(
        output_dir / "screened_base_bands_summary.json",
        {
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "output_png": str(output_dir / "screened_base_bands.png"),
            "output_pdf": str(output_dir / "screened_base_bands.pdf"),
            "hostname": socket.gethostname(),
            "points_per_segment": int(args.points_per_segment),
            "chunk_size": int(args.chunk_size),
            "spin_index": int(args.spin_index),
            "ylim_mev": [float(ylim_mev[0]), float(ylim_mev[1])],
            "hamiltonian": "pre-HF base h0 = projected physical H_sp(V) + fixed average-scheme remote-band contribution",
            "path_mode": str(args.path_mode),
            "panels": [
                {
                    "panel": str(result["panel"]),
                    "screened_u_mev": float(result["screened_u_mev"]),
                    "energy_zero_mev": float(result["mu_mev"]),
                    "basis_cache_key": str(result["basis_cache_key"]),
                    "basis_cache_hit": bool(result["basis_cache_hit"]),
                    "elapsed_sec": float(result["elapsed_sec"]),
                    "screening": None
                    if panel_sources[idx].screening_path is None
                    else screening_result_to_dict(screening_result_from_dict(_read_json(panel_sources[idx].screening_path))),
                }
                for idx, result in enumerate(panel_results)
            ],
            "elapsed_sec": float(perf_counter() - start),
        },
    )
    print(f"[done] output_png={output_dir / 'screened_base_bands.png'}", flush=True)


if __name__ == "__main__":
    main()
