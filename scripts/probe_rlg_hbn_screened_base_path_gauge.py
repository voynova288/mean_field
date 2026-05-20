#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import socket
from time import perf_counter

import numpy as np

from mean_field.devtools.plot_rlg_hbn_paper_hf_bands import (
    _build_interaction,
    _paper_hf_path,
    _parse_ylim,
    _plot_panel,
    _sector_energies,
)
from mean_field.systems.RnG_hBN import (
    RLGhBNModel,
    active_band_indices_for_interaction,
    build_kpath_from_nodes,
    build_moire_k_grid,
    build_rlg_hbn_projected_basis_for_kvec,
    rlg_hbn_occupied_state_count,
    screening_result_from_dict,
    screening_result_to_dict,
)
from mean_field.systems.RnG_hBN.hf import (
    _build_projected_basis_for_indices,
    _prepare_remote_average_source,
    _remote_average_hamiltonian_from_source,
)


PANEL_RE = re.compile(r"^xi(?P<xi>-?\d+)_V(?P<v_mev>-?\d+)meV$")


@dataclass(frozen=True)
class PanelSource:
    panel: str
    task_dir: Path
    config_path: Path
    screening_path: Path


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


def _discover_panel_sources(output_root: Path, requested_panels: set[str]) -> list[PanelSource]:
    tasks_root = output_root / "tasks"
    by_panel: dict[str, PanelSource] = {}
    for panel_dir in sorted(tasks_root.glob("task_*/*_V*meV")):
        if not panel_dir.is_dir() or PANEL_RE.match(panel_dir.name) is None:
            continue
        panel = panel_dir.name
        if panel not in requested_panels:
            continue
        task_dir = panel_dir.parent
        config_path = task_dir / "paper_hf_config.json"
        screening_path = panel_dir / "screening_result.json"
        if config_path.exists() and screening_path.exists():
            by_panel.setdefault(
                panel,
                PanelSource(
                    panel=panel,
                    task_dir=task_dir,
                    config_path=config_path,
                    screening_path=screening_path,
                ),
            )
    missing = sorted(requested_panels - set(by_panel))
    if missing:
        raise FileNotFoundError(f"No panel sources found for: {missing}")
    return [by_panel[key] for key in sorted(by_panel)]


def _diagonalize_blocks(hamiltonian: np.ndarray) -> np.ndarray:
    values = np.asarray(hamiltonian, dtype=np.complex128)
    out = np.zeros((values.shape[0], values.shape[2]), dtype=float)
    for ik in range(values.shape[2]):
        out[:, ik] = np.linalg.eigvalsh(values[:, :, ik])
    return out


def _path_zero(energies: np.ndarray, config: dict[str, object]) -> float:
    total_occupied = rlg_hbn_occupied_state_count(
        nt=int(energies.shape[0]),
        nk=int(energies.shape[1]),
        nu=float(config["nu"]),
        active_valence_bands=int(config["active_valence_bands"]),
        n_spin=2,
        n_eta=2,
    )
    values = np.sort(np.asarray(energies, dtype=float).reshape(-1))
    if total_occupied <= 0 or total_occupied >= values.size:
        return 0.0
    return float(0.5 * (values[total_occupied - 1] + values[total_occupied]))


def _build_source_basis(physical_model, basis_model, interaction, screening, valleys):
    mesh_size = int(interaction.k_mesh_size)
    k_grid_frac, kvec_grid = build_moire_k_grid(basis_model.lattice, mesh_size, endpoint=False)
    active_indices = active_band_indices_for_interaction(basis_model, interaction)
    return _build_projected_basis_for_indices(
        physical_model=physical_model,
        basis_model=basis_model,
        interaction=interaction,
        kvec=np.asarray(kvec_grid.reshape(-1), dtype=np.complex128),
        band_indices=active_indices,
        valleys=tuple(int(value) for value in valleys),
        mesh_size=mesh_size,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float).reshape(-1, 2),
        screening=screening,
        name="rlg_hbn_screened_active_path_probe_source",
        build_h0=True,
    )


def _validation_path(model: RLGhBNModel, points_per_segment: int, mode: str):
    if mode == "full":
        return _paper_hf_path(model, int(points_per_segment))
    if mode == "gamma_mprime":
        return build_kpath_from_nodes(
            (model.lattice.gamma_m, model.lattice.g_m2 / 2.0),
            ("$\\Gamma_M$", "$M'_M$"),
            (int(points_per_segment),),
        )
    raise ValueError(f"Unknown path mode {mode!r}")


def _path_hamiltonian(source_basis, path, *, chunk_size: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
    remote_start = perf_counter()
    print("[probe] remote source preparation start", flush=True)
    remote_source = _prepare_remote_average_source(source_basis)
    if remote_source is None:
        raise ValueError("Path probe requires the average scheme remote source")
    print(f"[probe] remote source preparation done elapsed={perf_counter() - remote_start:.1f}s", flush=True)
    hamiltonian_chunks: list[np.ndarray] = []
    energy_chunks: list[np.ndarray] = []
    kvec = np.asarray(path.kvec, dtype=np.complex128)
    for start in range(0, kvec.size, int(chunk_size)):
        stop = min(start + int(chunk_size), kvec.size)
        chunk_start = perf_counter()
        print(f"[probe] path chunk {start}:{stop} start", flush=True)
        target_basis = build_rlg_hbn_projected_basis_for_kvec(
            source_basis.basis_model,
            source_basis.interaction,
            kvec[start:stop],
            physical_model=source_basis.model,
            active_band_indices=source_basis.active_band_indices,
            valleys=source_basis.valleys,
        )
        fixed_remote = _remote_average_hamiltonian_from_source(
            target_basis,
            source_basis,
            remote_source,
            beta=float(beta),
        )
        h0 = np.asarray(target_basis.physical_h0, dtype=np.complex128) + fixed_remote
        hamiltonian_chunks.append(h0)
        energy_chunks.append(_diagonalize_blocks(h0))
        print(f"[probe] path chunk {start}:{stop} done elapsed={perf_counter() - chunk_start:.1f}s", flush=True)
    return np.concatenate(hamiltonian_chunks, axis=2), np.concatenate(energy_chunks, axis=1)


def _save_panel(panel_dir: Path, result: dict[str, object], *, dpi: int, ylim_mev: tuple[float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = result["path"]
    panel_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_dir / "screened_base_path_probe.npz",
        kdist=np.asarray(path.kdist, dtype=float),
        kvec_nm_inv=np.stack([np.asarray(path.kvec).real, np.asarray(path.kvec).imag], axis=-1),
        all_energies_mev=np.asarray(result["all_energies_mev"], dtype=float),
        spin_up_K_energies_mev=np.asarray(result["k_energies_mev"], dtype=float),
        spin_up_Kprime_energies_mev=np.asarray(result["kprime_energies_mev"], dtype=float),
        energy_zero_mev=np.asarray(float(result["mu_mev"])),
    )
    _write_json(
        panel_dir / "screened_base_path_probe_summary.json",
        {
            "panel": str(result["panel"]),
            "title": str(result["title"]),
            "points": int(np.asarray(path.kvec).size),
            "energy_zero_mev": float(result["mu_mev"]),
            "screened_u_mev": float(result["screened_u_mev"]),
            "output_png": str(panel_dir / "screened_base_path_probe.png"),
            "output_pdf": str(panel_dir / "screened_base_path_probe.pdf"),
        },
    )
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    _plot_panel(ax, result, ylim_mev=ylim_mev, show_ylabel=True)
    ax.set_title(str(result["title"]), fontsize=9)
    fig.savefig(panel_dir / "screened_base_path_probe.png", dpi=int(dpi))
    fig.savefig(panel_dir / "screened_base_path_probe.pdf")
    plt.close(fig)


def _save_combined(output_dir: Path, panel_results: list[dict[str, object]], *, dpi: int, ylim_mev: tuple[float, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panel_results), figsize=(max(3.4 * len(panel_results), 3.6), 3.35), sharey=True, constrained_layout=True)
    if len(panel_results) == 1:
        axes = [axes]
    for idx, (ax, result) in enumerate(zip(axes, panel_results, strict=True)):
        _plot_panel(ax, result, ylim_mev=ylim_mev, show_ylabel=idx == 0)
        ax.set_title(str(result["title"]), fontsize=9)
    axes[0].plot([], [], color="black", linewidth=1.0, label="$K$")
    axes[0].plot([], [], color="#c62828", linewidth=1.0, label="$K'$")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(output_dir / "screened_base_path_probe.png", dpi=int(dpi))
    fig.savefig(output_dir / "screened_base_path_probe.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-only RLG/hBN screened-base periodic-gauge validation.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panels", type=str, default="xi0_V064meV,xi1_V064meV")
    parser.add_argument("--points-per-segment", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--ylim-mev", type=str, default="-120,120")
    parser.add_argument("--path-mode", choices=("full", "gamma_mprime"), default="full")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    start_time = perf_counter()
    output_root = args.output_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = {value.strip() for value in args.panels.split(",") if value.strip()}
    sources = _discover_panel_sources(output_root, panels)
    ylim_mev = _parse_ylim(args.ylim_mev, "fig6")
    panel_results: list[dict[str, object]] = []
    for source in sources:
        panel_start = perf_counter()
        config = _read_json(source.config_path)
        screening = screening_result_from_dict(_read_json(source.screening_path))
        xi, v_mev = _parse_panel(source.panel)
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
            displacement_field_mev=float(screening.screened_u_mev),
            shell_count=int(config["shell_count"]),
        )
        interaction = _build_interaction(config)
        print(f"[probe] {source.panel} source basis start", flush=True)
        source_basis = _build_source_basis(physical_model, basis_model, interaction, screening, valleys=(1, -1))
        print(f"[probe] {source.panel} source basis done", flush=True)
        path = _validation_path(physical_model, int(args.points_per_segment), str(args.path_mode))
        path_hamiltonian, path_energies = _path_hamiltonian(
            source_basis,
            path,
            chunk_size=int(args.chunk_size),
            beta=float(config.get("beta", 1.0)),
        )
        n_spin = 2
        n_eta = 2
        n_band = int(path_hamiltonian.shape[0]) // (n_spin * n_eta)
        result = {
            "panel": source.panel,
            "title": f"$\\xi={xi}$, $V={v_mev:.0f}$ meV path probe, $U={screening.screened_u_mev:.2f}$ meV",
            "path": path,
            "all_energies_mev": path_energies,
            "k_energies_mev": _sector_energies(path_hamiltonian, n_spin=n_spin, n_eta=n_eta, n_band=n_band, spin=0, eta=0),
            "kprime_energies_mev": _sector_energies(path_hamiltonian, n_spin=n_spin, n_eta=n_eta, n_band=n_band, spin=0, eta=1),
            "mu_mev": _path_zero(path_energies, config),
            "screened_u_mev": float(screening.screened_u_mev),
            "elapsed_sec": float(perf_counter() - panel_start),
        }
        panel_results.append(result)
        _save_panel(output_dir / source.panel, result, dpi=int(args.dpi), ylim_mev=ylim_mev)
        print(f"[probe] {source.panel} done elapsed={result['elapsed_sec']:.1f}s", flush=True)

    _save_combined(output_dir, panel_results, dpi=int(args.dpi), ylim_mev=ylim_mev)
    _write_json(
        output_dir / "screened_base_path_probe_summary.json",
        {
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "output_png": str(output_dir / "screened_base_path_probe.png"),
            "hostname": socket.gethostname(),
            "path_mode": str(args.path_mode),
            "points_per_segment": int(args.points_per_segment),
            "chunk_size": int(args.chunk_size),
            "panels": [
                {
                    "panel": str(result["panel"]),
                    "screened_u_mev": float(result["screened_u_mev"]),
                    "energy_zero_mev": float(result["mu_mev"]),
                    "elapsed_sec": float(result["elapsed_sec"]),
                    "screening": screening_result_to_dict(
                        screening_result_from_dict(_read_json(sources[idx].screening_path))
                    ),
                }
                for idx, result in enumerate(panel_results)
            ],
            "elapsed_sec": float(perf_counter() - start_time),
        },
    )
    print(f"[probe] output_png={output_dir / 'screened_base_path_probe.png'}", flush=True)


if __name__ == "__main__":
    main()
