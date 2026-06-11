#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mean_field.core.io import write_text_artifact
from mean_field.devtools._runtime import (
    ensure_not_running_compute_on_login_node,
    select_energy_window_bands,
    write_json,
)
from mean_field.runtime import collect_runtime_environment, current_timestamp
from mean_field.systems.atmg import ATMGModel, ATMGParameters
from mean_field.systems.atmg.bilayer_map import build_atmg_via_tbg_sum


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "ATMG"
PAPER_REFERENCE = REPO_ROOT / "reference" / "1901.10485v2.pdf"
TBG_MAGIC_ALPHA_1 = 0.586
TBG_MAGIC_ALPHA_2 = 2.221


@dataclass(frozen=True)
class Fig3PanelSpec:
    key: str
    n_layers: int
    alpha: float
    alpha_label: str
    realistic_kappa: float
    y_window_scaled: float
    target_tbg_alpha: float

    @property
    def title(self) -> str:
        return f"n = {self.n_layers}, alpha = {self.alpha_label}"


@dataclass(frozen=True)
class CurveResult:
    kappa: float
    theta_deg: float
    w_ab_ev: float
    w_aa_ev: float
    moire_energy_scale_ev: float
    singular_values: np.ndarray
    subspace_labels: tuple[str, ...]
    combined_energies_ev: np.ndarray
    subspace_energies_ev: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PanelResult:
    spec: Fig3PanelSpec
    kdist: np.ndarray
    tick_positions: tuple[float, ...]
    tick_labels: tuple[str, ...]
    chiral: CurveResult
    realistic: CurveResult
    direct_node_max_abs_diff_ev: float | None
    elapsed_sec: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Khalaf et al. PRB 100, 085109 (2019) Fig. 3 ATMG band panels."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-shells", type=int, default=5)
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--w-ab-ev", type=float, default=0.110)
    parser.add_argument(
        "--no-direct-node-check",
        action="store_true",
        help="Skip the direct full-ATMG-vs-mapped-spectrum check at the high-symmetry path nodes.",
    )
    parser.add_argument(
        "--panel-workers",
        type=int,
        default=None,
        help="Number of independent Fig. 3 panel processes. Defaults to PANEL_WORKERS or 1.",
    )
    parser.add_argument(
        "--panel-threads",
        type=int,
        default=None,
        help="BLAS/OpenMP threads per panel worker. Defaults to PANEL_THREADS or cpus/workers.",
    )
    return parser.parse_args()


def _default_output_dir() -> Path:
    timestamp = current_timestamp().replace(":", "").replace("-", "")
    return DEFAULT_OUTPUT_ROOT / f"atmg_fig3_khalaf_{timestamp}"


def _fig3_specs() -> tuple[Fig3PanelSpec, ...]:
    phi = (math.sqrt(5.0) + 1.0) / 2.0
    return (
        Fig3PanelSpec(
            key="n3_alpha1",
            n_layers=3,
            alpha=TBG_MAGIC_ALPHA_1 / math.sqrt(2.0),
            alpha_label="0.414",
            realistic_kappa=0.8,
            y_window_scaled=1.0,
            target_tbg_alpha=TBG_MAGIC_ALPHA_1,
        ),
        Fig3PanelSpec(
            key="n3_alpha2",
            n_layers=3,
            alpha=TBG_MAGIC_ALPHA_2 / math.sqrt(2.0),
            alpha_label="1.57",
            realistic_kappa=0.35,
            y_window_scaled=0.5,
            target_tbg_alpha=TBG_MAGIC_ALPHA_2,
        ),
        Fig3PanelSpec(
            key="n4_alpha1",
            n_layers=4,
            alpha=TBG_MAGIC_ALPHA_1 / phi,
            alpha_label="0.362",
            realistic_kappa=0.8,
            y_window_scaled=1.0,
            target_tbg_alpha=TBG_MAGIC_ALPHA_1,
        ),
        Fig3PanelSpec(
            key="n4_alpha1_prime",
            n_layers=4,
            alpha=TBG_MAGIC_ALPHA_1 * phi,
            alpha_label="0.948",
            realistic_kappa=0.5,
            y_window_scaled=0.5,
            target_tbg_alpha=TBG_MAGIC_ALPHA_1,
        ),
    )


def _theta_deg_from_alpha(alpha: float, *, w_ab_ev: float, vf: float, graphene_a_nm: float) -> float:
    graphene_k_mag = 4.0 * math.pi / (3.0 * float(graphene_a_nm))
    theta_rad = float(w_ab_ev) / (float(alpha) * float(vf) * graphene_k_mag)
    return float(theta_rad * 180.0 / math.pi)


def build_khalaf_fig3_path(model: ATMGModel, points_per_segment: int):
    kprime = -complex(model.lattice.q0)
    kpoint = 0.0 + 0.0j
    gamma = -complex(model.lattice.q0) - complex(model.lattice.q_plus)
    gamma_prime = gamma + complex(model.lattice.g_m1)
    kprime_translated = kprime + complex(model.lattice.g_m1)
    return model.build_kpath(
        (kprime, kpoint, gamma, gamma_prime, kprime_translated),
        ("K'", "K", r"$\Gamma$", r"$\Gamma'$", "K'"),
        points_per_segment=points_per_segment,
    )


def _build_params(spec: Fig3PanelSpec, *, kappa: float, w_ab_ev: float) -> ATMGParameters:
    prototype = ATMGParameters.realistic(2, 1.0, w_ab=w_ab_ev)
    theta_deg = _theta_deg_from_alpha(
        spec.alpha,
        w_ab_ev=w_ab_ev,
        vf=prototype.vf,
        graphene_a_nm=prototype.graphene_lattice_constant_nm,
    )
    model_name = "fig3_chiral" if abs(float(kappa)) < 1.0e-15 else "fig3_realistic"
    return ATMGParameters(
        n_layers=spec.n_layers,
        theta_deg=theta_deg,
        w_ab=w_ab_ev,
        kappa=float(kappa),
        vf=prototype.vf,
        graphene_lattice_constant_nm=prototype.graphene_lattice_constant_nm,
        model_name=model_name,
    )


def _compute_curve(
    spec: Fig3PanelSpec,
    *,
    kappa: float,
    w_ab_ev: float,
    n_shells: int,
    points_per_segment: int,
) -> tuple[CurveResult, np.ndarray, tuple[float, ...], tuple[str, ...]]:
    params = _build_params(spec, kappa=kappa, w_ab_ev=w_ab_ev)
    model = ATMGModel.from_config(spec.n_layers, params.theta_deg, n_shells=n_shells, params=params)
    path = build_khalaf_fig3_path(model, points_per_segment)
    first_mapped = build_atmg_via_tbg_sum(complex(path.kvec[0]), model.lattice, model.params, valley=1)

    combined_energies = np.zeros((path.kvec.size, first_mapped.combined_energies.size), dtype=float)
    subspace_energies = [
        np.zeros((path.kvec.size, block.size), dtype=float) for block in first_mapped.subspace_energies
    ]

    for ik, kval in enumerate(path.kvec):
        mapped = build_atmg_via_tbg_sum(complex(kval), model.lattice, model.params, valley=1)
        combined_energies[ik, :] = np.asarray(mapped.combined_energies, dtype=float)
        for block_index, block in enumerate(mapped.subspace_energies):
            subspace_energies[block_index][ik, :] = np.asarray(block, dtype=float)

    tick_positions = tuple(float(path.kdist[index - 1]) for index in path.node_indices)
    tick_labels = tuple(str(label) for label in path.labels)
    result = CurveResult(
        kappa=float(kappa),
        theta_deg=float(params.theta_deg),
        w_ab_ev=float(params.w_ab),
        w_aa_ev=float(params.kappa * params.w_ab),
        moire_energy_scale_ev=float(params.moire_energy_scale),
        singular_values=np.asarray(first_mapped.singular_values, dtype=float),
        subspace_labels=tuple(first_mapped.labels),
        combined_energies_ev=np.asarray(combined_energies, dtype=float),
        subspace_energies_ev=tuple(np.asarray(item, dtype=float) for item in subspace_energies),
    )
    return result, np.asarray(path.kdist, dtype=float), tick_positions, tick_labels


def _direct_node_max_abs_diff(
    spec: Fig3PanelSpec,
    *,
    w_ab_ev: float,
    n_shells: int,
    points_per_segment: int,
) -> float:
    max_abs = 0.0
    for kappa in (0.0, spec.realistic_kappa):
        params = _build_params(spec, kappa=kappa, w_ab_ev=w_ab_ev)
        model = ATMGModel.from_config(spec.n_layers, params.theta_deg, n_shells=n_shells, params=params)
        path = build_khalaf_fig3_path(model, points_per_segment)
        for node in path.nodes:
            direct_evals, _ = model.diagonalize(node.kvec, valley=1)
            mapped_evals = model.mapped_spectrum(node.kvec, valley=1).combined_energies
            diff = np.max(np.abs(np.sort(direct_evals) - np.sort(mapped_evals)))
            max_abs = max(max_abs, float(diff))
    return float(max_abs)


def _compute_panel(
    spec: Fig3PanelSpec,
    *,
    w_ab_ev: float,
    n_shells: int,
    points_per_segment: int,
    direct_node_check: bool,
) -> PanelResult:
    start = perf_counter()
    chiral, kdist, tick_positions, tick_labels = _compute_curve(
        spec,
        kappa=0.0,
        w_ab_ev=w_ab_ev,
        n_shells=n_shells,
        points_per_segment=points_per_segment,
    )
    realistic, kdist_realistic, tick_positions_realistic, tick_labels_realistic = _compute_curve(
        spec,
        kappa=spec.realistic_kappa,
        w_ab_ev=w_ab_ev,
        n_shells=n_shells,
        points_per_segment=points_per_segment,
    )
    if not np.allclose(kdist, kdist_realistic):
        raise RuntimeError(f"Path mismatch between chiral and realistic runs for {spec.key}")
    if tick_positions != tick_positions_realistic or tick_labels != tick_labels_realistic:
        raise RuntimeError(f"Path tick mismatch between chiral and realistic runs for {spec.key}")

    direct_diff = None
    if direct_node_check:
        direct_diff = _direct_node_max_abs_diff(
            spec,
            w_ab_ev=w_ab_ev,
            n_shells=n_shells,
            points_per_segment=points_per_segment,
        )

    return PanelResult(
        spec=spec,
        kdist=kdist,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        chiral=chiral,
        realistic=realistic,
        direct_node_max_abs_diff_ev=direct_diff,
        elapsed_sec=float(perf_counter() - start),
    )


def _read_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_parallelism(
    *,
    panel_workers_arg: int | None,
    panel_threads_arg: int | None,
    panel_count: int,
) -> tuple[int, int]:
    requested_workers = panel_workers_arg if panel_workers_arg is not None else (_read_env_int("PANEL_WORKERS") or 1)
    panel_workers = max(1, min(int(requested_workers), int(panel_count)))

    requested_threads = panel_threads_arg if panel_threads_arg is not None else _read_env_int("PANEL_THREADS")
    if requested_threads is None:
        allocated_cpus = _read_env_int("SLURM_CPUS_PER_TASK") or _read_env_int("OMP_NUM_THREADS") or (os.cpu_count() or 1)
        requested_threads = max(1, int(allocated_cpus) // int(panel_workers))
    panel_threads = max(1, int(requested_threads))
    return panel_workers, panel_threads


def _configure_worker_threads(panel_threads: int) -> None:
    value = str(max(1, int(panel_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value

    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]

        threadpool_limits(limits=int(value))
    except Exception:
        pass


def _compute_panel_worker(payload: tuple[Fig3PanelSpec, float, int, int, bool, int]) -> PanelResult:
    spec, w_ab_ev, n_shells, points_per_segment, direct_node_check, panel_threads = payload
    _configure_worker_threads(panel_threads)
    return _compute_panel(
        spec,
        w_ab_ev=w_ab_ev,
        n_shells=n_shells,
        points_per_segment=points_per_segment,
        direct_node_check=direct_node_check,
    )


def _plot_curve(axis, panel: PanelResult, curve: CurveResult, *, color: str, linestyle: str, label: str) -> None:
    scale = float(curve.moire_energy_scale_ev)
    energies_scaled = np.asarray(curve.combined_energies_ev, dtype=float) / scale
    band_indices = select_energy_window_bands(
        energies_scaled,
        emin=-float(panel.spec.y_window_scaled),
        emax=float(panel.spec.y_window_scaled),
        fallback_each_side=12,
        fallback_include_center=False,
    )
    for band in band_indices:
        axis.plot(
            panel.kdist,
            energies_scaled[:, band],
            color=color,
            linestyle=linestyle,
            linewidth=0.72,
            alpha=0.9,
        )
    axis.plot([], [], color=color, linestyle=linestyle, linewidth=1.0, label=label)


def _plot_fig3(panels: tuple[PanelResult, ...], output_png: Path, output_pdf: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=False, sharey=False)
    panel_order = ("n3_alpha1", "n3_alpha2", "n4_alpha1", "n4_alpha1_prime")
    panel_by_key = {panel.spec.key: panel for panel in panels}
    axes_flat = tuple(axes.ravel())

    for axis, key in zip(axes_flat, panel_order, strict=True):
        panel = panel_by_key[key]
        for xpos in panel.tick_positions[1:-1]:
            axis.axvline(xpos, color="0.75", linewidth=0.65, zorder=0)
        axis.axhline(0.0, color="0.62", linewidth=0.55, zorder=0)

        _plot_curve(axis, panel, panel.chiral, color="red", linestyle="-", label=r"$\kappa = 0$")
        _plot_curve(
            axis,
            panel,
            panel.realistic,
            color="blue",
            linestyle="--",
            label=rf"$\kappa = {panel.spec.realistic_kappa:g}$",
        )

        axis.set_title(panel.spec.title, fontsize=12)
        axis.set_xticks(panel.tick_positions, panel.tick_labels)
        axis.set_xlim(float(panel.kdist[0]), float(panel.kdist[-1]))
        axis.set_ylim(-float(panel.spec.y_window_scaled), float(panel.spec.y_window_scaled))
        axis.grid(True, alpha=0.18, linewidth=0.55)
        axis.legend(frameon=False, loc="upper right", fontsize=9)

    axes[0, 0].set_ylabel(r"$E/(v_F k_D \theta)$")
    axes[1, 0].set_ylabel(r"$E/(v_F k_D \theta)$")
    fig.suptitle("Khalaf et al. Fig. 3 ATMG band structure reproduction", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def _central_pair_bandwidth_scaled(curve: CurveResult, *, target_alpha: float) -> float | None:
    if not curve.subspace_energies_ev:
        return None
    singular_values = np.asarray(curve.singular_values, dtype=float)
    if singular_values.size == 0:
        return None
    target_index = int(np.argmin(np.abs(singular_values - float(target_alpha))))
    bands_ev = np.asarray(curve.subspace_energies_ev[target_index], dtype=float)
    center = bands_ev.shape[1] // 2
    pair = bands_ev[:, max(0, center - 1) : min(bands_ev.shape[1], center + 1)]
    if pair.size == 0:
        return None
    return float((np.max(pair) - np.min(pair)) / float(curve.moire_energy_scale_ev))


def _save_panel_npz(panel_dir: Path, panel: PanelResult) -> Path:
    panel_dir.mkdir(parents=True, exist_ok=True)
    output_path = panel_dir / "bands_path.npz"
    payload: dict[str, object] = {
        "kdist": np.asarray(panel.kdist, dtype=float),
        "tick_positions": np.asarray(panel.tick_positions, dtype=float),
        "tick_labels": np.asarray(panel.tick_labels, dtype=object),
        "n_layers": np.asarray(panel.spec.n_layers, dtype=int),
        "alpha_dimless": np.asarray(panel.spec.alpha, dtype=float),
        "alpha_label": np.asarray(panel.spec.alpha_label, dtype=object),
        "target_tbg_alpha": np.asarray(panel.spec.target_tbg_alpha, dtype=float),
        "realistic_kappa": np.asarray(panel.spec.realistic_kappa, dtype=float),
        "chiral_energies_ev": np.asarray(panel.chiral.combined_energies_ev, dtype=float),
        "realistic_energies_ev": np.asarray(panel.realistic.combined_energies_ev, dtype=float),
        "chiral_energies_scaled": np.asarray(panel.chiral.combined_energies_ev, dtype=float)
        / float(panel.chiral.moire_energy_scale_ev),
        "realistic_energies_scaled": np.asarray(panel.realistic.combined_energies_ev, dtype=float)
        / float(panel.realistic.moire_energy_scale_ev),
        "chiral_singular_values": np.asarray(panel.chiral.singular_values, dtype=float),
        "realistic_singular_values": np.asarray(panel.realistic.singular_values, dtype=float),
        "subspace_labels": np.asarray(panel.chiral.subspace_labels, dtype=object),
        "direct_node_max_abs_diff_ev": np.asarray(
            np.nan if panel.direct_node_max_abs_diff_ev is None else panel.direct_node_max_abs_diff_ev,
            dtype=float,
        ),
    }
    for prefix, curve in (("chiral", panel.chiral), ("realistic", panel.realistic)):
        payload[f"{prefix}_theta_deg"] = np.asarray(curve.theta_deg, dtype=float)
        payload[f"{prefix}_kappa"] = np.asarray(curve.kappa, dtype=float)
        payload[f"{prefix}_w_ab_ev"] = np.asarray(curve.w_ab_ev, dtype=float)
        payload[f"{prefix}_w_aa_ev"] = np.asarray(curve.w_aa_ev, dtype=float)
        payload[f"{prefix}_moire_energy_scale_ev"] = np.asarray(curve.moire_energy_scale_ev, dtype=float)
        for index, energies in enumerate(curve.subspace_energies_ev, start=1):
            payload[f"{prefix}_subspace_{index}_energies_ev"] = np.asarray(energies, dtype=float)
    np.savez(output_path, **payload)
    return output_path


def _curve_summary(curve: CurveResult, *, target_tbg_alpha: float) -> dict[str, object]:
    return {
        "kappa": float(curve.kappa),
        "theta_deg": float(curve.theta_deg),
        "w_ab_ev": float(curve.w_ab_ev),
        "w_aa_ev": float(curve.w_aa_ev),
        "moire_energy_scale_ev": float(curve.moire_energy_scale_ev),
        "singular_values": [float(value) for value in curve.singular_values],
        "subspace_labels": list(curve.subspace_labels),
        "central_pair_bandwidth_scaled": _central_pair_bandwidth_scaled(curve, target_alpha=target_tbg_alpha),
    }


def _write_report(
    path: Path,
    *,
    output_dir: Path,
    figure_png: Path,
    figure_pdf: Path,
    summary_json: Path,
    panels: tuple[PanelResult, ...],
    panel_npz_paths: dict[str, str],
) -> None:
    lines = [
        "# ATMG Fig. 3 Reproduction",
        "",
        f"- `paper = {PAPER_REFERENCE}`",
        "- `target = Khalaf et al. PRB 100, 085109 (2019), Fig. 3`",
        "- `path = Kprime(-Q0) -> K(0) -> Gamma(-Q0-Qplus) -> GammaPrime(-Q0-Qplus+G_M1) -> Kprime(-Q0+G_M1)`",
        f"- `output_dir = {output_dir}`",
        f"- `figure_png = {figure_png}`",
        f"- `figure_pdf = {figure_pdf}`",
        f"- `summary_json = {summary_json}`",
        "",
        "## Panel Parameters",
        "",
    ]
    for panel in panels:
        lines.extend(
            [
                f"### {panel.spec.key}",
                "",
                f"- `n_layers = {panel.spec.n_layers}`",
                f"- `alpha = {panel.spec.alpha:.12g}`",
                f"- `theta_deg = {panel.chiral.theta_deg:.12g}`",
                f"- `kappa_values = 0, {panel.spec.realistic_kappa:g}`",
                f"- `singular_values = {[float(value) for value in panel.chiral.singular_values]}`",
                f"- `target_tbg_alpha = {panel.spec.target_tbg_alpha:.12g}`",
                f"- `chiral_central_pair_bandwidth_scaled = {_central_pair_bandwidth_scaled(panel.chiral, target_alpha=panel.spec.target_tbg_alpha)}`",
                f"- `realistic_central_pair_bandwidth_scaled = {_central_pair_bandwidth_scaled(panel.realistic, target_alpha=panel.spec.target_tbg_alpha)}`",
                f"- `direct_node_max_abs_diff_ev = {panel.direct_node_max_abs_diff_ev}`",
                f"- `bands_npz = {panel_npz_paths[panel.spec.key]}`",
                f"- `elapsed_sec = {panel.elapsed_sec:.3f}`",
                "",
            ]
        )
    write_text_artifact("\n".join(lines), path)


def main() -> int:
    total_start = perf_counter()
    args = _parse_args()
    ensure_not_running_compute_on_login_node("ATMG Fig. 3 band reproduction")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _fig3_specs()
    panel_workers, panel_threads = _resolve_parallelism(
        panel_workers_arg=args.panel_workers,
        panel_threads_arg=args.panel_threads,
        panel_count=len(specs),
    )
    print(f"[parallel] panel_workers={panel_workers}, panel_threads={panel_threads}", flush=True)

    panels_by_key: dict[str, PanelResult] = {}
    direct_node_check = not bool(args.no_direct_node_check)

    for spec in specs:
        print(
            f"[panel] {spec.key}: n={spec.n_layers}, alpha={spec.alpha:.12g}, "
            f"kappa=0/{spec.realistic_kappa:g}",
            flush=True,
        )

    if panel_workers == 1:
        _configure_worker_threads(panel_threads)
        for spec in specs:
            panels_by_key[spec.key] = _compute_panel(
                spec,
                w_ab_ev=float(args.w_ab_ev),
                n_shells=int(args.n_shells),
                points_per_segment=int(args.points_per_segment),
                direct_node_check=direct_node_check,
            )
            print(f"[panel-done] {spec.key}: elapsed_sec={panels_by_key[spec.key].elapsed_sec:.3f}", flush=True)
    else:
        payloads = [
            (
                spec,
                float(args.w_ab_ev),
                int(args.n_shells),
                int(args.points_per_segment),
                direct_node_check,
                int(panel_threads),
            )
            for spec in specs
        ]
        with ProcessPoolExecutor(
            max_workers=int(panel_workers),
            initializer=_configure_worker_threads,
            initargs=(int(panel_threads),),
        ) as executor:
            futures = {executor.submit(_compute_panel_worker, payload): payload[0].key for payload in payloads}
            for future in as_completed(futures):
                key = futures[future]
                panel = future.result()
                panels_by_key[key] = panel
                print(f"[panel-done] {key}: elapsed_sec={panel.elapsed_sec:.3f}", flush=True)

    panel_results = tuple(panels_by_key[spec.key] for spec in specs)
    figure_png = output_dir / "atmg_fig3_bands.png"
    figure_pdf = output_dir / "atmg_fig3_bands.pdf"
    _plot_fig3(panel_results, figure_png, figure_pdf)

    panel_npz_paths: dict[str, str] = {}
    for panel in panel_results:
        panel_npz_paths[panel.spec.key] = str(_save_panel_npz(output_dir / panel.spec.key, panel))

    elapsed_sec = float(perf_counter() - total_start)
    summary_json = output_dir / "summary.json"
    report_md = output_dir / "fig3_reproduction_report.md"
    summary_payload = {
        "generated_at": current_timestamp(),
        "elapsed_sec": elapsed_sec,
        "paper_reference": str(PAPER_REFERENCE),
        "output_dir": str(output_dir),
        "runtime_environment": asdict(collect_runtime_environment()),
        "settings": {
            "n_shells": int(args.n_shells),
            "points_per_segment": int(args.points_per_segment),
            "w_ab_ev": float(args.w_ab_ev),
            "direct_node_check": not bool(args.no_direct_node_check),
            "panel_workers": int(panel_workers),
            "panel_threads": int(panel_threads),
            "energy_unit": "E/(v_F*k_D*theta)",
            "path_convention": "Kprime(-Q0) -> K(0) -> Gamma(-Q0-Qplus) -> GammaPrime(-Q0-Qplus+G_M1) -> Kprime(-Q0+G_M1)",
        },
        "figures": {
            "png": str(figure_png),
            "pdf": str(figure_pdf),
        },
        "panel_npz_paths": panel_npz_paths,
        "panels": {
            panel.spec.key: {
                "n_layers": int(panel.spec.n_layers),
                "alpha_dimless": float(panel.spec.alpha),
                "alpha_label": panel.spec.alpha_label,
                "target_tbg_alpha": float(panel.spec.target_tbg_alpha),
                "realistic_kappa": float(panel.spec.realistic_kappa),
                "y_window_scaled": float(panel.spec.y_window_scaled),
                "tick_labels": list(panel.tick_labels),
                "chiral": _curve_summary(panel.chiral, target_tbg_alpha=panel.spec.target_tbg_alpha),
                "realistic": _curve_summary(panel.realistic, target_tbg_alpha=panel.spec.target_tbg_alpha),
                "direct_node_max_abs_diff_ev": panel.direct_node_max_abs_diff_ev,
                "elapsed_sec": float(panel.elapsed_sec),
            }
            for panel in panel_results
        },
    }
    write_json(summary_json, summary_payload)
    _write_report(
        report_md,
        output_dir=output_dir,
        figure_png=figure_png,
        figure_pdf=figure_pdf,
        summary_json=summary_json,
        panels=panel_results,
        panel_npz_paths=panel_npz_paths,
    )

    print(f"output_dir={output_dir}")
    print(f"figure_png={figure_png}")
    print(f"figure_pdf={figure_pdf}")
    print(f"summary_json={summary_json}")
    print(f"report_md={report_md}")
    print(f"elapsed_sec={elapsed_sec:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
