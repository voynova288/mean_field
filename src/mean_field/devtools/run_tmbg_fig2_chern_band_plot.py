from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, select_flat_pair_window, write_json
from mean_field.runtime import collect_runtime_environment, current_timestamp
from mean_field.systems.tmbg import (
    PathBandsResult,
    TMBGBandPlotPanel,
    TMBGModel,
    TMBGParameters,
    TopologyResult,
    compute_topology_from_grid_result,
    infer_flat_band_indices,
    write_tmbg_lattice_plot,
    write_tmbg_paper_band_figure,
)
from mean_field.systems.tmbg.topology_sewn import compute_sewn_grid, fhs_chern_sewn


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "TMBG"
DEFAULT_DELTAS_EV = (0.0, 0.060, -0.040)
PARK_FIG2_CHERN: dict[float, dict[str, int]] = {
    0.000: {"valence": 2, "conduction": -3, "cnp_pair_total": -1},
    0.060: {"valence": -2, "conduction": 1},
    -0.040: {"valence": 1, "conduction": -2},
}


@dataclass(frozen=True)
class PanelComputation:
    delta_ev: float
    model_name: str
    flat_band_indices: tuple[int, int]
    selected_band_indices: tuple[int, ...]
    flat_gap_mev: float
    flat_gap_location: str
    topology_method: str
    topology_mode: str
    chern_payload: dict[str, object]
    expected_payload: dict[str, int]
    status: str
    annotation: str


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def _parse_positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Park 2020 Fig. 2 tMBG band panels with K-valley Chern annotations. "
            "For Delta=0 the default topology target is the total Chern of the two CNP bands."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-deg", type=float, default=1.21)
    parser.add_argument("--n-shells", type=int, default=5)
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--topology-mesh-size", type=int, default=24)
    parser.add_argument(
        "--topology-method",
        choices=("standard", "sewn"),
        default="standard",
        help="Use the standard periodic FHS links or moire-BZ boundary-sewn FHS links for Chern numbers.",
    )
    parser.add_argument(
        "--sewn-orientation",
        choices=("raw", "physical"),
        default="raw",
        help="Orientation convention for --topology-method sewn.",
    )
    parser.add_argument("--valley", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--bands-per-side", type=int, default=6)
    parser.add_argument("--deltas-ev", type=_parse_csv_floats, default=DEFAULT_DELTAS_EV)
    parser.add_argument(
        "--panel-workers",
        type=_parse_positive_int,
        default=None,
        help="Number of independent Delta panels to compute concurrently; defaults to PANEL_WORKERS or all panels.",
    )
    parser.add_argument(
        "--worker-threads",
        type=_parse_positive_int,
        default=None,
        help="BLAS/OpenMP threads assigned to each panel worker; defaults to PANEL_THREADS or cpus/workers.",
    )
    parser.add_argument(
        "--path-kind",
        choices=("park-fig2", "standard"),
        default="park-fig2",
        help="Use the extended-zone K-Kprime-Gamma-GammaPrime-K path from Park 2020 Fig. 2.",
    )
    parser.add_argument(
        "--gamma-prime-choice",
        choices=("g1", "minus_g1", "g2", "minus_g2", "g1_minus_g2", "minus_g1_plus_g2"),
        default="minus_g1",
        help="Translated Gamma point used by the extended-zone Park Fig. 2 path.",
    )
    parser.add_argument(
        "--plot-valleys",
        choices=("selected", "both"),
        default="selected",
        help="Plot only --valley or overlay the time-reversed valley bands in each Fig. 2 panel.",
    )
    parser.add_argument(
        "--individual-delta-zero",
        action="store_true",
        help="Compute separate Chern numbers for the Delta=0 valence/conduction bands instead of the CNP-pair total.",
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Only generate band panels and path data; skip Chern calculations.",
    )
    parser.add_argument(
        "--compare-paper-chern",
        action="store_true",
        help="Report pass/fail against the Chern labels printed in Park 2020 Fig. 2.",
    )
    return parser.parse_args()


def _positive_int_from_env(name: str) -> int | None:
    text = os.environ.get(name)
    if text is None or not text.strip():
        return None
    value = int(text)
    if value < 1:
        raise SystemExit(f"{name} must be a positive integer, got {text!r}.")
    return value


def _allocated_cpus() -> int:
    slurm_cpus = _positive_int_from_env("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return slurm_cpus
    return max(1, os.cpu_count() or 1)


def _resolve_panel_workers(args: argparse.Namespace, delta_count: int) -> int:
    if delta_count < 1:
        raise SystemExit("Expected at least one Delta panel.")
    requested = args.panel_workers
    if requested is None:
        requested = _positive_int_from_env("PANEL_WORKERS")
    if requested is None:
        requested = min(delta_count, _allocated_cpus())
    return max(1, min(int(requested), int(delta_count)))


def _resolve_worker_threads(args: argparse.Namespace, panel_workers: int) -> int:
    requested = args.worker_threads
    if requested is None:
        requested = _positive_int_from_env("PANEL_THREADS")
    if requested is None:
        requested = max(1, _allocated_cpus() // max(1, int(panel_workers)))
    return max(1, int(requested))


def _configure_worker_threads(worker_threads: int) -> None:
    text = str(max(1, int(worker_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = text


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"tmbg_fig2_chern_{job_id}"
    else:
        stem = f"tmbg_fig2_chern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DEFAULT_OUTPUT_ROOT / stem


def _panel_label(delta_ev: float) -> str:
    delta_mev = int(round(delta_ev * 1.0e3))
    if delta_mev > 0:
        return f"Delta = +{delta_mev} meV"
    if delta_mev < 0:
        return f"Delta = {delta_mev} meV"
    return "Delta = 0 meV"


def _delta_panel_dirname(delta_ev: float) -> str:
    return f"delta_{int(round(delta_ev * 1.0e3)):+04d}mev"


def _delta_key(delta_ev: float) -> float:
    return round(float(delta_ev), 3)


def _display_k_label(label: str) -> str:
    return {"Gamma": "Gamma", "GammaPrime": "Gamma'", "M": "M", "K": "K", "Kprime": "K'", "KPrime": "K'"}.get(
        label, label
    )


def _path_gap_summary(path_result: PathBandsResult, flat_pair: tuple[int, int]) -> tuple[float, str]:
    energies = np.asarray(path_result.energies, dtype=float)
    flat_gaps = energies[:, int(flat_pair[1])] - energies[:, int(flat_pair[0])]
    gap_index = int(np.argmin(flat_gaps))
    gap_mev = float(flat_gaps[gap_index] * 1.0e3)

    location = None
    for node in path_result.path.nodes:
        if int(node.index - 1) == gap_index:
            location = _display_k_label(node.label)
            break
    if location is None:
        kvec = complex(path_result.path.kvec[gap_index])
        location = f"({kvec.real:+.4f}, {kvec.imag:+.4f}) nm^-1"
    return gap_mev, location


def _topology_payload(result: TopologyResult, *, extra: dict[str, object] | None = None) -> dict[str, object]:
    payload = {
        "band_indices": list(int(index) for index in result.band_indices),
        "valley": int(result.valley),
        "chern_number": float(result.chern_number),
        "rounded_chern_number": int(result.rounded_chern_number),
        "integer_residual": float(result.integer_residual),
        "is_nearly_integer": bool(result.is_nearly_integer),
    }
    if extra:
        payload.update(extra)
    return payload


def _rounded(payload: dict[str, object], key: str) -> int | None:
    entry = payload.get(key)
    if not isinstance(entry, dict):
        return None
    rounded = entry.get("rounded_chern_number")
    if rounded is None:
        return None
    return int(rounded)


def _max_integer_residual(payload: dict[str, object]) -> float | None:
    residuals: list[float] = []
    for entry in payload.values():
        if isinstance(entry, dict) and "integer_residual" in entry:
            residuals.append(float(entry["integer_residual"]))
    if not residuals:
        return None
    return float(max(residuals))


def _chern_status(
    delta_ev: float,
    chern_payload: dict[str, object],
    *,
    skip_topology: bool,
    individual_delta_zero: bool,
    compare_paper_chern: bool,
) -> tuple[str, dict[str, int]]:
    if skip_topology:
        return "skipped", {}
    expected = PARK_FIG2_CHERN.get(_delta_key(delta_ev), {}) if compare_paper_chern else {}
    if not expected:
        return "computed", {}

    if abs(delta_ev) < 5.0e-10 and not individual_delta_zero:
        if "cnp_pair_total" not in expected:
            return "computed", expected
        observed = _rounded(chern_payload, "cnp_pair_total")
        return ("pass" if observed == expected.get("cnp_pair_total") else "fail"), expected

    observed_valence = _rounded(chern_payload, "valence")
    observed_conduction = _rounded(chern_payload, "conduction")
    passed = observed_valence == expected.get("valence") and observed_conduction == expected.get("conduction")
    return ("pass" if passed else "fail"), expected


def _chern_annotation(
    delta_ev: float,
    chern_payload: dict[str, object],
    *,
    skip_topology: bool,
    individual_delta_zero: bool,
) -> str:
    if skip_topology:
        return "Chern: skipped"
    if abs(delta_ev) < 5.0e-10 and not individual_delta_zero:
        total = _rounded(chern_payload, "cnp_pair_total")
        return f"C_CNP = {total if total is not None else 'n/a'}"
    valence = _rounded(chern_payload, "valence")
    conduction = _rounded(chern_payload, "conduction")
    return f"C_v = {valence if valence is not None else 'n/a'}, C_c = {conduction if conduction is not None else 'n/a'}"


def _compute_chern_payload(
    model: TMBGModel,
    flat_pair: tuple[int, int],
    *,
    delta_ev: float,
    mesh_size: int,
    valley: int,
    skip_topology: bool,
    individual_delta_zero: bool,
) -> tuple[str, dict[str, object]]:
    if skip_topology:
        return "skipped", {}

    max_band = max(int(flat_pair[0]), int(flat_pair[1]))
    n_bands = max_band + 1
    if abs(delta_ev) < 5.0e-10 and not individual_delta_zero:
        result = model.topology_on_grid(
            mesh_size,
            (int(flat_pair[0]), int(flat_pair[1])),
            valley=valley,
            n_bands=n_bands,
        )
        return "cnp_pair_total", {"cnp_pair_total": _topology_payload(result)}

    valence = model.topology_on_grid(mesh_size, int(flat_pair[0]), valley=valley, n_bands=n_bands)
    conduction = model.topology_on_grid(mesh_size, int(flat_pair[1]), valley=valley, n_bands=n_bands)
    return "individual_flat_bands", {
        "valence": _topology_payload(valence),
        "conduction": _topology_payload(conduction),
        "flat_pair_total_from_individual": int(valence.rounded_chern_number + conduction.rounded_chern_number),
    }


def _compute_topologies_from_shared_grid(
    model: TMBGModel,
    flat_pair: tuple[int, int],
    *,
    delta_ev: float,
    mesh_size: int,
    valley: int,
    individual_delta_zero: bool,
    topology_method: str,
    sewn_orientation: str,
) -> tuple[str, dict[str, TopologyResult], dict[str, dict[str, object]]]:
    if abs(delta_ev) < 5.0e-10 and not individual_delta_zero:
        targets: tuple[tuple[str, int | tuple[int, int]], ...] = (("cnp_pair_total", flat_pair),)
        topology_mode = "cnp_pair_total"
    else:
        targets = (("valence", int(flat_pair[0])), ("conduction", int(flat_pair[1])))
        topology_mode = "individual_flat_bands"

    if topology_method == "sewn":
        sewn_targets: tuple[tuple[str, str, tuple[int, ...]], ...]
        if abs(delta_ev) < 5.0e-10 and not individual_delta_zero:
            sewn_targets = (("cnp_pair_total", "pair", (int(flat_pair[0]), int(flat_pair[1]))),)
        else:
            sewn_targets = (
                ("valence", "valence", (int(flat_pair[0]),)),
                ("conduction", "conduction", (int(flat_pair[1]),)),
            )

        attempts = (
            (int(mesh_size), (0.0, 0.0)),
            (int(mesh_size), (0.5 / float(mesh_size), 0.5 / float(mesh_size))),
            (int(2 * mesh_size), (0.0, 0.0)),
        )
        last_error: ValueError | None = None
        for trial_mesh, frac_shift in attempts:
            try:
                grid = compute_sewn_grid(
                    model,
                    mesh=trial_mesh,
                    valley=valley,
                    subset_pad=1,
                    frac_shift=frac_shift,
                )
                central_pair = (int(grid.iv), int(grid.ic))
                if tuple(int(index) for index in flat_pair) != central_pair:
                    raise ValueError(
                        f"Sewn topology expects the inferred flat pair {flat_pair} to match the central "
                        f"band pair {central_pair}."
                    )

                results: dict[str, TopologyResult] = {}
                diagnostics: dict[str, dict[str, object]] = {}
                for name, selector, band_indices in sewn_targets:
                    chern, min_link, berry = fhs_chern_sewn(
                        grid,
                        selector,
                        orientation=sewn_orientation,
                        return_berry=True,
                    )
                    results[name] = TopologyResult(
                        band_indices=tuple(int(index) for index in band_indices),
                        valley=int(valley),
                        k_grid_frac=grid.k_grid_frac,
                        berry_curvature=np.asarray(berry, dtype=float),
                        chern_number=float(chern),
                        rounded_chern_number=int(np.rint(chern)),
                    )
                    diagnostics[name] = {
                        "topology_method": "sewn",
                        "sewn_orientation": str(sewn_orientation),
                        "min_link": float(min_link),
                        "trial_mesh_size": int(trial_mesh),
                        "frac_shift": [float(frac_shift[0]), float(frac_shift[1])],
                    }
                return topology_mode, results, diagnostics
            except ValueError as exc:
                last_error = exc
                continue

        assert last_error is not None
        raise last_error

    if topology_method != "standard":
        raise ValueError(f"Unsupported topology_method={topology_method!r}")

    max_band = max(int(flat_pair[0]), int(flat_pair[1]))
    attempts = (
        (int(mesh_size), (0.0, 0.0)),
        (int(mesh_size), (0.5 / float(mesh_size), 0.5 / float(mesh_size))),
        (int(2 * mesh_size), (0.0, 0.0)),
    )
    last_error: ValueError | None = None
    for trial_mesh, frac_shift in attempts:
        grid_result = model.bands_on_grid(
            trial_mesh,
            valley=valley,
            n_bands=max_band + 1,
            return_eigenvectors=True,
            endpoint=False,
            frac_shift=frac_shift,
        )
        try:
            return topology_mode, {
                name: compute_topology_from_grid_result(grid_result, target, valley=valley)
                for name, target in targets
            }, {name: {"topology_method": "standard"} for name, _ in targets}
        except ValueError as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def _write_panel_arrays(
    panel_dir: Path,
    path_result: PathBandsResult,
    *,
    overlay_path_results: tuple[PathBandsResult, ...] = (),
    selected_band_indices: tuple[int, ...],
    flat_pair: tuple[int, int],
    delta_ev: float,
    chern_payload: dict[str, object],
    metadata: PanelComputation,
) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, object] = {
        "k_distance": np.asarray(path_result.path.kdist, dtype=float),
        "energies": np.asarray(path_result.energies[:, selected_band_indices], dtype=float),
        "kvec_nm_inv": np.stack(
            [
                np.asarray(path_result.path.kvec.real, dtype=float),
                np.asarray(path_result.path.kvec.imag, dtype=float),
            ],
            axis=-1,
        ),
        "band_indices": np.asarray(selected_band_indices, dtype=int),
        "flat_band_indices": np.asarray(flat_pair, dtype=int),
        "k_labels": np.asarray(path_result.path.labels, dtype=object),
    }
    for overlay_index, overlay_result in enumerate(overlay_path_results, start=1):
        arrays[f"overlay_{overlay_index}_energies"] = np.asarray(
            overlay_result.energies[:, selected_band_indices],
            dtype=float,
        )
        arrays[f"overlay_{overlay_index}_k_distance"] = np.asarray(overlay_result.path.kdist, dtype=float)
        arrays[f"overlay_{overlay_index}_kvec_nm_inv"] = np.stack(
            [
                np.asarray(overlay_result.path.kvec.real, dtype=float),
                np.asarray(overlay_result.path.kvec.imag, dtype=float),
            ],
            axis=-1,
        )
    np.savez_compressed(
        panel_dir / "bands_path.npz",
        **arrays,
    )

    write_json(
        panel_dir / "chern_numbers.json",
        {
            "delta_ev": float(delta_ev),
            "flat_band_indices": list(int(index) for index in flat_pair),
            "selected_band_indices": list(int(index) for index in selected_band_indices),
            "topology_mode": metadata.topology_mode,
            "chern": chern_payload,
            "expected": metadata.expected_payload,
            "status": metadata.status,
            "annotation": metadata.annotation,
        },
        sort_keys=False,
    )

    berry_fields: dict[str, np.ndarray | int | float] = {}
    for key, entry in chern_payload.items():
        if isinstance(entry, dict) and "band_indices" in entry:
            # Berry fields are large; they are stored when the TopologyResult is
            # available in memory through a side-channel below.
            continue
    if berry_fields:
        np.savez_compressed(panel_dir / "berry_curvature.npz", **berry_fields)


def _save_berry_curvature(panel_dir: Path, topology_results: dict[str, TopologyResult]) -> None:
    if not topology_results:
        return
    arrays: dict[str, object] = {}
    for name, result in topology_results.items():
        arrays[f"berry_curvature_{name}"] = np.asarray(result.berry_curvature, dtype=float)
        arrays[f"band_indices_{name}"] = np.asarray(result.band_indices, dtype=int)
        arrays[f"chern_number_{name}"] = float(result.chern_number)
        arrays[f"rounded_chern_number_{name}"] = int(result.rounded_chern_number)
    np.savez_compressed(panel_dir / "berry_curvature.npz", **arrays)


def _compute_panel(
    delta_ev: float,
    *,
    args: argparse.Namespace,
) -> tuple[TMBGBandPlotPanel, PanelComputation, PathBandsResult, dict[str, TopologyResult]]:
    params = TMBGParameters.full(interlayer_potential=float(delta_ev), staggered_potential=0.0)
    model = TMBGModel.from_config(args.theta_deg, n_shells=args.n_shells, params=params)
    if args.path_kind == "park-fig2":
        kpath = model.park_fig2_kpath(
            points_per_segment=args.points_per_segment,
            gamma_prime_choice=args.gamma_prime_choice,
        )
    else:
        kpath = model.standard_kpath(points_per_segment=args.points_per_segment)
    path_result = model.bands_along_path(
        kpath,
        valley=args.valley,
        n_bands=model.lattice.matrix_dim,
    )
    overlay_path_results: tuple[PathBandsResult, ...] = ()
    overlay_label = None
    primary_label = None
    if args.plot_valleys == "both":
        other_valley = -int(args.valley)
        overlay_path_results = (
            model.bands_along_path(
                kpath,
                valley=other_valley,
                n_bands=model.lattice.matrix_dim,
            ),
        )
        primary_label = f"valley {int(args.valley):+d}"
        overlay_label = f"valley {other_valley:+d}"
    flat_pair = infer_flat_band_indices(path_result.energies)
    selected_band_indices = select_flat_pair_window(
        path_result.energies.shape[1],
        flat_pair,
        args.bands_per_side,
    )
    flat_gap_mev, flat_gap_location = _path_gap_summary(path_result, flat_pair)

    topology_mode = "skipped"
    topology_method = "skipped" if args.skip_topology else str(args.topology_method)
    topology_results: dict[str, TopologyResult] = {}
    topology_diagnostics: dict[str, dict[str, object]] = {}
    chern_payload: dict[str, object] = {}
    if not args.skip_topology:
        topology_mode, topology_results, topology_diagnostics = _compute_topologies_from_shared_grid(
            model,
            flat_pair,
            delta_ev=float(delta_ev),
            mesh_size=int(args.topology_mesh_size),
            valley=int(args.valley),
            individual_delta_zero=bool(args.individual_delta_zero),
            topology_method=str(args.topology_method),
            sewn_orientation=str(args.sewn_orientation),
        )
        chern_payload.update(
            {
                name: _topology_payload(result, extra=topology_diagnostics.get(name))
                for name, result in topology_results.items()
            }
        )
        if "valence" in topology_results and "conduction" in topology_results:
            chern_payload["flat_pair_total_from_individual"] = int(
                topology_results["valence"].rounded_chern_number + topology_results["conduction"].rounded_chern_number
            )

    status, expected_payload = _chern_status(
        float(delta_ev),
        chern_payload,
        skip_topology=bool(args.skip_topology),
        individual_delta_zero=bool(args.individual_delta_zero),
        compare_paper_chern=bool(args.compare_paper_chern),
    )
    chern_line = _chern_annotation(
        float(delta_ev),
        chern_payload,
        skip_topology=bool(args.skip_topology),
        individual_delta_zero=bool(args.individual_delta_zero),
    )
    gap_line = f"flat_gap: {flat_gap_mev:.2f} meV at {flat_gap_location}"
    annotation = f"{gap_line}\n{chern_line}"
    metadata = PanelComputation(
        delta_ev=float(delta_ev),
        model_name=params.model_name,
        flat_band_indices=(int(flat_pair[0]), int(flat_pair[1])),
        selected_band_indices=tuple(int(index) for index in selected_band_indices),
        flat_gap_mev=float(flat_gap_mev),
        flat_gap_location=str(flat_gap_location),
        topology_method=topology_method,
        topology_mode=topology_mode,
        chern_payload=chern_payload,
        expected_payload=expected_payload,
        status=status,
        annotation=annotation,
    )
    panel = TMBGBandPlotPanel(
        label=_panel_label(float(delta_ev)),
        path_result=path_result,
        band_indices=metadata.selected_band_indices,
        flat_band_indices=metadata.flat_band_indices,
        annotation=annotation,
        primary_label=primary_label,
        overlay_path_results=overlay_path_results,
        overlay_label=overlay_label,
    )
    return panel, metadata, path_result, topology_results


def _compute_panel_worker(
    delta_ev: float,
    args: argparse.Namespace,
    worker_threads: int,
) -> tuple[TMBGBandPlotPanel, PanelComputation, PathBandsResult, dict[str, TopologyResult]]:
    _configure_worker_threads(worker_threads)
    return _compute_panel(delta_ev, args=args)


def _report_lines(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    panel_metadata: list[PanelComputation],
    start_time: str,
    end_time: str,
    elapsed: float,
) -> list[str]:
    lines = [
        "# tMBG Park 2020 Fig. 2 Chern Reproduction",
        "",
        "## Parameters",
        "",
        f"- `theta_deg = {args.theta_deg}`",
        f"- `n_shells = {args.n_shells}`",
        f"- `points_per_segment = {args.points_per_segment}`",
        f"- `topology_mesh_size = {args.topology_mesh_size}`",
        f"- `topology_method = {args.topology_method}`",
        f"- `sewn_orientation = {args.sewn_orientation}`",
        f"- `valley = {args.valley}`",
        f"- `plot_valleys = {args.plot_valleys}`",
        f"- `delta_values_ev = {list(float(value) for value in args.deltas_ev)}`",
        f"- `panel_workers = {args.panel_workers}`",
        f"- `worker_threads = {args.worker_threads}`",
        f"- `path_kind = {args.path_kind}`",
        f"- `gamma_prime_choice = {args.gamma_prime_choice}`",
        f"- `delta_zero_mode = {'individual' if args.individual_delta_zero else 'cnp_pair_total'}`",
        f"- `skip_topology = {bool(args.skip_topology)}`",
        f"- `compare_paper_chern = {bool(args.compare_paper_chern)}`",
        "",
        "## Chern Summary",
        "",
        "| Delta (eV) | Bands | Target | Expected | Observed | Max residual | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in panel_metadata:
        bands = f"{item.flat_band_indices[0]}, {item.flat_band_indices[1]}"
        expected = item.expected_payload if item.expected_payload else "n/a"
        observed = {}
        for key in ("cnp_pair_total", "valence", "conduction"):
            value = _rounded(item.chern_payload, key)
            if value is not None:
                observed[key] = value
        if "flat_pair_total_from_individual" in item.chern_payload:
            observed["total"] = item.chern_payload["flat_pair_total_from_individual"]
        residual = _max_integer_residual(item.chern_payload)
        residual_text = "n/a" if residual is None else f"{residual:.3e}"
        lines.append(
            f"| {item.delta_ev:+.3f} | {bands} | {item.topology_mode} | `{expected}` | `{observed or 'n/a'}` | {residual_text} | {item.status} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `fig2_chern_bands.png = {output_dir / 'fig2_chern_bands.png'}`",
            f"- `fig2_chern_bands.pdf = {output_dir / 'fig2_chern_bands.pdf'}`",
            f"- `run_metadata.json = {output_dir / 'run_metadata.json'}`",
            f"- `lattice_info.json = {output_dir / 'lattice_info.json'}`",
            "",
            "Each `delta_*mev/` directory contains `bands_path.npz`, `chern_numbers.json`, `berry_curvature.npz`, and a one-panel `bands_chern` figure.",
            "",
            "## Runtime",
            "",
            f"- `start_time = {start_time}`",
            f"- `end_time = {end_time}`",
            f"- `total_elapsed_sec = {elapsed:.6f}`",
            "",
        ]
    )
    return lines


def main() -> None:
    args = _parse_args()
    ensure_not_running_compute_on_login_node("tMBG Fig. 2 Chern band plot")

    deltas = tuple(float(value) for value in args.deltas_ev)
    panel_workers = _resolve_panel_workers(args, len(deltas))
    worker_threads = _resolve_worker_threads(args, panel_workers)
    args.panel_workers = panel_workers
    args.worker_threads = worker_threads
    _configure_worker_threads(worker_threads)

    start_time = current_timestamp()
    start_counter = perf_counter()
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    completed: dict[float, tuple[TMBGBandPlotPanel, PanelComputation, PathBandsResult, dict[str, TopologyResult]]] = {}
    reference_model: TMBGModel | None = None
    print(
        f"[parallel] panel_workers={panel_workers} worker_threads={worker_threads} "
        f"deltas={','.join(f'{value:+.6f}' for value in deltas)}",
        flush=True,
    )

    if panel_workers > 1 and len(deltas) > 1:
        with ProcessPoolExecutor(max_workers=panel_workers) as executor:
            futures = {
                executor.submit(_compute_panel_worker, delta_ev, args, worker_threads): delta_ev
                for delta_ev in deltas
            }
            for delta_ev in deltas:
                print(f"[panel] queued delta_ev={delta_ev:+.6f}", flush=True)
            for future in as_completed(futures):
                delta_ev = futures[future]
                print(f"[panel] collect delta_ev={delta_ev:+.6f}", flush=True)
                completed[delta_ev] = future.result()
    else:
        for delta_ev in deltas:
            print(f"[panel] start delta_ev={delta_ev:+.6f}", flush=True)
            completed[delta_ev] = _compute_panel(delta_ev, args=args)

    panels: list[TMBGBandPlotPanel] = []
    panel_metadata: list[PanelComputation] = []
    for delta_ev in deltas:
        panel, metadata, path_result, topology_results = completed[delta_ev]
        panels.append(panel)
        panel_metadata.append(metadata)
        if reference_model is None:
            reference_model = TMBGModel.from_config(
                args.theta_deg,
                n_shells=args.n_shells,
                params=TMBGParameters.full(interlayer_potential=delta_ev, staggered_potential=0.0),
            )

        panel_dir = output_dir / _delta_panel_dirname(delta_ev)
        _write_panel_arrays(
            panel_dir,
            path_result,
            overlay_path_results=panel.overlay_path_results,
            selected_band_indices=metadata.selected_band_indices,
            flat_pair=metadata.flat_band_indices,
            delta_ev=delta_ev,
            chern_payload=metadata.chern_payload,
            metadata=metadata,
        )
        _save_berry_curvature(panel_dir, topology_results)
        write_tmbg_paper_band_figure(
            panel_dir,
            (panel,),
            stem="bands_chern",
            title=f"Park 2020 Fig. 2, {panel.label}",
            ylim=(-0.100, 0.100),
        )
        print(
            f"[panel] done delta_ev={delta_ev:+.6f} status={metadata.status} "
            f"mode={metadata.topology_mode} bands={metadata.flat_band_indices}",
            flush=True,
        )

    if reference_model is not None:
        write_json(output_dir / "lattice_info.json", reference_model.lattice_summary(), sort_keys=False)
        write_tmbg_lattice_plot(
            output_dir,
            reference_model.lattice,
            title=f"tMBG moire reciprocal lattice, theta={args.theta_deg:.2f} deg",
        )

    valley_title = f"valleys={int(args.valley):+d},{-int(args.valley):+d}" if args.plot_valleys == "both" else f"valley={args.valley}"
    plot_paths = write_tmbg_paper_band_figure(
        output_dir,
        tuple(panels),
        stem="fig2_chern_bands",
        title=f"Park 2020 Fig. 2 tMBG, theta={args.theta_deg:.2f} deg, {valley_title}",
        ylim=(-0.100, 0.100),
    )

    elapsed = perf_counter() - start_counter
    end_time = current_timestamp()
    report_path = output_dir / "fig2_chern_report.md"
    report_path.write_text(
        "\n".join(
            _report_lines(
                args=args,
                output_dir=output_dir,
                panel_metadata=panel_metadata,
                start_time=start_time,
                end_time=end_time,
                elapsed=elapsed,
            )
        ),
        encoding="utf-8",
    )

    env = collect_runtime_environment()
    metadata = {
        "implementation": "python_tmbg",
        "runner_kind": "tmbg_fig2_chern_band_plot",
        "parameters": {
            "theta_deg": float(args.theta_deg),
            "n_shells": int(args.n_shells),
            "points_per_segment": int(args.points_per_segment),
            "topology_mesh_size": int(args.topology_mesh_size),
            "topology_method": str(args.topology_method),
            "sewn_orientation": str(args.sewn_orientation),
            "valley": int(args.valley),
            "plot_valleys": str(args.plot_valleys),
            "bands_per_side": int(args.bands_per_side),
            "deltas_ev": [float(value) for value in args.deltas_ev],
            "path_kind": str(args.path_kind),
            "gamma_prime_choice": str(args.gamma_prime_choice),
            "individual_delta_zero": bool(args.individual_delta_zero),
            "skip_topology": bool(args.skip_topology),
            "compare_paper_chern": bool(args.compare_paper_chern),
        },
        "runtime": {
            "start_time": start_time,
            "end_time": end_time,
            "total_elapsed_sec": float(elapsed),
            "environment": asdict(env),
        },
        "panels": [asdict(item) for item in panel_metadata],
        "artifacts": {
            "fig2_chern_bands_png": str(plot_paths["paper_band_plot_png"]),
            "fig2_chern_bands_pdf": str(plot_paths["paper_band_plot_pdf"]),
            "fig2_chern_report_md": str(report_path),
            "run_metadata_json": str(output_dir / "run_metadata.json"),
            "lattice_info_json": str(output_dir / "lattice_info.json"),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata, sort_keys=False)

    failed = [item for item in panel_metadata if item.status == "fail"]
    status = "fail" if failed else "pass"
    print(f"status={status}\tpanels={len(panel_metadata)}\tfailures={len(failed)}\ttotal_elapsed_sec={elapsed:.6f}")
    print(f"output_dir={output_dir}")
    print(f"fig2_chern_bands_png={plot_paths['paper_band_plot_png']}")
    print(f"fig2_chern_report_md={report_path}")


if __name__ == "__main__":
    main()
