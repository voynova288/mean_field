from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Iterable

import numpy as np

from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.htg import (
    HTGModel,
    HTGHartreeFockRun,
    HTGParams,
    InteractionParams,
    KWAN_2023_FERMI_VELOCITY_M_PER_S,
    KWAN_2023_TUNNELING_EV,
    classify_htg_strong_coupling_state,
    normalize_htg_init_mode,
    scan_htg_ground_state,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "HTG"
DEFAULT_INIT_MODES = ("d3b", "d3a", "fi", "flavor", "vp", "sp", "chern", "perturbed", "random")
DEFAULT_SEEDS = (1, 2)
DEFAULT_CASES = (
    "1.80:75",
    "1.60:85",
    "1.80:50",
    "1.95:40",
)


@dataclass(frozen=True)
class AnchorCase:
    theta_deg: float
    wAA_mev: float
    label: str


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer.")
    return values


def _parse_csv_strings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated mode.")
    return values


def _parse_cases(text: str) -> tuple[AnchorCase, ...]:
    cases: list[AnchorCase] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        pieces = item.split(":")
        if len(pieces) not in {2, 3}:
            raise argparse.ArgumentTypeError(
                "Cases must be comma-separated theta:wAA[:label] entries, for example 1.80:75:paper."
            )
        theta = float(pieces[0])
        waa = float(pieces[1])
        label = pieces[2] if len(pieces) == 3 and pieces[2] else f"theta{theta:.3f}_wAA{waa:.1f}"
        cases.append(AnchorCase(theta_deg=theta, wAA_mev=waa, label=label))
    if not cases:
        raise argparse.ArgumentTypeError("Expected at least one anchor case.")
    return tuple(cases)


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"htg_fig9b_exact_anchor_scan_{job_id}"
    else:
        stem = f"htg_fig9b_exact_anchor_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DEFAULT_OUTPUT_ROOT / stem


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an exact-anchor HTG Fig. 9b multi-init scan with final_error audit fields. "
            "For full scans, the parameter centers must be passed explicitly; plotting ticks or cell edges are not "
            "calculation points."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cases", type=_parse_cases, default=_parse_cases(",".join(DEFAULT_CASES)))
    parser.add_argument("--artifact-prefix", default="fig9b_exact_anchor_scan")
    parser.add_argument("--report-title", default="HTG Fig. 9b Exact-Anchor Scan")
    parser.add_argument("--w-ev", type=float, default=KWAN_2023_TUNNELING_EV)
    parser.add_argument("--fermi-velocity-m-per-s", type=float, default=KWAN_2023_FERMI_VELOCITY_M_PER_S)
    parser.add_argument("--n-shells", type=int, default=3)
    parser.add_argument("--nu", type=float, default=3.0)
    parser.add_argument("--epsilon-r", type=float, default=8.0)
    parser.add_argument("--d-sc-nm", type=float, default=25.0)
    parser.add_argument("--u-ev", type=float, default=0.0)
    parser.add_argument("--n-k", type=int, default=12)
    parser.add_argument("--g-shells", type=int, default=1)
    parser.add_argument("--projected-band-count", type=int, default=2)
    parser.add_argument("--finite-zero-limit", action="store_true")
    parser.add_argument("--zero-cutoff-nm-inv", type=float, default=1.0e-12)
    parser.add_argument("--init-modes", type=_parse_csv_strings, default=DEFAULT_INIT_MODES)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--precision", type=float, default=1.0e-6)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--oda-stall-threshold", type=float, default=0.0)
    parser.add_argument("--energy-degeneracy-tolerance-ev", type=float, default=1.0e-10)
    parser.add_argument(
        "--reproduction-mode",
        choices=("auto", "qualitative", "paper-level"),
        default="auto",
        help="Report mode label. Auto marks <300 HF candidates/grid as qualitative and >=300 as paper-level.",
    )
    parser.add_argument("--disable-numba", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved cases and output path.")
    return parser.parse_args()


def _format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.16g}"


def _grid_edges(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        step = 1.0
        return [float(arr[0] - step / 2.0), float(arr[0] + step / 2.0)]
    mids = 0.5 * (arr[:-1] + arr[1:])
    first = arr[0] - (mids[0] - arr[0])
    last = arr[-1] + (arr[-1] - mids[-1])
    return [float(value) for value in np.concatenate([[first], mids, [last]])]


def _compact_class_label(label: object) -> str:
    compact = str(label).strip().strip("[]").replace(" ", "")
    return compact or str(label)


def _run_final_error(run: HTGHartreeFockRun) -> float | None:
    if run.iter_err.size == 0:
        return None
    return float(run.iter_err[-1])


def _run_energy(run: HTGHartreeFockRun) -> float:
    return float(run.state.diagnostics.get("hf_energy", np.nan))


def _run_gap_mev(run: HTGHartreeFockRun) -> float:
    return 1000.0 * float(run.state.diagnostics.get("hf_gap", np.nan))


def _lower_remote_count(n_band: int) -> int:
    n_band = int(n_band)
    if n_band < 2 or n_band % 2 != 0:
        raise ValueError(f"projected band count must be an even integer >=2, got {n_band}")
    return (n_band - 2) // 2


def _occupied_flavor_counts(run: HTGHartreeFockRun) -> tuple[int, ...] | None:
    counts = run.state.occupation_counts
    if counts is None:
        return None
    return tuple(int(value) for value in counts)


def _conduction_bandwidth_mev(run: HTGHartreeFockRun) -> tuple[float | None, int | None]:
    """Return Wcond for the single central-band-occupied flavor, when defined.

    Fig. 9b at nu=+3 uses the first unoccupied self-consistent HF band in the
    flavor with exactly one central-band occupation. The structured seeds carry
    this as ``state.occupation_counts``; unconstrained random/perturbed runs do
    not, so Wcond is intentionally left undefined for those representatives.
    """

    counts = _occupied_flavor_counts(run)
    if counts is None:
        return None, None
    n_spin = int(run.state.n_spin)
    n_eta = int(run.state.n_eta)
    n_band = int(run.state.n_band)
    single_central_count = _lower_remote_count(n_band) + 1
    flavor_indices = [index for index, count in enumerate(counts) if int(count) == single_central_count]
    if len(flavor_indices) != 1:
        return None, None
    flavor_index = int(flavor_indices[0])
    ispin, ieta = np.unravel_index(flavor_index, (n_spin, n_eta), order="C")
    idx = np.arange(n_spin * n_eta * n_band, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    local_band = int(counts[flavor_index])
    if local_band < 0 or local_band >= n_band:
        return None, flavor_index
    band_index = int(idx[int(ispin), int(ieta), local_band])
    band = np.asarray(run.state.energies[band_index, :], dtype=float)
    return 1000.0 * float(np.max(band) - np.min(band)), flavor_index


def _classification_payload(run: HTGHartreeFockRun) -> dict[str, object]:
    classification = classify_htg_strong_coupling_state(
        run.state.density,
        n_spin=run.state.n_spin,
        n_eta=run.state.n_eta,
        n_band=run.state.n_band,
    )
    return classification.to_dict()


def _run_payload(case: AnchorCase, run: HTGHartreeFockRun) -> dict[str, object]:
    classification = _classification_payload(run)
    class_label = str(classification["class_label"])
    wcond, wcond_flavor = _conduction_bandwidth_mev(run)
    counts = _occupied_flavor_counts(run)
    return {
        "case_label": case.label,
        "theta_deg": float(case.theta_deg),
        "wAA_mev": float(case.wAA_mev),
        "init_mode": run.init_mode,
        "seed": int(run.seed),
        "converged": bool(run.converged),
        "exit_reason": run.exit_reason,
        "iterations": int(run.iterations),
        "final_error": _run_final_error(run),
        "final_energy_ev": _run_energy(run),
        "hf_gap_mev": _run_gap_mev(run),
        "class_label": class_label,
        "class_compact_label": _compact_class_label(class_label),
        "family": str(classification["family"]),
        "nu_z": float(classification["nu_z"]),
        "wcond_mev": wcond,
        "wcond_flavor_index": wcond_flavor,
        "occupied_flavor_counts": "" if counts is None else str(list(counts)),
    }


def _select_best_run(
    runs: Iterable[HTGHartreeFockRun],
    *,
    energy_tolerance_ev: float,
) -> tuple[HTGHartreeFockRun, bool, bool]:
    run_list = list(runs)
    converged_pool = [run for run in run_list if run.converged]
    pool = converged_pool if converged_pool else run_list
    selected_from_converged_pool = bool(converged_pool)
    min_energy = min(_run_energy(run) for run in pool)
    degenerate = [run for run in pool if abs(_run_energy(run) - min_energy) <= float(energy_tolerance_ev)]
    with_wcond = [run for run in degenerate if _conduction_bandwidth_mev(run)[0] is not None]
    if with_wcond:
        return min(with_wcond, key=_run_energy), selected_from_converged_pool, len(with_wcond) != len(degenerate)
    return min(degenerate, key=_run_energy), selected_from_converged_pool, False


def _write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_optional_float(value) if isinstance(value, float) else ("" if value is None else value)
                    for key in fieldnames
                    for value in (row.get(key),)
                }
            )


def _sorted_unique(values: Iterable[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _mode_label(requested: str, n_hf_candidates_per_grid: int) -> str:
    if requested == "paper-level":
        return "paper-level exhaustive seed mode"
    if requested == "qualitative":
        return "qualitative reproduction mode"
    if int(n_hf_candidates_per_grid) >= 300:
        return "paper-level exhaustive seed mode"
    return "qualitative reproduction mode"


def _grid_metadata(
    *,
    cases: tuple[AnchorCase, ...],
    parameters: dict[str, object],
    runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    theta_values = _sorted_unique(case.theta_deg for case in cases)
    waa_values = _sorted_unique(case.wAA_mev for case in cases)
    theta_edges = _grid_edges(theta_values)
    waa_edges = _grid_edges(waa_values)
    init_modes = [str(mode) for mode in parameters["init_modes"]]
    seeds = [int(seed) for seed in parameters["seeds"]]
    n_hf_candidates_per_grid = len(init_modes) * len(seeds)
    n_parameter_points = len(cases)
    normalized_modes = [normalize_htg_init_mode(mode) for mode in init_modes]
    wcond_array_shape = [len(waa_values), len(theta_values)]
    return {
        "figure": "Kwan Fig. 9(b)",
        "reproduction_mode": _mode_label(
            str(parameters.get("requested_reproduction_mode", "auto")),
            n_hf_candidates_per_grid,
        ),
        "nu": float(parameters["nu"]),
        "theta_grid_deg": theta_values,
        "wAA_grid_meV": waa_values,
        "theta_values_deg": theta_values,
        "wAA_values_meV": waa_values,
        "theta_center_values_deg": theta_values,
        "wAA_center_values_meV": waa_values,
        "theta_edge_values_deg": theta_edges,
        "wAA_edge_values_meV": waa_edges,
        "n_theta": len(theta_values),
        "n_wAA": len(waa_values),
        "n_parameter_points": n_parameter_points,
        "wcond_array_shape": wcond_array_shape,
        "wcond_array_axis_order": ["wAA_grid_meV rows ascending", "theta_grid_deg columns ascending"],
        "wcond_array_rows": "wAA_grid_meV",
        "wcond_array_columns": "theta_grid_deg",
        "mesh_note": (
            "Calculation points are the explicit center arrays theta_grid_deg and wAA_grid_meV. "
            "Major tick labels are plotting ticks only and cell edges are derived from adjacent centers."
        ),
        "calculation_points_are_cell_centers": True,
        "cell_edges_are_derived_for_plotting_only": True,
        "cell_edges_are_calculation_points": False,
        "parameter_grid_formula": "n_parameter_points = n_theta * n_wAA for this rectangular scan",
        "hf_initial_samples_per_grid_point": {
            "init_modes": init_modes,
            "normalized_init_modes": normalized_modes,
            "init_mode_aliases": {"d3a": "fb", "d3b": "sublattice"},
            "n_init_modes": len(init_modes),
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_hf_candidates_per_grid": n_hf_candidates_per_grid,
            "candidate_count_formula": "n_hf_candidates_per_grid = n_init_modes * n_seeds",
        },
        "n_total_hf_runs": n_parameter_points * n_hf_candidates_per_grid,
        "total_hf_run_formula": "n_total_hf_runs = n_parameter_points * n_hf_candidates_per_grid",
        "paper_seed_note": (
            "Kwan et al. report >300 initial seeds of different types per parameter in their HF phase-diagram "
            "search. Runs with fewer candidates should be described as qualitative reproduction mode, not an "
            "independent exhaustive global-minimum search."
        ),
        "translation_symmetry": "primitive-cell translation-invariant HF scan; no doubled/tripled TSB sectors",
        "initial_state_scope": (
            "strong-coupling, BM, perturbed, and random primitive-cell seeds only; not the full paper TSB search space"
        ),
        "kwan_parameters": {
            "vF_m_per_s": float(parameters["fermi_velocity_m_per_s"]),
            "wAB_meV": 1000.0 * float(parameters["w_ev"]),
            "wAA_meV": "scanned",
            "epsilon_r": float(parameters["epsilon_r"]),
            "d_sc_nm": float(parameters["d_sc_nm"]),
            "U_ev": float(parameters["u_ev"]),
            "interaction_scheme": "average",
            "pauli_twist": False,
            "system_size_for_phase_map": f"{int(parameters['n_k'])}x{int(parameters['n_k'])}",
            "fig9a_validation_size": "18x18",
            "drop_q0_coulomb": bool(parameters["drop_q0_coulomb"]),
        },
        "observable": {
            "name": "Wcond",
            "definition": "max_k E_lowest_unoccupied(k) - min_k E_lowest_unoccupied(k)",
            "units": "meV",
            "computed_over": "full 2D self-consistent HF mBZ grid",
            "not_computed_from": "high-symmetry path bands or path_band_gap_ev",
            "implementation_note": str(parameters["bandwidth_definition"]),
        },
        "runtime": runtime or {},
    }


def _print_sample_budget(metadata: dict[str, object]) -> None:
    samples = metadata["hf_initial_samples_per_grid_point"]
    print(
        "[grid] figure={figure} mode={mode} n_theta={n_theta} n_wAA={n_wAA} "
        "n_parameter_points={n_parameter_points}".format(
            figure=metadata["figure"],
            mode=metadata["reproduction_mode"],
            n_theta=metadata["n_theta"],
            n_wAA=metadata["n_wAA"],
            n_parameter_points=metadata["n_parameter_points"],
        ),
        flush=True,
    )
    print(f"[grid] theta_values_deg={metadata['theta_values_deg']}", flush=True)
    print(f"[grid] wAA_values_meV={metadata['wAA_values_meV']}", flush=True)
    print(f"[grid] theta_edge_values_deg={metadata['theta_edge_values_deg']}", flush=True)
    print(f"[grid] wAA_edge_values_meV={metadata['wAA_edge_values_meV']}", flush=True)
    print(f"[grid] wcond_array_shape={metadata['wcond_array_shape']}", flush=True)
    print(
        "[samples] n_init_modes={n_init_modes} n_seeds={n_seeds} "
        "n_hf_candidates_per_grid={n_hf_candidates_per_grid} n_total_hf_runs={n_total}".format(
            n_init_modes=samples["n_init_modes"],
            n_seeds=samples["n_seeds"],
            n_hf_candidates_per_grid=samples["n_hf_candidates_per_grid"],
            n_total=metadata["n_total_hf_runs"],
        ),
        flush=True,
    )
    print(f"[samples] init_modes={samples['init_modes']}", flush=True)
    print(f"[samples] seeds={samples['seeds']}", flush=True)


def _write_report(
    output_dir: Path,
    *,
    report_path: Path,
    report_title: str,
    selected_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    parameters: dict[str, object],
    runtime: dict[str, object],
    metadata: dict[str, object],
) -> None:
    finite_wcond = [float(row["wcond_mev"]) for row in selected_rows if row.get("wcond_mev") is not None]
    selected_converged = sum(bool(row["converged"]) for row in selected_rows)
    detail_converged = sum(bool(row["converged"]) for row in detail_rows)
    samples = metadata["hf_initial_samples_per_grid_point"]
    lines = [
        f"# {report_title}",
        "",
        "## Purpose",
        "",
        "This run advances the Fig. 9b reproduction audit by rerunning selected multi-init points with final_error recorded.",
        "",
        "The parameter mesh used here is our numerical sampling choice for reproducing Fig. 9(b); "
        "Kwan et al. do not prescribe a unique plotting mesh for that panel.",
        "",
        "## Runtime",
        "",
        f"- `hostname = {runtime.get('hostname', '')}`",
        f"- `slurm_job_id = {runtime.get('slurm_job_id', '')}`",
        f"- `elapsed_sec = {float(runtime.get('elapsed_sec', 0.0)):.3f}`",
        "",
        "## Parameters",
        "",
        f"- `nu = {parameters['nu']}`",
        f"- `n_k = {parameters['n_k']}`",
        f"- `g_shells = {parameters['g_shells']}`",
        f"- `projected_band_count = {parameters['projected_band_count']}`",
        f"- `epsilon_r = {parameters['epsilon_r']}`",
        f"- `wAB_meV = {1000.0 * float(parameters['w_ev']):.6g}`",
        f"- `d_sc_nm = {parameters['d_sc_nm']}`",
        f"- `U_ev = {parameters['u_ev']}`",
        f"- `init_modes = {parameters['init_modes']}`",
        f"- `seeds = {parameters['seeds']}`",
        f"- `reproduction_mode = {metadata['reproduction_mode']}`",
        "",
        "## Grid And Sample Budget",
        "",
        f"- Parameter grid samples: `{metadata['n_theta']} * {metadata['n_wAA']} = {metadata['n_parameter_points']}`",
        f"- Wcond array shape: `{metadata['wcond_array_shape']}` with rows=`wAA_grid_meV`, columns=`theta_grid_deg`",
        f"- theta centers: `{metadata['theta_grid_deg']}`",
        f"- wAA centers: `{metadata['wAA_grid_meV']}`",
        f"- theta cell edges: `{metadata['theta_edge_values_deg']}`",
        f"- wAA cell edges: `{metadata['wAA_edge_values_meV']}`",
        f"- HF initial samples per grid point: `{samples['n_init_modes']} * {samples['n_seeds']} = {samples['n_hf_candidates_per_grid']}`",
        f"- Total HF runs: `{metadata['n_parameter_points']} * {samples['n_hf_candidates_per_grid']} = {metadata['n_total_hf_runs']}`",
        f"- Mesh note: {metadata['mesh_note']}",
        f"- Paper seed note: {metadata['paper_seed_note']}",
        "",
        "## Observable",
        "",
        f"- `Wcond = {metadata['observable']['definition']}`",
        f"- computed over `{metadata['observable']['computed_over']}`",
        f"- not computed from `{metadata['observable']['not_computed_from']}`",
        "",
        "## Summary",
        "",
        f"- Selected rows converged: {selected_converged}/{len(selected_rows)}",
        f"- Per-run rows converged: {detail_converged}/{len(detail_rows)}",
        f"- Selected rows with finite Wcond: {len(finite_wcond)}/{len(selected_rows)}",
    ]
    if finite_wcond:
        lines.append(f"- Selected Wcond min/max: {min(finite_wcond):.6g} / {max(finite_wcond):.6g} meV")
    lines.extend(
        [
            "",
            "## Selected Grid Points",
            "",
            "| case | theta | wAA | class | family | Wcond meV | HF gap meV | final error | init | converged |",
            "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in selected_rows:
        requested_init = str(row.get("best_requested_init_mode", row["best_init_mode"]))
        normalized_init = str(row["best_init_mode"])
        init_label = requested_init if requested_init == normalized_init else f"{requested_init}/{normalized_init}"
        lines.append(
            "| {case_label} | {theta_deg:.6g} | {wAA_mev:.6g} | {class_label} | {family} | {wcond} | "
            "{gap} | {err} | {init_mode}:{seed} | {converged} |".format(
                case_label=row["case_label"],
                theta_deg=float(row["theta_deg"]),
                wAA_mev=float(row["wAA_mev"]),
                class_label=row["class_label"],
                family=row["family"],
                wcond=_format_optional_float(row.get("wcond_mev")),
                gap=_format_optional_float(row.get("hf_gap_mev")),
                err=_format_optional_float(row.get("final_error")),
                init_mode=init_label,
                seed=row["best_seed"],
                converged=row["converged"],
            )
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    parameters = {
        "cases": [{"theta_deg": case.theta_deg, "wAA_mev": case.wAA_mev, "label": case.label} for case in args.cases],
        "w_ev": float(args.w_ev),
        "fermi_velocity_m_per_s": float(args.fermi_velocity_m_per_s),
        "n_shells": int(args.n_shells),
        "nu": float(args.nu),
        "epsilon_r": float(args.epsilon_r),
        "d_sc_nm": float(args.d_sc_nm),
        "u_ev": float(args.u_ev),
        "n_k": int(args.n_k),
        "g_shells": int(args.g_shells),
        "projected_band_count": int(args.projected_band_count),
        "finite_zero_limit": bool(args.finite_zero_limit),
        "drop_q0_coulomb": bool(not args.finite_zero_limit),
        "zero_cutoff_nm_inv": float(args.zero_cutoff_nm_inv),
        "init_modes": list(args.init_modes),
        "seeds": [int(seed) for seed in args.seeds],
        "max_iter": int(args.max_iter),
        "precision": float(args.precision),
        "beta": float(args.beta),
        "oda_stall_threshold": float(args.oda_stall_threshold),
        "energy_degeneracy_tolerance_ev": float(args.energy_degeneracy_tolerance_ev),
        "requested_reproduction_mode": str(args.reproduction_mode),
        "bandwidth_definition": (
            "first unoccupied self-consistent HF eigenband of the single central-band-occupied flavor, "
            "computed over the full 2D mBZ grid"
        ),
    }
    preflight_metadata = _grid_metadata(cases=args.cases, parameters=parameters)

    if args.dry_run:
        print(f"output_dir={output_dir}")
        _print_sample_budget(preflight_metadata)
        for case in args.cases:
            print(f"case={case.theta_deg}:{case.wAA_mev}:{case.label}")
        return

    ensure_not_running_compute_on_login_node("HTG Fig. 9b exact-anchor scan")
    if args.disable_numba:
        os.environ["MEAN_FIELD_HF_DISABLE_NUMBA"] = "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = str(args.artifact_prefix).strip()
    if not artifact_prefix:
        raise SystemExit("--artifact-prefix must not be empty")
    selected_tsv = output_dir / f"{artifact_prefix}.tsv"
    detail_tsv = output_dir / f"{artifact_prefix}_run_details.tsv"
    json_path = output_dir / f"{artifact_prefix}.json"
    report_path = output_dir / f"{artifact_prefix}_report.md"
    grid_metadata_path = output_dir / "grid_metadata.json"

    _print_sample_budget(preflight_metadata)

    start = perf_counter()
    selected_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for case in args.cases:
        print(f"[case] theta={case.theta_deg:g} wAA_mev={case.wAA_mev:g} label={case.label}", flush=True)
        w_aa_ev = float(case.wAA_mev) / 1000.0
        kappa = float(w_aa_ev / float(args.w_ev))
        params = HTGParams(
            fermi_velocity_m_per_s=args.fermi_velocity_m_per_s,
            w_ev=args.w_ev,
            kappa=kappa,
            zeta_rad=0.0,
            model_name="kwan2023_hf",
        )
        interaction = InteractionParams(
            epsilon_r=args.epsilon_r,
            d_sc_nm=args.d_sc_nm,
            U_ev=args.u_ev,
            n_k=args.n_k,
            g_shells=args.g_shells,
            finite_zero_limit=args.finite_zero_limit,
            zero_cutoff_nm_inv=args.zero_cutoff_nm_inv,
        )
        model = HTGModel.from_config(case.theta_deg, n_shells=args.n_shells, params=params)
        scan = scan_htg_ground_state(
            model,
            interaction,
            nu=args.nu,
            init_modes=args.init_modes,
            seeds=args.seeds,
            beta=args.beta,
            max_iter=args.max_iter,
            precision=args.precision,
            oda_stall_threshold=args.oda_stall_threshold,
            projected_band_count=args.projected_band_count,
            use_numba=False if args.disable_numba else None,
        )
        run_requests = [(mode, int(seed)) for mode in args.init_modes for seed in args.seeds]
        if len(run_requests) != len(scan.runs):
            raise RuntimeError(f"expected {len(run_requests)} run requests, got {len(scan.runs)} scan runs")
        requested_init_by_run = {id(run): requested_mode for run, (requested_mode, _) in zip(scan.runs, run_requests)}

        for run, (requested_mode, _) in zip(scan.runs, run_requests):
            detail = _run_payload(case, run)
            detail["requested_init_mode"] = requested_mode
            detail_rows.append(detail)
        best, selected_from_converged_pool, selected_with_wcond_tiebreak = _select_best_run(
            scan.runs,
            energy_tolerance_ev=args.energy_degeneracy_tolerance_ev,
        )
        selected = _run_payload(case, best)
        selected.update(
            {
                "kappa": kappa,
                "wAB_mev": 1000.0 * float(args.w_ev),
                "epsilon_r": float(args.epsilon_r),
                "d_sc_nm": float(args.d_sc_nm),
                "U_ev": float(args.u_ev),
                "n_k": int(args.n_k),
                "g_shells": int(args.g_shells),
                "projected_band_count": int(args.projected_band_count),
                "n_hf_candidates": len(args.init_modes) * len(args.seeds),
                "seed_mode_count": len(args.init_modes),
                "seed_count": len(args.seeds),
                "init_modes": str(list(args.init_modes)),
                "seeds": str([int(seed) for seed in args.seeds]),
                "best_init_mode": best.init_mode,
                "best_requested_init_mode": requested_init_by_run.get(id(best), best.init_mode),
                "best_seed": int(best.seed),
                "strong_coupling_class": selected["class_label"],
                "scan_run_count": len(scan.runs),
                "converged_run_count": sum(1 for run in scan.runs if run.converged),
                "selected_from_converged_pool": selected_from_converged_pool,
                "selected_with_wcond_tiebreak": selected_with_wcond_tiebreak,
            }
        )
        selected_rows.append(selected)

    runtime = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "elapsed_sec": float(perf_counter() - start),
    }
    metadata = _grid_metadata(cases=args.cases, parameters=parameters, runtime=runtime)

    selected_fields = (
        "case_label",
        "theta_deg",
        "wAA_mev",
        "wAB_mev",
        "kappa",
        "epsilon_r",
        "d_sc_nm",
        "U_ev",
        "n_k",
        "g_shells",
        "projected_band_count",
        "n_hf_candidates",
        "seed_mode_count",
        "seed_count",
        "init_modes",
        "seeds",
        "converged",
        "exit_reason",
        "iterations",
        "final_error",
        "final_energy_ev",
        "hf_gap_mev",
        "class_label",
        "class_compact_label",
        "strong_coupling_class",
        "family",
        "nu_z",
        "wcond_mev",
        "wcond_flavor_index",
        "occupied_flavor_counts",
        "best_init_mode",
        "best_requested_init_mode",
        "best_seed",
        "scan_run_count",
        "converged_run_count",
        "selected_from_converged_pool",
        "selected_with_wcond_tiebreak",
    )
    detail_fields = (
        "case_label",
        "theta_deg",
        "wAA_mev",
        "requested_init_mode",
        "init_mode",
        "seed",
        "converged",
        "exit_reason",
        "iterations",
        "final_error",
        "final_energy_ev",
        "hf_gap_mev",
        "class_label",
        "class_compact_label",
        "family",
        "nu_z",
        "wcond_mev",
        "wcond_flavor_index",
        "occupied_flavor_counts",
    )
    _write_tsv(selected_tsv, selected_rows, selected_fields)
    _write_tsv(detail_tsv, detail_rows, detail_fields)
    write_json(grid_metadata_path, metadata)
    write_json(
        json_path,
        {
            "artifacts": {
                "tsv": str(selected_tsv),
                "run_details_tsv": str(detail_tsv),
                "report": str(report_path),
                "grid_metadata": str(grid_metadata_path),
            },
            "grid_metadata": metadata,
            "parameters": parameters,
            "runtime": runtime,
            "rows": selected_rows,
            "run_details": detail_rows,
        },
    )
    _write_report(
        output_dir,
        report_path=report_path,
        report_title=str(args.report_title),
        selected_rows=selected_rows,
        detail_rows=detail_rows,
        parameters=parameters,
        runtime=runtime,
        metadata=metadata,
    )
    print(f"[done] output_dir={output_dir}", flush=True)
    print(f"selected_tsv={selected_tsv}", flush=True)
    print(f"run_details_tsv={detail_tsv}", flush=True)


if __name__ == "__main__":
    main()
