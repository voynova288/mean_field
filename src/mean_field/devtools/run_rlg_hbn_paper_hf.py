from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from time import perf_counter

import numpy as np

from mean_field.core.hf import summarize_hf_state_archive, validate_hf_archive_shapes
from mean_field.devtools._runtime import (
    complex_to_pairs as _complex_to_pairs,
    ensure_not_running_compute_on_login_node,
    parse_csv_floats as _parse_csv_floats,
    parse_csv_ints as _parse_csv_ints,
    parse_csv_strings as _parse_csv_strings,
    write_json,
)
from mean_field.systems.RnG_hBN import (
    RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
    RLGhBNInteractionParams,
    RLGhBNModel,
    RLGhBNParams,
    load_or_build_layer_overlap_blocks,
    load_or_build_projected_basis,
    load_or_solve_screening,
    normalize_rlg_hbn_init_mode,
    run_rlg_hbn_hartree_fock,
    screening_result_to_dict,
    table_ii_moire_parameters,
    update_cache_manifest_file,
)
from mean_field.systems.RnG_hBN.hf import RLG_HBN_FORM_FACTOR_CONVENTION_VERSION


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "RnG_hBN"


PAPER_CONFIGS = {
    "fig5": {
        "description": "2312.11617v1 Fig. 5 HF band-structure source states",
        "layer_count": 5,
        "theta_deg": 0.77,
        "shell_count": 4,
        "xi_values": (1,),
        "v_values_mev": (40.0, 48.0, 56.0, 64.0),
        "epsilon_r": 6.25,
        "gate_distance_nm": 10.0,
        "active_valence_bands": 4,
        "active_conduction_bands": 4,
        "k_mesh_size": 12,
        "interaction_cutoff_q1": 3.0,
        "nu": 1.0,
        "scheme": "average",
        "use_screened_basis": True,
    },
    "fig6": {
        "description": "2312.11617v1 Fig. 6 HF detail source states",
        "layer_count": 5,
        "theta_deg": 0.77,
        "shell_count": 4,
        "xi_values": (0, 1),
        "v_values_mev": (64.0,),
        "epsilon_r": 5.0,
        "gate_distance_nm": 10.0,
        "active_valence_bands": 3,
        "active_conduction_bands": 3,
        "k_mesh_size": 18,
        "interaction_cutoff_q1": 3.0,
        "nu": 1.0,
        "scheme": "average",
        "use_screened_basis": True,
    },
}


def _parse_run_specs(text: str) -> tuple[tuple[str, int], ...]:
    specs: list[tuple[str, int]] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise argparse.ArgumentTypeError(
                f"Expected run spec entries as init_mode:seed, got {stripped!r}."
            )
        init_mode, seed_text = stripped.split(":", 1)
        init_mode = init_mode.strip()
        seed_text = seed_text.strip()
        if not init_mode:
            raise argparse.ArgumentTypeError(f"Missing init mode in run spec {stripped!r}.")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid seed in run spec {stripped!r}.") from exc
        specs.append((init_mode, seed))
    if not specs:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated run spec.")
    return tuple(specs)


DETERMINISTIC_INIT_MODES = {"bm"}


def rlg_hbn_run_specs_for_modes(
    init_modes: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    specs: list[tuple[str, int]] = []
    for init_mode in init_modes:
        mode = str(init_mode)
        mode_seeds = seeds
        if mode.strip().lower() in DETERMINISTIC_INIT_MODES and len(mode_seeds) > 1:
            mode_seeds = mode_seeds[:1]
        for seed in mode_seeds:
            specs.append((mode, int(seed)))
    return tuple(specs)


def default_rlg_hbn_run_specs(paper_target: str) -> tuple[tuple[str, int], ...]:
    if paper_target == "fig6":
        return (
            ("flavor", 1),
            ("flavor", 2),
            ("bm", 1),
            ("perturbed", 1),
            ("perturbed", 2),
            ("perturbed", 3),
            ("perturbed", 4),
            ("random", 1),
            ("random", 2),
            ("random", 3),
            ("random", 4),
        )
    return rlg_hbn_run_specs_for_modes(("flavor", "bm", "perturbed"), (1,))


def _serialize_run_specs(run_specs: tuple[tuple[str, int], ...]) -> list[dict[str, object]]:
    return [{"init_mode": str(init_mode), "seed": int(seed)} for init_mode, seed in run_specs]


def _run_specs_from_config(config: dict[str, object]) -> tuple[tuple[str, int], ...]:
    raw_specs = config.get("run_specs")
    if raw_specs is None:
        return rlg_hbn_run_specs_for_modes(
            tuple(str(mode) for mode in config["init_modes"]),
            tuple(int(seed) for seed in config["seeds"]),
        )
    specs: list[tuple[str, int]] = []
    if not isinstance(raw_specs, (list, tuple)):
        raise TypeError("run_specs must be a list of {init_mode, seed} entries.")
    for raw in raw_specs:
        if isinstance(raw, dict):
            specs.append((str(raw["init_mode"]), int(raw["seed"])))
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            specs.append((str(raw[0]), int(raw[1])))
        else:
            raise TypeError(f"Invalid run spec entry: {raw!r}")
    if not specs:
        raise ValueError("run_specs must not be empty.")
    return tuple(specs)


def _preflight_run_specs(config: dict[str, object]) -> dict[str, object]:
    """Validate cheap run-spec invariants before building screening/basis/overlap caches."""

    specs = _run_specs_from_config(config)
    validated: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for init_mode, seed in specs:
        try:
            normalized = normalize_rlg_hbn_init_mode(str(init_mode))
        except ValueError as exc:
            errors.append({"init_mode": str(init_mode), "seed": int(seed), "error": str(exc)})
            continue
        validated.append(
            {
                "init_mode": str(init_mode),
                "normalized_init_mode": str(normalized),
                "seed": int(seed),
            }
        )
    if errors:
        bad = ", ".join(f"{entry['init_mode']}:{entry['seed']}" for entry in errors)
        details = "; ".join(str(entry["error"]) for entry in errors)
        raise ValueError(
            "Invalid RLG/hBN HF run_specs before expensive setup: "
            f"{bad}. {details}. See run_preflight_failure.json for details."
        )

    active_valence = int(config["active_valence_bands"])
    active_conduction = int(config["active_conduction_bands"])
    k_mesh_size = int(config["k_mesh_size"])
    if active_valence < 0 or active_conduction < 0:
        raise ValueError(
            "active_valence_bands and active_conduction_bands must be non-negative; "
            f"got {active_valence=} {active_conduction=}"
        )
    if active_valence + active_conduction <= 0:
        raise ValueError("At least one active band is required for RLG/hBN HF.")
    if k_mesh_size <= 0:
        raise ValueError(f"k_mesh_size must be positive, got {k_mesh_size}")

    return {
        "status": "ok",
        "run_specs": validated,
        "active_valence_bands": active_valence,
        "active_conduction_bands": active_conduction,
        "k_mesh_size": k_mesh_size,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-config projected HF source states for R5G/hBN Figs. 5 and 6."
    )
    parser.add_argument("--paper-target", choices=tuple(PAPER_CONFIGS), default="fig5")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "cache")
    parser.add_argument("--cache-policy", choices=("reuse", "refresh", "off"), default="reuse")
    parser.add_argument("--screening-solver", choices=("grid", "fixed_point"), default="grid")
    parser.add_argument("--screening-u-min-mev", type=float, default=-100.0)
    parser.add_argument("--screening-u-max-mev", type=float, default=200.0)
    parser.add_argument("--screening-u-grid-points", type=int, default=121)
    parser.add_argument("--skip-screening-check", action="store_true")
    parser.add_argument("--v-values-mev", type=_parse_csv_floats, default=None)
    parser.add_argument("--xi-values", type=_parse_csv_ints, default=None)
    parser.add_argument("--epsilon-r", type=float, default=None)
    parser.add_argument("--gate-distance-nm", type=float, default=None)
    parser.add_argument("--scheme", choices=("average", "cn"), default=None)
    parser.add_argument("--interaction-cutoff-q1", type=float, default=None)
    parser.add_argument(
        "--hbn-moire-scale",
        "--kappa-hbn",
        type=float,
        default=1.0,
        help="Scale the Table-II hBN moire potential amplitudes V0,V1. Use 0 for the κ_hBN=0 branch.",
    )
    parser.add_argument("--active-valence-bands", type=int, default=None)
    parser.add_argument("--active-conduction-bands", type=int, default=None)
    parser.add_argument("--k-mesh-size", type=int, default=None)
    parser.add_argument("--init-modes", type=_parse_csv_strings, default=None)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=None)
    parser.add_argument(
        "--run-specs",
        type=_parse_run_specs,
        default=None,
        help="Comma-separated exact init_mode:seed list. Overrides the init-modes x seeds expansion.",
    )
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--precision", type=float, default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--oda-stall-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Write full latest-checkpoint archives every N HF iterations; trace/progress logs remain per iteration.",
    )
    parser.add_argument("--screening-mesh-size", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the resolved paper configuration without building bases or running HF.",
    )
    return parser.parse_args()


def _default_output_dir(paper_target: str) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"rlg_hbn_{paper_target}_hf_paper_{job_id}"
    else:
        stem = f"rlg_hbn_{paper_target}_hf_paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DEFAULT_OUTPUT_ROOT / stem


def _atomic_write_json(path: Path, payload: object, *, sort_keys: bool = True) -> None:
    write_json(path, payload, sort_keys=sort_keys)


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _git_commit_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _append_progress_event(panel_dir: Path, payload: dict[str, object]) -> None:
    event = {"timestamp": _timestamp(), **payload}
    _atomic_write_json(panel_dir / "hf_progress.json", event)
    with (panel_dir / "hf_progress_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _run_key(init_mode: str, seed: int) -> str:
    return f"{str(init_mode)}_seed{int(seed)}"


def _trace_from_arrays(
    *,
    iter_energy: np.ndarray | None = None,
    iter_err: np.ndarray | None = None,
    iter_oda: np.ndarray | None = None,
) -> dict[str, list[float] | list[int]]:
    energy = [] if iter_energy is None else [float(value) for value in np.asarray(iter_energy, dtype=float).reshape(-1)]
    err = [] if iter_err is None else [float(value) for value in np.asarray(iter_err, dtype=float).reshape(-1)]
    oda = [] if iter_oda is None else [float(value) for value in np.asarray(iter_oda, dtype=float).reshape(-1)]
    n = max(len(energy), len(err), len(oda))
    return {
        "iteration": list(range(1, n + 1)),
        "energy_mev": energy,
        "err": err,
        "oda": oda,
    }


def _trace_arrays(trace: dict[str, list[float] | list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(trace.get("energy_mev", []), dtype=float),
        np.asarray(trace.get("err", []), dtype=float),
        np.asarray(trace.get("oda", []), dtype=float),
    )


def _save_state_archive(
    path: Path,
    run,
    trace: dict[str, list[float] | list[int]],
    *,
    cache_metadata: dict[str, object] | None = None,
    ) -> None:
    iter_energy, iter_err, iter_oda = _trace_arrays(trace)
    payload = {
        "density": np.asarray(run.state.density, dtype=np.complex128),
        "hamiltonian": np.asarray(run.state.hamiltonian, dtype=np.complex128),
        "h0": np.asarray(run.state.h0, dtype=np.complex128),
        "energies_mev": np.asarray(run.state.energies, dtype=float),
        "reference_density": np.asarray(run.state.reference_density, dtype=np.complex128),
        "nu": np.asarray([float(run.state.nu)], dtype=float),
        "active_valence_bands": np.asarray([int(run.state.active_valence_bands)], dtype=int),
        "scheme": np.asarray(str(run.state.scheme)),
        "n_spin": np.asarray([int(run.state.n_spin)], dtype=int),
        "n_eta": np.asarray([int(run.state.n_eta)], dtype=int),
        "n_band": np.asarray([int(run.state.n_band)], dtype=int),
        "occupation_counts": np.asarray(
            [] if run.state.occupation_counts is None else tuple(int(v) for v in run.state.occupation_counts),
            dtype=int,
        ),
        "mu_mev": np.asarray([float(run.state.mu)], dtype=float),
        "kvec_nm_inv": _complex_to_pairs(run.basis_data.kvec),
        "k_grid_frac": np.asarray(run.basis_data.k_grid_frac, dtype=float),
        "band_energies_mev": np.asarray(run.basis_data.band_energies, dtype=float),
        "active_band_indices": np.asarray(run.basis_data.active_band_indices, dtype=int),
        "flat_band_indices": np.asarray(run.basis_data.flat_band_indices, dtype=int),
        "iter_energy_mev": iter_energy,
        "iter_err": iter_err,
        "iter_oda": iter_oda,
    }
    if cache_metadata:
        for key, value in cache_metadata.items():
            if isinstance(value, (str, Path)):
                payload[key] = np.asarray(str(value))
            elif value is None:
                payload[key] = np.asarray("")
            else:
                payload[key] = np.asarray(value)
    _atomic_savez(path, **payload)


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    state,
    basis_data,
    trace: dict[str, list[float] | list[int]],
    init_mode: str,
    seed: int,
    iteration: int,
    energy: float,
    err: float,
    oda: float,
) -> None:
    iter_energy, iter_err, iter_oda = _trace_arrays(trace)
    _atomic_savez(
        checkpoint_dir / "hf_checkpoint_latest.npz",
        density=np.asarray(state.density, dtype=np.complex128),
        hamiltonian=np.asarray(state.hamiltonian, dtype=np.complex128),
        h0=np.asarray(state.h0, dtype=np.complex128),
        energies_mev=np.asarray(state.energies, dtype=float),
        reference_density=np.asarray(state.reference_density, dtype=np.complex128),
        nu=np.asarray([float(state.nu)], dtype=float),
        active_valence_bands=np.asarray([int(state.active_valence_bands)], dtype=int),
        scheme=np.asarray(str(state.scheme)),
        n_spin=np.asarray([int(state.n_spin)], dtype=int),
        n_eta=np.asarray([int(state.n_eta)], dtype=int),
        n_band=np.asarray([int(state.n_band)], dtype=int),
        occupation_counts=np.asarray(
            [] if state.occupation_counts is None else tuple(int(v) for v in state.occupation_counts),
            dtype=int,
        ),
        mu_mev=np.asarray([float(state.mu)], dtype=float),
        kvec_nm_inv=_complex_to_pairs(basis_data.kvec),
        k_grid_frac=np.asarray(basis_data.k_grid_frac, dtype=float),
        active_band_indices=np.asarray(basis_data.active_band_indices, dtype=int),
        zero_literal_q0_fock=np.asarray(
            [_env_flag_enabled("MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK", default=False)],
            dtype=bool,
        ),
        iteration=np.asarray([int(iteration)], dtype=int),
        iter_energy_mev=iter_energy,
        iter_err=iter_err,
        iter_oda=iter_oda,
    )
    _atomic_write_json(
        checkpoint_dir / "hf_checkpoint_latest.json",
        {
            "init_mode": str(init_mode),
            "seed": int(seed),
            "iteration": int(iteration),
            "energy_mev": float(energy),
            "err": float(err),
            "oda": float(oda),
            "checkpoint_npz": str(checkpoint_dir / "hf_checkpoint_latest.npz"),
            "trace": trace,
        },
    )


def _load_trace_json(path: Path) -> dict[str, list[float] | list[int]]:
    if not path.exists():
        return _trace_from_arrays()
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace = payload.get("trace", {})
    if not isinstance(trace, dict):
        return _trace_from_arrays()
    return {
        "iteration": [int(value) for value in trace.get("iteration", [])],
        "energy_mev": [float(value) for value in trace.get("energy_mev", [])],
        "err": [float(value) for value in trace.get("err", [])],
        "oda": [float(value) for value in trace.get("oda", [])],
    }


def _load_archive_density(path: Path, expected_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, list[float] | list[int]]]:
    summary = summarize_hf_state_archive(path)
    validate_hf_archive_shapes(summary)
    if summary.density_shape != tuple(int(value) for value in expected_shape):
        raise ValueError(
            f"Checkpoint density shape {summary.density_shape} does not match current basis {expected_shape}"
        )
    with np.load(path, allow_pickle=False) as archive:
        density = np.asarray(archive["density"], dtype=np.complex128)
        return density, _trace_from_arrays(
            iter_energy=archive["iter_energy_mev"] if "iter_energy_mev" in archive.files else None,
            iter_err=archive["iter_err"] if "iter_err" in archive.files else None,
            iter_oda=archive["iter_oda"] if "iter_oda" in archive.files else None,
        )


def _run_payload(run, trace: dict[str, list[float] | list[int]] | None = None) -> dict[str, object]:
    iterations = int(run.iterations)
    final_error = float(run.iter_err[-1]) if run.iter_err.size else None
    if trace is not None:
        trace_err = trace.get("err", [])
        iterations = len(trace_err)
        final_error = float(trace_err[-1]) if trace_err else final_error
    return {
        "init_mode": run.init_mode,
        "seed": int(run.seed),
        "converged": bool(run.converged),
        "exit_reason": run.exit_reason,
        "iterations": int(iterations),
        "final_error": final_error,
        "final_energy_mev": float(run.state.diagnostics.get("hf_energy", np.nan)),
        "hf_gap_mev": float(run.state.diagnostics.get("hf_gap", np.nan)),
        "filling": float(run.state.diagnostics.get("filling", np.nan)),
        "projector_idempotency_residual": float(
            run.state.diagnostics.get("projector_idempotency_residual", np.nan)
        ),
        "density_hermitian_residual": float(run.state.diagnostics.get("density_hermitian_residual", np.nan)),
        "hamiltonian_hermitian_residual": float(
            run.state.diagnostics.get("hamiltonian_hermitian_residual", np.nan)
        ),
    }


def _panel_name(*, xi: int, v_mev: float) -> str:
    return f"xi{int(xi)}_V{int(round(float(v_mev))):03d}meV"


def _rlg_hbn_params_with_moire_scale(
    *,
    layer_count: int,
    xi: int,
    displacement_field_mev: float,
    hbn_moire_scale: float,
) -> RLGhBNParams:
    """Build RLG/hBN parameters for the paper κ_hBN branch.

    κ_hBN scales only the hBN moire potential amplitudes. The lattice mismatch
    still defines the moire reciprocal lattice, so κ_hBN=0 sets V0=V1=0 in the
    same folded basis rather than changing the moire cell.
    """

    scale = float(hbn_moire_scale)
    table_v0, table_v1, table_phase_deg = table_ii_moire_parameters(int(layer_count), int(xi))
    return RLGhBNParams(
        layer_count=int(layer_count),
        xi=int(xi),
        displacement_field_mev=float(displacement_field_mev),
        moire_v0_mev=scale * float(table_v0),
        moire_v1_mev=scale * float(table_v1),
        moire_phase_deg=float(table_phase_deg),
    )


def _completed_run_summary(run_dir: Path, *, max_iter: int) -> dict[str, object] | None:
    summary_path = run_dir / "hf_run_summary.json"
    archive_path = run_dir / "hf_run_state.npz"
    if not summary_path.exists() or not archive_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if bool(summary.get("converged", False)):
        return summary
    if int(summary.get("iterations", 0)) >= int(max_iter):
        return summary
    return None


def _write_panel_convergence(
    panel_dir: Path,
    *,
    panel: str,
    panel_start: float,
    run_payloads: list[dict[str, object]],
    screening: object,
    cache_metadata: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not run_payloads:
        return None
    best = min(run_payloads, key=lambda payload: float(payload.get("final_energy_mev", np.inf)))
    convergence_payload = {
        "panel": panel,
        "elapsed_sec": float(perf_counter() - panel_start),
        "runs": run_payloads,
        "best": best,
        "screening": screening,
    }
    if cache_metadata:
        convergence_payload.update(cache_metadata)
    _atomic_write_json(panel_dir / "hf_convergence.json", convergence_payload)
    selected_archive = panel_dir / "runs" / _run_key(str(best["init_mode"]), int(best["seed"])) / "hf_run_state.npz"
    if selected_archive.exists():
        shutil.copy2(selected_archive, panel_dir / "hf_ground_state.npz")
    return convergence_payload


def _run_panel_with_incremental_outputs(
    *,
    output_dir: Path,
    cache_dir: Path,
    cache_policy: str,
    panel_dir: Path,
    panel: str,
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    config: dict[str, object],
    panel_start: float,
) -> tuple[dict[str, object], Path]:
    _append_progress_event(panel_dir, {"stage": "screening_start", "panel": panel})
    print(f"[stage] {panel} screening_start", flush=True)
    screening_result = None
    screening_payload = None
    screening_cache_key = ""
    screening_cache_hit = False
    if interaction.use_screened_basis:
        screening_cache = load_or_solve_screening(
            model,
            interaction,
            cache_dir=cache_dir,
            cache_policy=cache_policy,
            solver=str(config["screening_solver"]),
            mesh_size=config["screening_mesh_size"],
            u_min_mev=float(config["screening_u_min_mev"]),
            u_max_mev=float(config["screening_u_max_mev"]),
            n_grid=int(config["screening_u_grid_points"]),
            root_tolerance_mev=1.0e-5,
        )
        screening_result = screening_cache.value
        screening_cache_key = str(screening_cache.key)
        screening_cache_hit = bool(screening_cache.hit)
        screening_payload = screening_result_to_dict(screening_result)  # type: ignore[arg-type]
        _atomic_write_json(panel_dir / "screening_result.json", screening_payload)
        update_cache_manifest_file(
            output_dir / "cache_manifest.json",
            cache_dir=cache_dir,
            kind="screening",
            key=screening_cache_key,
            hit=screening_cache_hit,
            path=screening_cache.path,
            panel=panel,
        )
        if screening_cache_hit:
            print(f"[cache-hit] screening {screening_cache_key}", flush=True)
        else:
            print(f"[cache-miss] screening {screening_cache_key}", flush=True)
    _append_progress_event(
        panel_dir,
        {
            "stage": "screening_done",
            "panel": panel,
            "screening": screening_payload,
            "cache_key": screening_cache_key,
            "cache_hit": screening_cache_hit,
        },
    )

    _append_progress_event(panel_dir, {"stage": "basis_start", "panel": panel})
    print(f"[stage] {panel} basis_start", flush=True)
    basis_cache = load_or_build_projected_basis(
        model,
        interaction,
        cache_dir=cache_dir,
        cache_policy=cache_policy,
        mesh_size=int(config["k_mesh_size"]),
        screening=screening_result,  # type: ignore[arg-type]
        screening_solver=str(config["screening_solver"]),
        screening_mesh_size=config["screening_mesh_size"],
        screening_u_min_mev=float(config["screening_u_min_mev"]),
        screening_u_max_mev=float(config["screening_u_max_mev"]),
        screening_u_grid_points=int(config["screening_u_grid_points"]),
    )
    basis_data = basis_cache.value
    update_cache_manifest_file(
        output_dir / "cache_manifest.json",
        cache_dir=cache_dir,
        kind="basis",
        key=str(basis_cache.key),
        hit=bool(basis_cache.hit),
        path=basis_cache.path,
        panel=panel,
    )
    if basis_cache.hit:
        print(f"[cache-hit] basis {basis_cache.key}", flush=True)
    else:
        print(f"[cache-miss] basis {basis_cache.key}", flush=True)
    _append_progress_event(
        panel_dir,
        {
            "stage": "basis_done",
            "panel": panel,
            "nk": int(basis_data.nk),
            "nt": int(basis_data.nt),
            "n_band": int(basis_data.n_band),
            "screening": screening_payload,
            "cache_key": str(basis_cache.key),
            "cache_hit": bool(basis_cache.hit),
        },
    )
    print(f"[stage] {panel} basis_done nk={basis_data.nk} nt={basis_data.nt}", flush=True)

    _append_progress_event(panel_dir, {"stage": "overlap_start", "panel": panel})
    print(f"[stage] {panel} overlap_start", flush=True)
    overlap_cache = load_or_build_layer_overlap_blocks(
        basis_data,
        cache_dir=cache_dir,
        cache_policy=cache_policy,
        basis_cache_key=str(basis_cache.key),
    )
    overlap_blocks = overlap_cache.value
    update_cache_manifest_file(
        output_dir / "cache_manifest.json",
        cache_dir=cache_dir,
        kind="overlap",
        key=str(overlap_cache.key),
        hit=bool(overlap_cache.hit),
        path=overlap_cache.path,
        panel=panel,
    )
    if overlap_cache.hit:
        print(f"[cache-hit] overlap {overlap_cache.key}", flush=True)
    else:
        print(f"[cache-miss] overlap {overlap_cache.key}", flush=True)
    _append_progress_event(
        panel_dir,
        {
            "stage": "overlap_done",
            "panel": panel,
            "shift_count": len(overlap_blocks.shifts),
            "cache_key": str(overlap_cache.key),
            "cache_hit": bool(overlap_cache.hit),
        },
    )
    print(f"[stage] {panel} overlap_done shifts={len(overlap_blocks.shifts)}", flush=True)

    cache_metadata = {
        "basis_cache_key": str(basis_cache.key),
        "overlap_cache_key": str(overlap_cache.key),
        "screening_cache_key": screening_cache_key,
        "cache_dir": str(cache_dir),
        "basis_periodic_gauge": RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
        "basis_periodic_gauge_padding": int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING),
        "form_factor_convention": RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
        "cache_hits": {
            "screening": bool(screening_cache_hit),
            "basis": bool(basis_cache.hit),
            "overlap": bool(overlap_cache.hit),
        },
    }
    archive_cache_metadata = {
        "cache_key_basis": str(basis_cache.key),
        "cache_key_overlap": str(overlap_cache.key),
        "cache_key_screening": screening_cache_key,
        "screened_u_mev": float(basis_data.basis_model.params.displacement_field_mev),
        "physical_v_mev": float(model.params.displacement_field_mev),
        "cache_dir": str(cache_dir),
        "basis_periodic_gauge": RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
        "basis_periodic_gauge_padding": int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING),
        "form_factor_convention": RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
        "zero_literal_q0_fock": _env_flag_enabled("MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK", default=False),
        "hbn_moire_scale": float(config.get("hbn_moire_scale", 1.0)),
        "hbn_moire_v0_mev": float(model.params.moire_v0_mev),
        "hbn_moire_v1_mev": float(model.params.moire_v1_mev),
        "hbn_moire_phase_deg": float(model.params.moire_phase_deg),
    }

    run_payloads: list[dict[str, object]] = []
    for init_mode, seed in _run_specs_from_config(config):
            run_dir = panel_dir / "runs" / _run_key(init_mode, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            completed = _completed_run_summary(run_dir, max_iter=int(config["max_iter"]))
            if completed is not None:
                print(f"[skip] {panel} init={init_mode} seed={seed} already has usable run archive", flush=True)
                run_payloads.append(completed)
                _write_panel_convergence(
                    panel_dir,
                    panel=panel,
                    panel_start=panel_start,
                    run_payloads=run_payloads,
                    screening=screening_payload,
                    cache_metadata=cache_metadata,
                )
                continue

            checkpoint_dir = run_dir / "checkpoints"
            final_archive = run_dir / "hf_run_state.npz"
            latest_checkpoint = checkpoint_dir / "hf_checkpoint_latest.npz"
            latest_checkpoint_json = checkpoint_dir / "hf_checkpoint_latest.json"
            initial_density = None
            trace = _trace_from_arrays()
            resume_source = None
            if final_archive.exists():
                initial_density, trace = _load_archive_density(final_archive, basis_data.h0.shape)
                resume_source = final_archive
            elif latest_checkpoint.exists():
                initial_density, trace = _load_archive_density(latest_checkpoint, basis_data.h0.shape)
                json_trace = _load_trace_json(latest_checkpoint_json)
                if len(json_trace.get("err", [])) >= len(trace.get("err", [])):
                    trace = json_trace
                resume_source = latest_checkpoint
            iteration_offset = len(trace.get("err", []))
            remaining_iter = max(1, int(config["max_iter"]) - int(iteration_offset))
            _append_progress_event(
                panel_dir,
                {
                    "stage": "run_start",
                    "panel": panel,
                    "init_mode": init_mode,
                    "seed": int(seed),
                    "iteration_offset": int(iteration_offset),
                    "remaining_iter": int(remaining_iter),
                    "resume_source": "" if resume_source is None else str(resume_source),
                },
            )
            print(
                f"[run] start {panel} init={init_mode} seed={seed} "
                f"offset={iteration_offset} remaining={remaining_iter}",
                flush=True,
            )

            checkpoint_interval = max(1, int(config.get("checkpoint_interval", 1)))

            def step_callback(state, step, *, _trace=trace, _offset=iteration_offset) -> None:
                absolute_iteration = int(_offset) + int(step.iteration)
                _trace.setdefault("iteration", []).append(absolute_iteration)
                _trace.setdefault("energy_mev", []).append(float(step.energy))
                _trace.setdefault("err", []).append(float(step.norm_selected))
                _trace.setdefault("oda", []).append(float(step.oda_lambda))
                if absolute_iteration == 1 or absolute_iteration % checkpoint_interval == 0:
                    _write_checkpoint(
                        checkpoint_dir,
                        state=state,
                        basis_data=basis_data,
                        trace=_trace,
                        init_mode=init_mode,
                        seed=int(seed),
                        iteration=absolute_iteration,
                        energy=float(step.energy),
                        err=float(step.norm_selected),
                        oda=float(step.oda_lambda),
                    )
                _append_progress_event(
                    panel_dir,
                    {
                        "stage": "hf_iteration",
                        "panel": panel,
                        "init_mode": init_mode,
                        "seed": int(seed),
                        "iteration": absolute_iteration,
                        "energy_mev": float(step.energy),
                        "err": float(step.norm_selected),
                        "oda": float(step.oda_lambda),
                    },
                )
                print(
                    f"[iter] {panel} init={init_mode} seed={seed} "
                    f"iter={absolute_iteration} err={step.norm_selected:.6e} "
                    f"oda={step.oda_lambda:.6g} energy={step.energy:.9g}",
                    flush=True,
                )

            run = run_rlg_hbn_hartree_fock(
                basis_data,
                overlap_blocks=overlap_blocks,
                nu=float(config["nu"]),
                init_mode=init_mode,
                seed=int(seed),
                beta=float(config["beta"]),
                max_iter=remaining_iter,
                precision=float(config["precision"]),
                oda_stall_threshold=float(config["oda_stall_threshold"]),
                initial_density=initial_density,
                step_callback=step_callback,
            )
            payload = _run_payload(run, trace)
            _save_state_archive(final_archive, run, trace, cache_metadata=archive_cache_metadata)
            if trace.get("iteration"):
                _write_checkpoint(
                    checkpoint_dir,
                    state=run.state,
                    basis_data=basis_data,
                    trace=trace,
                    init_mode=init_mode,
                    seed=int(seed),
                    iteration=int(trace["iteration"][-1]),
                    energy=float(payload["final_energy_mev"]),
                    err=float(payload["final_error"]) if payload["final_error"] is not None else float("nan"),
                    oda=float(trace["oda"][-1]) if trace.get("oda") else float("nan"),
                )
            _atomic_write_json(
                run_dir / "hf_run_summary.json",
                {
                    **payload,
                    "run_dir": str(run_dir),
                    "state_npz": str(final_archive),
                    "checkpoint_dir": str(checkpoint_dir),
                    "resumed_from": "" if resume_source is None else str(resume_source),
                    **archive_cache_metadata,
                },
            )
            run_payloads.append(payload)
            _append_progress_event(
                panel_dir,
                {
                    "stage": "run_done",
                    "panel": panel,
                    "init_mode": init_mode,
                    "seed": int(seed),
                    "converged": bool(payload["converged"]),
                    "exit_reason": str(payload["exit_reason"]),
                    "iterations": int(payload["iterations"]),
                    "final_error": payload["final_error"],
                    "final_energy_mev": float(payload["final_energy_mev"]),
                },
            )
            _write_panel_convergence(
                panel_dir,
                panel=panel,
                panel_start=panel_start,
                run_payloads=run_payloads,
                screening=screening_payload,
                cache_metadata=cache_metadata,
            )

    convergence_payload = _write_panel_convergence(
        panel_dir,
        panel=panel,
        panel_start=panel_start,
        run_payloads=run_payloads,
        screening=screening_payload,
        cache_metadata=cache_metadata,
    )
    if convergence_payload is None:
        raise RuntimeError(f"No HF runs completed for panel {panel}")
    best_archive = panel_dir / "hf_ground_state.npz"
    if not best_archive.exists():
        raise FileNotFoundError(best_archive)
    return convergence_payload, best_archive


def _resolved_config(args: argparse.Namespace) -> dict[str, object]:
    base = dict(PAPER_CONFIGS[str(args.paper_target)])
    if args.v_values_mev is not None:
        base["v_values_mev"] = tuple(float(v) for v in args.v_values_mev)
    if args.xi_values is not None:
        base["xi_values"] = tuple(int(xi) for xi in args.xi_values)
    if args.epsilon_r is not None:
        base["epsilon_r"] = float(args.epsilon_r)
    if args.gate_distance_nm is not None:
        base["gate_distance_nm"] = float(args.gate_distance_nm)
    if args.scheme is not None:
        base["scheme"] = str(args.scheme)
    if args.interaction_cutoff_q1 is not None:
        base["interaction_cutoff_q1"] = float(args.interaction_cutoff_q1)
    base["hbn_moire_scale"] = float(args.hbn_moire_scale)
    if args.active_valence_bands is not None:
        base["active_valence_bands"] = int(args.active_valence_bands)
    if args.active_conduction_bands is not None:
        base["active_conduction_bands"] = int(args.active_conduction_bands)
    if args.k_mesh_size is not None:
        base["k_mesh_size"] = int(args.k_mesh_size)
    base["paper_target"] = str(args.paper_target)
    if args.init_modes is None:
        default_init_modes = ("flavor", "bm", "perturbed", "random") if args.paper_target == "fig6" else ("flavor", "bm", "perturbed")
    else:
        default_init_modes = tuple(str(mode) for mode in args.init_modes)
    if args.seeds is None:
        default_seeds = (1, 2, 3, 4) if args.paper_target == "fig6" else (1,)
    else:
        default_seeds = tuple(int(seed) for seed in args.seeds)
    if args.run_specs is None and args.init_modes is None and args.seeds is None:
        run_specs = default_rlg_hbn_run_specs(str(args.paper_target))
        default_init_modes = tuple(dict.fromkeys(init_mode for init_mode, _ in run_specs))
        default_seeds = tuple(dict.fromkeys(int(seed) for _, seed in run_specs))
    elif args.run_specs is None:
        run_specs = rlg_hbn_run_specs_for_modes(default_init_modes, default_seeds)
    else:
        run_specs = tuple((str(init_mode), int(seed)) for init_mode, seed in args.run_specs)
        default_init_modes = tuple(dict.fromkeys(init_mode for init_mode, _ in run_specs))
        default_seeds = tuple(dict.fromkeys(int(seed) for _, seed in run_specs))
    base["init_modes"] = tuple(str(mode) for mode in default_init_modes)
    base["seeds"] = tuple(int(seed) for seed in default_seeds)
    base["run_specs"] = _serialize_run_specs(run_specs)
    base["candidate_count"] = len(run_specs)
    base["max_iter"] = int(args.max_iter)
    if args.precision is None:
        base["precision"] = 1.0e-4 if str(args.paper_target) == "fig6" else 1.0e-6
    else:
        base["precision"] = float(args.precision)
    base["beta"] = float(args.beta)
    base["oda_stall_threshold"] = float(args.oda_stall_threshold)
    base["checkpoint_interval"] = max(1, int(args.checkpoint_interval))
    base["screening_mesh_size"] = (
        int(base["k_mesh_size"]) if args.screening_mesh_size is None else int(args.screening_mesh_size)
    )
    base["screening_solver"] = str(args.screening_solver)
    base["screening_u_min_mev"] = float(args.screening_u_min_mev)
    base["screening_u_max_mev"] = float(args.screening_u_max_mev)
    base["screening_u_grid_points"] = int(args.screening_u_grid_points)
    base["cache_policy"] = str(args.cache_policy)
    base["skip_screening_check"] = bool(args.skip_screening_check)
    base["zero_literal_q0_fock"] = os.environ.get(
        "MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"}
    return base


def main() -> None:
    start = perf_counter()
    args = _parse_args()
    config = _resolved_config(args)
    if not args.dry_run:
        ensure_not_running_compute_on_login_node(f"RLG/hBN {args.paper_target} paper HF")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir(args.paper_target).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_policy != "off":
        cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_preflight = _preflight_run_specs(config)
    except Exception as exc:
        write_json(
            output_dir / "run_preflight_failure.json",
            {
                "message": "RLG/hBN paper HF preflight failed before screening/basis/overlap setup.",
                "error": str(exc),
                "paper_target": str(args.paper_target),
                "run_specs": config.get("run_specs"),
                "init_modes": config.get("init_modes"),
                "seeds": config.get("seeds"),
            },
        )
        raise

    write_json(
        output_dir / "paper_hf_config.json",
        {
            **config,
            "run_preflight": run_preflight,
            "git_commit_sha": _git_commit_sha(),
            "cache_dir": str(cache_dir),
            "paper_reference": str(REPO_ROOT / "reference" / "2312.11617v1.pdf"),
            "runtime": {
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
                "dry_run": bool(args.dry_run),
            },
        },
    )

    if args.dry_run:
        print(f"[dry-run] output_dir={output_dir}")
        print(f"[dry-run] target={args.paper_target} config={config}")
        return

    if args.paper_target == "fig6" and not args.skip_screening_check:
        from mean_field.devtools.validate_rlg_hbn_fig6_prereqs import validate_fig6_screening_checkpoints

        prereq_payload = validate_fig6_screening_checkpoints(
            cache_dir=cache_dir,
            cache_policy=str(args.cache_policy),
            screening_solver=str(args.screening_solver),
            screening_u_min_mev=float(args.screening_u_min_mev),
            screening_u_max_mev=float(args.screening_u_max_mev),
            screening_u_grid_points=int(args.screening_u_grid_points),
            tolerance_mev=3.0,
        )
        write_json(output_dir / "prereq_screening_checkpoint.json", prereq_payload)
        failed = [entry for entry in prereq_payload["checks"] if not bool(entry["passed"])]
        if failed:
            write_json(
                output_dir / "prereq_screening_failure.json",
                {
                    "message": "Fig. 6 screening checkpoint failed; HF run was not started.",
                    "failed": failed,
                    "prereq": prereq_payload,
                },
            )
            raise RuntimeError("Fig. 6 screening checkpoint failed; see prereq_screening_failure.json")

    panel_summaries: list[dict[str, object]] = []
    for xi in tuple(int(value) for value in config["xi_values"]):
        for v_mev in tuple(float(value) for value in config["v_values_mev"]):
            panel_start = perf_counter()
            panel = _panel_name(xi=xi, v_mev=v_mev)
            panel_dir = output_dir / panel
            panel_dir.mkdir(parents=True, exist_ok=True)
            print(f"[panel] start {panel}", flush=True)

            model_params = _rlg_hbn_params_with_moire_scale(
                layer_count=int(config["layer_count"]),
                xi=int(xi),
                displacement_field_mev=float(v_mev),
                hbn_moire_scale=float(config["hbn_moire_scale"]),
            )
            model = RLGhBNModel.from_config(
                layer_count=int(config["layer_count"]),
                xi=int(xi),
                theta_deg=float(config["theta_deg"]),
                displacement_field_mev=float(v_mev),
                shell_count=int(config["shell_count"]),
                params=model_params,
            )
            interaction = RLGhBNInteractionParams(
                epsilon_r=float(config["epsilon_r"]),
                gate_distance_nm=float(config["gate_distance_nm"]),
                scheme=str(config["scheme"]),
                active_valence_bands=int(config["active_valence_bands"]),
                active_conduction_bands=int(config["active_conduction_bands"]),
                k_mesh_size=int(config["k_mesh_size"]),
                interaction_cutoff_q1=float(config["interaction_cutoff_q1"]),
                use_screened_basis=bool(config.get("use_screened_basis", True)),
            )
            write_json(
                panel_dir / "panel_config.json",
                {
                    "panel": panel,
                    "model": model.lattice_summary(),
                    "interaction": interaction.to_summary_dict(),
                    "nu": float(config["nu"]),
                    "init_modes": list(config["init_modes"]),
                    "seeds": list(config["seeds"]),
                    "run_specs": _serialize_run_specs(_run_specs_from_config(config)),
                    "candidate_count": int(config["candidate_count"]),
                    "hbn_moire_scale": float(config["hbn_moire_scale"]),
                    "max_iter": int(config["max_iter"]),
                    "precision": float(config["precision"]),
                    "beta": float(config["beta"]),
                    "cache_dir": str(cache_dir),
                    "cache_policy": str(args.cache_policy),
                    "screening_solver": str(args.screening_solver),
                },
            )

            convergence_payload, _ = _run_panel_with_incremental_outputs(
                output_dir=output_dir,
                cache_dir=cache_dir,
                cache_policy=str(args.cache_policy),
                panel_dir=panel_dir,
                panel=panel,
                model=model,
                interaction=interaction,
                config=config,
                panel_start=panel_start,
            )
            panel_elapsed = perf_counter() - panel_start
            panel_summaries.append(
                {
                    "panel": panel,
                    "panel_dir": str(panel_dir),
                    "elapsed_sec": float(panel_elapsed),
                    "best": convergence_payload["best"],
                }
            )
            print(f"[panel] done {panel} elapsed_sec={panel_elapsed:.3f}", flush=True)

    elapsed = perf_counter() - start
    write_json(
        output_dir / "paper_hf_summary.json",
        {
            "output_dir": str(output_dir),
            "paper_target": str(args.paper_target),
            "elapsed_sec": float(elapsed),
            "panels": panel_summaries,
        },
    )
    latest_path = DEFAULT_OUTPUT_ROOT / f"LATEST_{str(args.paper_target).upper()}_HF_PAPER.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(output_dir) + "\n", encoding="utf-8")
    print(f"[done] output_dir={output_dir}")


if __name__ == "__main__":
    main()
