from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mean_field.api.artifacts import ModelRecord, write_contract_artifacts
from mean_field.core.hf import (
    check_single_flavor_simplification,
    summarize_hf_state_archive,
    validate_hf_archive_shapes,
)

# Historical RnG/hBN paper-HF runner code was retired from tracked surface and
# archived under local_archive/retired_surface/ff38584_rlg_hbn_paper_hf_devtool/.
# This module intentionally keeps only lightweight compatibility helpers used by
# canonical sidecar/archive tests and the parallel-merge metadata tool.  It must
# not run HF, build projected bases, or import mean_field.systems.RnG_hBN.

RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION = "centered_cell_reciprocal_relabel_pad1_v2"
RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING = 1
RLG_HBN_FORM_FACTOR_CONVENTION_VERSION = "physical_q_plus_g_valley_signed_raw_shift_v2"

PAPER_CONFIGS = {
    "fig5": {
        "description": "2312.11617v1 Fig. 5 HF band-structure source states",
        "layer_count": 5,
        "theta_deg": 0.77,
        "shell_count": 4,
        "xi_values": (1,),
        "v_values_mev": (40.0, 48.0, 56.0, 64.0),
        "hbn_moire_scale": 1.0,
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
        "hbn_moire_scale": 1.0,
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

DETERMINISTIC_INIT_MODES = {"bm"}
SUPPORTED_INIT_MODES = {"bm", "flavor", "flavor_polarized", "perturbed", "random", "diag_random"}


def _parse_run_specs(text: str) -> tuple[tuple[str, int], ...]:
    specs: list[tuple[str, int]] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise argparse.ArgumentTypeError(f"Expected run spec entries as init_mode:seed, got {stripped!r}.")
        init_mode, seed_text = stripped.split(":", 1)
        try:
            seed = int(seed_text.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid seed in run spec {stripped!r}.") from exc
        specs.append((init_mode.strip(), seed))
    if not specs:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated run spec.")
    return tuple(specs)


def rlg_hbn_run_specs_for_modes(
    init_modes: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    specs: list[tuple[str, int]] = []
    for init_mode in init_modes:
        mode = str(init_mode)
        mode_seeds = seeds[:1] if mode.strip().lower() in DETERMINISTIC_INIT_MODES and len(seeds) > 1 else seeds
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
    if not isinstance(raw_specs, (list, tuple)):
        raise TypeError("run_specs must be a list of {init_mode, seed} entries.")
    specs: list[tuple[str, int]] = []
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


def _normalize_init_mode_for_preflight(init_mode: str) -> str:
    normalized = str(init_mode).strip().lower().replace("-", "_")
    aliases = {"diag_random": "random"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_INIT_MODES:
        raise ValueError(f"Unsupported RLG/hBN HF init mode {init_mode!r}.")
    return normalized


def _preflight_run_specs(config: dict[str, object]) -> dict[str, object]:
    """Validate cheap run-spec invariants without building HF inputs."""

    specs = _run_specs_from_config(config)
    validated: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for init_mode, seed in specs:
        try:
            normalized = _normalize_init_mode_for_preflight(str(init_mode))
        except ValueError as exc:
            errors.append({"init_mode": str(init_mode), "seed": int(seed), "error": str(exc)})
            continue
        validated.append({"init_mode": str(init_mode), "normalized_init_mode": normalized, "seed": int(seed)})
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
            f"got active_valence={active_valence} active_conduction={active_conduction}"
        )
    if active_valence + active_conduction <= 0:
        raise ValueError("At least one active band is required for RLG/hBN HF.")
    if k_mesh_size <= 0:
        raise ValueError(f"k_mesh_size must be positive, got {k_mesh_size}")
    return {"status": "ok", "run_specs": validated, "run_count": len(validated)}


def write_rlg_hbn_paper_hf_contract_sidecars(
    output_dir: Path,
    *,
    paper_target: str,
    config: dict[str, object],
    run_preflight: dict[str, object],
    cache_dir: Path,
    runtime_metadata: dict[str, object],
    workflow_statuses: dict[str, str],
    workflow_messages: dict[str, str],
    summary_payload: dict[str, object] | None,
) -> dict[str, Path]:
    """Write public metadata sidecars for a historical RLG/hBN paper-HF result root."""

    dry_run = bool(runtime_metadata.get("dry_run", False))
    panel_count = len(summary_payload.get("panels", [])) if summary_payload is not None else 0
    failed_jobs = sorted(name for name, status in workflow_statuses.items() if status == "failed")
    status = "dry_run" if dry_run else ("pass" if not failed_jobs and summary_payload is not None else "warning")
    files: dict[str, object] = {
        "paper_hf_config": "paper_hf_config.json",
        "workflow_manifest": "workflow_manifest.json",
        "workflow_run_state": "workflow_run_state.json",
        "workflow_run_state_markdown": "workflow_run_state.md",
    }
    if summary_payload is not None:
        files["paper_hf_summary"] = "paper_hf_summary.json"
    return write_contract_artifacts(
        output_dir,
        workflow="rlg_hbn.paper_hf",
        system_name="rlg_hbn",
        model=ModelRecord(
            system_name="rlg_hbn",
            params={
                "paper_target": str(paper_target),
                "layer_count": int(config["layer_count"]),
                "xi_values": [int(value) for value in config["xi_values"]],
                "v_values_mev": [float(value) for value in config["v_values_mev"]],
                "hbn_moire_scale": float(config.get("hbn_moire_scale", 1.0)),
            },
            lattice={"theta_deg": float(config["theta_deg"]), "shell_count": int(config["shell_count"])},
        ),
        config={**config, "run_preflight": run_preflight, "cache_dir": str(cache_dir)},
        conventions={
            "energy_unit": "meV",
            "density_convention": "stored_delta",
            "density_axis_order": "abk",
            "system": "RLG/hBN",
            "paper_target": str(paper_target),
            "basis_periodic_gauge": RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
            "basis_periodic_gauge_padding": int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING),
            "form_factor_convention": RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
        },
        environment=runtime_metadata,
        validation={
            "status": status,
            "run_preflight": run_preflight,
            "workflow_statuses": dict(workflow_statuses),
            "workflow_messages": dict(workflow_messages),
            "failed_jobs": failed_jobs,
            "panel_count": int(panel_count),
        },
        observables={
            "paper_target": str(paper_target),
            "elapsed_sec": None if summary_payload is None else summary_payload.get("elapsed_sec"),
            "panels": [] if summary_payload is None else summary_payload.get("panels", []),
        },
        files=files,
        metadata={"cache_dir": str(cache_dir), "paper_target": str(paper_target)},
    )


def _complex_to_pairs(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    return np.stack((arr.real, arr.imag), axis=-1)


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


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
    return {"iteration": list(range(1, n + 1)), "energy_mev": energy, "err": err, "oda": oda}


def _trace_arrays(trace: dict[str, list[float] | list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(trace.get("energy_mev", []), dtype=float),
        np.asarray(trace.get("err", []), dtype=float),
        np.asarray(trace.get("oda", []), dtype=float),
    )


def save_rlg_hbn_paper_hf_state_archive(
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
        "density_convention": np.asarray("stored_delta"),
        "density_axis_order": np.asarray("abk"),
        "reference_density_convention": np.asarray(str(run.state.scheme)),
        "basis_periodic_gauge": np.asarray(RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION),
        "basis_periodic_gauge_padding": np.asarray([int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING)], dtype=int),
        "form_factor_convention": np.asarray(RLG_HBN_FORM_FACTOR_CONVENTION_VERSION),
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


def load_rlg_hbn_paper_hf_archive_density(path: Path, expected_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, list[float] | list[int]]]:
    summary = summarize_hf_state_archive(path)
    validate_hf_archive_shapes(summary)
    if summary.density_shape != tuple(int(value) for value in expected_shape):
        raise ValueError(f"Checkpoint density shape {summary.density_shape} does not match current basis {expected_shape}")
    with np.load(path, allow_pickle=False) as archive:
        density = np.asarray(archive["density"], dtype=np.complex128)
        return density, _trace_from_arrays(
            iter_energy=archive["iter_energy_mev"] if "iter_energy_mev" in archive.files else None,
            iter_err=archive["iter_err"] if "iter_err" in archive.files else None,
            iter_oda=archive["iter_oda"] if "iter_oda" in archive.files else None,
        )


# Historical RnG/hBN parallel paper-HF merge workflow was retired from tracked
# command surface and archived under
# local_archive/retired_surface/09b4946_rlg_hbn_parallel_merge_devtool/.
# Keep only the metadata sidecar compatibility helper used by schema tests.


def write_rlg_hbn_parallel_hf_merge_contract_sidecars(
    source_root: Path,
    *,
    paper_target: str,
    merged_config: dict[str, object],
    selected_rows: list[dict[str, object]],
    ignored_panel_dirs: list[dict[str, object]],
    tasks_root: Path,
) -> dict[str, Path]:
    return write_contract_artifacts(
        source_root,
        workflow="rlg_hbn.parallel_hf_merge",
        system_name="rlg_hbn",
        model=ModelRecord(
            system_name="rlg_hbn",
            params={
                "paper_target": str(paper_target),
                "layer_count": int(merged_config["layer_count"]),
                "xi_values": [int(value) for value in merged_config["xi_values"]],
                "v_values_mev": [float(value) for value in merged_config["v_values_mev"]],
                "hbn_moire_scale": float(merged_config.get("hbn_moire_scale", 1.0)),
            },
            lattice={"theta_deg": float(merged_config["theta_deg"]), "shell_count": int(merged_config["shell_count"])},
        ),
        config=merged_config,
        conventions={
            "energy_unit": "meV",
            "density_convention": "stored_delta",
            "density_axis_order": "abk",
            "system": "RLG/hBN",
            "paper_target": str(paper_target),
        },
        validation={
            "status": "pass",
            "selected_panel_count": int(len(selected_rows)),
            "ignored_candidate_count": int(len(ignored_panel_dirs)),
            "tasks_root": str(tasks_root),
        },
        observables={
            "paper_target": str(paper_target),
            "selected": selected_rows,
            "ignored_candidates": ignored_panel_dirs,
        },
        files={
            "paper_hf_config": "paper_hf_config.json",
            "cache_manifest": "cache_manifest.json",
            "parallel_selection_summary": "parallel_selection_summary.json",
            "selected_panels": [str(row["panel"]) for row in selected_rows],
        },
        metadata={"tasks_root": str(tasks_root), "paper_target": str(paper_target)},
    )


# Historical dense RnG/hBN q=0 TDHF runner was retired from tracked command
# surface and archived under
# local_archive/retired_surface/7606a1a_rlg_hbn_tdhf_q0_devtool/.
# Keep only lightweight compatibility helpers used by schema/adapter tests.


def _flavor_counts(state) -> dict[tuple[int, int], int]:
    if getattr(state, "occupation_counts", None) is None:
        return {}
    counts = np.asarray(state.occupation_counts, dtype=int).reshape((int(state.n_spin), int(state.n_eta)), order="C")
    return {(int(s), int(e)): int(counts[s, e]) for s in range(counts.shape[0]) for e in range(counts.shape[1])}


def rlg_hbn_tdhf_q0_shortcut_decision(state, mode: str, channel: str) -> tuple[bool, str]:
    if channel == "all":
        return False, "single-flavor shortcut is not applied to mixed all-channel blocks"
    if channel == "intraflavor":
        return False, "single-flavor shortcut is not applied to intraflavor blocks"
    if mode == "off":
        return False, "disabled by --single-flavor-shortcut=off"
    counts = _flavor_counts(state)
    if not counts:
        status = check_single_flavor_simplification(active_space_has_valence=bool(int(state.active_valence_bands) > 0), occupied_flavor_counts={}, polarized_flavor=(0, 0))
        if mode == "on":
            raise ValueError("--single-flavor-shortcut=on requires saved occupation_counts metadata")
        return False, status.reason
    polarized_candidates = [flavor for flavor, count in counts.items() if int(count) > 0]
    polarized = polarized_candidates[0] if polarized_candidates else next(iter(counts))
    status = check_single_flavor_simplification(active_space_has_valence=bool(int(state.active_valence_bands) > 0), occupied_flavor_counts=counts, polarized_flavor=polarized)
    if mode == "on" and not status.allowed:
        raise ValueError(f"single-flavor shortcut requested but illegal: {status.reason}")
    return bool(status.allowed and mode in {"auto", "on"}), status.reason


def write_rlg_hbn_tdhf_q0_contract_sidecars(output_dir: Path, *, config_payload: dict[str, object], summary_payload: dict[str, object] | None, spectrum_path: Path | None) -> dict[str, Path]:
    """Write public contract sidecars for a historical RLG/hBN q=0 TDHF result."""

    hf_archive = str(config_payload.get("hf_archive", ""))
    channel = str(config_payload.get("channel", "all"))
    files: dict[str, object] = {"tdhf_config": "tdhf_q0_config.json", "source_hf_archive": hf_archive}
    array_files: tuple[str | Path, ...] = ()
    if summary_payload is None:
        validation_payload = {"status": "dry_run", "source_hf_converged": config_payload.get("summary_converged")}
        observables_payload: dict[str, object] = {}
    else:
        files["tdhf_summary"] = "tdhf_q0_summary.json"
        if spectrum_path is not None:
            files["tdhf_spectrum"] = spectrum_path.name
            array_files = (spectrum_path,)
        structure = dict(summary_payload.get("structure", {})) if isinstance(summary_payload.get("structure"), dict) else {}
        spectrum = dict(summary_payload.get("spectrum", {})) if isinstance(summary_payload.get("spectrum"), dict) else {}
        source_converged = config_payload.get("summary_converged")
        validation_payload = {
            "status": "pass" if bool(structure.get("ok", False)) and source_converged is not False else "warning",
            "source_hf_converged": source_converged,
            "structure": structure,
            "spectrum_residuals": {"max_residual": spectrum.get("max_residual"), "pairing_residual": spectrum.get("pairing_residual"), "selected_count": spectrum.get("selected_count")},
            "dense_memory_guard": {"max_dense_memory_gb": config_payload.get("max_dense_memory_gb"), "estimated_dense_memory_gib": summary_payload.get("estimated_dense_memory_gib"), "max_pairs": config_payload.get("max_pairs"), "n_pairs": summary_payload.get("n_pairs")},
        }
        observables_payload = {
            "channel": summary_payload.get("channel"),
            "channel_counts": summary_payload.get("channel_counts"),
            "n_pairs": summary_payload.get("n_pairs"),
            "liouvillian_dim": summary_payload.get("liouvillian_dim"),
            "first_positive_energies_mev": spectrum.get("first_positive_energies_mev"),
            "single_flavor_shortcut_used": summary_payload.get("single_flavor_shortcut_used"),
            "single_flavor_shortcut_reason": summary_payload.get("single_flavor_shortcut_reason"),
            "hf_summary": summary_payload.get("hf_summary"),
        }
    return write_contract_artifacts(
        output_dir,
        workflow="rlg_hbn.tdhf_q0",
        system_name="rlg_hbn",
        model=ModelRecord(system_name="rlg_hbn", params={"source_hf_archive": hf_archive}),
        config=config_payload,
        conventions={"energy_unit": "meV", "density_convention": "stored_delta", "density_axis_order": "abk", "q_sector": "q0", "tdhf_pair_convention": "core_hf_tdhf", "channel": channel, "source_hf_archive": hf_archive},
        environment=dict(config_payload.get("runtime", {})) if isinstance(config_payload.get("runtime"), dict) else {},
        validation=validation_payload,
        observables=observables_payload,
        files=files,
        metadata={"source_hf_archive": hf_archive, "channel": channel},
        array_files=array_files,
    )


__all__ = [
    "PAPER_CONFIGS",
    "default_rlg_hbn_run_specs",
    "load_rlg_hbn_paper_hf_archive_density",
    "rlg_hbn_run_specs_for_modes",
    "rlg_hbn_tdhf_q0_shortcut_decision",
    "save_rlg_hbn_paper_hf_state_archive",
    "write_rlg_hbn_paper_hf_contract_sidecars",
    "write_rlg_hbn_parallel_hf_merge_contract_sidecars",
    "write_rlg_hbn_tdhf_q0_contract_sidecars",
]
