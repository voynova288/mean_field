from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import socket
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from mean_field.core.hf import solve_tdhf_matrices, split_pair_indices_by_flavor_channel
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.RnG_hBN import (
    RLGhBNLayerOverlapBlockSet,
    RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_tdhf_c3_quotient_cycle,
    build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs,
    build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs,
    build_rlg_hbn_tdhf_finite_q_quotient_context,
    build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs,
    build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs,
    build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_q_pairs,
    center_reciprocal_fractional_coordinates,
    finite_q_shift_cartesian_nm_inv,
    load_rlg_hbn_tdhf_run_from_archive,
    required_rlg_hbn_tdhf_finite_q_overlap_shifts,
    required_rlg_hbn_tdhf_full_finite_q_overlap_shifts,
    validate_rlg_hbn_hf_single_representative_source_closure,
)
from mean_field.workflows import collect_slurm_metadata

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "RnG_hBN" / "tdhf"
FINITE_Q_CHANNELS = ("intraflavor", "intervalley", "interspin", "inter_spin_valley")
UMKLAPP_COMPLETION_CHOICES = ("strict", "build", "allow-incomplete")


def _parse_channels(text: str) -> tuple[str, ...]:
    channels = tuple(item.strip() for item in text.split(",") if item.strip())
    if not channels:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated channel")
    invalid = sorted(set(channels).difference(FINITE_Q_CHANNELS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Finite-q channels must be one of {FINITE_Q_CHANNELS}; got {invalid}"
        )
    return channels


def _parse_q_shifts(text: str) -> tuple[tuple[int, int], ...]:
    shifts: list[tuple[int, int]] = []
    for raw_item in text.replace(" ", "").split(";"):
        if not raw_item:
            continue
        parts = raw_item.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "q shifts must be a semicolon-separated list of integer pairs, e.g. '0,0;1,0;2,0'"
            )
        try:
            shifts.append((int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid q-shift item {raw_item!r}") from exc
    if not shifts:
        raise argparse.ArgumentTypeError("Expected at least one q shift")
    return tuple(shifts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dense finite-q RLG/hBN TDHF/RPA postprocessing. Typed variational-"
            "quotient archives use the microscopic all-channel signed-q matrix API; "
            "legacy archives retain the older intraflavor/shortcut paths."
        )
    )
    parser.add_argument("--hf-archive", type=Path, required=True, help="Path to hf_run_state.npz or hf_ground_state.npz.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional hf_run_summary.json for run metadata.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override cache directory if the archive lacks cache_dir.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults under results/RnG_hBN/tdhf.")
    parser.add_argument(
        "--channels",
        type=_parse_channels,
        default=("intervalley", "interspin"),
        help="Comma-separated finite-q channels. Choices: intraflavor,intervalley,interspin,inter_spin_valley.",
    )
    parser.add_argument(
        "--q-shifts",
        type=_parse_q_shifts,
        required=True,
        help="Semicolon-separated integer mesh momentum shifts, e.g. '0,0;1,0;2,0'.",
    )
    parser.add_argument(
        "--umklapp-completion",
        choices=UMKLAPP_COMPLETION_CHOICES,
        default="build",
        help=(
            "How to handle finite-q wrapped form-factor keys. 'strict' requires the archive cache to already contain "
            "all stored_shift = G + W_target - W_source keys; 'build' computes missing keys on the compute node; "
            "'allow-incomplete' skips missing keys and is diagnostic-only."
        ),
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--c3-quotient-provider",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Typed quotient archives automatically use "
            "the derived microscopic finite-q quotient API for every channel."
        ),
    )
    parser.add_argument("--periodic-gauge-padding", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=4096, help="Refuse dense assembly above this ph-pair count.")
    parser.add_argument("--max-dense-memory-gb", type=float, default=8.0, help="Conservative dense TDHF memory estimate limit per q/channel block.")
    parser.add_argument("--structure-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--imag-tol", type=float, default=1.0e-8)
    parser.add_argument("--energy-tol", type=float, default=1.0e-10)
    parser.add_argument("--norm-tol", type=float, default=1.0e-10)
    parser.add_argument("--allow-unconverged", action="store_true", help="Do not reject archives whose summary says converged=false.")
    parser.add_argument(
        "--allow-untyped-legacy",
        action="store_true",
        help=(
            "Diagnostic only: permit an archive without typed HF interaction "
            "provenance. Production runs fail closed by default."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Write/print resolved config only; do not load cache or solve TDHF.")
    parser.add_argument("--no-save-vectors", action="store_true", help="Do not store X/Y mode vectors in NPZ outputs.")
    parser.add_argument(
        "--no-save-matrices",
        action="store_true",
        help="Do not store dense A/B/L matrices in NPZ outputs; summaries and spectra are still saved.",
    )
    return parser.parse_args()


def _default_output_dir(hf_archive: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    stem = hf_archive.stem
    suffix = f"{stem}_finite_q"
    if job_id:
        suffix += f"_{job_id}"
    else:
        suffix += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / suffix


def _load_summary(summary_path: Path | None, archive_path: Path) -> dict[str, object]:
    path = summary_path if summary_path is not None else archive_path.with_name("hf_run_summary.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def _dense_memory_estimate_bytes(n_pairs: int, *, save_vectors: bool) -> int:
    p2 = int(n_pairs) * int(n_pairs)
    multiplier = 192 if save_vectors else 160
    return int(multiplier * p2)


def _raw_real_eigenvalue_summary(raw_eigenvalues: np.ndarray, *, imag_tol: float, energy_tol: float) -> dict[str, object]:
    values = np.asarray(raw_eigenvalues, dtype=np.complex128).reshape(-1)
    finite_mask = np.isfinite(values.real) & np.isfinite(values.imag)
    finite_values = values[finite_mask]
    complex_values = finite_values[np.abs(finite_values.imag) > float(imag_tol)]
    finite_real = finite_values[
        np.abs(finite_values.imag) <= float(imag_tol)
    ]
    real_values = np.sort(finite_real.real.astype(float, copy=False))
    near_zero = real_values[np.abs(real_values) <= float(energy_tol)]
    positive = real_values[real_values > float(energy_tol)]
    negative = real_values[real_values < -float(energy_tol)]
    low_abs = real_values[np.argsort(np.abs(real_values))[:20]] if real_values.size else np.asarray([], dtype=float)
    return {
        "total_count": int(values.size),
        "finite_count": int(finite_values.size),
        "nonfinite_count": int(values.size - finite_values.size),
        "real_count": int(real_values.size),
        "positive_real_count": int(positive.size),
        "negative_real_count": int(negative.size),
        "near_zero_count": int(near_zero.size),
        "complex_count": int(complex_values.size),
        "max_imaginary_abs_mev": (
            float(np.max(np.abs(finite_values.imag)))
            if finite_values.size
            else 0.0
        ),
        "near_zero_eigenvalues_mev": [float(value) for value in near_zero[:40]],
        "lowest_abs_real_eigenvalues_mev": [float(value) for value in low_abs],
        "lowest_real_eigenvalues_mev": [float(value) for value in real_values[:20]],
        "highest_real_eigenvalues_mev": [float(value) for value in real_values[-20:]],
    }


def _spectrum_assignment_residual(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left_values = np.asarray(left, dtype=np.complex128).reshape(-1)
    right_values = np.asarray(right, dtype=np.complex128).reshape(-1)
    if left_values.size != right_values.size:
        raise ValueError(
            f"spectrum sizes differ: {left_values.size} != {right_values.size}"
        )
    if left_values.size == 0:
        return 0.0
    cost = np.abs(left_values[:, None] - right_values[None, :])
    rows, columns = linear_sum_assignment(cost)
    return float(np.max(cost[rows, columns]))


def _filter_pairs(pairs, channel: str):
    groups = split_pair_indices_by_flavor_channel(pairs)
    indices = groups[channel]
    return tuple(pairs[int(index)] for index in indices), {name: int(len(values)) for name, values in groups.items()}


def _merge_overlap_blocks(
    base: RLGhBNLayerOverlapBlockSet,
    extra: RLGhBNLayerOverlapBlockSet,
) -> RLGhBNLayerOverlapBlockSet:
    return RLGhBNLayerOverlapBlockSet(
        shifts=base.shifts,
        gvecs=base.gvecs,
        layer_overlaps={**base.layer_overlaps, **extra.layer_overlaps},
        layer_diagonal_overlaps={**base.layer_diagonal_overlaps, **extra.layer_diagonal_overlaps},
        hartree_layer_coulomb={**base.hartree_layer_coulomb, **extra.hartree_layer_coulomb},
        fock_layer_coulomb={**base.fock_layer_coulomb, **extra.fock_layer_coulomb},
    )


def _run_with_completed_umklapp(run, missing_shifts: tuple[tuple[int, int], ...]):
    if not missing_shifts:
        return run
    extra = build_rlg_hbn_layer_overlap_blocks(run.basis_data, shifts=missing_shifts)
    merged = _merge_overlap_blocks(run.overlap_blocks, extra)
    return replace(run, overlap_blocks=merged)


def _q_label(q_shift: tuple[int, int]) -> str:
    return f"qx{int(q_shift[0]):+d}_qy{int(q_shift[1]):+d}".replace("+", "p").replace("-", "m")


def _c3_shift_mod(q_shift: tuple[int, int], mesh_size: int) -> tuple[int, int]:
    m, n = (int(q_shift[0]), int(q_shift[1]))
    return (-n) % int(mesh_size), (m - n) % int(mesh_size)


def _c3_orbit_representative(q_shift: tuple[int, int], mesh_size: int) -> tuple[int, int]:
    q0 = (int(q_shift[0]) % int(mesh_size), int(q_shift[1]) % int(mesh_size))
    q1 = _c3_shift_mod(q0, mesh_size)
    q2 = _c3_shift_mod(q1, mesh_size)
    return min(q0, q1, q2)


def _mesh_shape_from_frac(k_grid_frac: np.ndarray) -> tuple[int, int]:
    frac = np.asarray(k_grid_frac, dtype=float)
    nx = int(np.unique(np.round(frac[:, 0], decimals=12)).size)
    ny = int(np.unique(np.round(frac[:, 1], decimals=12)).size)
    return nx, ny


def main() -> None:
    start = perf_counter()
    args = _parse_args()
    archive_path = args.hf_archive.expanduser().resolve()
    summary = _load_summary(args.summary_path, archive_path)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else _default_output_dir(archive_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slurm_metadata = collect_slurm_metadata()
    runtime_metadata: dict[str, object] = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }
    if slurm_metadata:
        runtime_metadata["slurm"] = slurm_metadata
    config_payload = {
        "hf_archive": str(archive_path),
        "summary_path": "" if args.summary_path is None else str(args.summary_path.expanduser().resolve()),
        "cache_dir": "" if args.cache_dir is None else str(args.cache_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "channels": list(args.channels),
        "q_shifts": [list(q) for q in args.q_shifts],
        "umklapp_completion": str(args.umklapp_completion),
        "beta": float(args.beta),
        "c3_quotient_provider": bool(args.c3_quotient_provider),
        "periodic_gauge_padding": int(args.periodic_gauge_padding),
        "max_pairs": int(args.max_pairs),
        "max_dense_memory_gb": float(args.max_dense_memory_gb),
        "structure_tolerance": float(args.structure_tolerance),
        "imag_tol": float(args.imag_tol),
        "energy_tol": float(args.energy_tol),
        "norm_tol": float(args.norm_tol),
        "allow_unconverged": bool(args.allow_unconverged),
        "allow_untyped_legacy": bool(args.allow_untyped_legacy),
        "dry_run": bool(args.dry_run),
        "save_vectors": not bool(args.no_save_vectors),
        "save_matrices": not bool(args.no_save_matrices),
        "summary_converged": bool(summary.get("converged", False)) if summary else None,
        "runtime": runtime_metadata,
    }
    write_json(output_dir / "tdhf_finite_q_config.json", config_payload)
    if args.dry_run:
        print(f"[dry-run] output_dir={output_dir}")
        print(f"[dry-run] config={config_payload}")
        return

    ensure_not_running_compute_on_login_node("RLG/hBN finite-q TDHF postprocessing")
    if summary and not bool(summary.get("converged", False)) and not args.allow_unconverged:
        raise SystemExit(
            "Refusing to run finite-q TDHF on an unconverged HF archive; pass --allow-unconverged for diagnostics only."
        )

    run = load_rlg_hbn_tdhf_run_from_archive(
        archive_path,
        cache_dir=args.cache_dir,
        summary_path=args.summary_path,
    )
    if not run.converged and not args.allow_unconverged:
        raise SystemExit(
            "Refusing finite-q TDHF because the restored HF run is not converged"
        )
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    mesh_shape = _mesh_shape_from_frac(run.basis_data.k_grid_frac)
    provenance = run.interaction_provenance
    typed_quotient = bool(
        provenance is not None and provenance.quotient_enabled
    )
    typed_single_representative = bool(
        provenance is not None
        and not provenance.quotient_enabled
        and provenance.convention
        == RLG_HBN_HF_SINGLE_REPRESENTATIVE_INTERACTION_CONVENTION_VERSION
    )
    if provenance is None and not bool(args.allow_untyped_legacy):
        raise SystemExit(
            "Refusing untyped legacy HF archive; pass --allow-untyped-legacy "
            "only for an explicitly diagnostic run"
        )
    if provenance is not None and not (
        typed_quotient or typed_single_representative
    ):
        raise SystemExit(
            "Refusing unsupported typed HF interaction convention "
            f"{provenance.convention!r}"
        )
    prepared_quotient = None
    single_representative_source_closure = None
    if typed_quotient:
        if str(args.umklapp_completion) == "allow-incomplete":
            raise SystemExit(
                "Typed quotient finite-q assembly forbids diagnostic incomplete Umklapp sums"
            )
        prepared_quotient = build_rlg_hbn_tdhf_finite_q_quotient_context(
            run,
            periodic_gauge_padding=int(args.periodic_gauge_padding),
            beta=float(args.beta),
            require_provenance=True,
        )
        physical_shifts = tuple(prepared_quotient.physical_shifts)
    else:
        physical_shifts = tuple(
            (int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts
        )
        if typed_single_representative:
            if str(args.umklapp_completion) == "allow-incomplete":
                raise SystemExit(
                    "Typed single-representative finite-q assembly forbids "
                    "diagnostic incomplete Umklapp sums"
                )
            single_representative_source_closure = (
                validate_rlg_hbn_hf_single_representative_source_closure(run)
            )
    if bool(args.c3_quotient_provider) and typed_single_representative:
        raise SystemExit(
            "--c3-quotient-provider is incompatible with the typed "
            "single-representative functional"
        )
    if bool(args.c3_quotient_provider) and not typed_quotient:
        if mesh_shape[0] != mesh_shape[1]:
            raise SystemExit(
                "--c3-quotient-provider requires a square regular momentum mesh"
            )
        if str(args.umklapp_completion) == "allow-incomplete":
            raise SystemExit(
                "--c3-quotient-provider forbids --umklapp-completion allow-incomplete"
            )
    config_payload["typed_finite_q_quotient"] = typed_quotient
    config_payload["typed_finite_q_single_representative"] = (
        typed_single_representative
    )
    config_payload["finite_q_matrix_api"] = (
        "build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs"
        if typed_quotient
        else (
            "build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs"
            if typed_single_representative
            else "legacy_pair_assembly"
        )
    )
    write_json(output_dir / "tdhf_finite_q_config.json", config_payload)
    block_summaries: list[dict[str, object]] = []
    quotient_matrix_cache: dict[tuple[int, int], object] = {}
    quotient_cycle_metadata: dict[tuple[int, int], dict[str, object]] = {}
    for q_shift in args.q_shifts:
        all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift)
        for channel in args.channels:
            pairs, channel_counts = _filter_pairs(all_pairs, str(channel))
            n_pairs = int(len(pairs))
            if n_pairs > int(args.max_pairs):
                raise SystemExit(
                    f"Refusing dense finite-q TDHF assembly for q={q_shift} channel={channel}: "
                    f"{n_pairs} ph pairs above max_pairs={args.max_pairs}."
                )
            estimated_bytes = _dense_memory_estimate_bytes(n_pairs, save_vectors=not bool(args.no_save_vectors))
            memory_limit_bytes = int(float(args.max_dense_memory_gb) * 1024**3)
            if estimated_bytes > memory_limit_bytes:
                raise SystemExit(
                    f"Refusing dense finite-q TDHF assembly for q={q_shift} channel={channel}: "
                    f"estimated memory {estimated_bytes / 1024**3:.2f} GiB exceeds "
                    f"--max-dense-memory-gb={float(args.max_dense_memory_gb):.2f}."
                )

            if typed_quotient:
                required_shifts = tuple(physical_shifts)
                missing: tuple[tuple[int, int], ...] = ()
                run_for_block = run
                built_missing = False
                signed_matrices = (
                    build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs(
                        run,
                        orbitals,
                        pairs,
                        q_shift,
                        prepared_context=prepared_quotient,
                        beta=float(args.beta),
                        periodic_gauge_padding=int(args.periodic_gauge_padding),
                        structure_tolerance=float(args.structure_tolerance),
                        physical_shifts=physical_shifts,
                        require_provenance=True,
                    )
                )
                matrices = signed_matrices.plus
                partner_matrices = signed_matrices.minus
                quotient_block_meta: dict[str, object] | None = {
                    "matrix_api": (
                        "build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs"
                    ),
                    "ordinary_boundary_lift": (
                        "analytic_periodic_gauge_relabel_v1"
                    ),
                    "fixed_fixed_branch_rule": "same_puncture_copy_v1",
                    "fixed_branch_weight": 1.0 / 3.0,
                    "source_provenance_validated": True,
                }
            else:
                if str(channel) == "intraflavor":
                    required_shifts = required_rlg_hbn_tdhf_full_finite_q_overlap_shifts(
                        orbitals,
                        run.basis_data,
                        pairs,
                        q_shift,
                        physical_shifts=physical_shifts,
                    )
                else:
                    required_shifts = required_rlg_hbn_tdhf_finite_q_overlap_shifts(
                        orbitals,
                        run.basis_data,
                        pairs,
                        q_shift,
                        physical_shifts=physical_shifts,
                    )
                available = set(
                    tuple(int(v) for v in key)
                    for key in run.overlap_blocks.layer_overlaps
                )
                missing = tuple(
                    shift for shift in required_shifts if shift not in available
                )
                if missing and str(args.umklapp_completion) == "strict":
                    raise SystemExit(
                        f"Missing finite-q wrapped overlap shifts for q={q_shift} channel={channel}: {list(missing)[:20]}. "
                        "Rerun with --umklapp-completion build on a compute node to construct closure keys."
                    )
                run_for_block = run
                built_missing = False
                if missing and str(args.umklapp_completion) == "build":
                    run = _run_with_completed_umklapp(run, tuple(sorted(missing)))
                    run_for_block = run
                    built_missing = True

                quotient_block_meta = None
                if typed_single_representative:
                    signed_matrices = (
                        build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs(
                            run_for_block,
                            orbitals,
                            pairs,
                            q_shift,
                            channel=str(channel),
                            beta=float(args.beta),
                            structure_tolerance=float(args.structure_tolerance),
                            require_complete_umklapp=True,
                            physical_shifts=physical_shifts,
                            require_provenance=True,
                        )
                    )
                    matrices = signed_matrices.plus
                    partner_matrices = signed_matrices.minus
                    quotient_block_meta = {
                        "matrix_api": (
                            "build_rlg_hbn_tdhf_finite_q_single_representative_matrix_pair_from_pairs"
                        ),
                        "finite_cutoff_rule": "fixed_physical_G_shell",
                        "fixed_node_rule": "single_stored_torus_representative",
                        "source_provenance_validated": True,
                        "source_closure": single_representative_source_closure,
                    }
                elif str(channel) == "intraflavor" and bool(
                    args.c3_quotient_provider
                ):
                    q_key = (
                        int(q_shift[0]) % int(mesh_shape[0]),
                        int(q_shift[1]) % int(mesh_shape[1]),
                    )
                    representative = _c3_orbit_representative(
                        q_key, int(mesh_shape[0])
                    )
                    if q_key not in quotient_matrix_cache:
                        cycle = build_rlg_hbn_tdhf_c3_quotient_cycle(
                            run_for_block,
                            orbitals,
                            representative,
                            beta=float(args.beta),
                            physical_shifts=physical_shifts,
                            structure_tolerance=float(args.structure_tolerance),
                            closure_tolerance=1.0e-9,
                        )
                        quotient_matrix_cache.update(cycle.matrices)
                        cycle_meta = {
                            "representative_shift": [
                                int(x) for x in representative
                            ],
                            "cycle_shifts": [
                                [int(x) for x in shift]
                                for shift in cycle.shifts
                            ],
                            "closure_residuals": {
                                name: float(value)
                                for name, value in cycle.closure_residuals.items()
                            },
                            "step_metadata": [
                                step.metadata for step in cycle.steps
                            ],
                        }
                        for cycle_shift in cycle.shifts:
                            quotient_cycle_metadata[cycle_shift] = cycle_meta
                    matrices = quotient_matrix_cache[q_key]
                    quotient_block_meta = quotient_cycle_metadata[q_key]
                elif str(channel) == "intraflavor":
                    matrices = build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
                        run_for_block,
                        orbitals,
                        pairs,
                        q_shift,
                        beta=float(args.beta),
                        structure_tolerance=float(args.structure_tolerance),
                        require_complete_umklapp=(
                            str(args.umklapp_completion) != "allow-incomplete"
                        ),
                        physical_shifts=physical_shifts,
                    )
                else:
                    matrices = build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
                        run_for_block,
                        orbitals,
                        pairs,
                        q_shift,
                        beta=float(args.beta),
                        structure_tolerance=float(args.structure_tolerance),
                        require_complete_umklapp=(
                            str(args.umklapp_completion) != "allow-incomplete"
                        ),
                        physical_shifts=physical_shifts,
                    )
                if not typed_single_representative:
                    partner_matrices = None
            spectrum = solve_tdhf_matrices(
                matrices,
                energy_tol=float(args.energy_tol),
                imag_tol=float(args.imag_tol),
                norm_tol=float(args.norm_tol),
            )
            if partner_matrices is None:
                partner_spectrum = None
                particle_hole_q_minus_residual = None
                quartet_residual = None
            elif tuple(int(v) for v in q_shift) == (0, 0):
                partner_spectrum = spectrum
                particle_hole_q_minus_residual = _spectrum_assignment_residual(
                    spectrum.raw_eigenvalues,
                    -np.conj(spectrum.raw_eigenvalues),
                )
                quartet_residual = particle_hole_q_minus_residual
            else:
                partner_spectrum = solve_tdhf_matrices(
                    partner_matrices,
                    energy_tol=float(args.energy_tol),
                    imag_tol=float(args.imag_tol),
                    norm_tol=float(args.norm_tol),
                )
                particle_hole_q_minus_residual = _spectrum_assignment_residual(
                    spectrum.raw_eigenvalues,
                    -np.conj(partner_spectrum.raw_eigenvalues),
                )
                quartet_residual = particle_hole_q_minus_residual

            pair_particle = np.asarray([pair.particle for pair in matrices.pairs], dtype=int)
            pair_hole = np.asarray([pair.hole for pair in matrices.pairs], dtype=int)
            pair_particle_k = np.asarray(
                [pair.particle_momentum if pair.particle_momentum is not None else -1 for pair in matrices.pairs],
                dtype=int,
            )
            pair_hole_k = np.asarray(
                [pair.hole_momentum if pair.hole_momentum is not None else -1 for pair in matrices.pairs],
                dtype=int,
            )
            q_array = np.asarray(q_shift, dtype=int)
            q_frac = np.asarray([float(q_shift[0]) / float(mesh_shape[0]), float(q_shift[1]) / float(mesh_shape[1])])
            q_centered_frac = center_reciprocal_fractional_coordinates(q_frac)
            q_cartesian_nm_inv = finite_q_shift_cartesian_nm_inv(
                run.basis_data.basis_model.lattice,
                q_shift,
                mesh_shape,
                centered=True,
            )
            arrays: dict[str, object] = {
                "energies_mev": spectrum.energies,
                "eigenvalues": spectrum.eigenvalues,
                "eta_norms": spectrum.eta_norms,
                "residuals": spectrum.residuals,
                "raw_eigenvalues": spectrum.raw_eigenvalues,
                "raw_eta_norms": spectrum.raw_eta_norms,
                "raw_solver_residuals": spectrum.raw_residuals,
                "selected_indices": spectrum.selected_indices,
                "pair_particle": pair_particle,
                "pair_hole": pair_hole,
                "pair_particle_k": pair_particle_k,
                "pair_hole_k": pair_hole_k,
                "q_shift": q_array,
                "q_frac": q_frac,
                "q_centered_frac": q_centered_frac,
                "q_cartesian_nm_inv": q_cartesian_nm_inv,
            }
            if partner_spectrum is not None:
                assert partner_matrices is not None
                minus_pair_particle = np.asarray(
                    [pair.particle for pair in partner_matrices.pairs], dtype=int
                )
                minus_pair_hole = np.asarray(
                    [pair.hole for pair in partner_matrices.pairs], dtype=int
                )
                minus_pair_particle_k = np.asarray(
                    [
                        pair.particle_momentum
                        if pair.particle_momentum is not None
                        else -1
                        for pair in partner_matrices.pairs
                    ],
                    dtype=int,
                )
                minus_pair_hole_k = np.asarray(
                    [
                        pair.hole_momentum
                        if pair.hole_momentum is not None
                        else -1
                        for pair in partner_matrices.pairs
                    ],
                    dtype=int,
                )
                arrays.update(
                    {
                        "minus_q_shift": -q_array,
                        "minus_pair_particle": minus_pair_particle,
                        "minus_pair_hole": minus_pair_hole,
                        "minus_pair_particle_k": minus_pair_particle_k,
                        "minus_pair_hole_k": minus_pair_hole_k,
                        "minus_energies_mev": partner_spectrum.energies,
                        "minus_eigenvalues": partner_spectrum.eigenvalues,
                        "minus_eta_norms": partner_spectrum.eta_norms,
                        "minus_residuals": partner_spectrum.residuals,
                        "minus_raw_eigenvalues": partner_spectrum.raw_eigenvalues,
                        "minus_raw_eta_norms": partner_spectrum.raw_eta_norms,
                        "minus_raw_solver_residuals": partner_spectrum.raw_residuals,
                        "minus_selected_indices": partner_spectrum.selected_indices,
                    }
                )
            if not args.no_save_matrices:
                arrays["A"] = matrices.A
                arrays["B"] = matrices.B
                arrays["L"] = matrices.L
                if partner_matrices is not None:
                    arrays["A_minus_q"] = partner_matrices.A
                    arrays["B_minus_q"] = partner_matrices.B
                    arrays["L_minus_q"] = partner_matrices.L
            if not args.no_save_vectors:
                arrays["X"] = spectrum.X
                arrays["Y"] = spectrum.Y
                if partner_spectrum is not None:
                    arrays["X_minus_q"] = partner_spectrum.X
                    arrays["Y_minus_q"] = partner_spectrum.Y
            spectrum_name = f"tdhf_finite_q_{channel}_{_q_label(q_shift)}_spectrum.npz"
            spectrum_path = output_dir / spectrum_name
            _atomic_savez(spectrum_path, **arrays)

            block_summaries.append(
                {
                    "q_shift": [int(q_shift[0]), int(q_shift[1])],
                    "q_frac": [float(q_frac[0]), float(q_frac[1])],
                    "q_centered_frac": [float(q_centered_frac[0]), float(q_centered_frac[1])],
                    "q_cartesian_nm_inv": [float(q_cartesian_nm_inv[0]), float(q_cartesian_nm_inv[1])],
                    "q_coordinate_convention": "centered repeated-zone Cartesian q in nm^-1; componentwise centered reciprocal coordinates, no Wigner-Seitz folding",
                    "channel": str(channel),
                    "channel_counts": channel_counts,
                    "n_pairs": n_pairs,
                    "liouvillian_dim": int(matrices.L.shape[0]),
                    "estimated_dense_memory_gib": float(estimated_bytes / 1024**3),
                    "saved_matrices": not bool(args.no_save_matrices),
                    "saved_vectors": not bool(args.no_save_vectors),
                    "required_overlap_shifts": [list(s) for s in required_shifts],
                    "missing_overlap_shifts": [list(s) for s in missing],
                    "built_missing_overlap_shifts": bool(built_missing),
                    "c3_quotient_provider": (
                        quotient_block_meta if typed_quotient else None
                    ),
                    "single_representative_provider": (
                        quotient_block_meta
                        if typed_single_representative
                        else None
                    ),
                    "spectrum_npz": str(spectrum_path),
                    "liouvillian_convention": (
                        "finite_q_partner_plus_minus"
                        if partner_matrices is not None
                        and tuple(int(v) for v in q_shift) != (0, 0)
                        else "standard_q0_style"
                    ),
                    "structure": {
                        "A_hermitian": float(matrices.structure.a_hermitian),
                        "B_symmetric": float(matrices.structure.b_symmetric),
                        "particle_hole_symmetry": float(matrices.structure.particle_hole_symmetry),
                        "tolerance": float(matrices.structure.tolerance),
                        "ok": bool(matrices.structure.ok),
                    },
                    "spectrum": {
                        "selected_count": int(spectrum.energies.size),
                        "first_positive_energies_mev": [float(value) for value in spectrum.energies[:20]],
                        "raw_eigenvalue_summary": _raw_real_eigenvalue_summary(
                            spectrum.raw_eigenvalues,
                            imag_tol=float(args.imag_tol),
                            energy_tol=max(float(args.energy_tol), 1.0e-6),
                        ),
                        "same_q_plus_minus_nonreciprocity_diagnostic_mev": float(
                            spectrum.pairing_residual
                        ),
                        "particle_hole_q_minus_assignment_residual_mev": (
                            None
                            if particle_hole_q_minus_residual is None
                            else float(particle_hole_q_minus_residual)
                        ),
                        "quartet_residual_mev": (
                            None
                            if quartet_residual is None
                            else float(quartet_residual)
                        ),
                        "minus_raw_eigenvalue_summary": (
                            None
                            if partner_spectrum is None
                            else _raw_real_eigenvalue_summary(
                                partner_spectrum.raw_eigenvalues,
                                imag_tol=float(args.imag_tol),
                                energy_tol=max(float(args.energy_tol), 1.0e-6),
                            )
                        ),
                        "max_residual": float(np.max(spectrum.residuals)) if spectrum.residuals.size else 0.0,
                        "raw_eta_norm_range": [
                            float(np.min(spectrum.raw_eta_norms)),
                            float(np.max(spectrum.raw_eta_norms)),
                        ] if spectrum.raw_eta_norms.size else [0.0, 0.0],
                        "max_raw_solver_residual": (
                            float(np.max(spectrum.raw_residuals))
                            if spectrum.raw_residuals.size
                            else 0.0
                        ),
                    },
                }
            )
            print(f"[block] q={q_shift} channel={channel} n_pairs={n_pairs} modes={spectrum.energies.size}")

    summary_payload = {
        "hf_archive": str(archive_path),
        "output_dir": str(output_dir),
        "channels": list(args.channels),
        "q_shifts": [list(q) for q in args.q_shifts],
        "umklapp_completion": str(args.umklapp_completion),
        "scope": (
            "typed variational-v2 microscopic finite-q quotient matrices for separated channels"
            if typed_quotient
            else (
                "typed fixed-G single-representative signed finite-q matrices for separated channels"
                if typed_single_representative
                else "legacy full intraflavor plus polarized flavor-flip shortcuts"
            )
        ),
        "implemented_channels": list(FINITE_Q_CHANNELS),
        "remaining_limitations": (
            "separated channels only; mixed all-channel blocks and iterative eigensolvers are not assembled"
        ),
        "typed_finite_q_quotient": typed_quotient,
        "typed_finite_q_single_representative": typed_single_representative,
        "finite_q_matrix_api": config_payload["finite_q_matrix_api"],
        "single_representative_source_closure": (
            single_representative_source_closure
        ),
        "c3_quotient_provider_compatibility_flag": bool(
            args.c3_quotient_provider
        ),
        "physical_shifts": [list(s) for s in physical_shifts],
        "mesh_shape": [int(mesh_shape[0]), int(mesh_shape[1])],
        "q_coordinate_convention": "q_shift / mesh_shape is centered componentwise to [-1/2, 1/2) and converted with the RLG/hBN moire reciprocal vectors to Cartesian nm^-1; do not Wigner-Seitz fold Fig. S45 q-mesh points",
        "blocks": block_summaries,
        "hf_summary": summary,
        "elapsed_sec": float(perf_counter() - start),
    }
    write_json(output_dir / "tdhf_finite_q_summary.json", summary_payload)
    print(f"[done] output_dir={output_dir}")
    print(f"[done] blocks={len(block_summaries)} elapsed_sec={summary_payload['elapsed_sec']:.3f}")


if __name__ == "__main__":
    main()
