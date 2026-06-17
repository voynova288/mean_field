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

from mean_field.core.hf import solve_tdhf_matrices, split_pair_indices_by_flavor_channel
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.RnG_hBN import (
    RLGhBNLayerOverlapBlockSet,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs,
    build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_q_pairs,
    load_rlg_hbn_tdhf_run_from_archive,
    required_rlg_hbn_tdhf_finite_q_overlap_shifts,
)
from mean_field.workflows import collect_slurm_metadata

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "RnG_hBN" / "tdhf"
FINITE_Q_CHANNELS = ("intervalley", "interspin", "inter_spin_valley")
UMKLAPP_COMPLETION_CHOICES = ("strict", "build", "allow-incomplete")


def _parse_channels(text: str) -> tuple[str, ...]:
    channels = tuple(item.strip() for item in text.split(",") if item.strip())
    if not channels:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated channel")
    invalid = sorted(set(channels).difference(FINITE_Q_CHANNELS))
    if invalid:
        raise argparse.ArgumentTypeError(
            "Finite-q shortcut channels must be flavor-flip channels "
            f"{FINITE_Q_CHANNELS}; got {invalid}"
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
            "Run dense finite-q RLG/hBN TDHF/RPA postprocessing for the implemented "
            "conduction-only flavor-flip exchange shortcut."
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
        help="Comma-separated finite-q shortcut channels. Choices: intervalley,interspin,inter_spin_valley.",
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
    parser.add_argument("--max-pairs", type=int, default=4096, help="Refuse dense assembly above this ph-pair count.")
    parser.add_argument("--max-dense-memory-gb", type=float, default=8.0, help="Conservative dense TDHF memory estimate limit per q/channel block.")
    parser.add_argument("--structure-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--imag-tol", type=float, default=1.0e-8)
    parser.add_argument("--energy-tol", type=float, default=1.0e-10)
    parser.add_argument("--norm-tol", type=float, default=1.0e-10)
    parser.add_argument("--allow-unconverged", action="store_true", help="Do not reject archives whose summary says converged=false.")
    parser.add_argument("--dry-run", action="store_true", help="Write/print resolved config only; do not load cache or solve TDHF.")
    parser.add_argument("--no-save-vectors", action="store_true", help="Do not store X/Y mode vectors in NPZ outputs.")
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
    finite_real = values[np.isfinite(values.real) & np.isfinite(values.imag) & (np.abs(values.imag) <= float(imag_tol))]
    real_values = np.sort(finite_real.real.astype(float, copy=False))
    near_zero = real_values[np.abs(real_values) <= float(energy_tol)]
    low_abs = real_values[np.argsort(np.abs(real_values))[:20]] if real_values.size else np.asarray([], dtype=float)
    return {
        "real_count": int(real_values.size),
        "near_zero_count": int(near_zero.size),
        "near_zero_eigenvalues_mev": [float(value) for value in near_zero[:40]],
        "lowest_abs_real_eigenvalues_mev": [float(value) for value in low_abs],
        "lowest_real_eigenvalues_mev": [float(value) for value in real_values[:20]],
        "highest_real_eigenvalues_mev": [float(value) for value in real_values[-20:]],
    }


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
        "max_pairs": int(args.max_pairs),
        "max_dense_memory_gb": float(args.max_dense_memory_gb),
        "structure_tolerance": float(args.structure_tolerance),
        "imag_tol": float(args.imag_tol),
        "energy_tol": float(args.energy_tol),
        "norm_tol": float(args.norm_tol),
        "allow_unconverged": bool(args.allow_unconverged),
        "dry_run": bool(args.dry_run),
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
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    physical_shifts = tuple((int(g[0]), int(g[1])) for g in run.overlap_blocks.shifts)
    mesh_shape = _mesh_shape_from_frac(run.basis_data.k_grid_frac)
    completed_umklapp_cache: dict[tuple[tuple[int, int], ...], object] = {}

    block_summaries: list[dict[str, object]] = []
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

            required_shifts = required_rlg_hbn_tdhf_finite_q_overlap_shifts(
                orbitals,
                run.basis_data,
                pairs,
                q_shift,
                physical_shifts=physical_shifts,
            )
            available = set(tuple(int(v) for v in key) for key in run.overlap_blocks.layer_overlaps)
            missing = tuple(shift for shift in required_shifts if shift not in available)
            if missing and str(args.umklapp_completion) == "strict":
                raise SystemExit(
                    f"Missing finite-q wrapped overlap shifts for q={q_shift} channel={channel}: {list(missing)[:20]}. "
                    "Rerun with --umklapp-completion build on a compute node to construct closure keys."
                )
            run_for_block = run
            built_missing = False
            if missing and str(args.umklapp_completion) == "build":
                missing_key = tuple(sorted(missing))
                if missing_key not in completed_umklapp_cache:
                    completed_umklapp_cache[missing_key] = _run_with_completed_umklapp(run, missing_key)
                    built_missing = True
                run_for_block = completed_umklapp_cache[missing_key]

            matrices = build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
                run_for_block,
                orbitals,
                pairs,
                q_shift,
                beta=float(args.beta),
                structure_tolerance=float(args.structure_tolerance),
                require_complete_umklapp=str(args.umklapp_completion) != "allow-incomplete",
                physical_shifts=physical_shifts,
            )
            spectrum = solve_tdhf_matrices(
                matrices,
                energy_tol=float(args.energy_tol),
                imag_tol=float(args.imag_tol),
                norm_tol=float(args.norm_tol),
            )

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
            arrays: dict[str, object] = {
                "energies_mev": spectrum.energies,
                "eigenvalues": spectrum.eigenvalues,
                "eta_norms": spectrum.eta_norms,
                "residuals": spectrum.residuals,
                "raw_eigenvalues": spectrum.raw_eigenvalues,
                "selected_indices": spectrum.selected_indices,
                "pair_particle": pair_particle,
                "pair_hole": pair_hole,
                "pair_particle_k": pair_particle_k,
                "pair_hole_k": pair_hole_k,
                "q_shift": q_array,
                "q_frac": q_frac,
                "A": matrices.A,
                "B": matrices.B,
            }
            if not args.no_save_vectors:
                arrays["X"] = spectrum.X
                arrays["Y"] = spectrum.Y
            spectrum_name = f"tdhf_finite_q_{channel}_{_q_label(q_shift)}_spectrum.npz"
            spectrum_path = output_dir / spectrum_name
            _atomic_savez(spectrum_path, **arrays)

            block_summaries.append(
                {
                    "q_shift": [int(q_shift[0]), int(q_shift[1])],
                    "q_frac": [float(q_frac[0]), float(q_frac[1])],
                    "channel": str(channel),
                    "channel_counts": channel_counts,
                    "n_pairs": n_pairs,
                    "liouvillian_dim": int(matrices.L.shape[0]),
                    "estimated_dense_memory_gib": float(estimated_bytes / 1024**3),
                    "required_overlap_shifts": [list(s) for s in required_shifts],
                    "missing_overlap_shifts": [list(s) for s in missing],
                    "built_missing_overlap_shifts": bool(built_missing),
                    "spectrum_npz": str(spectrum_path),
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
                        "pairing_residual": float(spectrum.pairing_residual),
                        "max_residual": float(np.max(spectrum.residuals)) if spectrum.residuals.size else 0.0,
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
        "shortcut_scope": "conduction-only fully spin-valley-polarized flavor-flip exchange-only finite-q TDHF",
        "implemented_channels": list(FINITE_Q_CHANNELS),
        "not_implemented": "full finite-q intraflavor A/B Eq. D19 bookkeeping",
        "physical_shifts": [list(s) for s in physical_shifts],
        "mesh_shape": [int(mesh_shape[0]), int(mesh_shape[1])],
        "blocks": block_summaries,
        "hf_summary": summary,
        "elapsed_sec": float(perf_counter() - start),
    }
    write_json(output_dir / "tdhf_finite_q_summary.json", summary_payload)
    print(f"[done] output_dir={output_dir}")
    print(f"[done] blocks={len(block_summaries)} elapsed_sec={summary_payload['elapsed_sec']:.3f}")


if __name__ == "__main__":
    main()
