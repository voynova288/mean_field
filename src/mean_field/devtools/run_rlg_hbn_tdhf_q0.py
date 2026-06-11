from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
from time import perf_counter

import numpy as np

from mean_field.core.hf import (
    check_single_flavor_simplification,
    solve_tdhf_matrices,
    split_pair_indices_by_flavor_channel,
)
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.systems.RnG_hBN import (
    build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_q0_matrices_from_pairs,
    build_rlg_hbn_tdhf_q0_pairs,
    load_rlg_hbn_tdhf_run_from_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "RnG_hBN" / "tdhf"
CHANNEL_CHOICES = ("all", "intraflavor", "intervalley", "interspin", "inter_spin_valley")
SHORTCUT_CHOICES = ("auto", "on", "off")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense q=0 TDHF/RPA postprocessing from a saved RLG/hBN HF archive."
    )
    parser.add_argument("--hf-archive", type=Path, required=True, help="Path to hf_run_state.npz or hf_ground_state.npz.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Optional hf_run_summary.json for run metadata.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override cache directory if the archive lacks cache_dir.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults under results/RnG_hBN/tdhf.")
    parser.add_argument("--channel", choices=CHANNEL_CHOICES, default="all")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=2048, help="Refuse dense assembly above this ph-pair count.")
    parser.add_argument("--max-dense-memory-gb", type=float, default=8.0, help="Conservative dense TDHF memory estimate limit.")
    parser.add_argument("--structure-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--imag-tol", type=float, default=1.0e-8)
    parser.add_argument("--energy-tol", type=float, default=1.0e-10)
    parser.add_argument("--norm-tol", type=float, default=1.0e-10)
    parser.add_argument("--single-flavor-shortcut", choices=SHORTCUT_CHOICES, default="auto")
    parser.add_argument("--assembly", choices=("vectorized", "generic"), default="vectorized")
    parser.add_argument("--allow-unconverged", action="store_true", help="Do not reject archives whose summary says converged=false.")
    parser.add_argument("--dry-run", action="store_true", help="Write/print resolved config only; do not load cache or solve TDHF.")
    parser.add_argument("--no-save-vectors", action="store_true", help="Do not store X/Y mode vectors in the NPZ output.")
    return parser.parse_args()


def _default_output_dir(hf_archive: Path, channel: str) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    stem = hf_archive.stem
    suffix = f"{stem}_{channel}_q0"
    if job_id:
        suffix += f"_{job_id}"
    else:
        suffix += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / suffix


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def _load_summary(summary_path: Path | None, archive_path: Path) -> dict[str, object]:
    path = summary_path if summary_path is not None else archive_path.with_name("hf_run_summary.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _flavor_counts(state) -> dict[tuple[int, int], int]:
    if state.occupation_counts is None:
        return {}
    counts = np.asarray(state.occupation_counts, dtype=int).reshape((int(state.n_spin), int(state.n_eta)), order="C")
    return {(int(s), int(e)): int(counts[s, e]) for s in range(counts.shape[0]) for e in range(counts.shape[1])}


def _shortcut_decision(state, mode: str, channel: str) -> tuple[bool, str]:
    if channel == "all":
        return False, "single-flavor shortcut is not applied to mixed all-channel blocks"
    if channel == "intraflavor":
        return False, "single-flavor shortcut is not applied to intraflavor blocks"
    if mode == "off":
        return False, "disabled by --single-flavor-shortcut=off"
    counts = _flavor_counts(state)
    if not counts:
        status = check_single_flavor_simplification(
            active_space_has_valence=bool(int(state.active_valence_bands) > 0),
            occupied_flavor_counts={},
            polarized_flavor=(0, 0),
        )
        if mode == "on":
            raise ValueError("--single-flavor-shortcut=on requires saved occupation_counts metadata")
        return False, status.reason
    polarized_candidates = [flavor for flavor, count in counts.items() if int(count) > 0]
    polarized = polarized_candidates[0] if polarized_candidates else next(iter(counts))
    status = check_single_flavor_simplification(
        active_space_has_valence=bool(int(state.active_valence_bands) > 0),
        occupied_flavor_counts=counts,
        polarized_flavor=polarized,
    )
    if mode == "on" and not status.allowed:
        raise ValueError(f"single-flavor shortcut requested but illegal: {status.reason}")
    return bool(status.allowed and mode in {"auto", "on"}), status.reason


def _filter_pairs(pairs, channel: str):
    if channel == "all":
        return tuple(pairs), {name: int(len(indices)) for name, indices in split_pair_indices_by_flavor_channel(pairs).items()}
    groups = split_pair_indices_by_flavor_channel(pairs)
    indices = groups[channel]
    return tuple(pairs[int(index)] for index in indices), {name: int(len(values)) for name, values in groups.items()}


def _dense_memory_estimate_bytes(n_pairs: int, *, save_vectors: bool) -> int:
    # Conservative resident-memory proxy for A/B, L, raw eigvecs/workspace and
    # optionally saved X/Y.  It is not a LAPACK guarantee, but it prevents
    # accidental all-channel dense runs that are clearly too large.
    p2 = int(n_pairs) * int(n_pairs)
    multiplier = 192 if save_vectors else 160
    return int(multiplier * p2)



def _raw_real_eigenvalue_summary(raw_eigenvalues: np.ndarray, *, imag_tol: float, energy_tol: float) -> dict[str, object]:
    """Summarize unfiltered Liouvillian eigenvalues, including exact/near-zero modes."""

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


def main() -> None:
    start = perf_counter()
    args = _parse_args()
    archive_path = args.hf_archive.expanduser().resolve()
    summary = _load_summary(args.summary_path, archive_path)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else _default_output_dir(archive_path, args.channel).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "hf_archive": str(archive_path),
        "summary_path": "" if args.summary_path is None else str(args.summary_path.expanduser().resolve()),
        "cache_dir": "" if args.cache_dir is None else str(args.cache_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "channel": str(args.channel),
        "beta": float(args.beta),
        "max_pairs": int(args.max_pairs),
        "max_dense_memory_gb": float(args.max_dense_memory_gb),
        "structure_tolerance": float(args.structure_tolerance),
        "imag_tol": float(args.imag_tol),
        "energy_tol": float(args.energy_tol),
        "norm_tol": float(args.norm_tol),
        "single_flavor_shortcut": str(args.single_flavor_shortcut),
        "assembly": str(args.assembly),
        "allow_unconverged": bool(args.allow_unconverged),
        "dry_run": bool(args.dry_run),
        "summary_converged": bool(summary.get("converged", False)) if summary else None,
        "runtime": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        },
    }
    write_json(output_dir / "tdhf_q0_config.json", config_payload)
    if args.dry_run:
        print(f"[dry-run] output_dir={output_dir}")
        print(f"[dry-run] config={config_payload}")
        return

    ensure_not_running_compute_on_login_node("RLG/hBN q=0 TDHF postprocessing")
    if summary and not bool(summary.get("converged", False)) and not args.allow_unconverged:
        raise SystemExit(
            "Refusing to run TDHF on an unconverged HF archive; pass --allow-unconverged for diagnostics only."
        )

    run = load_rlg_hbn_tdhf_run_from_archive(
        archive_path,
        cache_dir=args.cache_dir,
        summary_path=args.summary_path,
    )
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    pairs, channel_counts = _filter_pairs(all_pairs, str(args.channel))
    if len(pairs) > int(args.max_pairs):
        raise SystemExit(
            f"Refusing dense TDHF assembly for {len(pairs)} ph pairs above max_pairs={args.max_pairs}. "
            "Use channel filtering, raise --max-pairs on a compute node, or implement block/matvec TDHF."
        )
    estimated_bytes = _dense_memory_estimate_bytes(len(pairs), save_vectors=not bool(args.no_save_vectors))
    memory_limit_bytes = int(float(args.max_dense_memory_gb) * 1024**3)
    if estimated_bytes > memory_limit_bytes:
        raise SystemExit(
            f"Refusing dense TDHF assembly: estimated memory {estimated_bytes / 1024**3:.2f} GiB "
            f"exceeds --max-dense-memory-gb={float(args.max_dense_memory_gb):.2f}. "
            "Use channel filtering, --no-save-vectors, or an explicit higher Slurm memory budget."
        )

    use_shortcut, shortcut_reason = _shortcut_decision(run.state, str(args.single_flavor_shortcut), str(args.channel))
    matrices = build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        beta=float(args.beta),
        include_direct_terms=not use_shortcut,
        include_exchange_terms=True,
        include_b_terms=not use_shortcut,
        structure_tolerance=float(args.structure_tolerance),
        assembly=str(args.assembly),
    )
    spectrum = solve_tdhf_matrices(
        matrices,
        energy_tol=float(args.energy_tol),
        imag_tol=float(args.imag_tol),
        norm_tol=float(args.norm_tol),
    )

    pair_particle = np.asarray([pair.particle for pair in matrices.pairs], dtype=int)
    pair_hole = np.asarray([pair.hole for pair in matrices.pairs], dtype=int)
    pair_k = np.asarray([pair.hole_momentum if pair.hole_momentum is not None else -1 for pair in matrices.pairs], dtype=int)
    arrays: dict[str, object] = {
        "energies_mev": spectrum.energies,
        "eigenvalues": spectrum.eigenvalues,
        "eta_norms": spectrum.eta_norms,
        "residuals": spectrum.residuals,
        "raw_eigenvalues": spectrum.raw_eigenvalues,
        "selected_indices": spectrum.selected_indices,
        "pair_particle": pair_particle,
        "pair_hole": pair_hole,
        "pair_k": pair_k,
        "A": matrices.A,
        "B": matrices.B,
    }
    if not args.no_save_vectors:
        arrays["X"] = spectrum.X
        arrays["Y"] = spectrum.Y
    spectrum_path = output_dir / "tdhf_q0_spectrum.npz"
    _atomic_savez(spectrum_path, **arrays)

    summary_payload = {
        "hf_archive": str(archive_path),
        "output_dir": str(output_dir),
        "spectrum_npz": str(spectrum_path),
        "channel": str(args.channel),
        "channel_counts": channel_counts,
        "n_pairs": int(len(matrices.pairs)),
        "liouvillian_dim": int(matrices.L.shape[0]),
        "estimated_dense_memory_gib": float(estimated_bytes / 1024**3),
        "single_flavor_shortcut_used": bool(use_shortcut),
        "single_flavor_shortcut_reason": str(shortcut_reason),
        "assembly": str(args.assembly),
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
        "hf_summary": summary,
        "elapsed_sec": float(perf_counter() - start),
    }
    write_json(output_dir / "tdhf_q0_summary.json", summary_payload)
    print(f"[done] output_dir={output_dir}")
    print(f"[done] n_pairs={len(matrices.pairs)} modes={spectrum.energies.size}")


if __name__ == "__main__":
    main()
