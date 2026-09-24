"""Qualify the accepted TPT bulk DFT/Wannier wavefunction bridge.

This is a read-only scientific qualification.  It reconstructs the executed
600-band -> 320-Wannier gauge from ``wannier90.chk``, compares it to the
accepted HR on the exact 12x4x4 source mesh, checks WAVECAR eigenvalues against
``wannier90.eig``, diagnoses smeared occupations, and compares direct
pseudo-wavefunction overlaps with PAW-aware ``wannier90.mmn`` neighbour blocks
at interior and reciprocal-boundary k points.

The report cannot authorize Hartree--Fock.  In particular, a nonzero direct
WAVECAR ``Q=0`` identity residual is evidence that PAW augmentation is still
required rather than a tolerance to waive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np

from mean_field.core.io import (
    publish_staged_file_noreplace,
    unique_staging_path,
    write_json_artifact,
)
from mean_field.runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.tpt_bulk.source import (
    TPT_BULK_HR_CHECKPOINT_ACTIVE_ENERGY_TOLERANCE_EV,
    TPT_BULK_HR_CHECKPOINT_ACTIVE_SUBSPACE_MIN_SV,
    TPT_BULK_HR_SHA256,
    TPT_BULK_HR_TEXT_COEFFICIENT_RESOLUTION_EV,
    TPT_BULK_SOURCE_MESH,
    build_tpt_bulk_active_eigensystem,
    load_tpt_bulk_source,
)
from mean_field.systems.tpt_bulk.wavefunctions import (
    TPT_BULK_WANNIER90_CHK_SHA256,
    TPT_BULK_WANNIER90_EIG_SHA256,
    TPT_BULK_WANNIER90_MMN_SHA256,
    TPT_BULK_WAVECAR_SHA256,
    Wannier90CheckpointReader,
    WavecarSpinorReader,
    contract_spinor_plane_wave_density_vertex,
    iter_wannier90_mmn_blocks,
    project_spinor_plane_wave_coefficients,
    read_wannier90_eig,
)

PARENT_QUALIFICATION_SELECTED_PROVENANCE_VERSION = (
    "parent_qualification_selected_provenance_v2"
)
PARENT_QUALIFICATION_SELECTED_PROVENANCE_SCOPE = (
    "parent_qualification_v2_selected_set_not_complete_python_import_closure"
)
_PARENT_QUALIFICATION_SELECTED_PROVENANCE_PATHS = (
    "scripts/mean_field_tools.py",
    "scripts/submit_mean_field.sbatch",
    "src/mean_field/runtime.py",
    "src/mean_field/core/io/__init__.py",
    "src/mean_field/core/io/artifacts.py",
    "src/mean_field/devtools/qualify_tpt_bulk_parent_wavefunctions.py",
    "src/mean_field/systems/tpt_bulk/__init__.py",
    "src/mean_field/systems/tpt_bulk/microscopic_hf.py",
    "src/mean_field/systems/tpt_bulk/source.py",
    "src/mean_field/systems/tpt_bulk/wavefunctions.py",
)


def _qualification_provenance_fields(
    hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "selected_provenance_version": (
            PARENT_QUALIFICATION_SELECTED_PROVENANCE_VERSION
        ),
        "selected_provenance_scope": PARENT_QUALIFICATION_SELECTED_PROVENANCE_SCOPE,
        "selected_provenance_hashes_at_qualification_start": hashes,
    }


def _qualification_report_identity(hashes: dict[str, str]) -> dict[str, object]:
    return {
        "schema": "tpt-bulk-parent-wavefunction-bridge-qualification-v5",
        "status": "pseudo_only_not_paw_complete",
        **_qualification_provenance_fields(hashes),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to replace existing report: {path}")
    staged = unique_staging_path(path)
    try:
        write_json_artifact(payload, staged)
        publish_staged_file_noreplace(staged, path)
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def qualify_parent(
    *,
    source_root: Path,
    parent_root: Path,
    active_ranks_1based: tuple[int, int],
    anchor_k_indices: tuple[int, ...],
    mmn_source_k_indices: tuple[int, ...],
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    selected_provenance_hashes = {
        relative: _sha256(repository_root / relative)
        for relative in _PARENT_QUALIFICATION_SELECTED_PROVENANCE_PATHS
    }
    source = load_tpt_bulk_source(source_root)
    eigensystem = build_tpt_bulk_active_eigensystem(
        source,
        mesh=TPT_BULK_SOURCE_MESH,
        active_ranks_1based=active_ranks_1based,
    )
    mmn_path = parent_root / "wannier90.mmn"

    with Wannier90CheckpointReader(
        parent_root / "wannier90.chk",
        expected_sha256=TPT_BULK_WANNIER90_CHK_SHA256,
    ) as checkpoint, WavecarSpinorReader(
        parent_root / "WAVECAR",
        expected_sha256=TPT_BULK_WAVECAR_SHA256,
    ) as wavecar:
        chk = checkpoint.metadata
        wav = wavecar.metadata
        if chk.mp_grid != TPT_BULK_SOURCE_MESH:
            raise ValueError("checkpoint mesh is not the accepted 12x4x4 source mesh")
        if (chk.num_kpts, chk.num_bands, chk.num_wann) != (192, 600, 320):
            raise ValueError("checkpoint dimensions are not 192 x 600 x 320")
        if (wav.num_kpts, wav.num_bands) != (chk.num_kpts, chk.num_bands):
            raise ValueError("WAVECAR and checkpoint dimensions disagree")
        if not np.allclose(
            chk.kpoints_fractional,
            eigensystem.k_fractional,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("checkpoint and accepted-HR k-point order disagree")
        if not np.allclose(
            wav.kpoints_fractional,
            eigensystem.k_fractional,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("WAVECAR and accepted-HR k-point order disagree")
        if not np.allclose(
            wav.lattice_angstrom,
            source.lattice_angstrom,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError("WAVECAR and accepted POSCAR lattices disagree")

        eig = read_wannier90_eig(
            parent_root / "wannier90.eig",
            num_bands=chk.num_bands,
            num_kpts=chk.num_kpts,
            expected_sha256=TPT_BULK_WANNIER90_EIG_SHA256,
        )
        wav_eig_error = float(np.max(np.abs(wav.eigenvalues_ev - eig)))

        gauge_by_k: dict[int, np.ndarray] = {}
        active_dft_by_k: dict[int, np.ndarray] = {}
        max_gauge_semiunitarity = 0.0
        max_active_energy_error = 0.0
        min_active_subspace_singular_value = 1.0
        active_smeared_occupation_traces: list[float] = []
        active_smeared_occupation_eigenvalue_min = 1.0
        active_smeared_occupation_eigenvalue_max = 0.0
        active_smeared_occupation_counts_above_half: list[int] = []
        first, last = active_ranks_1based
        for ik in range(chk.num_kpts):
            gauge = checkpoint.dft_to_wannier_gauge(ik)
            max_gauge_semiunitarity = max(
                max_gauge_semiunitarity,
                float(np.max(np.abs(gauge.conj().T @ gauge - np.eye(chk.num_wann)))),
            )
            h_wannier = gauge.conj().T @ (eig[ik, :, None] * gauge)
            h_wannier = 0.5 * (h_wannier + h_wannier.conj().T)
            values, vectors = np.linalg.eigh(h_wannier)
            selected_values = values[first - 1 : last]
            selected_vectors = vectors[:, first - 1 : last]
            max_active_energy_error = max(
                max_active_energy_error,
                float(
                    np.max(
                        np.abs(selected_values - eigensystem.active_energies_ev[:, ik])
                    )
                ),
            )
            overlap = (
                eigensystem.active_eigenvectors[:, :, ik].conj().T
                @ selected_vectors
            )
            min_active_subspace_singular_value = min(
                min_active_subspace_singular_value,
                float(np.linalg.svd(overlap, compute_uv=False)[-1]),
            )
            active_dft = gauge @ eigensystem.active_eigenvectors[:, :, ik]
            active_occupation = active_dft.conj().T @ (
                wav.occupations[ik, :, None] * active_dft
            )
            active_occupation = 0.5 * (
                active_occupation + active_occupation.conj().T
            )
            active_occupation_eigenvalues = np.linalg.eigvalsh(active_occupation)
            active_smeared_occupation_traces.append(
                float(np.trace(active_occupation).real)
            )
            active_smeared_occupation_eigenvalue_min = min(
                active_smeared_occupation_eigenvalue_min,
                float(active_occupation_eigenvalues[0]),
            )
            active_smeared_occupation_eigenvalue_max = max(
                active_smeared_occupation_eigenvalue_max,
                float(active_occupation_eigenvalues[-1]),
            )
            active_smeared_occupation_counts_above_half.append(
                int(np.count_nonzero(active_occupation_eigenvalues > 0.5))
            )
            if ik in anchor_k_indices:
                gauge_by_k[ik] = gauge
                active_dft_by_k[ik] = active_dft

        active_pw_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def active_pw(ik: int) -> tuple[np.ndarray, np.ndarray]:
            cached = active_pw_cache.get(ik)
            if cached is not None:
                return cached
            gauge = gauge_by_k.get(ik)
            if gauge is None:
                gauge = checkpoint.dft_to_wannier_gauge(ik)
                gauge_by_k[ik] = gauge
            active_dft = active_dft_by_k.get(ik)
            if active_dft is None:
                active_dft = gauge @ eigensystem.active_eigenvectors[:, :, ik]
                active_dft_by_k[ik] = active_dft
            outer = np.flatnonzero(chk.lwindow[:, ik])
            dft_coefficients = wavecar.read_spinor_coefficients(ik, outer)
            active_coefficients = project_spinor_plane_wave_coefficients(
                dft_coefficients,
                active_dft[outer, :],
            )
            result = (active_coefficients, wavecar.g_vectors(ik))
            active_pw_cache[ik] = result
            return result

        q0_checks: list[dict[str, Any]] = []
        for ik in anchor_k_indices:
            coefficients, g_vectors = active_pw(ik)
            rho_pseudo = contract_spinor_plane_wave_density_vertex(
                source_coefficients=coefficients,
                source_g_integer=g_vectors,
                target_coefficients=coefficients,
                target_g_integer=g_vectors,
                reciprocal_shift_integer=(0, 0, 0),
            )
            hermitian = 0.5 * (rho_pseudo + rho_pseudo.conj().T)
            eigenvalues = np.linalg.eigvalsh(hermitian)
            q0_checks.append(
                {
                    "k_index_0based": ik,
                    "k_fractional": wav.kpoints_fractional[ik].tolist(),
                    "max_abs_identity_residual": float(
                        np.max(np.abs(rho_pseudo - np.eye(last - first + 1)))
                    ),
                    "pseudo_overlap_eigenvalue_min": float(eigenvalues[0]),
                    "pseudo_overlap_eigenvalue_max": float(eigenvalues[-1]),
                    "pseudo_overlap_trace_real": float(np.trace(rho_pseudo).real),
                }
            )

        mmn_checks: list[dict[str, Any]] = []
        requested_mmn_sources = frozenset(mmn_source_k_indices)
        expected_mmn_checks = chk.nntot * len(requested_mmn_sources)
        for block_index, block in enumerate(
            iter_wannier90_mmn_blocks(
                mmn_path,
                expected_num_bands=chk.num_bands,
                expected_num_kpts=chk.num_kpts,
                expected_nntot=chk.nntot,
                expected_sha256=TPT_BULK_WANNIER90_MMN_SHA256,
            )
        ):
            if block.source_k_0based not in requested_mmn_sources:
                continue
            source_k = block.source_k_0based
            target_k = block.target_k_0based
            for ik in (source_k, target_k):
                if ik not in gauge_by_k:
                    gauge_by_k[ik] = checkpoint.dft_to_wannier_gauge(ik)
                if ik not in active_dft_by_k:
                    active_dft_by_k[ik] = (
                        gauge_by_k[ik]
                        @ eigensystem.active_eigenvectors[:, :, ik]
                    )
            source_active = active_dft_by_k[source_k]
            target_active = active_dft_by_k[target_k]
            augmented_source_bra_target_ket = (
                source_active.conj().T
                @ block.overlap_source_bra_target_ket
                @ target_active
            )
            rho_paw_aware = augmented_source_bra_target_ket.conj().T
            source_coefficients, source_g = active_pw(source_k)
            target_coefficients, target_g = active_pw(target_k)
            rho_pseudo = contract_spinor_plane_wave_density_vertex(
                source_coefficients=source_coefficients,
                source_g_integer=source_g,
                target_coefficients=target_coefficients,
                target_g_integer=target_g,
                reciprocal_shift_integer=block.target_cell_shift_integer,
            )
            difference = rho_pseudo - rho_paw_aware
            denominator = float(np.linalg.norm(rho_paw_aware))
            mmn_checks.append(
                {
                    "block_index_0based": block_index,
                    "source_k_0based": source_k,
                    "target_k_0based": target_k,
                    "target_cell_shift_integer": list(
                        block.target_cell_shift_integer
                    ),
                    "max_abs_pseudo_minus_paw_aware": float(
                        np.max(np.abs(difference))
                    ),
                    "relative_frobenius_pseudo_minus_paw_aware": float(
                        np.linalg.norm(difference) / denominator
                    ),
                    "paw_aware_frobenius_norm": denominator,
                }
            )
            if len(mmn_checks) == expected_mmn_checks:
                break
        if len(mmn_checks) != expected_mmn_checks:
            raise ValueError("did not read all requested neighbour blocks from wannier90.mmn")

        max_q0_residual = max(item["max_abs_identity_residual"] for item in q0_checks)
        max_mmn_relative = max(
            item["relative_frobenius_pseudo_minus_paw_aware"]
            for item in mmn_checks
        )
        report: dict[str, Any] = {
            **_qualification_report_identity(selected_provenance_hashes),
            "runtime": {
                "host": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "python": sys.version,
                "numpy": np.__version__,
            },
            "scope": {
                "system": "periodic_40_atom_Ta8Te20Pd12_bulk",
                "mesh": list(TPT_BULK_SOURCE_MESH),
                "active_ranks_1based": [first, last],
                "wannier_functions": chk.num_wann,
                "dft_bands": chk.num_bands,
                "kpoints": chk.num_kpts,
                "encut_ev": wav.encut_ev,
            },
            "source_hashes": {
                "wannier90_hr.dat": TPT_BULK_HR_SHA256,
                "WAVECAR": wavecar.sha256,
                "wannier90.chk": chk.sha256,
                "wannier90.eig": TPT_BULK_WANNIER90_EIG_SHA256,
                "wannier90.mmn": TPT_BULK_WANNIER90_MMN_SHA256,
            },
            "gauge_validation": {
                "ndimwin_min": int(chk.ndimwin.min()),
                "ndimwin_max": int(chk.ndimwin.max()),
                "max_semiunitarity_residual": max_gauge_semiunitarity,
                "hr_text_coefficient_resolution_ev": (
                    TPT_BULK_HR_TEXT_COEFFICIENT_RESOLUTION_EV
                ),
                "active_energy_tolerance_ev": (
                    TPT_BULK_HR_CHECKPOINT_ACTIVE_ENERGY_TOLERANCE_EV
                ),
                "active_energy_tolerance_provenance": {
                    "basis": (
                        "pre-existing accepted projector-refinement serialized-HR "
                        "path replay gate"
                    ),
                    "artifact": (
                        "results/tpt_bulk_mean_field/"
                        "wannier320_projector_candidate233_248_refinement_v1/"
                        "runs/job-505304/REPORT.json"
                    ),
                    "artifact_sha256": (
                        "4093b7bd526e96c575eb991f02267e929134d30198662dffdc7a731fcb31d99b"
                    ),
                },
                "active_subspace_min_singular_value_tolerance": (
                    TPT_BULK_HR_CHECKPOINT_ACTIVE_SUBSPACE_MIN_SV
                ),
                "max_active_energy_error_vs_hr_ev": max_active_energy_error,
                "min_active_subspace_singular_value_vs_hr": (
                    min_active_subspace_singular_value
                ),
            },
            "wavecar_validation": {
                "max_eigenvalue_error_vs_wannier90_eig_ev": wav_eig_error,
                "spinor_coefficients_per_k_min": int(
                    wav.plane_wave_coefficients_per_k.min()
                ),
                "spinor_coefficients_per_k_max": int(
                    wav.plane_wave_coefficients_per_k.max()
                ),
                "g_vectors_per_k_min": int(wav.g_vectors_per_k.min()),
                "g_vectors_per_k_max": int(wav.g_vectors_per_k.max()),
            },
            "smeared_occupation_diagnostic": {
                "authority": "diagnostic_only_not_fixed_rank_filling_authority",
                "full_dft_occupation_sum_min": float(
                    np.min(np.sum(wav.occupations, axis=1))
                ),
                "full_dft_occupation_sum_max": float(
                    np.max(np.sum(wav.occupations, axis=1))
                ),
                "full_dft_band_count_above_half_min": int(
                    np.min(np.count_nonzero(wav.occupations > 0.5, axis=1))
                ),
                "full_dft_band_count_above_half_max": int(
                    np.max(np.count_nonzero(wav.occupations > 0.5, axis=1))
                ),
                "active16_projected_trace_min": float(
                    min(active_smeared_occupation_traces)
                ),
                "active16_projected_trace_max": float(
                    max(active_smeared_occupation_traces)
                ),
                "active16_occupation_eigenvalue_min": (
                    active_smeared_occupation_eigenvalue_min
                ),
                "active16_occupation_eigenvalue_max": (
                    active_smeared_occupation_eigenvalue_max
                ),
                "active16_count_above_half_min": min(
                    active_smeared_occupation_counts_above_half
                ),
                "active16_count_above_half_max": max(
                    active_smeared_occupation_counts_above_half
                ),
            },
            "pseudo_q0_identity_checks": q0_checks,
            "paw_aware_mmn_neighbour_checks": mmn_checks,
            "verdict": {
                "checkpoint_gauge_bridge_validated": bool(
                    max_gauge_semiunitarity <= 1.0e-10
                    and max_active_energy_error
                    <= TPT_BULK_HR_CHECKPOINT_ACTIVE_ENERGY_TOLERANCE_EV
                    and min_active_subspace_singular_value
                    >= TPT_BULK_HR_CHECKPOINT_ACTIVE_SUBSPACE_MIN_SV
                    and wav_eig_error <= 1.0e-9
                ),
                "direct_wavecar_vertices_are_paw_complete": False,
                "max_sampled_q0_identity_residual": max_q0_residual,
                "max_sampled_mmn_relative_pseudo_difference": max_mmn_relative,
                "production_hf_authorized": False,
                "blocking_reasons": [
                    "direct WAVECAR contractions omit PAW augmentation",
                    "periodic-bulk screened Coulomb authority is absent",
                    "neutral filling/reference and DFT-HF double counting remain unclosed",
                ],
            },
        }
        return report


def _parse_indices(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("anchor indices must be a unique comma list")
    if any(value < 0 or value >= 192 for value in values):
        raise argparse.ArgumentTypeError("anchor indices must lie in [0,191]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-first", type=int, default=233)
    parser.add_argument("--active-last", type=int, default=248)
    parser.add_argument(
        "--anchor-k-indices",
        type=_parse_indices,
        default=_parse_indices("0,1,11,12,48,144,191"),
    )
    parser.add_argument(
        "--mmn-source-k-indices",
        type=_parse_indices,
        default=_parse_indices("0,6,11,191"),
        help="source k points whose six PAW-aware MMN neighbours are checked",
    )
    args = parser.parse_args()
    ensure_not_running_compute_on_login_node(
        "TPT bulk accepted-parent wavefunction qualification"
    )
    report = qualify_parent(
        source_root=args.source_root.expanduser().resolve(),
        parent_root=args.parent_root.expanduser().resolve(),
        active_ranks_1based=(args.active_first, args.active_last),
        anchor_k_indices=args.anchor_k_indices,
        mmn_source_k_indices=args.mmn_source_k_indices,
    )
    _atomic_write_new_json(args.output, report)
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
