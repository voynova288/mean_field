"""Source-sealed cache, profiling, and branch runner for TPT Q=0 Wannier HF.

Heavy commands in this module are intended for Slurm compute nodes only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time
from typing import Iterator, Mapping

import numpy as np
import scipy

from mean_field.core.hf.problem import run_hartree_fock_problem
from mean_field.core.hf.real_space_density_density import PeriodicDensityDensityKernel
from mean_field.core.io import write_json_artifact
from mean_field.runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.tpt.wannier_keldysh_hf import (
    TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
    build_gaussian_keldysh_density_density_kernel,
    build_spin_doubled_h0,
    evaluate_wannier_hr_on_mesh,
    parse_final_wannier_gaussian_densities,
    parse_wannier90_hr,
    rectangular_gaussian_keldysh_zero_cell,
    regular_fractional_mesh,
)
from mean_field.systems.tpt.wannier_q0_hf import (
    TPTWannierQ0HFInputs,
    TPTWannierQ0HFState,
    build_projector_seed_from_source,
    build_tpt_wannier_q0_hf_problem,
)

MESH_SHAPE = (40, 10)
N_SCALAR = 80
N_OCCUPIED_SCALAR = 60
N_SPINFUL = 160
N_OCCUPIED_SPINFUL = 120
LATTICE_2D_ANGSTROM = np.diag([3.7037999629999998, 18.5991001129000004])
SOURCE_HASHES = {
    "POSCAR": "e34bd32485729b31c8ec56a134f438f2568e2d4a2dd2a205f304811888a7e541",
    "wannier90_hr.dat": "46aa589893d1d24670fce5872aeef6b0b18f75a488b6c0fcec4da9ccaaf5a54d",
    "wannier90.wout": "ca06009459e0d7217e1e07688d918ea60a5547e5fc2a7de177416fece44049c8",
    "wannier90_centres.xyz": "d2b64a5354fdf4a85b1d983fb05885daddddf18f3220c127f1edb1c40f1f45e3",
}
EXPECTED_GAMMA_VALENCE_EV = -0.5005148303371362
EXPECTED_GAMMA_CONDUCTION_EV = -0.4053711492123373


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_inventory(directory: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    omitted = set() if exclude is None else set(exclude)
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and str(path.relative_to(directory)) not in omitted
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_json_artifact(payload, path)


def _runtime_inventory() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


@contextmanager
def _stage(timings: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - start


def _verify_sources(source_directory: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in SOURCE_HASHES.items():
        path = source_directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = _sha256(path)
        if observed[name] != expected:
            raise RuntimeError(
                f"source hash mismatch for {name}: {observed[name]} != {expected}"
            )
    return observed


def _parse_poscar_lattice(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if len(lines) < 5:
        raise ValueError("truncated POSCAR")
    scale = float(lines[1].split()[0])
    lattice = scale * np.asarray(
        [[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)]
    )
    if not np.allclose(
        lattice[:2, :2], LATTICE_2D_ANGSTROM, rtol=0.0, atol=2.0e-12
    ):
        raise ValueError("POSCAR in-plane lattice does not match the locked contract")
    if np.max(np.abs(lattice[:2, 2])) > 2.0e-12:
        raise ValueError("v1 expects in-plane lattice vectors in the xy plane")
    return lattice


def _parse_xyz_wannier_centres(path: Path, expected: int) -> np.ndarray:
    records: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines()[2:]:
        fields = line.split()
        if len(fields) == 4 and fields[0] == "X":
            records.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if len(records) != expected:
        raise ValueError(f"expected {expected} X centre records, found {len(records)}")
    return np.asarray(records, dtype=float)


def _scalar_reference(
    scalar_h0_abk: np.ndarray,
    *,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, float]]:
    h_ket = np.asarray(scalar_h0_abk, dtype=np.complex128).transpose(2, 0, 1)
    nk, nb, _ = h_ket.shape
    if nb != N_SCALAR:
        raise ValueError("scalar basis dimension mismatch")
    energies = np.empty((nk, nb), dtype=float)
    projectors = np.empty_like(h_ket)

    def solve(index: int) -> tuple[int, np.ndarray, np.ndarray]:
        values, vectors = np.linalg.eigh(h_ket[index])
        occupied = vectors[:, :N_OCCUPIED_SCALAR]
        return index, values, occupied @ occupied.conj().T

    if workers == 1:
        iterator = map(solve, range(nk))
        for index, values, projector in iterator:
            energies[index] = values
            projectors[index] = projector
    else:
        with ThreadPoolExecutor(max_workers=min(workers, nk)) as executor:
            for index, values, projector in executor.map(solve, range(nk)):
                energies[index] = values
                projectors[index] = projector
    highest = float(np.max(energies[:, N_OCCUPIED_SCALAR - 1]))
    lowest = float(np.min(energies[:, N_OCCUPIED_SCALAR]))
    gap = lowest - highest
    if gap <= 1.0e-10:
        raise RuntimeError(f"scalar parent global rank-60 gap is not positive: {gap}")
    gamma_values = energies[0]
    if abs(gamma_values[59] - EXPECTED_GAMMA_VALENCE_EV) > 5.0e-8:
        raise RuntimeError("Gamma rank-60 energy failed archived replay")
    if abs(gamma_values[60] - EXPECTED_GAMMA_CONDUCTION_EV) > 5.0e-8:
        raise RuntimeError("Gamma rank-61 energy failed archived replay")
    projector_residual = float(
        np.max(np.abs(projectors @ projectors - projectors), initial=0.0)
    )
    commutator = h_ket @ projectors - projectors @ h_ket
    diagnostics = {
        "scalar_reference_idempotency_residual": projector_residual,
        "scalar_reference_commutator_residual_ev": float(
            np.max(np.abs(commutator), initial=0.0)
        ),
        "scalar_parent_global_gap_ev": gap,
        "gamma_rank60_ev": float(gamma_values[59]),
        "gamma_rank61_ev": float(gamma_values[60]),
    }
    return projectors, energies, 0.5 * (highest + lowest), gap, diagnostics


def _spin_double_reference(
    scalar_projector_ket: np.ndarray,
    scalar_energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nk = scalar_projector_ket.shape[0]
    reference_ket = np.zeros((nk, N_SPINFUL, N_SPINFUL), dtype=np.complex128)
    reference_ket[:, :N_SCALAR, :N_SCALAR] = scalar_projector_ket
    reference_ket[:, N_SCALAR:, N_SCALAR:] = scalar_projector_ket
    spinful_energies = np.sort(
        np.concatenate([scalar_energies, scalar_energies], axis=1), axis=1
    ).T
    reference_stored = np.ascontiguousarray(
        reference_ket.transpose(0, 2, 1).transpose(1, 2, 0)
    )
    return reference_stored, spinful_energies


def _publish_cache(staging: Path, final: Path, completion_payload: Mapping[str, object]) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to replace existing cache {final}")
    final.mkdir(mode=0o755, parents=False, exist_ok=False)
    payload_names = sorted(path.name for path in staging.iterdir() if path.is_file())
    try:
        for name in payload_names:
            os.rename(staging / name, final / name)
        completed = final / "CACHE_COMPLETED.json"
        descriptor = os.open(completed, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(completion_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.rmdir(staging)
    except BaseException:
        raise


def build_cache(args: argparse.Namespace) -> None:
    ensure_not_running_compute_on_login_node("TPT Wannier--Keldysh cache build")
    source_directory = Path(args.source_directory).resolve()
    final = Path(args.output_directory).resolve()
    staging = final.parent / f".{final.name}.staging-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    if final.exists() or staging.exists():
        raise FileExistsError("cache final or staging path already exists")
    staging.mkdir(parents=True, exist_ok=False)
    timings: dict[str, float] = {}
    report: dict[str, object] = {
        "schema": "tpt-wannier-keldysh-scalar80-spinful-cache-v1",
        "status": "building",
        "runtime": _runtime_inventory(),
        "source_directory": str(source_directory),
    }
    try:
        with _stage(timings, "source_verification_seconds"):
            report["source_hashes"] = _verify_sources(source_directory)
            _parse_poscar_lattice(source_directory / "POSCAR")
            wannier = parse_final_wannier_gaussian_densities(
                source_directory / "wannier90.wout", expected_num_wann=N_SCALAR
            )
            xyz_centres = _parse_xyz_wannier_centres(
                source_directory / "wannier90_centres.xyz", N_SCALAR
            )
            centre_delta = xyz_centres - wannier.centres_cartesian_angstrom
            fractional_shift = centre_delta[:, :2] @ np.linalg.inv(
                LATTICE_2D_ANGSTROM
            )
            integer_shift = np.rint(fractional_shift)
            centre_residual = centre_delta.copy()
            centre_residual[:, :2] -= integer_shift @ LATTICE_2D_ANGSTROM
            centre_replay = float(
                np.max(np.abs(centre_residual), initial=0.0)
            )
            if centre_replay > 1.0e-6:
                raise RuntimeError(
                    "WOUT/XYZ centres disagree even modulo in-plane lattice vectors: "
                    f"{centre_replay}"
                )
            report["wout_xyz_centre_mod_lattice_max_abs_angstrom"] = centre_replay
            report["wout_xyz_max_integer_lattice_shift"] = int(
                np.max(np.abs(integer_shift), initial=0.0)
            )
            report["interaction_centre_representative"] = (
                "literal unwrapped final WOUT centres; individual wrapping would "
                "require a matching Wannier-basis Bloch-gauge transformation"
            )

        with _stage(timings, "hr_parse_seconds"):
            hr = parse_wannier90_hr(source_directory / "wannier90_hr.dat")
            if hr.n_wannier != N_SCALAR:
                raise RuntimeError("source HR does not contain 80 Wannier functions")

        with _stage(timings, "h0_fourier_seconds"):
            kpoints = regular_fractional_mesh(MESH_SHAPE)
            scalar_h0 = evaluate_wannier_hr_on_mesh(hr, kpoints)
            spinful_h0 = build_spin_doubled_h0(scalar_h0)

        with _stage(timings, "reference_eigensolve_seconds"):
            scalar_projector, scalar_energies, parent_mu, parent_gap, reference_diag = (
                _scalar_reference(scalar_h0, workers=int(args.workers))
            )
            reference_stored, parent_energies = _spin_double_reference(
                scalar_projector, scalar_energies
            )
            report["reference_diagnostics"] = reference_diag

        shell_builds: dict[float, object] = {}
        with _stage(timings, "interaction_shell_ladder_seconds"):
            for tail in (1.0e-9, 1.0e-11, 1.0e-13):
                shell_builds[tail] = build_gaussian_keldysh_density_density_kernel(
                    lattice_2d_angstrom=LATTICE_2D_ANGSTROM,
                    wannier=wannier,
                    mesh_shape=MESH_SHAPE,
                    alpha_2d_angstrom=TPT_PUBLISHED_ALPHA_2D_ANGSTROM,
                    kappa=1.0,
                    spin_blocks=2,
                    gaussian_tail_tolerance=tail,
                    zero_cell_angular_order=64,
                    zero_cell_radial_order=24,
                    shell_covariance_tolerance_ev=1.0e-8,
                    workers=int(args.workers),
                )
        coarse = shell_builds[1.0e-9].kernel
        selected_build = shell_builds[1.0e-11]
        selected = selected_build.kernel
        tight = shell_builds[1.0e-13].kernel
        shell_metrics = {
            "tail_1e-9_to_1e-11_fock_max_abs_ev": float(
                np.max(np.abs(coarse.fock_kernel_q - selected.fock_kernel_q))
            ),
            "tail_1e-11_to_1e-13_fock_max_abs_ev": float(
                np.max(np.abs(selected.fock_kernel_q - tight.fock_kernel_q))
            ),
            "tail_1e-9_to_1e-11_hartree_max_abs_ev": float(
                np.max(np.abs(coarse.hartree_kernel - selected.hartree_kernel))
            ),
            "tail_1e-11_to_1e-13_hartree_max_abs_ev": float(
                np.max(np.abs(selected.hartree_kernel - tight.hartree_kernel))
            ),
        }
        if max(
            shell_metrics["tail_1e-11_to_1e-13_fock_max_abs_ev"],
            shell_metrics["tail_1e-11_to_1e-13_hartree_max_abs_ev"],
        ) > float(args.shell_tolerance_ev):
            raise RuntimeError(f"reciprocal shell convergence failed: {shell_metrics}")
        report["shell_convergence"] = shell_metrics

        with _stage(timings, "zero_cell_quadrature_ladder_seconds"):
            zero_cells = {
                f"{angular}x{radial}": rectangular_gaussian_keldysh_zero_cell(
                    lattice_2d_angstrom=LATTICE_2D_ANGSTROM,
                    wannier=wannier,
                    mesh_shape=MESH_SHAPE,
                    angular_order=angular,
                    radial_order=radial,
                )
                for angular, radial in ((24, 12), (48, 20), (64, 24), (96, 40))
            }
        quadrature_metrics = {
            "24x12_to_48x20_max_abs_ev": float(
                np.max(np.abs(zero_cells["24x12"] - zero_cells["48x20"]))
            ),
            "48x20_to_64x24_max_abs_ev": float(
                np.max(np.abs(zero_cells["48x20"] - zero_cells["64x24"]))
            ),
            "64x24_to_96x40_max_abs_ev": float(
                np.max(np.abs(zero_cells["64x24"] - zero_cells["96x40"]))
            ),
        }
        if quadrature_metrics["64x24_to_96x40_max_abs_ev"] > float(
            args.quadrature_tolerance_ev
        ):
            raise RuntimeError(f"zero-cell quadrature convergence failed: {quadrature_metrics}")
        report["zero_cell_quadrature_convergence"] = quadrature_metrics

        with _stage(timings, "input_contract_validation_seconds"):
            inputs = TPTWannierQ0HFInputs(
                h0_abk=spinful_h0,
                reference_projector_stored=reference_stored,
                parent_energies_ev=parent_energies,
                parent_chemical_potential_ev=parent_mu,
                parent_fermi_gap_ev=parent_gap,
                interaction_kernel=selected,
                total_occupied_states=int(N_OCCUPIED_SPINFUL * np.prod(MESH_SHAPE)),
                mesh_shape=MESH_SHAPE,
            )
            zero_action = selected.apply(np.zeros_like(spinful_h0), fft_workers=1)
            if np.count_nonzero(zero_action.interaction_h) != 0:
                raise RuntimeError("reference-relative interaction is nonzero at D=0")
            report["parent_spinful_gap_ev"] = float(inputs.parent_fermi_gap_ev)

        with _stage(timings, "cache_serialization_seconds"):
            arrays = {
                "fractional_kpoints.npy": kpoints,
                "scalar_h0_abk.npy": scalar_h0,
                "spinful_h0_abk.npy": spinful_h0,
                "reference_projector_stored.npy": reference_stored,
                "parent_energies_ev.npy": parent_energies,
                "fock_kernel_q.npy": selected.fock_kernel_q,
                "hartree_kernel.npy": selected.hartree_kernel,
                "zero_cell_matrix_ev.npy": selected_build.zero_cell_matrix_ev,
                "wannier_centres_cartesian_angstrom.npy": wannier.centres_cartesian_angstrom,
                "wannier_total_spreads_angstrom2.npy": wannier.total_spreads_angstrom2,
                "reciprocal_shell_indices.npy": selected_build.reciprocal_shell_indices,
            }
            for name, values in arrays.items():
                np.save(staging / name, np.asarray(values), allow_pickle=False)
            report["kernel_metadata"] = selected_build.metadata
            report["mesh_shape"] = list(MESH_SHAPE)
            report["basis_order"] = "scalar80_block_up_then_scalar80_block_down"
            report["occupied_per_k"] = N_OCCUPIED_SPINFUL
            report["timings_seconds"] = timings
            report["peak_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            report["status"] = "validated"
            _write_json(staging / "METADATA.json", report)
            inventory = _sha256_inventory(staging)
            _write_json(staging / "SHA256SUMS.json", inventory)
            completion_payload = {
                "schema": "tpt-wannier-keldysh-cache-completion-v1",
                "status": "completed",
                "payload_sha256": _sha256_inventory(staging),
                "metadata_sha256": _sha256(staging / "METADATA.json"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
        _publish_cache(staging, final, completion_payload)
        print(json.dumps({"status": "completed", "cache": str(final)}, sort_keys=True))
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["timings_seconds"] = timings
        report["peak_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _write_json(staging / "FAILED.json", report)
        raise


def load_cache(cache_directory: Path) -> tuple[TPTWannierQ0HFInputs, dict[str, object]]:
    cache = cache_directory.resolve()
    completed = cache / "CACHE_COMPLETED.json"
    if not completed.is_file():
        raise RuntimeError("cache lacks terminal CACHE_COMPLETED.json")
    completion = json.loads(completed.read_text(encoding="utf-8"))
    expected_inventory = completion["payload_sha256"]
    observed_inventory = _sha256_inventory(cache, exclude={"CACHE_COMPLETED.json"})
    if observed_inventory != expected_inventory:
        raise RuntimeError("cache payload hash inventory mismatch")
    metadata = json.loads((cache / "METADATA.json").read_text(encoding="utf-8"))
    kernel = PeriodicDensityDensityKernel(
        mesh_shape=MESH_SHAPE,
        fock_kernel_q=np.load(cache / "fock_kernel_q.npy", allow_pickle=False),
        hartree_kernel=np.load(cache / "hartree_kernel.npy", allow_pickle=False),
        spin_blocks=2,
        require_neutral_density=True,
        neutrality_tolerance=1.0e-10,
    )
    parent_energies = np.load(cache / "parent_energies_ev.npy", allow_pickle=False)
    inputs = TPTWannierQ0HFInputs(
        h0_abk=np.load(cache / "spinful_h0_abk.npy", allow_pickle=False),
        reference_projector_stored=np.load(
            cache / "reference_projector_stored.npy", allow_pickle=False
        ),
        parent_energies_ev=parent_energies,
        parent_chemical_potential_ev=0.5
        * (
            float(np.max(parent_energies[119]))
            + float(np.min(parent_energies[120]))
        ),
        parent_fermi_gap_ev=float(metadata["parent_spinful_gap_ev"]),
        interaction_kernel=kernel,
        total_occupied_states=int(N_OCCUPIED_SPINFUL * np.prod(MESH_SHAPE)),
        mesh_shape=MESH_SHAPE,
    )
    return inputs, metadata


def _random_hermitian_source(shape: tuple[int, int, int], seed: int, amplitude: float) -> np.ndarray:
    nb, _, nk = shape
    rng = np.random.default_rng(int(seed))
    source = np.empty(shape, dtype=np.complex128)
    for index in range(nk):
        raw = rng.normal(size=(nb, nb)) + 1.0j * rng.normal(size=(nb, nb))
        hermitian = 0.5 * (raw + raw.conj().T)
        norm = float(np.linalg.norm(hermitian))
        source[:, :, index] = float(amplitude) * hermitian / norm
    return source


def profile_cache(args: argparse.Namespace) -> None:
    ensure_not_running_compute_on_login_node("TPT Wannier--Keldysh profile")
    inputs, metadata = load_cache(Path(args.cache_directory))
    timings: dict[str, float] = {}
    rng = np.random.default_rng(9917)
    raw = rng.normal(size=inputs.h0_abk.shape) + 1.0j * rng.normal(
        size=inputs.h0_abk.shape
    )
    direction = 0.5 * (raw + raw.conj().swapaxes(0, 1))
    diagonal = np.einsum("aak->a", direction, optimize=True) / inputs.nk
    correction = float(np.sum(diagonal.real)) / inputs.nb
    for basis in range(inputs.nb):
        direction[basis, basis, :] -= correction
    inputs.interaction_kernel.average_basis_charges(direction)

    with _stage(timings, "one_interaction_action_seconds"):
        action = inputs.interaction_kernel.apply(
            direction, fft_workers=int(args.fft_workers)
        )
    source = _random_hermitian_source(
        inputs.h0_abk.shape, seed=17, amplitude=2.0e-3
    )
    with _stage(timings, "source_projector_seed_seconds"):
        seed_density = build_projector_seed_from_source(
            inputs,
            source,
            eigensolver_workers=int(args.eigensolver_workers),
        )
    problem = build_tpt_wannier_q0_hf_problem(
        inputs,
        initial_densities={"random": seed_density},
        eigensolver_workers=int(args.eigensolver_workers),
        fft_workers=int(args.fft_workers),
    )
    state = TPTWannierQ0HFState.create(inputs, precision=1.0e-8)
    with _stage(timings, "one_scf_iteration_seconds"):
        run = run_hartree_fock_problem(
            state,
            problem,
            init_mode="random",
            seed=17,
            max_iter=1,
            oda_stall_threshold=0.0,
        )
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema": "tpt-wannier-keldysh-profile-v1",
        "runtime": _runtime_inventory(),
        "cache_metadata_sha256": _sha256(Path(args.cache_directory) / "METADATA.json"),
        "eigensolver_workers": int(args.eigensolver_workers),
        "fft_workers": int(args.fft_workers),
        "timings_seconds": timings,
        "interaction_norm": float(np.linalg.norm(action.interaction_h)),
        "seed_norm": float(np.linalg.norm(seed_density)),
        "one_iteration_exit_reason": run.exit_reason,
        "one_iteration_error": float(run.iter_err[-1]),
        "peak_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cache_schema": metadata["schema"],
    }
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-cache")
    build.add_argument("--source-directory", required=True)
    build.add_argument("--output-directory", required=True)
    build.add_argument("--workers", type=int, default=56)
    build.add_argument("--shell-tolerance-ev", type=float, default=1.0e-8)
    build.add_argument("--quadrature-tolerance-ev", type=float, default=1.0e-8)
    build.set_defaults(function=build_cache)

    profile = subparsers.add_parser("profile-cache")
    profile.add_argument("--cache-directory", required=True)
    profile.add_argument("--output-json", required=True)
    profile.add_argument("--eigensolver-workers", type=int, required=True)
    profile.add_argument("--fft-workers", type=int, required=True)
    profile.set_defaults(function=profile_cache)

    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
