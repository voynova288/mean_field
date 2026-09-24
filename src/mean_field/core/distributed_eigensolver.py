"""Distributed complex generalized-Hermitian SLEPc solves with root-only vectors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import scipy.sparse


@dataclass(frozen=True)
class DistributedEigenResult:
    energies: np.ndarray
    vectors: np.ndarray
    converged: int
    converged_reason: int
    communicator_size: int
    matrix_dimension: int
    h_csr_sha256: str
    s_csr_sha256: str


def _canonical_csr(matrix: scipy.sparse.spmatrix) -> scipy.sparse.csr_matrix:
    result = matrix.tocsr().astype(np.complex128, copy=False)
    result.sum_duplicates()
    result.sort_indices()
    return result


def _csr_sha256(matrix: scipy.sparse.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(matrix.data, dtype=np.complex128).tobytes())
    return digest.hexdigest()


def _prepare_root_hermitian(root_matrix, mpi_comm, *, label: str):
    rank = mpi_comm.Get_rank()
    ownership_valid = (rank == 0 and root_matrix is not None) or (
        rank != 0 and root_matrix is None
    )
    if not all(mpi_comm.allgather(ownership_valid)):
        raise ValueError(f"only communicator root may provide full {label}")
    if rank == 0:
        try:
            prepared = _canonical_csr((root_matrix + root_matrix.getH()) * 0.5)
            status = (True, "")
        except Exception as exc:
            prepared = None
            status = (False, f"{type(exc).__name__}: {exc}")
    else:
        prepared = None
        status = None
    success, error = mpi_comm.bcast(status, root=0)
    if not success:
        raise RuntimeError(f"root {label} preparation failed: {error}")
    return prepared


def _distribute_csr_rows(root_matrix, mpi_comm, *, tag: int):
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    ownership_valid = (rank == 0 and root_matrix is not None) or (
        rank != 0 and root_matrix is None
    )
    if not all(mpi_comm.allgather(ownership_valid)):
        raise ValueError("only communicator root may provide the full CSR matrix")
    if rank == 0:
        try:
            matrix = _canonical_csr(root_matrix)
            nrows, ncols = matrix.shape
            if nrows != ncols:
                raise ValueError("distributed eigensolver requires square matrices")
            preparation = (True, "", (nrows, ncols))
        except Exception as exc:
            matrix = None
            preparation = (False, f"{type(exc).__name__}: {exc}", None)
    else:
        matrix = None
        preparation = None
    prepared, error, shape = mpi_comm.bcast(preparation, root=0)
    if not prepared:
        raise RuntimeError(f"root CSR preparation failed: {error}")
    nrows, ncols = shape
    starts = [(rank_index * nrows) // size for rank_index in range(size + 1)]
    if rank == 0:
        local = None
        for destination in range(size):
            block = matrix[starts[destination] : starts[destination + 1], :].tocsr()
            payload = (
                np.asarray(block.indptr, dtype=np.int32),
                np.asarray(block.indices, dtype=np.int32),
                np.asarray(block.data, dtype=np.complex128),
            )
            if destination == 0:
                local = payload
            else:
                mpi_comm.send(payload, dest=destination, tag=tag)
        assert local is not None
    else:
        local = mpi_comm.recv(source=0, tag=tag)
    return nrows, ncols, starts[rank], starts[rank + 1], local


def solve_hep_extreme_root_csr(
    matrix_root: scipy.sparse.spmatrix | None,
    *,
    mpi_comm: Any,
    which: str,
    tolerance: float = 1.0e-9,
    max_iterations: int = 5000,
) -> float | None:
    """Return one Hermitian extremal eigenvalue on communicator root."""

    from petsc4py import PETSc  # type: ignore
    from slepc4py import SLEPc  # type: ignore

    rank = mpi_comm.Get_rank()
    full = _prepare_root_hermitian(matrix_root, mpi_comm, label="HEP matrix")
    nrows, ncols, row_start, row_end, local = _distribute_csr_rows(
        full, mpi_comm, tag=40991
    )
    local_rows = row_end - row_start
    indptr, indices, data = local
    matrix = None
    eps = None
    try:
        matrix = PETSc.Mat().createAIJ(
            size=((local_rows, nrows), (local_rows, ncols)),
            csr=(indptr, indices, np.asarray(data, dtype=PETSc.ScalarType)),
            comm=mpi_comm,
        )
        matrix.assemble()
        matrix.setOption(PETSc.Mat.Option.HERMITIAN, True)
        eps = SLEPc.EPS().create(comm=mpi_comm)
        eps.setOperators(matrix)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        eps.setType("krylovschur")
        eps.setDimensions(1, PETSc.DECIDE)
        choices = {
            "smallest_real": SLEPc.EPS.Which.SMALLEST_REAL,
            "largest_real": SLEPc.EPS.Which.LARGEST_REAL,
            "largest_magnitude": SLEPc.EPS.Which.LARGEST_MAGNITUDE,
        }
        try:
            eps.setWhichEigenpairs(choices[which])
        except KeyError as exc:
            raise ValueError(f"unsupported extremum selector: {which}") from exc
        eps.setTolerances(float(tolerance), int(max_iterations))
        eps.solve()
        converged = int(eps.getConverged())
        reason = int(eps.getConvergedReason())
        if converged < 1 or reason <= 0:
            raise RuntimeError(
                f"distributed extremal solve failed: which={which}, reason={reason}, converged={converged}"
            )
        value = float(np.real(eps.getEigenvalue(0)))
        if not np.isfinite(value):
            raise RuntimeError(f"distributed extremal solve returned nonfinite {which}")
    finally:
        if eps is not None:
            eps.destroy()
        if matrix is not None:
            matrix.destroy()
    return value if rank == 0 else None


def solve_ghep_root_csr(
    h_root: scipy.sparse.spmatrix | None,
    s_root: scipy.sparse.spmatrix | None,
    *,
    mpi_comm: Any,
    count: int,
    target_energy: float,
    tolerance: float = 1.0e-10,
    max_iterations: int = 5000,
    factor_solver: str = "mumps",
    mumps_extra_workspace_percent: int = 100,
    overlap_positive_certified: bool = False,
) -> DistributedEigenResult | None:
    """Solve ``H C = S C E`` collectively; return full vectors on group root only.

    ``h_root`` and ``s_root`` must be supplied only on ``mpi_comm`` rank zero.
    No metric repair or diagonal shift is applied.
    """

    from petsc4py import PETSc  # type: ignore
    from slepc4py import SLEPc  # type: ignore

    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()
    if count <= 0:
        raise ValueError("count must be positive")
    if not overlap_positive_certified:
        raise ValueError(
            "GHEP requires an independently completed positive-overlap extremal gate"
        )
    if mumps_extra_workspace_percent < 0:
        raise ValueError("MUMPS extra-workspace percentage must be nonnegative")
    h_full = _prepare_root_hermitian(h_root, mpi_comm, label="Hamiltonian")
    s_full = _prepare_root_hermitian(s_root, mpi_comm, label="overlap")
    if rank == 0:
        shape_valid = h_full.shape == s_full.shape
        h_hash = _csr_sha256(h_full)
        s_hash = _csr_sha256(s_full)
    else:
        shape_valid = None
        h_hash = None
        s_hash = None
    shape_valid = mpi_comm.bcast(shape_valid, root=0)
    if not shape_valid:
        raise ValueError("H/S shape mismatch")

    n_h, ncols_h, row_start, row_end, h_local = _distribute_csr_rows(
        h_full, mpi_comm, tag=41001
    )
    n_s, ncols_s, s_row_start, s_row_end, s_local = _distribute_csr_rows(
        s_full, mpi_comm, tag=41002
    )
    if (n_h, ncols_h, row_start, row_end) != (
        n_s,
        ncols_s,
        s_row_start,
        s_row_end,
    ):
        raise ValueError("distributed H/S layouts differ")
    local_rows = row_end - row_start

    def make_matrix(local_payload):
        indptr, indices, data = local_payload
        matrix = PETSc.Mat().createAIJ(
            size=((local_rows, n_h), (local_rows, n_h)),
            csr=(indptr, indices, np.asarray(data, dtype=PETSc.ScalarType)),
            comm=mpi_comm,
        )
        matrix.assemble()
        matrix.setOption(PETSc.Mat.Option.HERMITIAN, True)
        return matrix

    a_matrix = None
    b_matrix = None
    eps = None
    work = None
    scatter = None
    root_vector = None
    try:
        a_matrix = make_matrix(h_local)
        b_matrix = make_matrix(s_local)
        b_matrix.setOption(PETSc.Mat.Option.SPD, True)
        eps = SLEPc.EPS().create(comm=mpi_comm)
        eps.setOperators(a_matrix, b_matrix)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType("krylovschur")
        eps.setDimensions(count, PETSc.DECIDE)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(float(target_energy))
        spectral_transform = eps.getST()
        spectral_transform.setType("sinvert")
        spectral_transform.setShift(float(target_energy))
        linear_solver = spectral_transform.getKSP()
        linear_solver.setType("preonly")
        preconditioner = linear_solver.getPC()
        preconditioner.setType("lu")
        preconditioner.setFactorSolverType(factor_solver)
        if factor_solver == "mumps":
            option_prefix = linear_solver.getOptionsPrefix() or ""
            PETSc.Options()[f"{option_prefix}mat_mumps_icntl_14"] = str(
                int(mumps_extra_workspace_percent)
            )
        eps.setTolerances(float(tolerance), int(max_iterations))
        eps.setFromOptions()
        eps.solve()

        converged = int(eps.getConverged())
        reason = int(eps.getConvergedReason())
        if reason <= 0 or converged < count:
            raise RuntimeError(
                f"distributed SLEPc failed: reason={reason}, converged={converged}, requested={count}"
            )
        work = a_matrix.createVecRight()
        scatter, root_vector = PETSc.Scatter.toZero(work)
        energies = np.empty(count, dtype=np.float64)
        vectors = (
            np.empty((n_h, count), dtype=np.complex128) if rank == 0 else None
        )
        for index in range(count):
            eigenvalue = eps.getEigenpair(index, work)
            scatter.scatter(
                work,
                root_vector,
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD,
            )
            energies[index] = float(np.real(eigenvalue))
            if rank == 0:
                vectors[:, index] = np.asarray(
                    root_vector.getArray(readonly=True), dtype=np.complex128
                ).copy()
        if not np.all(np.isfinite(energies)):
            raise RuntimeError("distributed SLEPc returned nonfinite eigenvalues")
        order = np.argsort(energies)
        energies = energies[order]
        if rank == 0:
            vectors = vectors[:, order]
    finally:
        if scatter is not None:
            scatter.destroy()
        if root_vector is not None:
            root_vector.destroy()
        if work is not None:
            work.destroy()
        if eps is not None:
            eps.destroy()
        if b_matrix is not None:
            b_matrix.destroy()
        if a_matrix is not None:
            a_matrix.destroy()

    if rank != 0:
        return None
    return DistributedEigenResult(
        energies=energies,
        vectors=vectors,
        converged=converged,
        converged_reason=reason,
        communicator_size=size,
        matrix_dimension=n_h,
        h_csr_sha256=h_hash,
        s_csr_sha256=s_hash,
    )
