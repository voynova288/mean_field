from __future__ import annotations

import numpy as np

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field import calculate_overlap, calculate_overlap_between, compute_density_overlap_trace
from mean_field.systems.tbg.zero_field.model import BMSolution


def _fake_overlap_solution() -> BMSolution:
    params = TBGParameters.from_degrees(1.2)
    uk = np.zeros((4, 1, 2, 2), dtype=np.complex128)
    uk[:, 0, 0, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    uk[:, 0, 0, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    uk[:, 0, 1, 0] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
    uk[:, 0, 1, 1] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)

    return BMSolution(
        params=params,
        lattice_kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.0j], dtype=np.complex128),
        lg=1,
        nlocal=4,
        n_eta=2,
        n_spin=2,
        nb=1,
        hamiltonian=np.zeros((4, 4, 2, 2), dtype=np.complex128),
        sigma_z=np.zeros((4, 4, 2), dtype=np.complex128),
        uk=uk,
        spectrum=np.zeros((1, 2, 2), dtype=float),
        gvec=np.asarray([0.0 + 0.0j], dtype=np.complex128),
    )


def test_calculate_overlap_between_matches_square_overlap_for_same_solution() -> None:
    solution = _fake_overlap_solution()

    overlap_square = calculate_overlap(solution, 0, 0)
    overlap_between = calculate_overlap_between(solution, solution, 0, 0)

    assert overlap_between.shape == (solution.nt, solution.nk, solution.nt, solution.nk)
    assert np.allclose(overlap_square, np.eye(solution.nt * solution.nk, dtype=np.complex128))
    assert np.allclose(overlap_between.reshape(solution.nt * solution.nk, solution.nt * solution.nk, order="F"), overlap_square)


def test_compute_density_overlap_trace_reads_diagonal_k_blocks() -> None:
    solution = _fake_overlap_solution()
    overlap_between = calculate_overlap_between(solution, solution, 0, 0)

    density = np.zeros((solution.nt, solution.nt, solution.nk), dtype=np.complex128)
    density[:, :, 0] = np.diag([1.0, 2.0, 3.0, 4.0])
    density[:, :, 1] = np.diag([5.0, 6.0, 7.0, 8.0])

    assert compute_density_overlap_trace(density, overlap_between) == 36.0 + 0.0j
