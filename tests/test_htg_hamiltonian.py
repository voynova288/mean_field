from __future__ import annotations

import numpy as np

from mean_field.systems.htg import HTGParams, build_hamiltonian, build_htg_lattice, moire_coupling_matrix


def test_htg_hamiltonian_is_hermitian_for_small_cutoff() -> None:
    lattice = build_htg_lattice(1.5, n_shells=1)
    params = HTGParams.default()
    hmat = build_hamiltonian(lattice.gamma_m, lattice, params, valley=1)
    assert hmat.shape == (lattice.matrix_dim, lattice.matrix_dim)
    assert np.max(np.abs(hmat - hmat.conjugate().T)) < 1.0e-12


def test_htg_moire_coupling_phase_anchors() -> None:
    params = HTGParams.default()
    t0 = moire_coupling_matrix(0, params, valley=1)
    t1 = moire_coupling_matrix(1, params, valley=1)
    t2 = moire_coupling_matrix(2, params, valley=1)
    assert np.max(np.abs(t0.imag)) < 1.0e-12
    assert np.max(np.abs(t1 - t2.conjugate())) < 1.0e-12
