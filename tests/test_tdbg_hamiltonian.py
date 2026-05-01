from __future__ import annotations

import importlib.util
import numpy as np

from mean_field.systems.tdbg import TDBGParameters, build_bilayer_block, build_hamiltonian, build_tdbg_lattice, diagonalize_hamiltonian


def test_tdbg_ab_and_ba_local_bilayer_blocks_swap_dimer_sites_correctly() -> None:
    params = TDBGParameters.full()
    block_ab = build_bilayer_block(0.0 + 0.0j, 0.0 + 0.0j, params, upper_layer_potential=0.0, lower_layer_potential=0.0, stacking_order="AB")
    block_ba = build_bilayer_block(0.0 + 0.0j, 0.0 + 0.0j, params, upper_layer_potential=0.0, lower_layer_potential=0.0, stacking_order="BA")

    assert np.allclose(np.diag(block_ab).real, [0.0, params.delta_prime, params.delta_prime, 0.0], atol=1.0e-12)
    assert np.allclose(np.diag(block_ba).real, [params.delta_prime, 0.0, 0.0, params.delta_prime], atol=1.0e-12)
    assert np.isclose(block_ab[1, 2], params.gamma1, atol=1.0e-12)
    assert np.isclose(block_ba[0, 3], params.gamma1, atol=1.0e-12)


def test_tdbg_full_hamiltonian_matches_pytwist_reference_at_generic_k() -> None:
    lattice = build_tdbg_lattice(1.33, cut=1.0)
    params = TDBGParameters.full(Delta=0.0, stacking="AB-AB")
    k_tilde = lattice.gamma_m / 7.0 + lattice.kprime_m / 11.0

    hamiltonian = build_hamiltonian(k_tilde, lattice, params, valley=1)
    assert hamiltonian.shape == (lattice.matrix_dim, lattice.matrix_dim)
    assert np.allclose(hamiltonian, hamiltonian.conjugate().T, atol=1.0e-10)

    spec = importlib.util.spec_from_file_location("pytwist_local", "/data/home/ziyuzhu/pytwist/pytwist.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    reference = module.TDBGModel(1.33, 0.0, 0.0, 0.0, cut=1.0)

    for valley in (-1, 1):
        reference_hamiltonian = reference.gen_ham(k_tilde.real, k_tilde.imag, xi=valley)
        assert np.allclose(build_hamiltonian(k_tilde, lattice, params, valley=valley), reference_hamiltonian, atol=1.0e-12)

        evals, _ = diagonalize_hamiltonian(k_tilde, lattice, params, valley=valley, n_bands=lattice.matrix_dim)
        reference_evals = np.linalg.eigvalsh(reference_hamiltonian)
        assert np.allclose(evals, reference_evals, atol=1.0e-12)
