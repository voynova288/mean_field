from __future__ import annotations

import importlib.util
import numpy as np

from mean_field.systems.tdbg import (
    TDBGParameters,
)
from mean_field.systems.tdbg.hamiltonian import (
    build_bilayer_block,
    build_hamiltonian,
    build_site_block,
    diagonalize_hamiltonian,
)
from mean_field.systems.tdbg.lattice import (
    build_tdbg_lattice,
)


def test_tdbg_ab_and_ba_local_bilayer_blocks_swap_dimer_sites_correctly() -> None:
    params = TDBGParameters.full()
    block_ab = build_bilayer_block(0.0 + 0.0j, 0.0 + 0.0j, params, upper_layer_potential=0.0, lower_layer_potential=0.0, stacking_order="AB")
    block_ba = build_bilayer_block(0.0 + 0.0j, 0.0 + 0.0j, params, upper_layer_potential=0.0, lower_layer_potential=0.0, stacking_order="BA")

    assert np.allclose(np.diag(block_ab).real, [0.0, params.delta_prime, params.delta_prime, 0.0], atol=1.0e-12)
    assert np.allclose(np.diag(block_ba).real, [params.delta_prime, 0.0, 0.0, params.delta_prime], atol=1.0e-12)
    assert np.isclose(block_ab[1, 2], params.gamma1, atol=1.0e-12)
    assert np.isclose(block_ba[0, 3], params.gamma1, atol=1.0e-12)


def test_tdbg_ba_ab_domain_assigns_ba_top_and_ab_bottom_bilayers() -> None:
    lattice = build_tdbg_lattice(1.05, cut=1.0)
    params = TDBGParameters.full(stacking="BA-AB", Delta=0.0)
    top_index = next(i for i, site in enumerate(lattice.q_sites) if int(round(float(site[2]))) == 0)
    bottom_index = next(i for i, site in enumerate(lattice.q_sites) if int(round(float(site[2]))) == 1)

    top = build_site_block(0.0 + 0.0j, top_index, lattice, params, valley=1)
    bottom = build_site_block(0.0 + 0.0j, bottom_index, lattice, params, valley=1)
    assert np.isclose(top[0, 3], params.gamma1, atol=1.0e-12)
    assert np.isclose(bottom[1, 2], params.gamma1, atol=1.0e-12)


def _load_pytwist_reference():
    spec = importlib.util.spec_from_file_location("pytwist_local", "/data/home/ziyuzhu/pytwist/pytwist.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tdbg_full_hamiltonian_matches_pytwist_reference_at_generic_k() -> None:
    lattice = build_tdbg_lattice(1.33, cut=1.0)
    params = TDBGParameters.full(Delta=0.0, stacking="AB-AB")
    k_tilde = lattice.gamma_m / 7.0 + lattice.kprime_m / 11.0

    hamiltonian = build_hamiltonian(k_tilde, lattice, params, valley=1)
    assert hamiltonian.shape == (lattice.matrix_dim, lattice.matrix_dim)
    assert np.allclose(hamiltonian, hamiltonian.conjugate().T, atol=1.0e-10)

    module = _load_pytwist_reference()
    reference = module.TDBGModel(1.33, 0.0, 0.0, 0.0, cut=1.0)

    for valley in (-1, 1):
        reference_hamiltonian = reference.gen_ham(k_tilde.real, k_tilde.imag, xi=valley)
        assert np.allclose(build_hamiltonian(k_tilde, lattice, params, valley=valley), reference_hamiltonian, atol=1.0e-12)

        evals, _ = diagonalize_hamiltonian(k_tilde, lattice, params, valley=valley, n_bands=lattice.matrix_dim)
        reference_evals = np.linalg.eigvalsh(reference_hamiltonian)
        assert np.allclose(evals, reference_evals, atol=1.0e-12)


def test_tdbg_nonzero_pytwist_displacement_matches_valley_signed_delta() -> None:
    lattice = build_tdbg_lattice(1.33, cut=1.0)
    k_tilde = lattice.gamma_m / 7.0 + lattice.kprime_m / 11.0
    reference_d = 0.017

    module = _load_pytwist_reference()
    reference = module.TDBGModel(1.33, 0.0, 0.0, reference_d, cut=1.0)

    for valley in (-1, 1):
        params = TDBGParameters.full(Delta=valley * reference_d, stacking="AB-AB")
        reference_hamiltonian = reference.gen_ham(k_tilde.real, k_tilde.imag, xi=valley)
        assert np.allclose(build_hamiltonian(k_tilde, lattice, params, valley=valley), reference_hamiltonian, atol=1.0e-12)
