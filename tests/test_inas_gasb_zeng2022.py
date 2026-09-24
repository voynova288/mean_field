from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.inas_gasb.zeng2022 import (
    Zeng2022Parameters,
    ZengSlabBasis,
    build_zeng2022_folded_h0,
    zeng2022_coulomb_components,
    zeng2022_reference_density,
    zeng2022_spin_block,
    zeng2022_uniform_hartree_limit_ry,
)


def test_zeng2022_eq2_spin_blocks_and_time_reversal() -> None:
    params = Zeng2022Parameters(eg_ry=0.1, hybridization_ab_ry=0.6, q_ab_inv=0.0)
    momentum = np.asarray([0.23, -0.17])
    up = zeng2022_spin_block(momentum, params, spin="up")
    down_minus = zeng2022_spin_block(-momentum, params, spin="down")
    assert np.allclose(up, up.conj().T, rtol=0.0, atol=1e-14)
    assert np.allclose(up, down_minus.conj(), rtol=0.0, atol=1e-14)
    assert up[0, 0] == pytest.approx(0.5 * np.dot(momentum, momentum) + 0.05)
    assert up[1, 1] == pytest.approx(-0.5 * np.dot(momentum, momentum) - 0.05)
    assert up[0, 1] == pytest.approx(0.6 * (momentum[0] + 1j * momentum[1]))


def test_zeng2022_field_shift_and_folded_slab_momenta() -> None:
    params = Zeng2022Parameters(eg_ry=-0.4, hybridization_ab_ry=0.5, q_ab_inv=1.8)
    basis = ZengSlabBasis((-1, 0, 1))
    kappa = np.asarray([[0.1, -0.2], [-0.4, 0.3]])
    h0 = build_zeng2022_folded_h0(kappa, basis, params)
    assert h0.shape == (12, 12, 2)
    assert np.max(np.abs(h0 - np.swapaxes(h0.conj(), 0, 1))) < 1e-14
    for ik, point in enumerate(kappa):
        for slab in basis.slab_indices:
            physical = point + np.asarray([slab * params.q_ab_inv, 0.0])
            for spin in ("up", "down"):
                indices = [basis.index("c", spin, slab), basis.index("v", spin, slab)]
                expected = zeng2022_spin_block(physical, params, spin=spin)
                assert np.allclose(h0[np.ix_(indices, indices, [ik])][:, :, 0], expected)
    # H0 is slab diagonal before Hartree-Fock density-wave coherence is added.
    assert h0[basis.index("c", "up", -1), basis.index("c", "up", 0), 0] == 0.0


def test_zeng2022_slab_basis_and_reference_density_match_eq6() -> None:
    basis = ZengSlabBasis((-2, -1, 0, 1, 2))
    for index in range(basis.dimension):
        band, spin, slab = basis.label(index)
        assert basis.index(band, spin, slab) == index
    reference = zeng2022_reference_density(basis, nk=7)
    assert reference.shape == (20, 20, 7)
    assert np.max(np.abs(reference - np.swapaxes(reference.conj(), 0, 1))) == 0.0
    for slab in basis.slab_indices:
        for spin in ("up", "down"):
            assert np.all(reference[basis.index("c", spin, slab), basis.index("c", spin, slab)] == 0.0)
            assert np.all(reference[basis.index("v", spin, slab), basis.index("v", spin, slab)] == 1.0)
    assert np.allclose(np.trace(reference, axis1=0, axis2=1), 2 * len(basis.slab_indices))


def test_zeng2022_dimensionless_coulomb_and_neutral_hartree_limit() -> None:
    q = np.asarray([0.2, 0.7, 1.9])
    intra, inter = zeng2022_coulomb_components(q, d_over_ab=0.3)
    assert np.allclose(intra, 4.0 * np.pi / q)
    assert np.allclose(inter / intra, np.exp(-0.3 * q))
    with pytest.raises(ValueError, match="q=0 handling"):
        zeng2022_coulomb_components(np.asarray([0.0]), d_over_ab=0.3)
    sigma_c, sigma_v = zeng2022_uniform_hartree_limit_ry(0.02, d_over_ab=0.3)
    assert sigma_c - sigma_v == pytest.approx(8.0 * np.pi * 0.3 * 0.02)
    assert sigma_c + sigma_v == pytest.approx(0.0)
