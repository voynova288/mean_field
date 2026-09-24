from __future__ import annotations

import numpy as np

from mean_field.systems.tbg.arora2021 import (
    Arora2021Fig2Parameters,
    analytic_arora2021_b0_dhdk,
    build_arora2021_b0_hamiltonian,
    circular_reciprocal_indices,
    finite_difference_arora2021_b0_dhdk,
    gvec_from_indices,
)
from mean_field.systems.tbg.arora2021 import _build_circular_tunnel


def test_arora2021_circular_cutoff_matches_paper_dimension() -> None:
    indices = circular_reciprocal_indices(4)
    assert indices.shape == (61, 2)
    assert 4 * indices.shape[0] == 244


def test_arora2021_circular_hamiltonian_is_hermitian_and_244_dimensional() -> None:
    paper = Arora2021Fig2Parameters()
    params = paper.b0_params()
    config = paper.config(valley=1)
    h = build_arora2021_b0_hamiltonian(0.0 + 0.0j, params, config)
    assert h.shape == (244, 244)
    np.testing.assert_allclose(h, h.conj().T, atol=1.0e-12)


def test_arora2021_analytic_dhdk_matches_finite_difference() -> None:
    paper = Arora2021Fig2Parameters()
    params = paper.b0_params()
    config = paper.config(valley=1)
    indices = circular_reciprocal_indices(4)
    gvec = gvec_from_indices(params, indices)
    tunnel = _build_circular_tunnel(params, indices, config.valley)
    analytic = analytic_arora2021_b0_dhdk(params, config, indices=indices)
    finite = finite_difference_arora2021_b0_dhdk(
        params,
        config,
        indices=indices,
        gvec=gvec,
        tunnel=tunnel,
        step_dimless=1.0e-6,
    )
    for a, f in zip(analytic, finite, strict=True):
        np.testing.assert_allclose(a, f, atol=1.0e-9, rtol=1.0e-9)
