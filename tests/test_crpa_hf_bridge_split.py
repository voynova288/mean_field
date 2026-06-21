from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import HFOverlapBlockSet, build_projected_interaction_hamiltonian
from mean_field.crpa.hf_interface import (
    build_bare_projected_interaction_components,
    crpa_split_mode,
    half_reference_delta_like,
    physical_projector_from_delta,
)


def _toy_overlap_blocks() -> HFOverlapBlockSet:
    nt = 2
    nk = 1
    shift = (0, 0)
    overlap = np.zeros((nt, nk, nt, nk), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.eye(nt, dtype=np.complex128)
    diagonal = np.zeros((nt, nt, nk), dtype=np.complex128)
    diagonal[:, :, 0] = np.eye(nt, dtype=np.complex128)
    return HFOverlapBlockSet(
        shifts=(shift,),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={shift: overlap},
        diagonal_overlaps={shift: diagonal},
        hartree_screening={shift: 0.7},
        fock_screening={shift: np.asarray([[0.4]], dtype=float)},
    )


def test_crpa_hf_density_delta_roundtrip_uses_half_reference() -> None:
    density = np.zeros((3, 3, 2), dtype=np.complex128)
    ref = half_reference_delta_like(density)
    assert ref.shape == density.shape
    for ik in range(ref.shape[2]):
        np.testing.assert_allclose(np.diag(ref[:, :, ik]), -0.5)
    projector = physical_projector_from_delta(ref)
    np.testing.assert_allclose(projector, np.zeros_like(projector))


def test_bare_projected_components_sum_to_generic_interaction() -> None:
    density = np.zeros((2, 2, 1), dtype=np.complex128)
    density[:, :, 0] = np.asarray([[0.8, 0.1], [0.1, 0.2]], dtype=np.complex128)
    overlap_blocks = _toy_overlap_blocks()
    hartree, fock = build_bare_projected_interaction_components(
        density,
        overlap_blocks,
        v0=2.0,
        beta=0.5,
        use_numba=False,
    )
    split_total = hartree + fock
    generic = build_projected_interaction_hamiltonian(
        density,
        overlap_blocks,
        v0=2.0,
        beta=0.5,
        use_numba=False,
    )
    np.testing.assert_allclose(split_total, generic, atol=1.0e-12)


def test_crpa_split_mode_default_is_production_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEAN_FIELD_CRPA_SPLIT_MODE", raising=False)
    assert crpa_split_mode() == "active_cnp_fock_reference_projector"
    monkeypatch.setenv("MEAN_FIELD_CRPA_SPLIT_MODE", "production")
    assert crpa_split_mode() == "active_cnp_fock_reference_projector"
