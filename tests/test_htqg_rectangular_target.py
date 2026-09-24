from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

import mean_field.systems.htqg.hf as htqg_hf
from mean_field.core.bands import centered_band_indices
from mean_field.core.hf.overlap import HFOverlapBlockSet
from mean_field.systems.htqg.hf import (
    HTQGInteractionSettings,
    HTQGProjectedHFConfig,
    build_htqg_hf_target_hamiltonian,
    _calculate_htqg_projected_overlap_between,
    _projected_basis,
    HTQGProjectedHFTargetInteractionContext,
    assemble_htqg_hf_target_hamiltonian,
    build_htqg_overlap_blocks,
    build_htqg_projected_hf_data,
    build_htqg_target_data,
    prepare_htqg_hf_target_interaction,
)
from mean_field.systems.htqg.optical import hf_optical_derivative_bundle
from mean_field.systems.htqg.params import HTQGParams


def _small_source(*, interactions: bool):
    params = HTQGParams.default(
        kappa=0.6,
        lambda_mdt_nm=0.0,
        include_dirac_rotation=False,
    )
    config = HTQGProjectedHFConfig(
        theta_deg=2.25,
        n_shells=0,
        mesh_size=1,
        active_band_count=2,
        domain="alpha_beta_alpha",
        filling=0,
        params=params,
        interaction=HTQGInteractionSettings(
            epsilon_r=10.0,
            d_sc_nm=25.0,
            g_shells=0,
            include_hartree=interactions,
            include_fock=interactions,
        ),
        max_iter=1,
        active_basis="energy",
        frac_shift=(0.0, 0.0),
    )
    return build_htqg_projected_hf_data(config)


def test_htqg_target_data_accepts_an_explicit_larger_centered_window() -> None:
    data = _small_source(interactions=False)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j, 1.0e-4 + 0.0j]),
        band_indices=target_bands,
    )

    assert target.band_indices == target_bands
    assert target.nt == 4 * len(target_bands)
    assert target.nk == 2
    assert target.h0.shape == (16, 16, 2)
    assert target.wavefunctions.shape[1:] == (4, 2, 2)
    np.testing.assert_allclose(
        target.h0,
        build_htqg_hf_target_hamiltonian(
            data,
            target,
            np.zeros_like(data.h0),
        ),
        atol=0.0,
        rtol=0.0,
    )


def test_htqg_rectangular_target_contracts_a_fixed_source_density() -> None:
    data = _small_source(interactions=True)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j]),
        band_indices=target_bands,
    )

    hamiltonian = build_htqg_hf_target_hamiltonian(
        data,
        target,
        np.zeros_like(data.h0),
    )

    assert hamiltonian.shape == target.h0.shape == (16, 16, 1)
    assert np.all(np.isfinite(hamiltonian))
    np.testing.assert_allclose(
        hamiltonian[:, :, 0],
        hamiltonian[:, :, 0].conjugate().T,
        atol=1.0e-12,
    )


def test_htqg_target_assembly_exposes_raw_components_and_reuses_context() -> None:
    data = _small_source(interactions=True)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j]),
        band_indices=target_bands,
    )
    density = np.zeros_like(data.h0)
    context = prepare_htqg_hf_target_interaction(data, target)
    assembly = assemble_htqg_hf_target_hamiltonian(
        data,
        target,
        density,
        target_interaction_context=context,
    )

    np.testing.assert_allclose(
        assembly.raw,
        assembly.h0 + assembly.hartree + assembly.fock,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        assembly.hermitian,
        build_htqg_hf_target_hamiltonian(
            data,
            target,
            density,
            target_interaction_context=context,
        ),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        assembly.hermitian_correction,
        assembly.hermitian - assembly.raw,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        assembly.hermitian,
        assembly.hermitian.swapaxes(0, 1).conjugate(),
        atol=1.0e-14,
    )


def test_htqg_target_action_matches_dense_components_without_dense_target_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _small_source(interactions=True)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j, 1.0e-4 + 0.0j]),
        band_indices=target_bands,
    )
    source_blocks = build_htqg_overlap_blocks(data)

    density_rng = np.random.default_rng(113)
    occupied_vector = density_rng.normal(size=data.nt) + 1.0j * density_rng.normal(
        size=data.nt
    )
    occupied_vector /= np.linalg.norm(occupied_vector)
    conventional_density = 0.4 * np.outer(occupied_vector, occupied_vector.conjugate())
    density = np.repeat(conventional_density.T[:, :, None], data.nk, axis=2)

    context = prepare_htqg_hf_target_interaction(data, target)
    dense = assemble_htqg_hf_target_hamiltonian(
        data,
        target,
        density,
        source_overlap_blocks=source_blocks,
        target_interaction_context=context,
        use_numba=False,
    )
    vector_rng = np.random.default_rng(127)
    vectors = vector_rng.normal(size=(target.nt, 3, target.nk)) + 1.0j * vector_rng.normal(
        size=(target.nt, 3, target.nk)
    )

    target_overlap_spy = Mock(
        side_effect=AssertionError("bounded-memory action called _target_overlap_blocks")
    )
    prepare_spy = Mock(
        side_effect=AssertionError(
            "bounded-memory action called prepare_htqg_hf_target_interaction"
        )
    )
    source_basis_spy = Mock(wraps=htqg_hf._projected_basis)
    source_blocks_spy = Mock(wraps=htqg_hf._build_htqg_overlap_blocks_from_basis)
    monkeypatch.setattr(htqg_hf, "_target_overlap_blocks", target_overlap_spy)
    monkeypatch.setattr(htqg_hf, "prepare_htqg_hf_target_interaction", prepare_spy)
    monkeypatch.setattr(htqg_hf, "_projected_basis", source_basis_spy)
    monkeypatch.setattr(
        htqg_hf,
        "_build_htqg_overlap_blocks_from_basis",
        source_blocks_spy,
    )

    action = htqg_hf.apply_htqg_hf_target_hamiltonian(
        data,
        target,
        density,
        vectors,
        use_numba=False,
    )

    assert action.raw.shape == vectors.shape
    for name in ("h0", "hartree", "fock", "raw"):
        expected = np.empty_like(vectors)
        for ik in range(target.nk):
            expected[:, :, ik] = getattr(dense, name)[:, :, ik] @ vectors[:, :, ik]
        np.testing.assert_allclose(getattr(action, name), expected, atol=3.0e-13)
    np.testing.assert_allclose(
        action.raw,
        action.h0 + action.hartree + action.fock,
        atol=0.0,
        rtol=0.0,
    )
    target_overlap_spy.assert_not_called()
    prepare_spy.assert_not_called()
    source_basis_spy.assert_called_once()
    source_blocks_spy.assert_called_once()


def test_htqg_target_action_rejects_invalid_vectors() -> None:
    data = _small_source(interactions=False)
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j, 1.0e-4 + 0.0j]),
    )
    density = np.zeros_like(data.h0)
    invalid_vectors = (
        np.zeros((target.nt, target.nk), dtype=np.complex128),
        np.zeros((target.nt + 1, 1, target.nk), dtype=np.complex128),
        np.full((target.nt, 1, target.nk), np.nan, dtype=np.complex128),
        np.full((target.nt, 1, target.nk), "not-numeric", dtype="U11"),
    )

    for vectors in invalid_vectors:
        with pytest.raises((TypeError, ValueError), match="vectors"):
            htqg_hf.apply_htqg_hf_target_hamiltonian(
                data,
                target,
                density,
                vectors,
                use_numba=False,
            )


def test_htqg_target_assembly_is_rectangular_basis_covariant() -> None:
    data = _small_source(interactions=True)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j]),
        band_indices=target_bands,
    )
    source_blocks = build_htqg_overlap_blocks(data)
    context = prepare_htqg_hf_target_interaction(data, target)
    density = np.asarray(data.reference_density, dtype=np.complex128).copy()
    density[0, 0, 0] += 0.125

    rng = np.random.default_rng(41)
    source_q, _ = np.linalg.qr(
        rng.normal(size=(data.nt, data.nt)) + 1.0j * rng.normal(size=(data.nt, data.nt))
    )
    target_q, _ = np.linalg.qr(
        rng.normal(size=(target.nt, target.nt)) + 1.0j * rng.normal(size=(target.nt, target.nt))
    )

    def rotate_matrices(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ia,ikjl,jb->akbl",
            left.conjugate(),
            np.asarray(values, dtype=np.complex128),
            right,
            optimize=True,
        )

    def rotate_blocks(
        blocks: HFOverlapBlockSet,
        left: np.ndarray,
        right: np.ndarray,
        diagonal_q: np.ndarray,
    ) -> HFOverlapBlockSet:
        return HFOverlapBlockSet(
            shifts=blocks.shifts,
            gvecs=blocks.gvecs,
            overlaps={
                shift: rotate_matrices(value, left, right)
                for shift, value in blocks.overlaps.items()
            },
            diagonal_overlaps={
                shift: np.einsum(
                    "ia,ijk,jb->abk",
                    diagonal_q.conjugate(),
                    value,
                    diagonal_q,
                    optimize=True,
                )
                for shift, value in blocks.diagonal_overlaps.items()
            },
            hartree_screening=blocks.hartree_screening,
            fock_screening=blocks.fock_screening,
        )

    source_rot = rotate_blocks(source_blocks, source_q, source_q, source_q)
    target_rot = rotate_blocks(
        context.target_overlap_blocks,
        target_q,
        target_q,
        target_q,
    )
    target_source_rot = rotate_blocks(
        context.target_source_overlap_blocks,
        target_q,
        source_q,
        target_q,
    )
    density_rot = np.asarray(
        [source_q.T @ density[:, :, ik] @ source_q.conjugate() for ik in range(data.nk)]
    ).transpose(1, 2, 0)
    reference_rot = np.asarray(
        [
            source_q.T @ data.reference_density[:, :, ik] @ source_q.conjugate()
            for ik in range(data.nk)
        ]
    ).transpose(1, 2, 0)
    h0_source_rot = np.asarray(
        [source_q.conjugate().T @ data.h0[:, :, ik] @ source_q for ik in range(data.nk)]
    ).transpose(1, 2, 0)
    h0_target_rot = np.asarray(
        [target_q.conjugate().T @ target.h0[:, :, ik] @ target_q for ik in range(target.nk)]
    ).transpose(1, 2, 0)
    data_rot = replace(data, h0=h0_source_rot, reference_density=reference_rot)
    target_data_rot = replace(target, h0=h0_target_rot)
    rotated_identity = prepare_htqg_hf_target_interaction(
        data_rot, target_data_rot
    ).identity
    context_rot = HTQGProjectedHFTargetInteractionContext(
        identity=rotated_identity,
        identity_sha256=htqg_hf._target_interaction_context_sha256(
            data_rot,
            target_data_rot,
            target_rot,
            target_source_rot,
        ),
        target_overlap_blocks=target_rot,
        target_source_overlap_blocks=target_source_rot,
    )

    reference = assemble_htqg_hf_target_hamiltonian(
        data,
        target,
        density,
        source_overlap_blocks=source_blocks,
        target_interaction_context=context,
        use_numba=False,
    )
    rotated = assemble_htqg_hf_target_hamiltonian(
        data_rot,
        target_data_rot,
        density_rot,
        source_overlap_blocks=source_rot,
        target_interaction_context=context_rot,
        use_numba=False,
    )
    for name in ("h0", "hartree", "fock", "raw", "hermitian", "hermitian_correction"):
        expected = np.asarray(
            [
                target_q.conjugate().T @ getattr(reference, name)[:, :, ik] @ target_q
                for ik in range(target.nk)
            ]
        ).transpose(1, 2, 0)
        np.testing.assert_allclose(getattr(rotated, name), expected, atol=3.0e-13)


def test_htqg_target_context_rejects_same_shape_wrong_target() -> None:
    data = _small_source(interactions=True)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j]),
        band_indices=target_bands,
    )
    wrong_target = build_htqg_target_data(
        data,
        np.asarray([1.0e-4 + 0.0j]),
        band_indices=target_bands,
    )
    context = prepare_htqg_hf_target_interaction(data, target)
    with pytest.raises(ValueError, match="does not belong"):
        assemble_htqg_hf_target_hamiltonian(
            data,
            wrong_target,
            np.zeros_like(data.h0),
            target_interaction_context=context,
        )


def test_htqg_ambient_target_projects_back_to_the_exact_source_operator() -> None:
    data = _small_source(interactions=True)
    k0 = np.asarray([complex(data.kvec[0])])
    source_target = build_htqg_target_data(data, k0)
    ambient_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    ambient_target = build_htqg_target_data(data, k0, band_indices=ambient_bands)
    density = np.zeros_like(data.h0)

    source_hamiltonian = build_htqg_hf_target_hamiltonian(data, source_target, density)
    ambient_hamiltonian = build_htqg_hf_target_hamiltonian(data, ambient_target, density)
    target_basis = _projected_basis(
        data,
        wavefunctions=ambient_target.wavefunctions,
        name="ambient-target",
    )
    source_basis = _projected_basis(
        data,
        wavefunctions=source_target.wavefunctions,
        name="source-target",
    )
    overlap = _calculate_htqg_projected_overlap_between(
        target_basis,
        source_basis,
        0,
        0,
    )[:, 0, :, 0]

    np.testing.assert_allclose(overlap.conjugate().T @ overlap, np.eye(data.nt), atol=1.0e-12)
    np.testing.assert_allclose(
        overlap.conjugate().T @ ambient_hamiltonian[:, :, 0] @ overlap,
        source_hamiltonian[:, :, 0],
        atol=1.0e-12,
    )


def test_htqg_optical_derivative_bundle_preserves_the_requested_target_window() -> None:
    data = _small_source(interactions=False)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    point = hf_optical_derivative_bundle(
        data,
        np.zeros_like(data.h0),
        0.0 + 0.0j,
        step_nm_inv=1.0e-4,
        target_band_indices=target_bands,
    )

    assert point.band_indices == target_bands
    assert point.hamiltonian.shape == (16, 16)
    assert point.dhdk.shape == (2, 16, 16)
    assert point.d2hdk.shape == (2, 2, 16, 16)
    assert point.energies_ev.shape == (16,)
    assert point.eigenvectors.shape == (16, 16)
    assert point.min_basis_overlap_singular_value > 0.99


def test_htqg_target_data_rejects_invalid_explicit_windows_before_solve() -> None:
    data = _small_source(interactions=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_htqg_target_data(data, np.asarray([0.0j]), band_indices=(3, 2))
    with pytest.raises(ValueError, match="nonempty and unique"):
        build_htqg_target_data(data, np.asarray([0.0j]), band_indices=(2, 2))
    with pytest.raises(ValueError, match="escape matrix dimension"):
        build_htqg_target_data(
            data,
            np.asarray([0.0j]),
            band_indices=(0, data.lattice.matrix_dim),
        )

def test_htqg_target_batch_action_matches_serial_and_reuses_target_overlaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _small_source(interactions=True)
    target_bands = tuple(centered_band_indices(data.lattice.matrix_dim, 4))
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j, 1.0e-4 + 0.0j]),
        band_indices=target_bands,
    )
    source_blocks = build_htqg_overlap_blocks(data)

    rng = np.random.default_rng(173)
    stored_densities = []
    for weight in (0.25, 0.55):
        occupied = rng.normal(size=data.nt) + 1.0j * rng.normal(size=data.nt)
        occupied /= np.linalg.norm(occupied)
        conventional = weight * np.outer(occupied, occupied.conjugate())
        stored_densities.append(
            np.repeat(conventional.T[:, :, None], data.nk, axis=2)
        )
    densities = np.stack(stored_densities, axis=0)
    vectors = rng.normal(size=(target.nt, 2, target.nk)) + 1.0j * rng.normal(
        size=(target.nt, 2, target.nk)
    )

    overlap_spy = Mock(wraps=htqg_hf._calculate_htqg_projected_overlap_between)
    monkeypatch.setattr(
        htqg_hf,
        "_calculate_htqg_projected_overlap_between",
        overlap_spy,
    )
    htqg_hf.apply_htqg_hf_target_hamiltonian_batch(
        data,
        target,
        densities[:1],
        vectors,
        source_overlap_blocks=source_blocks,
        use_numba=False,
    )
    one_batch_overlap_calls = overlap_spy.call_count

    overlap_spy.reset_mock()
    batch = htqg_hf.apply_htqg_hf_target_hamiltonian_batch(
        data,
        target,
        densities,
        vectors,
        source_overlap_blocks=source_blocks,
        use_numba=False,
    )
    two_batch_overlap_calls = overlap_spy.call_count

    overlap_spy.reset_mock()
    serial = [
        htqg_hf.apply_htqg_hf_target_hamiltonian(
            data,
            target,
            density,
            vectors,
            source_overlap_blocks=source_blocks,
            use_numba=False,
        )
        for density in densities
    ]
    two_serial_overlap_calls = overlap_spy.call_count

    expected_shape = (densities.shape[0], target.nt, vectors.shape[1], target.nk)
    for name in ("h0", "hartree", "fock", "raw"):
        expected = np.stack([getattr(action, name) for action in serial], axis=0)
        assert getattr(batch, name).shape == expected_shape
        np.testing.assert_allclose(
            getattr(batch, name),
            expected,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    np.testing.assert_allclose(
        batch.raw,
        batch.h0 + batch.hartree + batch.fock,
        atol=0.0,
        rtol=0.0,
    )
    expected_batch_overlap_calls = 2 * target.nk * len(data.shifts)
    assert one_batch_overlap_calls == expected_batch_overlap_calls
    assert two_batch_overlap_calls == expected_batch_overlap_calls
    assert two_serial_overlap_calls == densities.shape[0] * expected_batch_overlap_calls
    assert two_batch_overlap_calls < two_serial_overlap_calls

def test_htqg_target_batch_action_rejects_invalid_density_batches() -> None:
    data = _small_source(interactions=False)
    target = build_htqg_target_data(
        data,
        np.asarray([0.0 + 0.0j, 1.0e-4 + 0.0j]),
    )
    vectors = np.zeros((target.nt, 1, target.nk), dtype=np.complex128)
    valid_shape = (2, data.nt, data.nt, data.nk)
    invalid_densities = (
        np.zeros((data.nt, data.nt, data.nk), dtype=np.complex128),
        np.zeros((0, data.nt, data.nt, data.nk), dtype=np.complex128),
        np.zeros((2, data.nt + 1, data.nt, data.nk), dtype=np.complex128),
        np.full(valid_shape, np.nan, dtype=np.complex128),
        np.full(valid_shape, "not-numeric", dtype="U11"),
    )

    for densities in invalid_densities:
        with pytest.raises((TypeError, ValueError), match="densities"):
            htqg_hf.apply_htqg_hf_target_hamiltonian_batch(
                data,
                target,
                densities,
                vectors,
                use_numba=False,
            )
