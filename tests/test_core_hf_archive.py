from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.hf import (
    ProjectorConvention,
    ReferencePolicy,
    average_reference_density,
    load_projector_from_hf_archive,
    stored_density_to_projector,
    summarize_hf_state_archive,
    validate_hf_archive_shapes,
)


def test_hf_archive_public_policy_aliases_are_exported() -> None:
    assert ReferencePolicy is not None
    assert ProjectorConvention is not None


def test_hf_archive_summary_reads_shapes_and_metadata(tmp_path) -> None:
    path = tmp_path / "state.npz"
    density = np.zeros((4, 4, 3), dtype=np.complex128)
    np.savez_compressed(
        path,
        density=density,
        hamiltonian=np.zeros_like(density),
        h0=np.zeros_like(density),
        energies_mev=np.zeros((4, 3), dtype=float),
        k_grid_frac=np.zeros((3, 2), dtype=float),
        reference_density=np.zeros_like(density),
        n_spin=np.asarray([2]),
        n_eta=np.asarray([1]),
        n_band=np.asarray([2]),
        active_valence_bands=np.asarray([1]),
        cache_dir=np.asarray("cache-root"),
        cache_key_basis=np.asarray("basis-key"),
    )

    summary = summarize_hf_state_archive(path)
    validate_hf_archive_shapes(summary)

    assert summary.density_shape == (4, 4, 3)
    assert summary.n_spin == 2
    assert summary.n_eta == 1
    assert summary.n_band == 2
    assert summary.has_reference_density
    assert summary.cache_dir == "cache-root"
    assert summary.cache_key_basis == "basis-key"


def test_stored_density_to_projector_uses_reference_and_ket_transpose() -> None:
    density = np.zeros((2, 2, 1), dtype=np.complex128)
    density[0, 1, 0] = 2.0 + 3.0j
    reference = np.zeros_like(density)
    reference[1, 0, 0] = 5.0

    stored = stored_density_to_projector(density, reference, convention="stored")
    ket = stored_density_to_projector(density, reference, convention="ket")

    np.testing.assert_array_equal(stored, density + reference)
    np.testing.assert_array_equal(ket[:, :, 0], (density + reference)[:, :, 0].T)


def test_stored_density_to_projector_average_reference_and_require_policy() -> None:
    density = np.zeros((2, 2, 3), dtype=np.complex128)
    projector = stored_density_to_projector(density, reference_policy="average", convention="stored")
    np.testing.assert_allclose(projector, average_reference_density(2, 3))

    with pytest.raises(ValueError, match="reference_density is required"):
        stored_density_to_projector(density, reference_policy="require")


def test_load_projector_from_hf_archive(tmp_path) -> None:
    path = tmp_path / "state.npz"
    density = np.zeros((2, 2, 1), dtype=np.complex128)
    reference = np.zeros_like(density)
    density[0, 1, 0] = 1.0
    np.savez_compressed(path, density=density, reference_density=reference)

    projector = load_projector_from_hf_archive(path, convention="ket")
    assert projector[1, 0, 0] == 1.0
