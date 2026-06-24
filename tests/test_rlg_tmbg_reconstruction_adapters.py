from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from analysis.topology import assert_topology_eligible, compute_system_topology_from_bundle
from mean_field.core.hf import ProjectedWavefunctionBasis
from mean_field.systems.RnG_hBN.hf import (
    RLGhBNHartreeFockState,
    build_rlg_hbn_final_hf_eigensystem,
    expand_rlg_hbn_projected_micro_basis,
    reconstruct_rlg_hbn_projected_hf_micro_wavefunctions,
    rlg_hbn_projected_hf_active_index,
)
from mean_field.systems.tmbg import polshyn_supercell as polshyn_public
from mean_field.systems.tmbg._polshyn_reconstruction import (
    expand_polshyn_projected_micro_basis,
    polshyn_projected_hf_active_index,
    polshyn_wang_active_eigenvectors_from_state,
    reconstruct_polshyn_wang_hf_micro_wavefunctions,
)
from mean_field.systems.tmbg.polshyn_supercell import PolshynDoubledCell, PolshynProjectedBasis, PolshynWangHFState


def _unitary_phases(n_active: int, n_k: int) -> np.ndarray:
    coeffs = np.repeat(np.eye(n_active, dtype=np.complex128)[:, :, None], n_k, axis=2)
    for ik in range(n_k):
        coeffs[:, :, ik] = np.diag(np.exp(1j * 0.07 * (ik + 1) * np.arange(n_active)))
    return coeffs


def _toy_rlg_data() -> SimpleNamespace:
    local_basis_size = 2
    grid_shape = (2, 3)
    basis_dim = local_basis_size * grid_shape[0] * grid_shape[1]
    n_band = 2
    n_eta = 2
    n_k = 4
    raw = np.zeros((basis_dim, n_band, n_eta, n_k), dtype=np.complex128)
    for ibasis in range(basis_dim):
        for iband in range(n_band):
            for iflavor in range(n_eta):
                for ik in range(n_k):
                    raw[ibasis, iband, iflavor, ik] = (
                        1 + 1000 * iband + 100 * iflavor + 10 * ibasis + ik + 1j * (iband - iflavor + 0.01 * ibasis)
                    )
    basis = ProjectedWavefunctionBasis(
        wavefunctions=raw,
        grid_shape=grid_shape,
        n_spin=2,
        local_basis_size=local_basis_size,
        boundary_mode="zero_fill",
    )
    k_grid_frac = np.stack(
        np.meshgrid(np.asarray([0.0, 0.5]), np.asarray([0.0, 0.5]), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    return SimpleNamespace(
        basis=basis,
        mesh_size=2,
        kvec=np.asarray([0.0, 0.5j, 0.5, 0.5 + 0.5j], dtype=np.complex128),
        k_grid_frac=k_grid_frac,
        valleys=(1, -1),
        reciprocal_grid_shape=grid_shape,
        reciprocal_grid_origin=(0, 0),
        active_band_indices=(10, 11),
        flat_band_indices=(10, 11),
    )


def _manual_rlg_shift(vectors: np.ndarray, *, shift: tuple[int, int]) -> np.ndarray:
    local = 2
    nx = 2
    ny = 3
    basis_dim = local * nx * ny
    valleys = (1, -1)
    array = np.asarray(vectors, dtype=np.complex128)
    one_dimensional = array.ndim == 1
    matrix = array[:, None] if one_dimensional else array.reshape((array.shape[0], -1), order="F")
    expected = np.zeros_like(matrix)
    frames = int(matrix.shape[1])
    for ispin in range(2):
        for iflavor, valley in enumerate(valleys):
            start = (ispin * len(valleys) + iflavor) * basis_dim
            block = matrix[start : start + basis_dim, :].reshape((local, nx, ny, frames), order="F")
            shifted = np.zeros_like(block)
            sx = -int(valley) * int(shift[0])
            sy = -int(valley) * int(shift[1])
            for ix in range(nx):
                tx = ix + sx
                if tx < 0 or tx >= nx:
                    continue
                for iy in range(ny):
                    ty = iy + sy
                    if ty < 0 or ty >= ny:
                        continue
                    shifted[:, tx, ty, :] = block[:, ix, iy, :]
            expected[start : start + basis_dim, :] = shifted.reshape((basis_dim, frames), order="F")
    if one_dimensional:
        return expected[:, 0]
    return expected.reshape(array.shape, order="F")


def test_rlg_hbn_reconstruction_expands_spin_flavor_rows_and_uses_public_common_helper() -> None:
    data = _toy_rlg_data()
    expanded = expand_rlg_hbn_projected_micro_basis(data)
    active = rlg_hbn_projected_hf_active_index(2, 2, 2)
    raw = data.basis.wavefunctions
    basis_dim = int(data.basis.basis_dimension)
    micro_dim = int(data.basis.n_spin * data.basis.n_flavor * basis_dim)

    assert data.basis.local_basis_size == 2
    assert data.basis.grid_shape == (2, 3)
    assert expanded.shape == (4, micro_dim, 8)
    for ispin in range(2):
        for iflavor in range(2):
            row_start = (ispin * 2 + iflavor) * basis_dim
            for iband in range(2):
                col = int(active[ispin, iflavor, iband])
                np.testing.assert_allclose(expanded[:, row_start : row_start + basis_dim, col], raw[:, iband, iflavor, :].T)

    coeffs = _unitary_phases(8, 4)
    bundle = reconstruct_rlg_hbn_projected_hf_micro_wavefunctions(data, coeffs, include_sewing=True, as_grid=True)

    expected_flat = np.einsum("kba,ahk->kbh", expanded, coeffs, optimize=True)
    assert bundle.psi_micro.shape == (2, 2, micro_dim, 8)
    np.testing.assert_allclose(bundle.psi_micro.reshape(4, micro_dim, 8), expected_flat)
    assert len(bundle.sewing_transforms) == 2
    assert bundle.basis_metadata["system"] == "RnG_hBN"
    assert bundle.basis_metadata["reconstruction_adapter"].startswith("mean_field.systems.RnG_hBN.hf")
    assert bundle.basis_metadata["common_helper"].endswith("reconstruct_projected_micro_wavefunctions")
    assert bundle.basis_metadata["raw_wavefunctions_axis_order"] == "basis,band,flavor,k"
    assert bundle.basis_metadata["microscopic_row_order"] == "spin_major,flavor_inner,basis_F(local,nx,ny)"
    assert bundle.basis_metadata["active_column_order"].endswith("order='F')")
    assert bundle.basis_metadata["local_basis_size"] == 2
    assert bundle.basis_metadata["reciprocal_grid_shape"] == [2, 3]

    labelled_rows = np.arange(micro_dim * 3, dtype=float).reshape((micro_dim, 3), order="F").astype(np.complex128)
    np.testing.assert_allclose(bundle.sewing_transforms[0](labelled_rows), _manual_rlg_shift(labelled_rows, shift=(1, 0)))
    np.testing.assert_allclose(bundle.sewing_transforms[1](labelled_rows), _manual_rlg_shift(labelled_rows, shift=(0, 1)))


def _toy_rlg_state() -> RLGhBNHartreeFockState:
    nt = 8
    nk = 4
    hamiltonian = np.zeros((nt, nt, nk), dtype=np.complex128)
    for ik in range(nk):
        hamiltonian[:, :, ik] = np.diag(np.arange(nt, dtype=float) + 0.1 * ik)
    return RLGhBNHartreeFockState(
        h0=hamiltonian.copy(),
        density=np.zeros_like(hamiltonian),
        hamiltonian=hamiltonian.copy(),
        energies=np.zeros((nt, nk), dtype=float),
        reference_density=np.zeros_like(hamiltonian),
        nu=0.0,
        v0=1.0,
        active_valence_bands=1,
        scheme="average",
        mu=0.0,
        precision=1.0e-9,
        n_spin=2,
        n_eta=2,
        n_band=2,
        occupation_counts=(1, 0, 1, 0),
    )


def test_rlg_hbn_public_reconstruction_supports_selected_states_and_size_guard() -> None:
    data = _toy_rlg_data()
    expanded = expand_rlg_hbn_projected_micro_basis(data)
    coeffs = _unitary_phases(8, 4)
    micro_dim = int(data.basis.n_spin * data.basis.n_flavor * data.basis.basis_dimension)
    full_output_elements = 4 * micro_dim * 8
    selected_output_elements = 4 * micro_dim * 2

    with pytest.raises(ValueError, match="size guard"):
        reconstruct_rlg_hbn_projected_hf_micro_wavefunctions(
            data,
            coeffs,
            include_sewing=False,
            as_grid=False,
            max_dense_elements=full_output_elements - 1,
        )

    selected = reconstruct_rlg_hbn_projected_hf_micro_wavefunctions(
        data,
        coeffs,
        include_sewing=True,
        as_grid=False,
        state_indices=(1, 6),
        max_dense_elements=selected_output_elements,
    )
    expected_selected = np.einsum("kba,ahk->kbh", expanded, coeffs[:, [1, 6], :], optimize=True)
    assert selected.psi_micro.shape == (4, micro_dim, 2)
    np.testing.assert_allclose(selected.psi_micro, expected_selected)
    assert selected.basis_metadata["selected_hf_state_indices"] == [1, 6]
    assert selected.basis_metadata["n_reconstructed_states"] == 2
    np.testing.assert_allclose(selected.sewing_transforms[0](selected.psi_micro[0]), _manual_rlg_shift(selected.psi_micro[0], shift=(1, 0)))
    np.testing.assert_allclose(selected.sewing_transforms[1](selected.psi_micro[0]), _manual_rlg_shift(selected.psi_micro[0], shift=(0, 1)))

    rectangular = reconstruct_rlg_hbn_projected_hf_micro_wavefunctions(
        data,
        coeffs[:, [1, 6], :],
        include_sewing=False,
        as_grid=False,
        state_indices=(1, 6),
        max_dense_elements=selected_output_elements,
    )
    np.testing.assert_allclose(rectangular.psi_micro, expected_selected)
    assert rectangular.basis_metadata["selected_coefficients_from_full_eigensystem"] is False
    with pytest.raises(ValueError, match="Rectangular selected active_eigenvectors"):
        reconstruct_rlg_hbn_projected_hf_micro_wavefunctions(data, coeffs[:, [1, 6], :], include_sewing=False, as_grid=False)

    state = _toy_rlg_state()
    final_hf = build_rlg_hbn_final_hf_eigensystem(state)
    run_like = SimpleNamespace(basis_data=data, state=state)
    from_state = reconstruct_rlg_hbn_projected_hf_micro_wavefunctions(
        run_like,
        include_sewing=False,
        as_grid=False,
        state_indices=(1, 6),
        max_dense_elements=selected_output_elements,
    )
    expected_from_state = np.einsum("kba,ahk->kbh", expanded, final_hf.eigenvectors[:, [1, 6], :], optimize=True)
    np.testing.assert_allclose(from_state.psi_micro, expected_from_state)
    assert from_state.basis_metadata["eigenvector_source"].startswith("build_rlg_hbn_final_hf_eigensystem")
    assert "tdhf" not in from_state.basis_metadata["eigenvector_source"].lower()

    bad_state = _toy_rlg_state()
    bad_state.hamiltonian[0, 1, 0] = 1.0e-3
    with pytest.raises(ValueError, match="not Hermitian enough"):
        build_rlg_hbn_final_hf_eigensystem(bad_state)


def _toy_polshyn_basis(n_k: int = 2, embedding_shape: tuple[int, int] = (2, 3)) -> PolshynProjectedBasis:
    local_basis_size = 6
    nx, ny = (int(embedding_shape[0]), int(embedding_shape[1]))
    basis_dim = local_basis_size * nx * ny
    nb = 2
    n_eta = 2
    raw = np.zeros((basis_dim, nb, n_eta, n_k), dtype=np.complex128)
    for ilocal in range(local_basis_size):
        for ix in range(nx):
            for iy in range(ny):
                ibasis = ilocal + local_basis_size * (ix + nx * iy)
                for iband in range(nb):
                    for ieta in range(n_eta):
                        for ik in range(n_k):
                            raw[ibasis, iband, ieta, ik] = (
                                1
                                + 100000 * iy
                                + 10000 * ix
                                + 1000 * ilocal
                                + 100 * iband
                                + 10 * ieta
                                + ik
                                + 1j * (0.01 * ibasis + iband - ieta)
                            )
    h0_blocks = np.zeros((2, n_eta, nb, nb, n_k), dtype=np.complex128)
    frac_x = np.arange(n_k, dtype=float) / max(int(n_k), 1)
    return PolshynProjectedBasis(
        model=object(),
        supercell=PolshynDoubledCell(),
        kvec=np.asarray([complex(value, 0.0) for value in frac_x], dtype=np.complex128),
        k_grid_frac=np.stack([frac_x, np.zeros(n_k, dtype=float)], axis=1),
        projected_indices=(27,),
        target_band_index=27,
        wavefunctions=raw,
        h0_blocks=h0_blocks,
        reference_diagonal=np.zeros((nb,), dtype=float),
        super_b1=1.0 + 0.0j,
        super_b2=0.0 + 1.0j,
        embedding_shape=(nx, ny),
        embedding_origin=(0, 0),
        embedding_positions={(ix, iy, fold): (ix, iy) for ix in range(nx) for iy in range(ny) for fold in (0, 1)},
    )


def _toy_polshyn_state_from_hamiltonian(
    basis: PolshynProjectedBasis,
    hamiltonian: np.ndarray,
) -> PolshynWangHFState:
    active = polshyn_projected_hf_active_index(basis.n_spin, basis.n_eta, basis.nb)
    hmat = np.asarray(hamiltonian, dtype=np.complex128)
    energies = np.zeros((basis.n_spin * basis.n_eta * basis.nb, basis.nk), dtype=float)
    for ik in range(basis.nk):
        for ispin in range(basis.n_spin):
            for ieta in range(basis.n_eta):
                indices = np.asarray(active[ispin, ieta, :], dtype=int)
                evals, _evecs = np.linalg.eigh(hmat[:, :, ik][np.ix_(indices, indices)])
                energies[indices, ik] = evals
    return PolshynWangHFState(
        h0=hmat.copy(),
        density=np.conjugate(hmat) * 0.0,
        hamiltonian=hmat.copy(),
        energies=energies,
        mu=0.0,
        precision=1.0e-9,
        v0=0.0,
        diagnostics={},
    )


def test_polshyn_public_reconstruction_keeps_flat_k_order_and_records_missing_sewing() -> None:
    assert "reconstruct_polshyn_wang_hf_micro_wavefunctions" in polshyn_public.__all__
    assert polshyn_public.reconstruct_polshyn_wang_hf_micro_wavefunctions is reconstruct_polshyn_wang_hf_micro_wavefunctions

    basis = _toy_polshyn_basis(embedding_shape=(2, 3))
    expanded = expand_polshyn_projected_micro_basis(basis)
    active = polshyn_projected_hf_active_index(2, 2, 2)
    raw = basis.wavefunctions
    basis_dim = int(basis.basis_dimension)
    micro_dim = int(basis.n_spin * basis.n_eta * basis_dim)

    assert basis.embedding_shape == (2, 3)
    assert basis_dim == 6 * 2 * 3
    assert expanded.shape == (2, micro_dim, 8)
    for ispin in range(2):
        for ieta in range(2):
            row_start = (ispin * 2 + ieta) * basis_dim
            for iband in range(2):
                col = int(active[ispin, ieta, iband])
                np.testing.assert_allclose(expanded[:, row_start : row_start + basis_dim, col], raw[:, iband, ieta, :].T)

    probe_basis = 5 + 6 * (1 + 2 * 2)
    probe_col = int(active[1, 1, 1])
    probe_row = (1 * 2 + 1) * basis_dim + probe_basis
    assert probe_basis < basis_dim
    assert raw[probe_basis, 1, 1, 1] == pytest.approx(
        1 + 100000 * 2 + 10000 * 1 + 1000 * 5 + 100 + 10 + 1 + 1j * (0.01 * probe_basis)
    )
    assert expanded[1, probe_row, probe_col] == raw[probe_basis, 1, 1, 1]

    coeffs = _unitary_phases(8, 2)
    full_output_elements = basis.nk * micro_dim * 8
    bundle = reconstruct_polshyn_wang_hf_micro_wavefunctions(
        basis,
        active_eigenvectors=coeffs,
        max_dense_elements=full_output_elements,
    )

    expected = np.einsum("kba,ahk->kbh", expanded, coeffs, optimize=True)
    np.testing.assert_allclose(bundle.psi_micro, expected)
    assert bundle.sewing_transforms == ()
    assert bundle.basis_metadata["system"] == "tmbg_polshyn_doubled"
    assert bundle.basis_metadata["reconstruction_api_status"] == "public_flat_k_diagnostic_topology_ineligible"
    assert bundle.basis_metadata["public_facade_exported"] is True
    assert bundle.basis_metadata["public_facade"] == "mean_field.systems.tmbg.polshyn_supercell.reconstruct_polshyn_wang_hf_micro_wavefunctions"
    assert bundle.basis_metadata["raw_wavefunctions_axis_order"] == "basis,folded_band,valley,k"
    assert bundle.basis_metadata["grid_shape_attached"] is False
    assert bundle.basis_metadata["sewing_available"] is False
    assert bundle.basis_metadata["topology_status"] == "topology-ineligible"
    assert bundle.basis_metadata["topology_eligible"] is False
    assert "sewing" in bundle.basis_metadata["topology_ineligible_reason"]
    with pytest.raises(ValueError, match="topology_eligible=False.*Polshyn doubled-cell sewing"):
        assert_topology_eligible(bundle, context="polshyn-public-diagnostic")
    with pytest.raises(ValueError, match="topology_eligible=False.*Polshyn doubled-cell sewing"):
        compute_system_topology_from_bundle(bundle, 0, system="tmbg_polshyn")
    assert bundle.basis_metadata["embedding_shape"] == [2, 3]
    assert bundle.basis_metadata["selected_hf_state_indices"] == list(range(8))
    assert bundle.basis_metadata["n_reconstructed_states"] == 8
    assert bundle.basis_metadata["dense_reconstruction_estimated_elements"] == full_output_elements
    assert "iy/f2 outer" in bundle.basis_metadata["k_flat_order"]
    assert bundle.basis_metadata["folded_band_labels"][1]["fold_momentum"] == "k+super_b1"

    with pytest.raises(NotImplementedError, match="sewing"):
        reconstruct_polshyn_wang_hf_micro_wavefunctions(basis, active_eigenvectors=coeffs, include_sewing=True)


def test_polshyn_public_reconstruction_supports_selected_states_and_size_guard() -> None:
    basis = _toy_polshyn_basis(embedding_shape=(2, 3))
    expanded = expand_polshyn_projected_micro_basis(basis)
    coeffs = _unitary_phases(8, 2)
    micro_dim = int(basis.n_spin * basis.n_eta * basis.basis_dimension)
    selected_output_elements = basis.nk * micro_dim * 2

    with pytest.raises(ValueError, match="size guard"):
        reconstruct_polshyn_wang_hf_micro_wavefunctions(
            basis,
            active_eigenvectors=coeffs,
            state_indices=(1, 6),
            max_dense_elements=selected_output_elements - 1,
        )

    selected = reconstruct_polshyn_wang_hf_micro_wavefunctions(
        basis,
        active_eigenvectors=coeffs,
        state_indices=(1, 6),
        max_dense_elements=selected_output_elements,
    )
    expected_selected = np.einsum("kba,ahk->kbh", expanded, coeffs[:, [1, 6], :], optimize=True)
    assert selected.psi_micro.shape == (basis.nk, micro_dim, 2)
    np.testing.assert_allclose(selected.psi_micro, expected_selected)
    assert selected.basis_metadata["selected_hf_state_indices"] == [1, 6]
    assert selected.basis_metadata["n_reconstructed_states"] == 2
    assert selected.basis_metadata["selected_coefficients_from_full_eigensystem"] is True
    assert selected.basis_metadata["selected_state_allocation"] == "output_axis_contains_only_selected_hf_states"
    assert selected.basis_metadata["dense_reconstruction_estimated_elements"] == selected_output_elements
    assert selected.basis_metadata["dense_reconstruction_estimated_all_state_elements"] == basis.nk * micro_dim * 8

    rectangular = reconstruct_polshyn_wang_hf_micro_wavefunctions(
        basis,
        active_eigenvectors=coeffs[:, [1, 6], :],
        state_indices=(1, 6),
        max_dense_elements=selected_output_elements,
    )
    np.testing.assert_allclose(rectangular.psi_micro, expected_selected)
    assert rectangular.basis_metadata["selected_coefficients_from_full_eigensystem"] is False

    with pytest.raises(ValueError, match="Rectangular selected active_eigenvectors"):
        reconstruct_polshyn_wang_hf_micro_wavefunctions(
            basis,
            active_eigenvectors=coeffs[:, [1, 6], :],
            max_dense_elements=selected_output_elements,
        )
    with pytest.raises(ValueError, match="HF state indices"):
        reconstruct_polshyn_wang_hf_micro_wavefunctions(
            basis,
            active_eigenvectors=coeffs,
            state_indices=(8,),
            max_dense_elements=basis.nk * micro_dim * 8,
        )


def test_polshyn_state_diagonalization_uses_ket_eigenvectors_not_stored_density_orientation() -> None:
    basis = _toy_polshyn_basis(embedding_shape=(2, 2))
    active = polshyn_projected_hf_active_index(2, 2, 2)
    nt = 8
    nk = 2
    hamiltonian = np.zeros((nt, nt, nk), dtype=np.complex128)
    base_block = np.asarray([[0.25, -0.12j], [0.12j, 0.5]], dtype=np.complex128)
    for ik in range(nk):
        for ispin in range(2):
            for ieta in range(2):
                indices = np.asarray(active[ispin, ieta, :], dtype=int)
                hamiltonian[:, :, ik][np.ix_(indices, indices)] = base_block + 0.01 * (ik + ispin + ieta) * np.eye(2)
    state = _toy_polshyn_state_from_hamiltonian(basis, hamiltonian)

    coeffs, energies, diagnostics = polshyn_wang_active_eigenvectors_from_state(basis, state)

    assert diagnostics["hamiltonian_hermiticity_residual"] == pytest.approx(0.0)
    assert diagnostics["off_sector_hamiltonian_residual"] == pytest.approx(0.0)
    assert diagnostics["stored_energy_eigh_residual"] == pytest.approx(0.0)
    for ik in range(nk):
        np.testing.assert_allclose(coeffs[:, :, ik].conjugate().T @ coeffs[:, :, ik], np.eye(nt), atol=1.0e-14)
        np.testing.assert_allclose(hamiltonian[:, :, ik] @ coeffs[:, :, ik], coeffs[:, :, ik] @ np.diag(energies[:, ik]), atol=1.0e-14)
    sector = np.asarray(active[0, 0, :], dtype=int)
    evals, evecs = np.linalg.eigh(base_block)
    np.testing.assert_allclose(energies[sector, 0], evals)
    np.testing.assert_allclose(coeffs[np.ix_(sector, sector, [0])][:, :, 0], evecs)

    micro_dim = int(basis.n_spin * basis.n_eta * basis.basis_dimension)
    bundle = reconstruct_polshyn_wang_hf_micro_wavefunctions(
        basis,
        state=state,
        max_dense_elements=basis.nk * micro_dim * nt,
    )
    expected = np.einsum("kba,ahk->kbh", expand_polshyn_projected_micro_basis(basis), coeffs, optimize=True)
    np.testing.assert_allclose(bundle.psi_micro, expected)
    assert bundle.basis_metadata["eigenvector_source"].startswith("sector_np.linalg.eigh")
    assert bundle.basis_metadata["hamiltonian_hermiticity_residual"] == pytest.approx(0.0)
    assert bundle.basis_metadata["off_sector_hamiltonian_residual"] == pytest.approx(0.0)
    assert bundle.basis_metadata["stored_energy_eigh_residual"] == pytest.approx(0.0)

    selected = reconstruct_polshyn_wang_hf_micro_wavefunctions(
        basis,
        state=state,
        state_indices=(1, 6),
        max_dense_elements=basis.nk * micro_dim * 2,
    )
    expected_selected = np.einsum("kba,ahk->kbh", expand_polshyn_projected_micro_basis(basis), coeffs[:, [1, 6], :], optimize=True)
    np.testing.assert_allclose(selected.psi_micro, expected_selected)
    assert selected.basis_metadata["selected_hf_state_indices"] == [1, 6]

    bad_hermiticity = _toy_polshyn_state_from_hamiltonian(basis, hamiltonian)
    bad_hermiticity.hamiltonian[int(active[0, 0, 0]), int(active[0, 0, 1]), 0] += 1.0e-3
    with pytest.raises(ValueError, match="not Hermitian enough"):
        polshyn_wang_active_eigenvectors_from_state(basis, bad_hermiticity)

    bad_off_sector = _toy_polshyn_state_from_hamiltonian(basis, hamiltonian)
    bad_off_sector.hamiltonian[int(active[0, 0, 0]), int(active[1, 0, 0]), 0] = 1.0e-3
    bad_off_sector.hamiltonian[int(active[1, 0, 0]), int(active[0, 0, 0]), 0] = 1.0e-3
    with pytest.raises(ValueError, match="off-sector"):
        polshyn_wang_active_eigenvectors_from_state(basis, bad_off_sector)

    bad_stored_energies = _toy_polshyn_state_from_hamiltonian(basis, hamiltonian)
    bad_stored_energies.energies[0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="stored energies"):
        polshyn_wang_active_eigenvectors_from_state(basis, bad_stored_energies)
