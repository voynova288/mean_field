from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from mean_field.core.io import file_sha256
import mean_field.systems.inas_gasb.kane_poisson_archive as archive_module
from mean_field.systems.inas_gasb.kane_poisson_archive import (
    load_canonical_kane_poisson_archive,
    save_canonical_kane_poisson_archive,
)
from mean_field.systems.inas_gasb.kane_poisson import (
    CanonicalKanePoissonConfig,
    KaneParentSourceSpec,
    KaneWindowAtPotential,
    PLANE_WAVE_FLATTENING_LABEL,
    PLANE_WAVE_POTENTIAL_COUPLING_LABEL,
    canonical_kane_poisson_fingerprint,
    canonical_parent_matrix_sequence_sha256,
    kane_orbital_transfer_density_profiles,
    kane_poisson_array_sha256,
    kane_potential_operator_fingerprint,
    periodic_variable_dielectric_poisson_map,
    plane_wave_potential_operator_fingerprint,
    solve_canonical_split_zero_kane_poisson,
)


def _toy_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.arange(8, dtype=float)
    z_weights = np.ones_like(z)
    epsilon = np.full_like(z, 14.0)
    k_weights = np.asarray([0.01])
    return z, z_weights, epsilon, k_weights


def _provenance(*arrays: np.ndarray, tag: str = "toy") -> str:
    digest = hashlib.sha256(tag.encode())
    for array in arrays:
        digest.update(kane_poisson_array_sha256(np.asarray(array)).encode())
    return digest.hexdigest()


def test_plane_wave_parent_spec_separates_parent_dimension_from_poisson_grid() -> None:
    z = (np.arange(16, dtype=float) + 0.5) * 0.25
    spec = KaneParentSourceSpec(
        k_cart_nm_inv=np.zeros((1, 2)),
        k_weights_nm2=np.asarray([0.01]),
        z_nm=z,
        z_weights_nm=np.full(z.size, 0.25),
        epsilon_r=np.full(z.size, 14.0),
        parent_hilbert_dimension=8 * 7,
        selected_band_indices=(1, 2, -2, -1),
        candidate_eigenpair_count=8,
        target_energy_mev=0.0,
        window_selection_label="toy_plane_wave_rank_four",
        static_parent_sha256=_provenance(tag="pw_parent"),
        material_profile_sha256=_provenance(tag="pw_materials"),
        kdotpy_source_sha256=_provenance(tag="pw_source"),
        hamiltonian_options=(("N", "3"), ("split_mev", "0")),
        basis_flattening_label=PLANE_WAVE_FLATTENING_LABEL,
        potential_coupling_label=PLANE_WAVE_POTENTIAL_COUPLING_LABEL,
    )
    assert spec.parent_hilbert_dimension == 56
    assert spec.z_nm.size == 16
    potential = np.linspace(-1.0, 1.0, z.size)
    assert plane_wave_potential_operator_fingerprint(potential) != (
        kane_potential_operator_fingerprint(potential)
    )


def test_parent_spec_rejects_mixed_real_and_plane_wave_labels() -> None:
    z = np.arange(8, dtype=float)
    with pytest.raises(ValueError, match="supported parent representation"):
        KaneParentSourceSpec(
            k_cart_nm_inv=np.zeros((1, 2)),
            k_weights_nm2=np.asarray([0.01]),
            z_nm=z,
            z_weights_nm=np.ones_like(z),
            epsilon_r=np.full_like(z, 14.0),
            parent_hilbert_dimension=64,
            selected_band_indices=(1, -1),
            candidate_eigenpair_count=4,
            target_energy_mev=0.0,
            window_selection_label="invalid_mixed_representation",
            static_parent_sha256=_provenance(tag="mixed_parent"),
            material_profile_sha256=_provenance(tag="mixed_materials"),
            kdotpy_source_sha256=_provenance(tag="mixed_source"),
            hamiltonian_options=(("split_mev", "0"),),
            basis_flattening_label=PLANE_WAVE_FLATTENING_LABEL,
        )


def _toy_builder(
    *,
    separated: bool,
    split_mev: float = 0.0,
    potential_mode: str = "correct",
):
    z, z_weights, epsilon, k_weights = _toy_grid()
    parent_dimension = 8 * z.size
    electron_site = 1
    hole_site = 5 if separated else electron_site
    electron_index = 8 * electron_site
    hole_index = 8 * hole_site + 2
    static_parent = np.zeros((parent_dimension, parent_dimension), dtype=np.complex128)
    static_parent[electron_index, electron_index] = -0.10
    static_parent[hole_index, hole_index] = +0.10
    static_parent_sha256 = canonical_parent_matrix_sequence_sha256([static_parent])
    parent_spec = KaneParentSourceSpec(
        k_cart_nm_inv=np.zeros((1, 2)),
        k_weights_nm2=k_weights,
        z_nm=z,
        z_weights_nm=z_weights,
        epsilon_r=epsilon,
        parent_hilbert_dimension=parent_dimension,
        selected_band_indices=(1, -1),
        candidate_eigenpair_count=4,
        target_energy_mev=0.0,
        window_selection_label="toy_fixed_two_state_window",
        static_parent_sha256=static_parent_sha256,
        material_profile_sha256=_provenance(epsilon, tag="materials"),
        kdotpy_source_sha256=_provenance(tag="toy_kdotpy_source"),
        hamiltonian_options=(("axial", "true"), ("bia", "false"), ("split_mev", "0")),
    )
    calls: list[np.ndarray] = []
    frozen_full_parent: np.ndarray | None = None

    phi = np.zeros((1, z.size, 8, 2), dtype=np.complex128)
    phi[0, electron_site, 0, 0] = 1.0
    phi[0, hole_site, 2, 1] = 1.0

    def builder(potential_mev: np.ndarray) -> KaneWindowAtPotential:
        nonlocal frozen_full_parent
        potential = np.asarray(potential_mev, dtype=float)
        calls.append(potential.copy())
        potential_operator = np.diag(np.repeat(potential, 8)).astype(np.complex128)
        if potential_mode == "correct":
            full_parent = static_parent + potential_operator
        elif potential_mode == "ignored":
            full_parent = static_parent.copy()
        elif potential_mode == "wrong_sign":
            full_parent = static_parent - potential_operator
        elif potential_mode == "selected_only":
            full_parent = static_parent.copy()
            full_parent[electron_index, electron_index] += potential[electron_site]
            full_parent[hole_index, hole_index] += potential[hole_site]
        elif potential_mode == "frozen_first":
            if frozen_full_parent is None:
                frozen_full_parent = static_parent + potential_operator
            full_parent = frozen_full_parent.copy()
        else:
            raise ValueError("unsupported toy potential_mode")
        replayed_static = full_parent - potential_operator
        hamiltonian = np.zeros((2, 2, 1), dtype=np.complex128)
        hamiltonian[0, 0, 0] = full_parent[electron_index, electron_index]
        hamiltonian[1, 1, 0] = full_parent[hole_index, hole_index]
        return KaneWindowAtPotential(
            hamiltonian_mev=hamiltonian,
            micro_wavefunctions=phi.copy(),
            selected_band_indices=(1, -1),
            parent_hilbert_dimension=parent_dimension,
            calculation_split_mev=split_mev,
            energy_zero_label="toy_zero_mean_periodic_potential",
            input_potential_sha256=kane_poisson_array_sha256(potential),
            potential_operator_fingerprint=kane_potential_operator_fingerprint(potential),
            parent_spec_fingerprint=parent_spec.fingerprint,
            replayed_static_parent_sha256=canonical_parent_matrix_sequence_sha256(
                [replayed_static]
            ),
            static_parent_replay_max_error_mev=float(
                np.max(np.abs(replayed_static - static_parent))
            ),
            potential_operator_max_error_mev=float(
                np.max(np.abs(full_parent - static_parent - potential_operator))
            ),
            provenance_fingerprint=_provenance(
                potential, hamiltonian, phi, z_weights, tag="toy_source"
            ),
        )

    def verify_cold_start() -> bool:
        if calls:
            raise ValueError("toy builder is not cold")
        return True

    def fresh_factory():
        return _toy_builder(
            separated=separated,
            split_mev=split_mev,
            potential_mode=potential_mode,
        )[0]

    builder.parent_spec = parent_spec  # type: ignore[attr-defined]
    builder.calls = calls  # type: ignore[attr-defined]
    builder.verify_cold_start = verify_cold_start  # type: ignore[attr-defined]
    builder.fresh_factory = fresh_factory  # type: ignore[attr-defined]
    return builder, calls, parent_spec


def test_canonical_parent_hash_is_representation_invariant_and_does_not_mutate_csr() -> None:
    dense = np.asarray(
        [[1.0, 2.0e-13 + 0.5j], [2.0e-13 - 0.5j, -2.0]],
        dtype=np.complex128,
    )
    csr = sparse.csr_matrix(dense)
    indptr_before = csr.indptr.copy()
    indices_before = csr.indices.copy()
    data_before = csr.data.copy()
    sparse_hash = canonical_parent_matrix_sequence_sha256([csr])
    dense_hash = canonical_parent_matrix_sequence_sha256([dense])
    assert sparse_hash == dense_hash
    assert np.array_equal(csr.indptr, indptr_before)
    assert np.array_equal(csr.indices, indices_before)
    assert np.array_equal(csr.data, data_before)


def test_periodic_variable_dielectric_poisson_satisfies_gauss_and_zero_mean() -> None:
    z = np.arange(16, dtype=float)
    phase = 2.0 * np.pi * np.arange(z.size) / z.size
    modulation = 2.0e-6 * np.sin(phase)
    background = 3.0e-6
    electron = background - 0.5 * modulation
    hole = background + 0.5 * modulation
    result = periodic_variable_dielectric_poisson_map(
        z,
        electron,
        hole,
        np.full_like(z, 14.5),
    )
    assert abs(np.mean(result.electron_potential_mev)) < 1e-13
    assert abs(result.integrated_net_number_nm2) < 1e-14
    assert result.gauss_residual_mev_nm2 < 1e-12
    assert np.max(np.abs(result.electron_potential_mev)) > 0.0


def test_variable_dielectric_manufactured_flux_solution_has_correct_sign_and_energy() -> None:
    z = np.arange(24, dtype=float)
    dz = 1.0
    phase = 2.0 * np.pi * np.arange(z.size) / z.size
    epsilon = 13.5 + 1.7 * np.cos(phase + 0.31)
    expected_potential = 0.35 * np.cos(phase - 0.27)
    expected_potential -= np.mean(expected_potential)
    epsilon_face = 2.0 * epsilon * np.roll(epsilon, -1) / (
        epsilon + np.roll(epsilon, -1)
    )
    forward_flux = epsilon_face * (
        np.roll(expected_potential, -1) - expected_potential
    ) / dz
    manufactured_source = (forward_flux - np.roll(forward_flux, 1)) / dz
    net_number = manufactured_source / (4.0 * np.pi * 1439.96448)
    background = np.max(np.abs(net_number)) + 1e-7
    hole = background + 0.5 * net_number
    electron = background - 0.5 * net_number
    result = periodic_variable_dielectric_poisson_map(z, electron, hole, epsilon)
    assert np.max(np.abs(result.electron_potential_mev - expected_potential)) < 1e-11
    field_energy = float(
        np.sum(
            epsilon_face
            * ((np.roll(expected_potential, -1) - expected_potential) / dz) ** 2
        )
        * dz
    )
    source_energy = float(-np.dot(expected_potential, manufactured_source) * dz)
    assert field_energy > 0.0
    assert source_energy == pytest.approx(field_energy, rel=1e-12, abs=1e-12)


def test_local_profiles_reproduce_direct_microscopic_carriers() -> None:
    _z, z_weights, _epsilon, k_weights = _toy_grid()
    builder, _calls, _parent_spec = _toy_builder(separated=True)
    source = builder(np.zeros_like(z_weights))
    density = np.zeros((2, 2, 1), dtype=np.complex128)
    density[0, 0, 0] = 0.7
    density[1, 1, 0] = 0.3
    electron, hole = kane_orbital_transfer_density_profiles(
        source.micro_wavefunctions,
        density,
        k_weights,
        z_weights,
        expected_electron_density_nm2=0.007,
        expected_hole_density_nm2=0.007,
    )
    assert electron[1] == pytest.approx(0.007)
    assert hole[5] == pytest.approx(0.007)
    assert np.count_nonzero(electron) == 1
    assert np.count_nonzero(hole) == 1


def test_complex_offdiagonal_profiles_match_microscopic_trace_and_window_rotation() -> None:
    _z, z_weights, _epsilon, k_weights = _toy_grid()
    builder, _calls, _parent_spec = _toy_builder(separated=True)
    source = builder(np.zeros_like(z_weights))
    density_matrix = np.asarray([[0.68, 0.13 + 0.07j], [0.13 - 0.07j, 0.32]])
    density = density_matrix[:, :, None]
    electron, hole = kane_orbital_transfer_density_profiles(
        source.micro_wavefunctions, density, k_weights, z_weights
    )
    phi = source.micro_wavefunctions[0]
    direct_electron = k_weights[0] * np.einsum(
        "zma,ab,zmb->z", phi[:, :2, :], density_matrix, phi[:, :2, :].conj()
    ).real
    direct_hole = k_weights[0] * np.einsum(
        "zma,ab,zmb->z",
        phi[:, 2:, :],
        np.eye(2) - density_matrix,
        phi[:, 2:, :].conj(),
    ).real
    assert np.max(np.abs(electron - direct_electron)) < 1e-14
    assert np.max(np.abs(hole - direct_hole)) < 1e-14

    gauge = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    rotated_phi = np.einsum(
        "kzma,ab->kzmb", source.micro_wavefunctions, gauge, optimize=True
    )
    rotated_density = (gauge.conj().T @ density_matrix @ gauge)[:, :, None]
    rotated_electron, rotated_hole = kane_orbital_transfer_density_profiles(
        rotated_phi, rotated_density, k_weights, z_weights
    )
    assert np.max(np.abs(rotated_electron - electron)) < 1e-14
    assert np.max(np.abs(rotated_hole - hole)) < 1e-14


def test_zero_source_canonical_split_zero_poisson_converges_and_replays() -> None:
    z, _z_weights, _epsilon, _k_weights = _toy_grid()
    builder, calls, parent_spec = _toy_builder(separated=False)
    result = solve_canonical_split_zero_kane_poisson(
        builder,
        independent_replay_builder_factory=builder.fresh_factory,
        parent_spec=parent_spec,
        initial_potential_mev=np.zeros_like(z),
        config=CanonicalKanePoissonConfig(
            temperature_K=1.0,
            maximum_iterations=4,
            mixing=0.5,
            fixed_point_tolerance_mev=1e-10,
            density_profile_tolerance_nm3=1e-10,
        ),
    )
    assert result.converged and result.replay_verified
    assert result.iterations == 1
    assert len(calls) == result.iterations == 1
    assert np.max(np.abs(result.potential_mev)) < 1e-12
    assert result.reference.electron_density_nm2 == pytest.approx(
        result.reference.hole_density_nm2, abs=1e-12
    )
    assert result.fixed_point_residual_mev < 1e-10
    assert result.reference.contract.poisson_residual_mev == pytest.approx(
        result.fixed_point_residual_mev
    )
    assert "poisson_residual_is_missing" not in result.credibility_blockers
    assert "band_window_is_not_converged" in result.credibility_blockers
    assert "not_a_fixed_gate_device_reference" in result.credibility_blockers[0]
    assert result.claim_scope == "canonical_periodic_window_root_not_fixed_gate"
    fingerprint = canonical_kane_poisson_fingerprint(result)
    assert len(fingerprint) == 64
    for changes in (
        {"fixed_point_residual_mev": 1.0},
        {"gauss_residual_mev_nm2": 1.0},
        {"converged": False},
        {"replay_verified": False},
    ):
        with pytest.raises(ValueError):
            replace(result, **changes)
    with pytest.raises(ValueError):
        result.potential_mev[0] = 1.0
    with pytest.raises(ValueError):
        result.reference.density[0, 0, 0] = 0.0


def test_canonical_archive_round_trip_preserves_source_authority_and_representation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    z, _z_weights, _epsilon, _k_weights = _toy_grid()
    builder, _calls, parent_spec = _toy_builder(separated=False)
    result = solve_canonical_split_zero_kane_poisson(
        builder,
        independent_replay_builder_factory=builder.fresh_factory,
        parent_spec=parent_spec,
        initial_potential_mev=np.zeros_like(z),
        config=CanonicalKanePoissonConfig(
            temperature_K=1.0,
            maximum_iterations=4,
            mixing=0.5,
            fixed_point_tolerance_mev=1e-10,
            density_profile_tolerance_nm3=1e-10,
        ),
    )
    plane_wave_parent = replace(
        result.parent_spec,
        basis_flattening_label=PLANE_WAVE_FLATTENING_LABEL,
        potential_coupling_label=PLANE_WAVE_POTENTIAL_COUPLING_LABEL,
    )
    plane_wave_source = replace(
        result.final_eigensystem,
        parent_spec_fingerprint=plane_wave_parent.fingerprint,
        potential_operator_fingerprint=plane_wave_potential_operator_fingerprint(
            result.potential_mev
        ),
    )
    plane_wave_result = replace(
        result,
        parent_spec=plane_wave_parent,
        final_eigensystem=plane_wave_source,
    )
    npz_path = tmp_path / "canonical.npz"
    metadata_path = tmp_path / "canonical.json"
    receipt = save_canonical_kane_poisson_archive(
        plane_wave_result,
        npz_path,
        metadata_path,
    )
    loaded = load_canonical_kane_poisson_archive(npz_path, metadata_path)

    assert (
        canonical_kane_poisson_fingerprint(loaded.result)
        == receipt.result_fingerprint
    )
    assert (
        loaded.result.parent_spec.basis_flattening_label
        == PLANE_WAVE_FLATTENING_LABEL
    )
    assert (
        loaded.result.parent_spec.potential_coupling_label
        == PLANE_WAVE_POTENTIAL_COUPLING_LABEL
    )
    assert loaded.source_authority.hartree_fock_authorized is False
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_authority"]["hartree_fock_authorized"] is False
    assert metadata["source_authority"]["physical_fixed_gate_claim"] is False
    assert metadata["source_authority"]["credibility_blockers"] == list(
        plane_wave_result.credibility_blockers
    )

    original_metadata = metadata_path.read_text()
    metadata["source_authority"]["hartree_fock_authorized"] = True
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="cannot authorize a physical HF claim"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)

    metadata_path.write_text(original_metadata)
    metadata = json.loads(original_metadata)
    metadata["result"]["converged"] = "true"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(TypeError, match="exact booleans"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)

    metadata_path.write_text(original_metadata)
    metadata = json.loads(original_metadata)
    metadata["reference"]["root_converged"] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(TypeError, match="exact boolean"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)

    metadata_path.write_text(original_metadata)
    metadata = json.loads(original_metadata)
    del metadata["schema"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="archive schema is required"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)
    metadata_path.write_text(original_metadata)
    metadata = json.loads(original_metadata)
    metadata["unexpected"] = True
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata keys changed"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)

    metadata_path.write_text(original_metadata)
    original_npz = npz_path.read_bytes()
    npz_path.write_bytes(original_npz + b"tampered")
    with pytest.raises(ValueError, match="NPZ hash mismatch"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)
    npz_path.write_bytes(original_npz)

    metadata = json.loads(original_metadata)
    first_array = next(iter(metadata["array_manifest"]))
    metadata["array_manifest"][first_array]["unexpected"] = True
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manifest changed"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)

    metadata_path.write_text(original_metadata)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["unexpected"] = np.asarray([1.0])
    np.savez_compressed(npz_path, **arrays)
    metadata = json.loads(original_metadata)
    metadata["npz_sha256"] = file_sha256(npz_path)
    metadata["array_manifest"]["unexpected"] = {
        "dtype": "float64",
        "shape": [1],
        "sha256": kane_poisson_array_sha256(arrays["unexpected"]),
    }
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="archive array keys changed"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)

    npz_path.write_bytes(original_npz)
    metadata_path.write_text(original_metadata)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["electron_density_nm3"] = arrays["electron_density_nm3"] + 1.0e-4
    np.savez_compressed(npz_path, **arrays)
    metadata = json.loads(original_metadata)
    metadata["npz_sha256"] = file_sha256(npz_path)
    changed = arrays["electron_density_nm3"]
    metadata["array_manifest"]["electron_density_nm3"] = {
        "dtype": str(changed.dtype),
        "shape": list(changed.shape),
        "sha256": kane_poisson_array_sha256(changed),
    }
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="carrier profiles changed"):
        load_canonical_kane_poisson_archive(npz_path, metadata_path)

    npz_path.write_bytes(original_npz)
    metadata_path.write_text(original_metadata)
    with pytest.raises(ValueError, match="paths must be distinct"):
        save_canonical_kane_poisson_archive(
            plane_wave_result,
            tmp_path / "alias",
            tmp_path / "alias",
        )
    with pytest.raises(FileExistsError, match="must not already exist"):
        save_canonical_kane_poisson_archive(
            plane_wave_result,
            npz_path,
            metadata_path,
        )

    race_npz = tmp_path / "canonical-race.npz"
    race_metadata = tmp_path / "canonical-race.json"

    def competing_writer() -> object:
        try:
            return save_canonical_kane_poisson_archive(
                plane_wave_result,
                race_npz,
                race_metadata,
            )
        except Exception as error:  # returned for deterministic thread inspection
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        competing_results = tuple(pool.map(lambda _index: competing_writer(), range(2)))
    assert sum(not isinstance(item, Exception) for item in competing_results) == 1
    assert sum(isinstance(item, FileExistsError) for item in competing_results) == 1
    race_loaded = load_canonical_kane_poisson_archive(race_npz, race_metadata)
    assert (
        canonical_kane_poisson_fingerprint(race_loaded.result)
        == canonical_kane_poisson_fingerprint(plane_wave_result)
    )

    rebound_npz = tmp_path / "canonical-rebound.npz"
    rebound_metadata = tmp_path / "canonical-rebound.json"
    real_publish = archive_module.publish_staged_file_noreplace
    rebound_calls = 0

    def replace_path_after_publication(staged, destination):
        nonlocal rebound_calls
        rebound_calls += 1
        identity = real_publish(staged, destination)
        if rebound_calls == 2:
            destination.unlink()
            destination.write_bytes(b"post-publication-replacement")
        return identity

    monkeypatch.setattr(
        archive_module,
        "publish_staged_file_noreplace",
        replace_path_after_publication,
    )
    rebound_receipt = save_canonical_kane_poisson_archive(
        plane_wave_result,
        rebound_npz,
        rebound_metadata,
    )
    assert rebound_receipt.metadata_sha256 != file_sha256(rebound_metadata)

    partial_npz = tmp_path / "canonical-partial.npz"
    partial_metadata = tmp_path / "canonical-partial.json"
    monkeypatch.setattr(
        archive_module,
        "publish_staged_file_noreplace",
        real_publish,
    )
    publication_calls = 0

    def collide_on_second_member(staged, destination):
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls == 2:
            destination.write_bytes(b"competitor-owned-metadata")
        return real_publish(staged, destination)

    monkeypatch.setattr(
        archive_module,
        "publish_staged_file_noreplace",
        collide_on_second_member,
    )
    with pytest.raises(FileExistsError):
        save_canonical_kane_poisson_archive(
            plane_wave_result,
            partial_npz,
            partial_metadata,
        )
    assert partial_npz.is_file()
    assert partial_metadata.read_bytes() == b"competitor-owned-metadata"


def test_final_replay_accepts_exact_window_gauge_rotation_but_not_metadata_drift() -> None:
    z, _z_weights, _epsilon, _k_weights = _toy_grid()
    iteration_builder, iteration_calls, parent_spec = _toy_builder(separated=False)
    gauge = np.asarray([[1.0, 1.0j], [1.0j, 1.0]]) / np.sqrt(2.0)

    def transformed_factory(mode: str):
        factory_calls = 0
        created: list[object] = []

        def factory():
            nonlocal factory_calls
            raw, calls, fresh_parent = _toy_builder(separated=False)
            factory_calls += 1
            if factory_calls == 1:
                created.append(raw)
                return raw

            def transformed(potential: np.ndarray) -> KaneWindowAtPotential:
                source = raw(potential)
                if mode == "metadata":
                    return replace(source, energy_zero_label="drifted_energy_zero")
                wavefunctions = np.einsum(
                    "kzma,ab->kzmb", source.micro_wavefunctions, gauge, optimize=True
                )
                hamiltonian = source.hamiltonian_mev
                if mode == "gauge":
                    hamiltonian = np.empty_like(source.hamiltonian_mev)
                    for ik in range(hamiltonian.shape[2]):
                        hamiltonian[:, :, ik] = (
                            gauge.conj().T
                            @ source.hamiltonian_mev[:, :, ik]
                            @ gauge
                        )
                return replace(
                    source,
                    hamiltonian_mev=hamiltonian,
                    micro_wavefunctions=wavefunctions,
                    provenance_fingerprint=_provenance(
                        potential,
                        hamiltonian,
                        wavefunctions,
                        tag=f"{mode}_replay_source",
                    ),
                )

            transformed.parent_spec = fresh_parent  # type: ignore[attr-defined]
            transformed.calls = calls  # type: ignore[attr-defined]
            transformed.verify_cold_start = raw.verify_cold_start  # type: ignore[attr-defined]
            created.append(transformed)
            return transformed

        return factory, created

    gauge_factory, gauge_created = transformed_factory("gauge")
    result = solve_canonical_split_zero_kane_poisson(
        iteration_builder,
        independent_replay_builder_factory=gauge_factory,
        parent_spec=parent_spec,
        initial_potential_mev=np.zeros_like(z),
        config=CanonicalKanePoissonConfig(
            temperature_K=1.0,
            fixed_point_tolerance_mev=1e-10,
            density_profile_tolerance_nm3=1e-10,
        ),
    )
    assert result.replay_verified
    assert len(iteration_calls) == result.iterations == 1
    assert len(gauge_created) == 2

    bad_iteration, _bad_calls, bad_parent_spec = _toy_builder(separated=False)
    inconsistent_factory, _created = transformed_factory("inconsistent")
    with pytest.raises(ValueError, match="H/density covariance"):
        solve_canonical_split_zero_kane_poisson(
            bad_iteration,
            independent_replay_builder_factory=inconsistent_factory,
            parent_spec=bad_parent_spec,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(
                temperature_K=1.0,
                fixed_point_tolerance_mev=1e-10,
                density_profile_tolerance_nm3=1e-10,
            ),
        )

    metadata_iteration, _metadata_calls, metadata_parent = _toy_builder(
        separated=False
    )
    metadata_factory, _created = transformed_factory("metadata")
    with pytest.raises(ValueError, match="changed typed source metadata"):
        solve_canonical_split_zero_kane_poisson(
            metadata_iteration,
            independent_replay_builder_factory=metadata_factory,
            parent_spec=metadata_parent,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(
                temperature_K=1.0,
                fixed_point_tolerance_mev=1e-10,
                density_profile_tolerance_nm3=1e-10,
            ),
        )


def test_separated_profiles_reach_nontrivial_periodic_fixed_point() -> None:
    z, z_weights, _epsilon, _k_weights = _toy_grid()
    builder, calls, parent_spec = _toy_builder(separated=True)
    result = solve_canonical_split_zero_kane_poisson(
        builder,
        independent_replay_builder_factory=builder.fresh_factory,
        parent_spec=parent_spec,
        initial_potential_mev=np.zeros_like(z),
        config=CanonicalKanePoissonConfig(
            temperature_K=1.0,
            maximum_iterations=200,
            mixing=0.25,
            fixed_point_tolerance_mev=2e-7,
            density_profile_tolerance_nm3=2e-9,
        ),
    )
    assert result.iterations > 1
    assert len(calls) == result.iterations
    assert np.ptp(result.potential_mev) > 0.0
    assert result.fixed_point_residual_mev <= result.config.fixed_point_tolerance_mev
    assert result.gauss_residual_mev_nm2 <= result.config.gauss_residual_tolerance_mev_nm2
    assert np.dot(z_weights, result.electron_density_nm3) == pytest.approx(
        result.reference.electron_density_nm2, abs=1e-10
    )
    assert np.dot(z_weights, result.hole_density_nm3) == pytest.approx(
        result.reference.hole_density_nm2, abs=1e-10
    )
    assert abs(result.history[-1].integrated_net_number_nm2) < 1e-10


def test_canonical_kane_poisson_solves_one_common_mu_at_every_u_iterate() -> None:
    z, _z_weights, _epsilon, _k_weights = _toy_grid()
    builder, calls, parent_spec = _toy_builder(separated=True)
    result = solve_canonical_split_zero_kane_poisson(
        builder,
        independent_replay_builder_factory=builder.fresh_factory,
        parent_spec=parent_spec,
        initial_potential_mev=np.zeros_like(z),
        config=CanonicalKanePoissonConfig(
            temperature_K=1.0,
            maximum_iterations=200,
            mixing=0.25,
            fixed_point_tolerance_mev=2e-7,
            density_profile_tolerance_nm3=2e-9,
        ),
    )

    assert len(calls) == len(result.history) == result.iterations
    assert result.history[-1].mu_mev == pytest.approx(result.reference.mu_mev)
    tolerances = result.config.root_tolerances
    for record in result.history:
        scale = max(
            abs(record.electron_density_nm2),
            abs(record.hole_density_nm2),
        )
        allowed_residual = (
            tolerances.residual_atol_nm2 + tolerances.residual_rtol * scale
        )
        assert abs(
            record.electron_density_nm2 - record.hole_density_nm2
        ) <= allowed_residual
        assert np.isfinite(record.mu_mev)
        assert not hasattr(record, "mu_e_mev")
        assert not hasattr(record, "mu_h_mev")


def test_kane_poisson_rejects_nonzero_split_and_unbound_potential() -> None:
    z, _z_weights, _epsilon, _k_weights = _toy_grid()
    split_builder, _calls, split_parent = _toy_builder(separated=False, split_mev=0.01)
    with pytest.raises(ValueError, match="calculation_split_mev == 0"):
        solve_canonical_split_zero_kane_poisson(
            split_builder,
            independent_replay_builder_factory=split_builder.fresh_factory,
            parent_spec=split_parent,
            initial_potential_mev=np.zeros_like(z),
        )

    valid_builder, _calls, valid_parent = _toy_builder(separated=False)

    def stale_builder(potential: np.ndarray) -> KaneWindowAtPotential:
        source = valid_builder(potential)
        return replace(
            source,
            input_potential_sha256=kane_poisson_array_sha256(
                np.asarray(potential) + 1.0
            ),
        )

    with pytest.raises(ValueError, match="not bound"):
        solve_canonical_split_zero_kane_poisson(
            stale_builder,
            independent_replay_builder_factory=valid_builder.fresh_factory,
            parent_spec=valid_parent,
            initial_potential_mev=np.zeros_like(z),
        )

    def wrong_parent_builder(potential: np.ndarray) -> KaneWindowAtPotential:
        return replace(valid_builder(potential), parent_spec_fingerprint="0" * 64)

    with pytest.raises(ValueError, match="parent identity differs"):
        solve_canonical_split_zero_kane_poisson(
            wrong_parent_builder,
            independent_replay_builder_factory=valid_builder.fresh_factory,
            parent_spec=valid_parent,
            initial_potential_mev=np.zeros_like(z),
        )

    def wrong_operator_error_builder(potential: np.ndarray) -> KaneWindowAtPotential:
        return replace(valid_builder(potential), potential_operator_max_error_mev=1e-3)

    with pytest.raises(ValueError, match="canonical coupling"):
        solve_canonical_split_zero_kane_poisson(
            wrong_operator_error_builder,
            independent_replay_builder_factory=valid_builder.fresh_factory,
            parent_spec=valid_parent,
            initial_potential_mev=np.zeros_like(z),
        )

    def drifting_window_builder(potential: np.ndarray) -> KaneWindowAtPotential:
        return replace(valid_builder(potential), selected_band_indices=(2, -1))

    with pytest.raises(ValueError, match="window policy changed"):
        solve_canonical_split_zero_kane_poisson(
            drifting_window_builder,
            independent_replay_builder_factory=valid_builder.fresh_factory,
            parent_spec=valid_parent,
            initial_potential_mev=np.zeros_like(z),
        )

    nonzero_potential = np.linspace(-0.2, 0.2, z.size)
    nonzero_potential -= np.mean(nonzero_potential)
    for mode in ("ignored", "wrong_sign", "selected_only"):
        bad_builder, _bad_calls, bad_parent = _toy_builder(
            separated=False, potential_mode=mode
        )
        with pytest.raises(ValueError, match=r"actual H\(U\)-U replay"):
            solve_canonical_split_zero_kane_poisson(
                bad_builder,
                independent_replay_builder_factory=bad_builder.fresh_factory,
                parent_spec=bad_parent,
                initial_potential_mev=nonzero_potential,
            )

    frozen_builder, _frozen_calls, frozen_parent = _toy_builder(
        separated=True, potential_mode="frozen_first"
    )
    with pytest.raises(ValueError, match=r"actual H\(U\)-U replay"):
        solve_canonical_split_zero_kane_poisson(
            frozen_builder,
            independent_replay_builder_factory=frozen_builder.fresh_factory,
            parent_spec=frozen_parent,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(
                temperature_K=1.0,
                maximum_iterations=10,
                mixing=0.25,
            ),
        )


def test_canonical_replay_factory_requires_two_fresh_cold_matching_builders() -> None:
    z, _z_weights, _epsilon, _k_weights = _toy_grid()

    iteration, _calls, parent = _toy_builder(separated=False)
    with pytest.raises(ValueError, match="fresh builder"):
        solve_canonical_split_zero_kane_poisson(
            iteration,
            independent_replay_builder_factory=lambda: iteration,
            parent_spec=parent,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(temperature_K=1.0),
        )

    iteration, _calls, parent = _toy_builder(separated=False)
    reused, _reused_calls, _ = _toy_builder(separated=False)
    with pytest.raises(ValueError, match="fresh builder"):
        solve_canonical_split_zero_kane_poisson(
            iteration,
            independent_replay_builder_factory=lambda: reused,
            parent_spec=parent,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(temperature_K=1.0),
        )

    iteration, _calls, parent = _toy_builder(separated=False)
    warmed, _warmed_calls, _ = _toy_builder(separated=False)
    warmed(np.zeros_like(z))
    with pytest.raises(ValueError, match="not cold|warmed"):
        solve_canonical_split_zero_kane_poisson(
            iteration,
            independent_replay_builder_factory=lambda: warmed,
            parent_spec=parent,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(temperature_K=1.0),
        )

    iteration, _calls, parent = _toy_builder(separated=False)
    wrong, _wrong_calls, _wrong_parent = _toy_builder(separated=True)
    wrong.parent_spec = replace(  # type: ignore[attr-defined]
        wrong.parent_spec,
        material_profile_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="does not have the solver parent spec"):
        solve_canonical_split_zero_kane_poisson(
            iteration,
            independent_replay_builder_factory=lambda: wrong,
            parent_spec=parent,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(temperature_K=1.0),
        )


def test_nonconverged_fixed_point_fails_closed() -> None:
    z, _z_weights, _epsilon, _k_weights = _toy_grid()
    builder, _calls, parent_spec = _toy_builder(separated=True)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_canonical_split_zero_kane_poisson(
            builder,
            independent_replay_builder_factory=builder.fresh_factory,
            parent_spec=parent_spec,
            initial_potential_mev=np.zeros_like(z),
            config=CanonicalKanePoissonConfig(
                temperature_K=1.0,
                maximum_iterations=1,
                mixing=0.2,
                fixed_point_tolerance_mev=1e-12,
                density_profile_tolerance_nm3=1e-12,
            ),
        )
