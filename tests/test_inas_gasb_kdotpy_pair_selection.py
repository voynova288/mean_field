from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
from scipy import sparse

from mean_field.core.io import file_sha256
import mean_field.systems.inas_gasb.kdotpy_poisson_adapter as adapter
import mean_field.systems.inas_gasb.kdotpy_window_builder as builder_module
from mean_field.systems.inas_gasb.kane_poisson import KaneWindowAtPotential
from mean_field.systems.inas_gasb.kdotpy_poisson_adapter import (
    DEFAULT_MINIMUM_ASSIGNMENT_MARGIN,
    DECLARED_GAMMA_SEED_STATUS,
    KdotpyE1H1PairSelectionSpec,
    KdotpyPairResolvedWindow,
    KdotpyPreviousUSameKHomotopyError,
    KdotpyPreviousUSameKHomotopySpec,
    LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL,
    XML_RUNTIME_BINDING_STATUS,
    _compare_e1h1_endpoint_projectors,
    _select_e1h1_candidate_pairs_by_overlap,
)
from mean_field.systems.inas_gasb.kdotpy_window_builder import (
    KdotpyCanonicalWindowBuilder,
    KdotpyCanonicalWindowOnlyBuilder,
)


def _unitary2(theta: float, phase: float) -> np.ndarray:
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return np.asarray(
        [
            [cosine, -np.exp(1.0j * phase) * sine],
            [np.exp(-1.0j * phase) * sine, cosine],
        ],
        dtype=np.complex128,
    )


def _reference_pairs(dimension: int = 8) -> tuple[np.ndarray, np.ndarray]:
    basis = np.eye(dimension, dtype=np.complex128)
    return basis[:, :2], basis[:, 2:4]


def _pair_spec(
    *,
    minimum_principal_overlap: float = 0.99,
    minimum_assignment_margin: float = DEFAULT_MINIMUM_ASSIGNMENT_MARGIN,
    electron_pair_label: str = "E-like",
    hole_pair_label: str = "H-like",
    previous_u_same_k_homotopy_spec: KdotpyPreviousUSameKHomotopySpec | None = None,
) -> KdotpyE1H1PairSelectionSpec:
    return KdotpyE1H1PairSelectionSpec(
        gamma_electron_local_candidate_positions=(4, 5),
        gamma_hole_local_candidate_positions=(0, 1),
        electron_selected_labels=(1, 2),
        hole_selected_labels=(-2, -1),
        minimum_principal_overlap=minimum_principal_overlap,
        minimum_assignment_margin=minimum_assignment_margin,
        electron_pair_label=electron_pair_label,
        hole_pair_label=hole_pair_label,
        previous_u_same_k_homotopy_spec=previous_u_same_k_homotopy_spec,
    )


def _install_fake_kdotpy_builder(
    monkeypatch: pytest.MonkeyPatch,
    candidate_energies_mev: np.ndarray,
    *,
    rotate_pairs: bool,
    nk: int = 1,
) -> dict[str, object]:
    candidate_energies = np.asarray(candidate_energies_mev, dtype=np.float64)
    dimension = 32
    if candidate_energies.shape != (6,):
        raise ValueError("synthetic fixture requires six candidate energies")
    if nk < 1:
        raise ValueError("nk must be positive")
    static_energies = np.concatenate(
        (candidate_energies, 100.0 + np.arange(dimension - candidate_energies.size))
    )

    hamiltonian = SimpleNamespace()
    hamiltonian.hz_sparse = lambda *args, **kwargs: sparse.diags(
        static_energies, format="csr"
    )
    hamiltonian.hz_sparse_pot = lambda params, potential: sparse.diags(
        np.repeat(np.asarray(potential, dtype=np.float64), 8), format="csr"
    )
    fake_kdotpy = ModuleType("kdotpy")
    fake_kdotpy.hamiltonian = hamiltonian  # type: ignore[attr-defined]
    fake_vector = ModuleType("kdotpy.vector")
    fake_vector.Vector = lambda value, astype=None: (value, astype)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kdotpy", fake_kdotpy)
    monkeypatch.setitem(sys.modules, "kdotpy.vector", fake_vector)

    call_count = 0

    def synthetic_eigensystem(
        matrix: sparse.spmatrix,
        *,
        candidate_count: int,
        target_energy_mev: float,
        tolerance: float,
        maximum_iterations: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        nonlocal call_count
        del target_energy_mev, tolerance, maximum_iterations
        frame = np.eye(dimension, dtype=np.complex128)[:, :candidate_count]
        energies = np.asarray(
            sparse.csr_matrix(matrix).diagonal()[:candidate_count].real,
            dtype=np.float64,
        )
        if rotate_pairs:
            frame[:, :2] = frame[:, :2] @ _unitary2(
                0.11 * (call_count + 1), 0.23
            )
            frame[:, 4:6] = frame[:, 4:6] @ _unitary2(
                0.17 * (call_count + 1), -0.31
            )
        call_count += 1
        return energies, frame, 0.0, 0.0

    monkeypatch.setattr(
        builder_module, "_rayleigh_ritz_near_target", synthetic_eigensystem
    )
    z_nm = np.arange(4, dtype=np.float64)
    k_radius = 0.1 * np.arange(nk, dtype=np.float64)
    return {
        "params": SimpleNamespace(nz=4, norbitals=8),
        "k_cart_nm_inv": np.column_stack((k_radius, np.zeros_like(k_radius))),
        "k_weights_nm2": np.full(nk, 0.1, dtype=np.float64),
        "z_nm": z_nm,
        "z_weights_nm": np.ones_like(z_nm),
        "epsilon_r": np.full_like(z_nm, 14.0),
        "selected_band_indices": (1, 2, -2, -1),
        "candidate_eigenpair_count": 6,
        "target_energy_mev": 0.0,
        "material_profile_sha256": "a" * 64,
        "kdotpy_source_sha256": "b" * 64,
        "hamiltonian_options": (("axial", "true"),),
        "energy_zero_label": "synthetic_zero",
    }


def _install_avoided_crossing_homotopy_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    maximum_substeps: int,
    minimum_principal_overlap: float = 0.7,
    minimum_assignment_margin: float = 1.0e-3,
) -> KdotpyCanonicalWindowBuilder:
    """Smooth fixed-k model where direct/N2 alias E/H but N4/N5/N8 do not."""

    dimension = 32
    base_energies = np.concatenate(
        (
            np.asarray([0.0, 0.0, 0.0, 10.0, 11.0, 12.0]),
            100.0 + np.arange(dimension - 6, dtype=np.float64),
        )
    )

    def static_parent(k_vector: object, *args: object, **kwargs: object) -> sparse.spmatrix:
        del args, kwargs
        marker = int(round(10.0 * float(np.asarray(k_vector)[0])))
        diagonal = base_energies.copy()
        diagonal[6] += marker
        return sparse.diags(diagonal, format="csr")

    hamiltonian = SimpleNamespace()
    hamiltonian.hz_sparse = static_parent
    hamiltonian.hz_sparse_pot = lambda params, potential: sparse.diags(
        np.repeat(np.asarray(potential, dtype=np.float64), 8), format="csr"
    )
    fake_kdotpy = ModuleType("kdotpy")
    fake_kdotpy.hamiltonian = hamiltonian  # type: ignore[attr-defined]
    fake_vector = ModuleType("kdotpy.vector")
    fake_vector.Vector = lambda value, astype=None: (value, astype)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kdotpy", fake_kdotpy)
    monkeypatch.setitem(sys.modules, "kdotpy.vector", fake_vector)

    baseline_degrees = np.asarray([0.0, 30.0, 60.0, 80.0, 60.0])
    target_degrees = np.asarray([30.0, 60.0, 90.0, 120.0, 151.0])

    def synthetic_eigensystem(
        matrix: sparse.spmatrix,
        *,
        candidate_count: int,
        target_energy_mev: float,
        tolerance: float,
        maximum_iterations: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        del target_energy_mev, tolerance, maximum_iterations
        diagonal = np.asarray(
            sparse.csr_matrix(matrix).diagonal().real, dtype=np.float64
        )
        potential_value = float(diagonal[0])
        marker = int(round(diagonal[6] - base_energies[6] - potential_value))
        if marker == 0:
            angle_degrees = 0.0
        else:
            baseline = baseline_degrees[marker - 1]
            target = target_degrees[marker - 1]
            fraction = float(np.clip(potential_value, 0.0, 1.0))
            if marker == 5:
                fraction = fraction**1.6
            angle_degrees = baseline + (target - baseline) * fraction
        theta = np.deg2rad(angle_degrees)
        basis = np.eye(dimension, dtype=np.complex128)
        frame = basis[:, :candidate_count].copy()
        frame[:, 1] = np.cos(theta) * basis[:, 1] + np.sin(theta) * basis[:, 2]
        frame[:, 2] = -np.sin(theta) * basis[:, 1] + np.cos(theta) * basis[:, 2]
        return diagonal[:candidate_count], frame, 0.0, 0.0

    monkeypatch.setattr(
        builder_module, "_rayleigh_ritz_near_target", synthetic_eigensystem
    )
    z_nm = np.arange(4, dtype=np.float64)
    k_radius = 0.1 * (1.0 + np.arange(5, dtype=np.float64))
    homotopy_spec = KdotpyPreviousUSameKHomotopySpec(
        maximum_substeps=maximum_substeps
    )
    pair_spec = KdotpyE1H1PairSelectionSpec(
        gamma_electron_local_candidate_positions=(0, 1),
        gamma_hole_local_candidate_positions=(2, 3),
        electron_selected_labels=(1, 2),
        hole_selected_labels=(-2, -1),
        minimum_principal_overlap=minimum_principal_overlap,
        minimum_assignment_margin=minimum_assignment_margin,
        electron_pair_label="E-like",
        hole_pair_label="H-like",
        previous_u_same_k_homotopy_spec=homotopy_spec,
    )
    return KdotpyCanonicalWindowBuilder(
        params=SimpleNamespace(nz=4, norbitals=8),
        k_cart_nm_inv=np.column_stack((k_radius, np.zeros_like(k_radius))),
        k_weights_nm2=np.full(5, 0.1, dtype=np.float64),
        z_nm=z_nm,
        z_weights_nm=np.ones_like(z_nm),
        epsilon_r=np.full_like(z_nm, 14.0),
        selected_band_indices=(1, 2, -2, -1),
        candidate_eigenpair_count=6,
        target_energy_mev=0.0,
        material_profile_sha256="a" * 64,
        kdotpy_source_sha256="b" * 64,
        hamiltonian_options=(("axial", "true"),),
        energy_zero_label="synthetic_avoided_crossing_zero",
        pair_selection_spec=pair_spec,
    )


def test_kdotpy_window_builder_is_the_only_builder_owner() -> None:
    assert KdotpyCanonicalWindowBuilder.__module__ == (
        "mean_field.systems.inas_gasb.kdotpy_window_builder"
    )
    assert KdotpyCanonicalWindowOnlyBuilder.__module__ == (
        "mean_field.systems.inas_gasb.kdotpy_window_builder"
    )
    assert "KdotpyCanonicalWindowBuilder" not in adapter.__dict__
    assert "KdotpyCanonicalWindowOnlyBuilder" not in adapter.__dict__
    with pytest.raises(AttributeError):
        getattr(adapter, "KdotpyCanonicalWindowBuilder")


def test_pair_spec_validates_positions_labels_margins_and_length_delimits_labels() -> None:
    spec = _pair_spec(minimum_principal_overlap=0.8)
    spec.validate_candidate_count(6)
    spec.validate_ordered_selected_labels((1, 2, -2, -1))
    assert spec.minimum_assignment_margin == DEFAULT_MINIMUM_ASSIGNMENT_MARGIN
    assert spec.ordered_selected_labels == (1, 2, -2, -1)
    assert len(spec.fingerprint) == 64

    with pytest.raises(ValueError, match="rank-two"):
        KdotpyE1H1PairSelectionSpec(
            gamma_electron_local_candidate_positions=(4,),  # type: ignore[arg-type]
            gamma_hole_local_candidate_positions=(0, 1),
            electron_selected_labels=(1, 2),
            hole_selected_labels=(-2, -1),
            minimum_principal_overlap=0.8,
        )
    with pytest.raises(ValueError, match="disjoint"):
        KdotpyE1H1PairSelectionSpec(
            gamma_electron_local_candidate_positions=(1, 2),
            gamma_hole_local_candidate_positions=(2, 3),
            electron_selected_labels=(1, 2),
            hole_selected_labels=(-2, -1),
            minimum_principal_overlap=0.8,
        )
    with pytest.raises(ValueError, match="in range"):
        spec.validate_candidate_count(5)
    with pytest.raises(ValueError, match=r"\(0,1\]"):
        _pair_spec(minimum_principal_overlap=0.0)
    with pytest.raises(ValueError, match="minimum_assignment_margin"):
        _pair_spec(minimum_assignment_margin=np.nan)
    with pytest.raises(ValueError, match="minimum_assignment_margin"):
        _pair_spec(minimum_assignment_margin=0.0)
    with pytest.raises(ValueError, match="labels"):
        _pair_spec(electron_pair_label="pair", hole_pair_label="pair")
    with pytest.raises(ValueError, match="exact nonzero integer labels"):
        KdotpyE1H1PairSelectionSpec(
            gamma_electron_local_candidate_positions=(4, 5),
            gamma_hole_local_candidate_positions=(0, 1),
            electron_selected_labels=(1.0, 2),  # type: ignore[arg-type]
            hole_selected_labels=(-2, -1),
            minimum_principal_overlap=0.8,
        )
    with pytest.raises(ValueError, match="H-first"):
        spec.validate_ordered_selected_labels((-2, -1, 1, 2))
    with pytest.raises(ValueError, match="exactly match"):
        spec.validate_ordered_selected_labels((1, -2, 2, -1))

    # This pair collided under the old raw string concatenation ("ab" + "c"
    # versus "a" + "bc").  Length-delimited fields must distinguish it.
    split_one = _pair_spec(electron_pair_label="ab", hole_pair_label="c")
    split_two = _pair_spec(electron_pair_label="a", hole_pair_label="bc")
    assert split_one.fingerprint != split_two.fingerprint


def test_homotopy_spec_is_opt_in_frozen_validated_and_fingerprinted() -> None:
    disabled = _pair_spec()
    homotopy = KdotpyPreviousUSameKHomotopySpec(maximum_substeps=8)
    enabled = _pair_spec(previous_u_same_k_homotopy_spec=homotopy)
    assert disabled.previous_u_same_k_homotopy_spec is None
    assert disabled.fingerprint == (
        "d4c94dcf43daf401863560ec82e718958413da1c1263a1e205ff43fbd196fbd1"
    )
    assert len(homotopy.fingerprint) == 64
    assert len(enabled.fingerprint) == 64
    assert enabled.fingerprint != disabled.fingerprint
    assert (
        KdotpyPreviousUSameKHomotopySpec(maximum_substeps=16).fingerprint
        != homotopy.fingerprint
    )
    with pytest.raises(ValueError, match="power of two"):
        KdotpyPreviousUSameKHomotopySpec(maximum_substeps=6)
    with pytest.raises(ValueError, match="exact integer"):
        KdotpyPreviousUSameKHomotopySpec(maximum_substeps=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="homotopy_spec"):
        _pair_spec(previous_u_same_k_homotopy_spec=object())  # type: ignore[arg-type]
    with pytest.raises(Exception):
        homotopy.maximum_substeps = 16  # type: ignore[misc]


def test_endpoint_comparison_rejects_wrong_eh_split_even_for_identical_quartet() -> None:
    basis = np.eye(8, dtype=np.complex128)
    reference = basis[:, :4]
    transported = reference[:, [0, 2, 1, 3]]
    comparison = _compare_e1h1_endpoint_projectors(
        reference,
        transported,
        reference_electron_local_candidate_positions=(0, 1),
        reference_hole_local_candidate_positions=(2, 3),
        transported_electron_local_candidate_positions=(0, 2),
        transported_hole_local_candidate_positions=(1, 3),
    )
    assert np.allclose(comparison.quartet_principal_values, 1.0)
    assert np.min(comparison.electron_principal_values) == pytest.approx(0.0)
    assert np.min(comparison.hole_principal_values) == pytest.approx(0.0)
    assert not comparison.matches


def test_homotopy_certifies_n4_n5_n8_and_returns_current_u_radial_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _install_avoided_crossing_homotopy_builder(
        monkeypatch, maximum_substeps=8
    )
    first = builder(np.zeros(4))
    previous_frames = tuple(
        frame.copy() for frame in builder._previous_iteration_frames
    )
    second = builder(np.ones(4))
    assert isinstance(first, KdotpyPairResolvedWindow)
    assert isinstance(second, KdotpyPairResolvedWindow)
    receipt = second.diagnostics.previous_u_same_k_homotopy_receipt
    assert receipt is not None
    assert receipt.k_modes == ("direct", "direct", "direct", "direct", "homotopy")
    assert len(receipt.conflicts) == 1
    conflict = receipt.conflicts[0]
    assert conflict.k_index == 4
    assert conflict.attempted_dyadic_substeps == (2, 4)
    assert conflict.first_matching_substeps == 4
    assert conflict.confirmation_substeps == (5, 8)
    assert conflict.direct_electron_local_candidate_positions == (0, 2)
    assert conflict.direct_hole_local_candidate_positions == (1, 3)
    assert conflict.radial_electron_local_candidate_positions == (0, 1)
    assert conflict.radial_hole_local_candidate_positions == (2, 3)
    assert np.allclose(conflict.forward_endpoint_electron_principal_values, 1.0)
    assert np.allclose(conflict.forward_endpoint_hole_principal_values, 1.0)
    assert np.allclose(conflict.reverse_endpoint_electron_principal_values, 1.0)
    assert np.allclose(conflict.reverse_endpoint_hole_principal_values, 1.0)
    assert np.array_equal(
        second.diagnostics.previous_u_same_k_electron_selected_local_candidate_positions[
            4
        ],
        np.asarray([0, 2]),
    )
    assert np.array_equal(
        second.diagnostics.electron_selected_local_candidate_positions[4],
        np.asarray([0, 1]),
    )
    returned_frame = (
        second.window.micro_wavefunctions[4].reshape(32, 4)
        * np.sqrt(builder.parent_spec.z_weights_nm[0])
    )
    assert np.allclose(
        np.abs(returned_frame.conj().T @ builder._previous_iteration_frames[4]),
        np.eye(4),
        atol=1.0e-12,
    )
    assert not np.allclose(
        np.abs(previous_frames[4].conj().T @ builder._previous_iteration_frames[4]),
        np.eye(4),
    )
    assert builder._previous_successful_potential_mev is not None
    assert np.array_equal(builder._previous_successful_potential_mev, np.ones(4))
    assert not builder._previous_successful_potential_mev.flags.writeable
    with pytest.raises(ValueError):
        builder._previous_successful_potential_mev.setflags(write=True)
    options = dict(builder.parent_spec.hamiltonian_options)
    assert options[
        "pair_selection_previous_u_same_k_homotopy_spec_sha256"
    ] == builder._pair_selection_spec.previous_u_same_k_homotopy_spec.fingerprint
    assert options[
        "pair_selection_previous_u_same_k_homotopy_maximum_substeps"
    ] == "8"
    assert len(receipt.fingerprint) == 64
    assert len(conflict.transport_trace_sha256) == 64
    assert not conflict.forward_endpoint_electron_principal_values.flags.writeable
    with pytest.raises(ValueError):
        conflict.forward_endpoint_electron_principal_values.setflags(write=True)
    tampered_receipt = replace(receipt)
    object.__setattr__(tampered_receipt, "current_potential_sha256", "0" * 64)
    with pytest.raises(ValueError, match="stale|tampered"):
        replace(
            second.diagnostics,
            previous_u_same_k_homotopy_receipt=tampered_receipt,
        )


def test_homotopy_propagates_pair_thresholds_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlap = 0.703
    margin = 0.0017
    builder = _install_avoided_crossing_homotopy_builder(
        monkeypatch,
        maximum_substeps=8,
        minimum_principal_overlap=overlap,
        minimum_assignment_margin=margin,
    )
    real_selector = builder_module._select_e1h1_candidate_pairs_by_overlap
    observed: list[tuple[float, float]] = []

    def threshold_spy(*args: object, **kwargs: object) -> object:
        observed.append(
            (
                float(kwargs["minimum_principal_overlap"]),
                float(kwargs["minimum_assignment_margin"]),
            )
        )
        return real_selector(*args, **kwargs)

    monkeypatch.setattr(
        builder_module, "_select_e1h1_candidate_pairs_by_overlap", threshold_spy
    )
    builder(np.zeros(4))
    builder(np.ones(4))
    assert len(observed) > 20
    assert all(value == (overlap, margin) for value in observed)


def test_homotopy_maximum_failure_rolls_back_frames_potential_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _install_avoided_crossing_homotopy_builder(
        monkeypatch, maximum_substeps=4
    )
    builder(np.zeros(4))
    prior_anchor = builder._previous_anchor_frame.copy()
    prior_frames = tuple(frame.copy() for frame in builder._previous_iteration_frames)
    prior_potential = builder._previous_successful_potential_mev.copy()
    prior_diagnostics = builder.last_diagnostics
    prior_history = tuple(builder.diagnostics_history)
    with pytest.raises(KdotpyPreviousUSameKHomotopyError) as caught:
        builder(np.ones(4))
    assert caught.value.receipt.reason_code == "maximum_substeps_exhausted"
    assert caught.value.receipt.attempted_dyadic_substeps == (2,)
    assert np.array_equal(builder._previous_anchor_frame, prior_anchor)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(builder._previous_iteration_frames, prior_frames)
    )
    assert np.array_equal(builder._previous_successful_potential_mev, prior_potential)
    assert builder.last_diagnostics is prior_diagnostics
    assert tuple(builder.diagnostics_history) == prior_history




def test_pair_builder_rejects_legacy_h_first_selected_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _install_fake_kdotpy_builder(
        monkeypatch,
        np.asarray([-3.0, -3.0, -1.0, 1.0, 2.0, 2.0]),
        rotate_pairs=False,
    )
    kwargs["selected_band_indices"] = (-2, -1, 1, 2)
    with pytest.raises(ValueError, match="H-first"):
        KdotpyCanonicalWindowBuilder(**kwargs, pair_selection_spec=_pair_spec())


def test_builder_requires_pair_spec_and_rejects_retired_anchor_only_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _install_fake_kdotpy_builder(
        monkeypatch,
        np.asarray([-3.0, -3.0, -1.0, 1.0, 2.0, 2.0]),
        rotate_pairs=False,
    )
    with pytest.raises(TypeError, match="pair_selection_spec"):
        KdotpyCanonicalWindowBuilder(**kwargs)
    with pytest.raises(TypeError, match="anchor_selected_candidate_indices"):
        KdotpyCanonicalWindowBuilder(
            **kwargs,
            pair_selection_spec=_pair_spec(),
            anchor_selected_candidate_indices=(4, 5, 0, 1),
        )


def test_pure_pair_selector_is_covariant_to_candidate_permutation_and_pair_u2() -> None:
    electron, hole = _reference_pairs()
    basis = np.eye(8, dtype=np.complex128)
    electron_rotated = electron @ _unitary2(0.73, -0.41)
    hole_rotated = hole @ _unitary2(-0.52, 0.29)
    candidates = np.column_stack(
        (
            hole_rotated[:, 0],
            basis[:, 4],
            electron_rotated[:, 1],
            hole_rotated[:, 1],
            electron_rotated[:, 0],
            basis[:, 5],
        )
    )

    selection = _select_e1h1_candidate_pairs_by_overlap(
        electron,
        hole,
        candidates,
        minimum_principal_overlap=0.99,
        minimum_assignment_margin=0.1,
    )
    assert selection.electron_local_candidate_positions == (2, 4)
    assert selection.hole_local_candidate_positions == (0, 3)
    assert np.allclose(selection.electron_principal_singular_values, 1.0)
    assert np.allclose(selection.hole_principal_singular_values, 1.0)
    assert selection.electron_assignment_margin > 0.1
    assert selection.hole_assignment_margin > 0.1


def test_pair_selector_rejects_collision_exact_tie_low_overlap_and_finite_near_tie() -> None:
    electron, hole = _reference_pairs()

    rng = np.random.default_rng(2)
    raw = rng.normal(size=(8, 8)) + 1.0j * rng.normal(size=(8, 8))
    collision_candidates, _ = np.linalg.qr(raw)
    with pytest.raises(ValueError, match="collide"):
        _select_e1h1_candidate_pairs_by_overlap(
            electron,
            hole,
            collision_candidates,
            minimum_principal_overlap=0.01,
            minimum_assignment_margin=1.0e-12,
        )

    mixed = np.column_stack(
        (
            (electron[:, 0] + hole[:, 0]) / np.sqrt(2.0),
            (electron[:, 1] + hole[:, 1]) / np.sqrt(2.0),
            (electron[:, 0] - hole[:, 0]) / np.sqrt(2.0),
            (electron[:, 1] - hole[:, 1]) / np.sqrt(2.0),
        )
    )
    with pytest.raises(ValueError, match="near-tie"):
        _select_e1h1_candidate_pairs_by_overlap(
            electron,
            hole,
            mixed,
            minimum_principal_overlap=0.1,
        )

    basis = np.eye(8, dtype=np.complex128)
    low_overlap_candidates = np.column_stack(
        (
            0.8 * basis[:, 0] + 0.6 * basis[:, 4],
            0.7 * basis[:, 1] + np.sqrt(0.51) * basis[:, 5],
            -0.6 * basis[:, 0] + 0.8 * basis[:, 4],
            -np.sqrt(0.51) * basis[:, 1] + 0.7 * basis[:, 5],
            basis[:, 2],
            basis[:, 3],
        )
    )
    with pytest.raises(ValueError, match="below threshold"):
        _select_e1h1_candidate_pairs_by_overlap(
            electron,
            hole,
            low_overlap_candidates,
            minimum_principal_overlap=0.9,
        )

    theta = np.pi / 4.0 - 2.5e-4
    finite_near_tie_candidates = np.column_stack(
        (
            np.cos(theta) * basis[:, 0] + np.sin(theta) * basis[:, 4],
            basis[:, 1],
            -np.sin(theta) * basis[:, 0] + np.cos(theta) * basis[:, 4],
            basis[:, 2],
            basis[:, 3],
            basis[:, 5],
        )
    )
    with pytest.raises(ValueError, match="near-tie"):
        _select_e1h1_candidate_pairs_by_overlap(
            electron,
            hole,
            finite_near_tie_candidates,
            minimum_principal_overlap=0.5,
            minimum_assignment_margin=1.0e-3,
        )
    accepted = _select_e1h1_candidate_pairs_by_overlap(
        electron,
        hole,
        finite_near_tie_candidates,
        minimum_principal_overlap=0.5,
        minimum_assignment_margin=1.0e-5,
    )
    assert 1.0e-5 < accepted.electron_assignment_margin < 1.0e-3


def test_pair_mode_returns_typed_same_call_receipt_and_tracks_both_continuity_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _install_fake_kdotpy_builder(
        monkeypatch,
        np.asarray([-3.0, -3.0, -1.0, 1.0, 2.0, 2.0]),
        rotate_pairs=True,
        nk=3,
    )
    pair_spec = _pair_spec(minimum_assignment_margin=0.05)
    builder = KdotpyCanonicalWindowBuilder(**kwargs, pair_selection_spec=pair_spec)

    first = builder(np.zeros(4))
    second = builder(np.full(4, 0.2))
    assert isinstance(first, KdotpyPairResolvedWindow)
    assert isinstance(second, KdotpyPairResolvedWindow)
    assert first.same_call_fingerprint != second.same_call_fingerprint
    assert first.diagnostics.previous_u_same_k_electron_principal_singular_values is None

    diagnostics = second.diagnostics
    expected_positions = np.asarray(
        [[4, 5, 0, 1], [4, 5, 0, 1], [4, 5, 0, 1]], dtype=np.int64
    )
    assert np.array_equal(
        diagnostics.anchor_selected_local_candidate_positions,
        np.asarray([4, 5, 0, 1]),
    )
    assert np.array_equal(
        diagnostics.selected_local_candidate_positions, expected_positions
    )
    assert np.array_equal(
        diagnostics.anchor_electron_selected_local_candidate_positions,
        np.asarray([4, 5]),
    )
    assert np.array_equal(
        diagnostics.anchor_hole_selected_local_candidate_positions,
        np.asarray([0, 1]),
    )
    assert np.array_equal(
        diagnostics.electron_selected_local_candidate_positions,
        expected_positions[:, :2],
    )
    assert np.array_equal(
        diagnostics.hole_selected_local_candidate_positions,
        expected_positions[:, 2:],
    )
    assert np.allclose(
        diagnostics.current_u_adjacent_k_electron_principal_singular_values, 1.0
    )
    assert np.allclose(
        diagnostics.current_u_adjacent_k_hole_principal_singular_values, 1.0
    )
    assert np.allclose(
        diagnostics.previous_u_same_k_electron_principal_singular_values, 1.0
    )
    assert np.allclose(
        diagnostics.previous_u_same_k_hole_principal_singular_values, 1.0
    )
    assert np.all(
        diagnostics.current_u_adjacent_k_electron_assignment_margin >= 0.05
    )
    assert np.all(
        diagnostics.previous_u_same_k_electron_assignment_margin >= 0.05
    )
    assert diagnostics.pair_labels == ("E-like", "H-like")
    assert diagnostics.ordered_physical_selected_labels == (1, 2, -2, -1)
    assert diagnostics.gamma_seed_status == DECLARED_GAMMA_SEED_STATUS
    assert "not-independent-identity-evidence" in diagnostics.gamma_seed_status
    assert diagnostics.minimum_assignment_margin == 0.05
    assert diagnostics.pair_selection_spec_fingerprint == pair_spec.fingerprint
    assert diagnostics.pair_selection_diagnostics_sha256 is not None
    assert len(diagnostics.pair_selection_diagnostics_sha256) == 64
    with pytest.raises(AttributeError):
        _ = diagnostics.selected_candidate_indices
    below_threshold = np.asarray(
        diagnostics.anchor_electron_principal_singular_values
    ).copy()
    below_threshold[0] = diagnostics.minimum_principal_overlap - 1.0e-3
    with pytest.raises(ValueError, match="minimum_principal_overlap"):
        replace(
            diagnostics,
            anchor_electron_principal_singular_values=below_threshold,
            pair_selection_diagnostics_sha256=None,
        )

    assert second.window.selected_band_indices == (1, 2, -2, -1)
    assert np.array_equal(
        np.diag(second.window.hamiltonian_mev[:, :, 0]),
        np.asarray([2.2, 2.2, -2.8, -2.8]),
    )
    assert second.parent_spec_fingerprint == builder.parent_spec.fingerprint
    assert second.material_profile_sha256 == builder.parent_spec.material_profile_sha256
    assert second.xml_runtime_binding_status == XML_RUNTIME_BINDING_STATUS
    assert (
        second.local_candidate_positions_namespace_label
        == LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL
    )
    assert len(second.candidate_energy_table_sha256) == 64
    assert len(second.local_candidate_positions_namespace_fingerprint) == 64

    options = dict(builder.parent_spec.hamiltonian_options)
    assert options["pair_selection_spec_sha256"] == pair_spec.fingerprint
    assert options["selection_anchor_local_candidate_positions"] == "4,5,0,1"
    assert options["pair_selection_minimum_assignment_margin"] == "0.05"
    assert options["pair_selection_gamma_seed_status"] == DECLARED_GAMMA_SEED_STATUS
    assert not any("homotopy" in key for key in options)
    assert (
        options["pair_selection_local_candidate_positions_namespace"]
        == LOCAL_CANDIDATE_POSITIONS_NAMESPACE_LABEL
    )

    for field in fields(diagnostics):
        value = getattr(diagnostics, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable, field.name
    for value in (
        second.window.hamiltonian_mev,
        second.window.micro_wavefunctions,
        second.input_potential_mev,
        second.parent_spec.k_cart_nm_inv,
        second.parent_spec.k_weights_nm2,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        diagnostics.selected_local_candidate_positions[0, 0] = 0
    with pytest.raises(ValueError):
        diagnostics.selected_local_candidate_positions.setflags(write=True)


def test_canonical_window_only_adapter_retains_pair_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _install_fake_kdotpy_builder(
        monkeypatch,
        np.asarray([-3.0, -3.0, -1.0, 1.0, 2.0, 2.0]),
        rotate_pairs=True,
        nk=2,
    )
    raw = KdotpyCanonicalWindowBuilder(
        **kwargs, pair_selection_spec=_pair_spec(minimum_assignment_margin=0.05)
    )
    wrapped = KdotpyCanonicalWindowOnlyBuilder(raw)
    assert wrapped.verify_cold_start()
    window = wrapped(np.zeros(4))
    assert isinstance(window, KaneWindowAtPotential)
    assert window is wrapped.pair_resolved_history[0].window
    assert len(wrapped.pair_resolved_history) == 1
    assert len(wrapped.diagnostics_history) == 1
    assert wrapped.parent_spec.fingerprint == raw.parent_spec.fingerprint
    with pytest.raises(ValueError, match="not cold|retained pair"):
        wrapped.verify_cold_start()






def test_pair_receipt_rejects_stale_or_tampered_cross_call_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _install_fake_kdotpy_builder(
        monkeypatch,
        np.asarray([-3.0, -3.0, -1.0, 1.0, 2.0, 2.0]),
        rotate_pairs=True,
        nk=2,
    )
    builder = KdotpyCanonicalWindowBuilder(**kwargs, pair_selection_spec=_pair_spec())
    first = builder(np.zeros(4))
    second = builder(np.full(4, 0.3))
    assert isinstance(first, KdotpyPairResolvedWindow)
    assert isinstance(second, KdotpyPairResolvedWindow)

    with pytest.raises(ValueError, match="stale|same call"):
        replace(second, diagnostics=first.diagnostics)
    with pytest.raises(ValueError, match="stale|tampered"):
        replace(second, candidate_energy_table_sha256="0" * 64)

    changed_positions = second.diagnostics.selected_local_candidate_positions.copy()
    changed_positions[0] = np.asarray([5, 4, 0, 1])
    with pytest.raises(ValueError, match="ordered columns|stale|tampered"):
        replace(
            second.diagnostics,
            selected_local_candidate_positions=changed_positions,
        )

    forcibly_tampered = replace(second.diagnostics)
    object.__setattr__(forcibly_tampered, "pair_labels", ("tampered", "H-like"))
    with pytest.raises(ValueError, match="stale or tampered"):
        replace(second, diagnostics=forcibly_tampered)


def test_pair_builder_rolls_back_tracking_state_after_late_return_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _install_fake_kdotpy_builder(
        monkeypatch,
        np.asarray([-3.0, -3.0, -1.0, 1.0, 2.0, 2.0]),
        rotate_pairs=True,
        nk=2,
    )
    builder = KdotpyCanonicalWindowBuilder(
        **kwargs,
        pair_selection_spec=_pair_spec(
            previous_u_same_k_homotopy_spec=(
                KdotpyPreviousUSameKHomotopySpec(maximum_substeps=8)
            )
        ),
    )
    real_pair_type = builder_module.KdotpyPairResolvedWindow

    def fail_late_pair_construction(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("injected late pair-return construction failure")

    monkeypatch.setattr(
        builder_module, "KdotpyPairResolvedWindow", fail_late_pair_construction
    )
    with pytest.raises(RuntimeError, match="late pair-return"):
        builder(np.zeros(4))
    assert builder._previous_anchor_frame is None
    assert builder._previous_iteration_frames is None
    assert builder._previous_successful_potential_mev is None
    assert builder.last_diagnostics is None
    assert builder.diagnostics_history == []

    monkeypatch.setattr(builder_module, "KdotpyPairResolvedWindow", real_pair_type)
    recovered = builder(np.zeros(4))
    assert isinstance(recovered, KdotpyPairResolvedWindow)
    assert len(builder.diagnostics_history) == 1
    assert np.array_equal(builder._previous_successful_potential_mev, np.zeros(4))
    assert not builder._previous_successful_potential_mev.flags.writeable

    prior_anchor = builder._previous_anchor_frame.copy()
    prior_frames = tuple(frame.copy() for frame in builder._previous_iteration_frames)
    prior_potential = builder._previous_successful_potential_mev.copy()
    prior_diagnostics = builder.last_diagnostics
    prior_history = tuple(builder.diagnostics_history)
    monkeypatch.setattr(
        builder_module, "KdotpyPairResolvedWindow", fail_late_pair_construction
    )
    with pytest.raises(RuntimeError, match="late pair-return"):
        builder(np.full(4, 0.1))
    assert np.array_equal(builder._previous_anchor_frame, prior_anchor)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(builder._previous_iteration_frames, prior_frames)
    )
    assert np.array_equal(
        builder._previous_successful_potential_mev, prior_potential
    )
    assert builder.last_diagnostics is prior_diagnostics
    assert tuple(builder.diagnostics_history) == prior_history
