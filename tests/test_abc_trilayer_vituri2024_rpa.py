"""Focused tests for authority-limited local Vituri C9 Hessian elements."""

from dataclasses import FrozenInstanceError, replace
import inspect

import numpy as np
import pytest

import mean_field.systems.abc_trilayer as abc_trilayer_module
import mean_field.systems.abc_trilayer.vituri2024_rpa as rpa_module
from mean_field.systems.abc_trilayer import (
    DIAGONAL_HF_SOURCE_PROVIDER_STATUS,
    FINITE_AREA_PROVIDER_STATUS,
    RPA_AREA_UNITS,
    RPA_A_ELEMENT_EQUATION,
    RPA_B_ELEMENT_EQUATION,
    RPA_ELEMENT_AUTHORITY,
    RPA_ELEMENT_NO_GO_LIMITS,
    RPA_ELEMENT_UNITS,
    RPA_VERTEX_UNITS,
    SCALAR_HESSIAN_EQUATION,
    SM_TEX_SHA256,
    Vituri2024DiagonalHFTransitionReceipt,
    Vituri2024FiniteAreaReceipt,
    Vituri2024Flavor,
    Vituri2024FourPointKinematicsReceipt,
    Vituri2024InteractionChoiceReceipt,
    Vituri2024Orbital,
    vituri2024_antisymmetrized_projected_vertex,
    vituri2024_rpa_a_element,
    vituri2024_rpa_b_element,
)

_DELTA1 = 0.028
_DIAGONAL_SOURCE_SHA = "a" * 64
_DIAGONAL_SOURCE_TEXT = (
    "Caller-attested immutable diagonal source for tiny local C9 tests; "
    "stationarity not independently certified."
)
_KINEMATICS_PROVIDER_SHA = "b" * 64
_KINEMATICS_SOURCE_TEXT = (
    "Caller-attested exact local C9 quartet; no torus or carry authority."
)
_AREA_PROVIDER_SHA = "c" * 64


def _orbital(
    momentum: tuple[float, float],
    *,
    valley: int = 1,
    spin: int = 1,
) -> Vituri2024Orbital:
    return Vituri2024Orbital(
        flavor=Vituri2024Flavor(valley=valley, spin=spin),
        momentum_inverse_angstrom=momentum,
    )


def _transition(
    particle: Vituri2024Orbital,
    hole: Vituri2024Orbital,
    *,
    particle_energy_ev: float,
    hole_energy_ev: float,
    source_artifact_sha256: str = _DIAGONAL_SOURCE_SHA,
    source_text: str = _DIAGONAL_SOURCE_TEXT,
) -> Vituri2024DiagonalHFTransitionReceipt:
    return Vituri2024DiagonalHFTransitionReceipt(
        particle=particle,
        hole=hole,
        particle_energy_ev=particle_energy_ev,
        hole_energy_ev=hole_energy_ev,
        source_artifact_sha256=source_artifact_sha256,
        source_text=source_text,
    )


def _area() -> Vituri2024FiniteAreaReceipt:
    return Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=53.0,
        provider_sha256=_AREA_PROVIDER_SHA,
        source_text=(
            "Caller-attested finite test area only; no mesh, quadrature, "
            "torus, background, paper, or production authority."
        ),
    )


def _interaction() -> Vituri2024InteractionChoiceReceipt:
    return Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=250.0,
        coulomb_e2_ev_angstrom=14.3996454784255,
        q0_evaluation="analytic_kernel_limit_only",
        provider_sha256="d" * 64,
        source_sha256=SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text="Tiny test interaction; not an HF q=0 background.",
    )


def _a_transitions() -> tuple[
    Vituri2024DiagonalHFTransitionReceipt,
    Vituri2024DiagonalHFTransitionReceipt,
]:
    # q_left=q_right=(1,2)/128 exactly in binary float64.
    hole_left = _orbital((0.0, 0.0))
    particle_left = _orbital((1.0 / 128.0, 2.0 / 128.0))
    hole_right = _orbital((-3.0 / 128.0, 1.0 / 128.0))
    particle_right = _orbital((-2.0 / 128.0, 3.0 / 128.0))
    return (
        _transition(
            particle_left,
            hole_left,
            particle_energy_ev=0.43,
            hole_energy_ev=-0.17,
        ),
        _transition(
            particle_right,
            hole_right,
            particle_energy_ev=0.57,
            hole_energy_ev=-0.29,
        ),
    )


def _b_transitions() -> tuple[
    Vituri2024DiagonalHFTransitionReceipt,
    Vituri2024DiagonalHFTransitionReceipt,
]:
    # q_left=(1,2)/128 and q_right=-(1,2)/128 exactly.
    hole_left = _orbital((0.0, 0.0))
    particle_left = _orbital((1.0 / 128.0, 2.0 / 128.0))
    hole_right = _orbital((3.0 / 128.0, -1.0 / 128.0))
    particle_right = _orbital((2.0 / 128.0, -3.0 / 128.0))
    return (
        _transition(
            particle_left,
            hole_left,
            particle_energy_ev=0.43,
            hole_energy_ev=-0.17,
        ),
        _transition(
            particle_right,
            hole_right,
            particle_energy_ev=0.61,
            hole_energy_ev=-0.31,
        ),
    )


def _call_a(
    left: Vituri2024DiagonalHFTransitionReceipt,
    right: Vituri2024DiagonalHFTransitionReceipt,
):
    return vituri2024_rpa_a_element(
        left,
        right,
        _area(),
        _DELTA1,
        _interaction(),
        kinematics_provider_sha256=_KINEMATICS_PROVIDER_SHA,
        kinematics_source_text=_KINEMATICS_SOURCE_TEXT,
    )


def _call_b(
    left: Vituri2024DiagonalHFTransitionReceipt,
    right: Vituri2024DiagonalHFTransitionReceipt,
):
    return vituri2024_rpa_b_element(
        left,
        right,
        _area(),
        _DELTA1,
        _interaction(),
        kinematics_provider_sha256=_KINEMATICS_PROVIDER_SHA,
        kinematics_source_text=_KINEMATICS_SOURCE_TEXT,
    )


def _manual_kinematics(
    alpha: Vituri2024Orbital,
    beta: Vituri2024Orbital,
    gamma: Vituri2024Orbital,
    delta: Vituri2024Orbital,
) -> Vituri2024FourPointKinematicsReceipt:
    return Vituri2024FourPointKinematicsReceipt(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        momentum_tolerance_inverse_angstrom=0.0,
        provider_sha256=_KINEMATICS_PROVIDER_SHA,
        derivation_source_sm_sha256=SM_TEX_SHA256,
        source_text=_KINEMATICS_SOURCE_TEXT,
    )


def test001_a_and_b_match_hand_c9_expressions_without_extra_factor() -> None:
    area = _area()
    interaction = _interaction()
    left_a, right_a = _a_transitions()
    result_a = vituri2024_rpa_a_element(
        left_a,
        right_a,
        area,
        _DELTA1,
        interaction,
        kinematics_provider_sha256=_KINEMATICS_PROVIDER_SHA,
        kinematics_source_text=_KINEMATICS_SOURCE_TEXT,
    )
    vertex_a = vituri2024_antisymmetrized_projected_vertex(
        _manual_kinematics(
            left_a.particle,
            right_a.hole,
            left_a.hole,
            right_a.particle,
        ),
        _DELTA1,
        interaction,
    )
    expected_a = -vertex_a.value / area.area_angstrom_squared

    left_b, right_b = _b_transitions()
    result_b = vituri2024_rpa_b_element(
        left_b,
        right_b,
        area,
        _DELTA1,
        interaction,
        kinematics_provider_sha256=_KINEMATICS_PROVIDER_SHA,
        kinematics_source_text=_KINEMATICS_SOURCE_TEXT,
    )
    vertex_b = vituri2024_antisymmetrized_projected_vertex(
        _manual_kinematics(
            left_b.particle,
            right_b.particle,
            left_b.hole,
            right_b.hole,
        ),
        _DELTA1,
        interaction,
    )
    expected_b = -vertex_b.value / area.area_angstrom_squared

    assert abs(vertex_a.value) > 1.0e-8
    assert abs(vertex_b.value) > 1.0e-8
    assert result_a.value == pytest.approx(expected_a, rel=2.0e-14, abs=2.0e-15)
    assert result_b.value == pytest.approx(expected_b, rel=2.0e-14, abs=2.0e-15)
    assert result_a.one_body_contribution_ev == 0.0
    assert result_b.one_body_contribution_ev == 0.0
    assert result_a.interaction_contribution_ev == pytest.approx(expected_a)
    assert result_b.interaction_contribution_ev == pytest.approx(expected_b)
    for result, expected in ((result_a, expected_a), (result_b, expected_b)):
        assert result.value != pytest.approx(expected / 2.0, rel=1.0e-6, abs=1.0e-12)
        assert result.value != pytest.approx(expected / 4.0, rel=1.0e-6, abs=1.0e-12)
        assert result.vertex_divided_by_area_exactly_once is True
        assert result.extra_half_factor_applied is False
        assert result.extra_quarter_factor_applied is False


def test001_a_adds_gap_only_for_the_exact_same_particle_hole_pair() -> None:
    left, right = _a_transitions()
    off_diagonal = _call_a(left, right)
    diagonal = _call_a(left, left)

    assert off_diagonal.one_body_contribution_ev == 0.0
    assert diagonal.one_body_contribution_ev == left.gap_ev
    assert diagonal.value == pytest.approx(
        left.gap_ev
        - diagonal.vertex.value / diagonal.area.area_angstrom_squared,
        rel=2.0e-14,
        abs=2.0e-15,
    )


def test001_exact_transfer_matching_and_opposition_fail_closed() -> None:
    left_a, right_a = _a_transitions()
    left_b, right_b = _b_transitions()
    result_a = _call_a(left_a, right_a)
    result_b = _call_b(left_b, right_b)

    assert result_a.left_transfer_inverse_angstrom == (
        1.0 / 128.0,
        2.0 / 128.0,
    )
    assert result_a.right_transfer_inverse_angstrom == result_a.left_transfer_inverse_angstrom
    assert result_b.right_transfer_inverse_angstrom == tuple(
        -value for value in result_b.left_transfer_inverse_angstrom
    )
    assert result_a.kinematics.residual_vector_inverse_angstrom == (0.0, 0.0)
    assert result_b.kinematics.residual_vector_inverse_angstrom == (0.0, 0.0)
    assert result_a.kinematics.momentum_tolerance_inverse_angstrom == 0.0
    assert result_b.kinematics.momentum_tolerance_inverse_angstrom == 0.0

    with pytest.raises(ValueError, match="exact q_left=q_right"):
        _call_a(left_a, right_b)
    with pytest.raises(ValueError, match="exact q_left=-q_right"):
        _call_b(left_b, right_a)


def test001_source_mismatch_and_duplicate_energy_inconsistency_fail() -> None:
    left, right = _a_transitions()
    wrong_hash = _transition(
        right.particle,
        right.hole,
        particle_energy_ev=right.particle_energy_ev,
        hole_energy_ev=right.hole_energy_ev,
        source_artifact_sha256="e" * 64,
    )
    wrong_text = _transition(
        right.particle,
        right.hole,
        particle_energy_ev=right.particle_energy_ev,
        hole_energy_ev=right.hole_energy_ev,
        source_text="Different caller-attested diagonal source text.",
    )
    for mismatched in (wrong_hash, wrong_text):
        with pytest.raises(ValueError, match="same immutable source receipt"):
            _call_a(left, mismatched)

    duplicate_with_wrong_particle_energy = _transition(
        left.particle,
        left.hole,
        particle_energy_ev=left.particle_energy_ev + 0.01,
        hole_energy_ev=left.hole_energy_ev,
    )
    with pytest.raises(ValueError, match="duplicate orbital.*energies"):
        _call_a(left, duplicate_with_wrong_particle_energy)


def test001_strict_transition_and_area_receipts_and_authority_limits() -> None:
    left, _ = _a_transitions()
    area = _area()
    assert left.particle_occupation == 0
    assert left.hole_occupation == 1
    assert left.gap_ev > 0.0
    assert left.source_artifact_immutable is True
    assert left.source_provider_status == DIAGONAL_HF_SOURCE_PROVIDER_STATUS
    assert left.hf_stationarity_certified is False
    assert left.production_ready is False
    assert len(left.source_fingerprint) == 64
    assert len(left.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        left.hf_stationarity_certified = True  # type: ignore[misc]

    assert area.area_angstrom_squared > 0.0
    assert area.provider_status == FINITE_AREA_PROVIDER_STATUS
    assert area.caller_attested is True
    assert area.mesh_authority is False
    assert area.quadrature_authority is False
    assert area.torus_authority is False
    assert area.background_authority is False
    assert area.paper_authority is False
    assert area.production_authority is False
    assert len(area.fingerprint) == 64

    with pytest.raises(ValueError, match="gap must be positive"):
        _transition(
            left.particle,
            left.hole,
            particle_energy_ev=-0.2,
            hole_energy_ev=-0.1,
        )
    for bad_area in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            Vituri2024FiniteAreaReceipt(
                area_angstrom_squared=bad_area,
                provider_sha256=_AREA_PROVIDER_SHA,
                source_text="Invalid test area.",
            )
    with pytest.raises(TypeError):
        Vituri2024FiniteAreaReceipt(
            area_angstrom_squared=True,  # type: ignore[arg-type]
            provider_sha256=_AREA_PROVIDER_SHA,
            source_text="Invalid test area.",
        )


def test001_element_dimensions_fingerprints_equations_and_no_go_scope() -> None:
    left_a, right_a = _a_transitions()
    left_b, right_b = _b_transitions()
    result_a = _call_a(left_a, right_a)
    result_b = _call_b(left_b, right_b)

    assert result_a.element_kind == "A"
    assert result_b.element_kind == "B"
    assert result_a.equation == RPA_A_ELEMENT_EQUATION
    assert result_b.equation == RPA_B_ELEMENT_EQUATION
    assert result_a.quartet_index_order == "(a,B;A,b)"
    assert result_b.quartet_index_order == "(a,b;A,B)"
    for result in (result_a, result_b):
        assert result.scalar_hessian_equation == SCALAR_HESSIAN_EQUATION
        assert result.units == RPA_ELEMENT_UNITS == "eV"
        assert result.vertex_units == RPA_VERTEX_UNITS == "eV*Angstrom^2"
        assert result.area_units == RPA_AREA_UNITS == "Angstrom^2"
        assert result.authority == RPA_ELEMENT_AUTHORITY
        assert result.source_fingerprint == result.left_transition.source_fingerprint
        assert result.area_fingerprint == result.area.fingerprint
        assert result.vertex_fingerprint == result.vertex.fingerprint
        assert result.kinematics_fingerprint == result.kinematics.fingerprint
        assert result.interaction_receipt_fingerprint == (
            result.vertex.interaction_receipt_fingerprint
        )
        for fingerprint in (
            result.fingerprint,
            result.context_fingerprint,
            result.source_fingerprint,
            result.area_fingerprint,
            result.vertex_fingerprint,
        ):
            assert len(fingerprint) == 64
        assert result.post_hermitized is False
        assert result.dense_rpa_assembly is False
        assert result.typed_signed_q_sector_promotion is False
        assert result.hf_stationarity_certified is False
        assert result.area_mesh_quadrature_torus_certified is False
        assert result.q0_background_certified is False
        assert result.domain_cutoff_convergence_certified is False
        assert result.production_rpa_authority is False
        assert result.paper_numerical_parity is False
        assert result.no_go_limits == RPA_ELEMENT_NO_GO_LIMITS

    for exported in (
        "Vituri2024DiagonalHFTransitionReceipt",
        "Vituri2024FiniteAreaReceipt",
        "Vituri2024RPAElementReceipt",
        "vituri2024_rpa_a_element",
        "vituri2024_rpa_b_element",
    ):
        assert hasattr(abc_trilayer_module, exported)
    for forbidden in (
        "assemble_vituri2024_rpa",
        "vituri2024_rpa_matrix",
        "run_vituri2024_rpa",
        "promote_vituri2024_signed_q",
    ):
        assert not hasattr(rpa_module, forbidden)

    module_doc = rpa_module.__doc__ or ""
    assert "paper Eq. C9" in module_doc
    assert "paper-C3 inconsistency" in module_doc
    assert "vituri2024_vertex" in module_doc
    assert "divided by the finite area exactly once" in module_doc
    assert "No post-Hermitizing" in module_doc
    for blocked in (
        "source stationarity",
        "area/mesh/quadrature/torus",
        "q=0 background",
        "domain/cutoff",
    ):
        assert blocked in module_doc
    assert "Delta1" in inspect.signature(vituri2024_rpa_a_element).parameters
    assert "Delta1" in inspect.signature(vituri2024_rpa_b_element).parameters

    # The typed result binds one common Delta1/interaction/area context.
    with pytest.raises(ValueError, match="Delta1 mismatch"):
        replace(result_a, delta1_ev=result_a.delta1_ev + 0.001)
    with pytest.raises(ValueError, match="interaction fingerprint mismatch"):
        replace(result_a, interaction_receipt_fingerprint="f" * 64)
    other_area = Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=59.0,
        provider_sha256=_AREA_PROVIDER_SHA,
        source_text="A distinct caller-attested finite area.",
    )
    with pytest.raises(ValueError, match="interaction contribution"):
        replace(
            result_a,
            area=other_area,
            area_fingerprint=other_area.fingerprint,
        )


def test001_pauli_and_flavor_forbidden_vertices_give_exact_zeros() -> None:
    # B quartet (a,b;A,B) has a=b, so Pauli antisymmetry short-circuits.
    hole_left = _orbital((0.0, 0.0))
    repeated_particle = _orbital((1.0 / 128.0, 2.0 / 128.0))
    hole_right = _orbital((2.0 / 128.0, 4.0 / 128.0))
    left = _transition(
        repeated_particle,
        hole_left,
        particle_energy_ev=0.4,
        hole_energy_ev=-0.2,
    )
    right = _transition(
        repeated_particle,
        hole_right,
        particle_energy_ev=0.4,
        hole_energy_ev=-0.3,
    )
    pauli = _call_b(left, right)
    assert pauli.value == 0.0j
    assert pauli.interaction_contribution_ev == 0.0j
    assert pauli.vertex.pauli_short_circuit is not None
    assert pauli.vertex.pauli_short_circuit.reason == "alpha_equals_beta"

    # In A quartet (a,B;A,b), both direct and exchanged flavor deltas fail.
    f1 = {"valley": 1, "spin": 1}
    f2 = {"valley": 1, "spin": -1}
    hole_l = _orbital((0.0, 0.0), **f2)
    particle_l = _orbital((1.0 / 128.0, 2.0 / 128.0), **f1)
    hole_r = _orbital((-3.0 / 128.0, 1.0 / 128.0), **f1)
    particle_r = _orbital((-2.0 / 128.0, 3.0 / 128.0), **f2)
    flavor_zero = _call_a(
        _transition(
            particle_l,
            hole_l,
            particle_energy_ev=0.4,
            hole_energy_ev=-0.2,
        ),
        _transition(
            particle_r,
            hole_r,
            particle_energy_ev=0.5,
            hole_energy_ev=-0.3,
        ),
    )
    assert flavor_zero.value == 0.0j
    assert flavor_zero.vertex.value == 0.0j
    assert flavor_zero.vertex.direct_ordered is not None
    assert flavor_zero.vertex.exchange_ordered is not None
    assert flavor_zero.vertex.direct_ordered.value == 0.0j
    assert flavor_zero.vertex.exchange_ordered.value == 0.0j


def _random_hermitian_antisymmetrized_tensor(
    rng: np.random.Generator, dimension: int
) -> np.ndarray:
    """Independent test tensor v[alpha,beta;gamma,delta]."""

    pairs = [(i, j) for i in range(dimension) for j in range(i + 1, dimension)]
    raw = rng.normal(size=(len(pairs), len(pairs))) + 1j * rng.normal(
        size=(len(pairs), len(pairs))
    )
    wedge_matrix = (raw + raw.conj().T) / 2.0
    vertex = np.zeros((dimension,) * 4, dtype=np.complex128)
    for p, (i, j) in enumerate(pairs):
        for q, (k, ell) in enumerate(pairs):
            value = wedge_matrix[p, q]
            vertex[i, j, k, ell] = value
            vertex[j, i, k, ell] = -value
            vertex[i, j, ell, k] = -value
            vertex[j, i, ell, k] = value
    return vertex


def test001_independent_normalized_slater_chart_scalar_energy_oracle() -> None:
    """Check C9 signs/order/factors without calling either RPA element API."""

    rng = np.random.default_rng(240810309)
    n_particle = 3
    n_hole = 2
    dimension = n_particle + n_hole
    vertex = _random_hermitian_antisymmetrized_tensor(rng, dimension)
    assert np.max(np.abs(vertex + vertex.swapaxes(0, 1))) == 0.0
    assert np.max(np.abs(vertex + vertex.swapaxes(2, 3))) == 0.0
    assert np.max(
        np.abs(vertex - vertex.transpose(2, 3, 0, 1).conj())
    ) == 0.0

    energies = np.concatenate(
        (rng.uniform(0.4, 1.4, n_particle), rng.uniform(-1.2, -0.2, n_hole))
    )
    occupied = tuple(range(n_particle, dimension))
    rho0 = np.diag([0.0] * n_particle + [1.0] * n_hole).astype(
        np.complex128
    )

    # For E2=1/2*v[a,b;g,d]*rho[d,a]*rho[g,b], the Fock matrix is
    # F[i,j]=h[i,j]+sum_B v[i,B;B,j].  Choose h so F=diag(epsilon).
    occupied_contraction = np.array(
        [
            [sum(vertex[i, cap_b, cap_b, j] for cap_b in occupied) for j in range(dimension)]
            for i in range(dimension)
        ],
        dtype=np.complex128,
    )
    one_body = np.diag(energies) - occupied_contraction
    assert one_body == pytest.approx(one_body.conj().T, abs=2.0e-14)

    def scalar_energy(rho: np.ndarray) -> complex:
        return np.trace(one_body @ rho) + 0.5 * np.einsum(
            "abgd,da,gb->", vertex, rho, rho
        )

    def normalized_chart_density(z_matrix: np.ndarray) -> np.ndarray:
        # Exactly C=[Z;I](I+Z^dagger Z)^(-1/2), using the principal root.
        stacked = np.vstack((z_matrix, np.eye(n_hole)))
        metric = np.eye(n_hole) + z_matrix.conj().T @ z_matrix
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        inverse_sqrt = (
            eigenvectors * eigenvalues[np.newaxis, :] ** (-0.5)
        ) @ eigenvectors.conj().T
        occupied_chart = stacked @ inverse_sqrt
        assert occupied_chart.conj().T @ occupied_chart == pytest.approx(
            np.eye(n_hole), abs=2.0e-14
        )
        return occupied_chart @ occupied_chart.conj().T

    pair_dimension = n_particle * n_hole
    a_matrix = np.zeros((pair_dimension, pair_dimension), dtype=np.complex128)
    b_matrix = np.zeros_like(a_matrix)
    gap_matrix = np.zeros_like(a_matrix)
    wrong_a_sign = np.zeros_like(a_matrix)
    wrong_a_annihilator_order = np.zeros_like(a_matrix)
    for a in range(n_particle):
        for cap_a_local in range(n_hole):
            cap_a = n_particle + cap_a_local
            left_index = a * n_hole + cap_a_local
            for b in range(n_particle):
                for cap_b_local in range(n_hole):
                    cap_b = n_particle + cap_b_local
                    right_index = b * n_hole + cap_b_local
                    gap = (
                        energies[a] - energies[cap_a]
                        if a == b and cap_a_local == cap_b_local
                        else 0.0
                    )
                    gap_matrix[left_index, right_index] = gap
                    # These are direct, test-local transcription of paper C9.
                    a_matrix[left_index, right_index] = (
                        gap - vertex[a, cap_b, cap_a, b]
                    )
                    b_matrix[left_index, right_index] = -vertex[
                        a, b, cap_a, cap_b
                    ]
                    wrong_a_sign[left_index, right_index] = (
                        gap + vertex[a, cap_b, cap_a, b]
                    )
                    wrong_a_annihilator_order[left_index, right_index] = (
                        gap - vertex[a, cap_b, b, cap_a]
                    )
    assert a_matrix == pytest.approx(a_matrix.conj().T, abs=2.0e-14)
    assert b_matrix == pytest.approx(b_matrix.T, abs=2.0e-14)

    z_matrix = 0.37 * (
        rng.normal(size=(n_particle, n_hole))
        + 1j * rng.normal(size=(n_particle, n_hole))
    )
    z = z_matrix.reshape(-1)

    def c9_quadratic(a_block: np.ndarray, b_block: np.ndarray) -> float:
        return float(
            np.real(np.vdot(z, a_block @ z) + z.T @ b_block.conj() @ z)
        )

    expected = c9_quadratic(a_matrix, b_matrix)
    reference_energy = scalar_energy(rho0)

    def centered_quadratic_coefficient(step: float) -> float:
        """Return the t^2 coefficient Q; the literal E''(0) equals 2Q."""

        plus = scalar_energy(normalized_chart_density(step * z_matrix))
        minus = scalar_energy(normalized_chart_density(-step * z_matrix))
        return float(np.real(((plus + minus) / 2.0 - reference_energy) / step**2))

    # The centered quadratic coefficient is O(t^2); Richardson removes that
    # term while all energies still come from the exact normalized chart.
    coarse = centered_quadratic_coefficient(1.0e-3)
    fine = centered_quadratic_coefficient(5.0e-4)
    finite_chart_quadratic = (4.0 * fine - coarse) / 3.0
    assert finite_chart_quadratic == pytest.approx(expected, rel=2.0e-8, abs=2.0e-8)

    wrong_candidates = (
        c9_quadratic(wrong_a_sign, b_matrix),
        c9_quadratic(wrong_a_annihilator_order, b_matrix),
        c9_quadratic(a_matrix, -b_matrix),
        c9_quadratic(
            gap_matrix + 0.5 * (a_matrix - gap_matrix), 0.5 * b_matrix
        ),
        c9_quadratic(
            gap_matrix + 0.25 * (a_matrix - gap_matrix), 0.25 * b_matrix
        ),
    )
    for wrong in wrong_candidates:
        assert abs(finite_chart_quadratic - wrong) > 0.1
