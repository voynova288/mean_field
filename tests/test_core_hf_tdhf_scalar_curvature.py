"""Exact public-oracle tests for generic signed-q scalar certification.

The expected A+/A-/B(+,-)/B(-,+) blocks are literal double commutators in an
8-orbital, 4-particle Fock space.  Production A/B assembly is never used to
construct the oracle blocks or the exact Slater energy callback.
"""

from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
from itertools import combinations

import numpy as np
import pytest

from mean_field.core.hf.tdhf import ParticleHolePair
from mean_field.core.hf.tdhf_scalar_curvature import (
    TDHFEnergyConvention,
    TDHFPhysicalDirection,
    TDHFScalarCurvatureCertificate,
    TDHFScalarCurvatureFactoryStatus,
    TDHFScalarCurvatureStepLadder,
    TDHFScalarCurvatureTolerances,
    TDHFTransitionTangentBasis,
    TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM,
    TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER,
    TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM,
    canonical_tdhf_scalar_directions,
    canonical_tdhf_stationarity_directions,
    certify_tdhf_scalar_curvature,
    exact_tdhf_projector_path,
    make_tdhf_scalar_curvature_approval,
    make_tdhf_scalar_functional_manifest,
)
from mean_field.core.hf.tdhf_signed import (
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFSignedQBlocks,
    build_standard_nambu_sewing,
    build_tdhf_signed_q_matrices,
    fingerprint_tdhf_matrix,
    fingerprint_tdhf_pairs,
    fingerprint_tdhf_sector,
)

N_ORBITALS = 8
N_PARTICLES = 4
STATES = tuple(
    sum(1 << index for index in occupied)
    for occupied in combinations(range(N_ORBITALS), N_PARTICLES)
)
STATE_INDEX = {state: index for index, state in enumerate(STATES)}
REFERENCE = STATE_INDEX[sum(1 << index for index in range(N_PARTICLES))]
PLUS = ((4, 0), (5, 1))
MINUS = ((6, 2), (7, 3))
SOURCE_FINGERPRINT = sha256(b"literal-8-orbital-stationary-slater-source-v2").hexdigest()
INTERACTION_FINGERPRINT = sha256(b"literal-8-orbital-many-body-hamiltonian-v2").hexdigest()


def _annihilate(state: int, orbital: int) -> tuple[int, int] | None:
    if not state & (1 << orbital):
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return sign, state ^ (1 << orbital)


def _create(state: int, orbital: int) -> tuple[int, int] | None:
    if state & (1 << orbital):
        return None
    sign = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return sign, state | (1 << orbital)


def _one_body_operator(particle: int, hole: int) -> np.ndarray:
    result = np.zeros((len(STATES), len(STATES)), dtype=np.complex128)
    for column, state in enumerate(STATES):
        first = _annihilate(state, hole)
        if first is None:
            continue
        sign1, intermediate = first
        second = _create(intermediate, particle)
        if second is None:
            continue
        sign2, final = second
        if final in STATE_INDEX:
            result[STATE_INDEX[final], column] = sign1 * sign2
    return result


def _commutator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right - right @ left


def _reference_expectation(operator: np.ndarray) -> complex:
    return complex(operator[REFERENCE, REFERENCE])


def _literal_hamiltonian() -> tuple[
    np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray
]:
    number = tuple(_one_body_operator(index, index) for index in range(N_ORBITALS))
    one_body_energies = np.array(
        [-0.37, -0.21, -0.08, -0.03, 0.42, 0.67, 0.91, 1.16]
    )
    hamiltonian = sum(
        value * operator for value, operator in zip(one_body_energies, number)
    )
    plus = tuple(_one_body_operator(*pair) for pair in PLUS)
    minus = tuple(_one_body_operator(*pair) for pair in MINUS)

    a_plus_interaction = np.array(
        [[0.13, 0.021 + 0.037j], [0.021 - 0.037j, -0.04]]
    )
    a_minus_interaction = np.array(
        [[-0.07, -0.031 + 0.019j], [-0.031 - 0.019j, 0.09]]
    )
    for operators, block in (
        (plus, a_plus_interaction),
        (minus, a_minus_interaction),
    ):
        for row, left in enumerate(operators):
            for column, right in enumerate(operators):
                hamiltonian += block[row, column] * left @ right.conj().T

    # Complex and nonsymmetric: transpose, conjugation, sign, and lane-order
    # mistakes are all observable.
    b_plus_minus_seed = np.array(
        [
            [0.041 + 0.027j, -0.033 + 0.012j],
            [0.018 - 0.029j, 0.052 + 0.007j],
        ]
    )
    for row, left in enumerate(plus):
        for column, right in enumerate(minus):
            pair_create = left @ right
            hamiltonian += b_plus_minus_seed[row, column] * pair_create
            hamiltonian += (
                b_plus_minus_seed[row, column].conjugate()
                * pair_create.conj().T
            )
    assert np.linalg.norm(hamiltonian - hamiltonian.conj().T) < 1.0e-13
    return hamiltonian, plus, minus, one_body_energies


def _literal_double_commutator_blocks(
    hamiltonian: np.ndarray,
    plus: tuple[np.ndarray, ...],
    minus: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def a_block(operators: tuple[np.ndarray, ...]) -> np.ndarray:
        return np.asarray(
            [
                [
                    _reference_expectation(
                        _commutator(left.conj().T, _commutator(hamiltonian, right))
                    )
                    for right in operators
                ]
                for left in operators
            ],
            dtype=np.complex128,
        )

    # For K=Z-Z† and v=(x,y*), B(+,-) is the conjugate of
    # < [[H,R_plus],R_minus] >.  The reverse block is independently evaluated.
    b_plus_minus = np.asarray(
        [
            [
                _reference_expectation(
                    _commutator(_commutator(hamiltonian, left), right)
                ).conjugate()
                for right in minus
            ]
            for left in plus
        ],
        dtype=np.complex128,
    )
    b_minus_plus = np.asarray(
        [
            [
                _reference_expectation(
                    _commutator(_commutator(hamiltonian, left), right)
                ).conjugate()
                for right in plus
            ]
            for left in minus
        ],
        dtype=np.complex128,
    )
    return a_block(plus), b_plus_minus, a_block(minus), b_minus_plus


def _pairs() -> tuple[tuple[ParticleHolePair, ...], tuple[ParticleHolePair, ...]]:
    plus = tuple(
        ParticleHolePair(
            particle,
            hole,
            particle_momentum=(1, index),
            hole_momentum=(0, index),
            particle_flavor="literal-plus",
            hole_flavor="literal-reference",
        )
        for index, (particle, hole) in enumerate(PLUS)
    )
    minus = tuple(
        ParticleHolePair(
            particle,
            hole,
            particle_momentum=(-1, index),
            hole_momentum=(0, index + len(PLUS)),
            particle_flavor="literal-minus",
            hole_flavor="literal-reference",
        )
        for index, (particle, hole) in enumerate(MINUS)
    )
    return plus, minus


def _sector_from_blocks(
    blocks: TDHFSignedQBlocks,
    *,
    source_fingerprint: str = SOURCE_FINGERPRINT,
    interaction_fingerprint: str = INTERACTION_FINGERPRINT,
) -> TDHFGenericSignedQSector:
    sewing = build_standard_nambu_sewing(
        blocks.plus_pairs,
        blocks.minus_pairs,
        source_fingerprint=source_fingerprint,
        construction="literal_fock_space_block_swap_v2",
    )
    return TDHFGenericSignedQSector(
        q=TDHFGenericSignedQ(
            plus_raw=(1, 0),
            minus_raw=(-1, 0),
            plus_canonical=(1, 0),
            minus_canonical=(-1, 0),
            provenance="literal distinct signed-q orbit",
        ),
        blocks=blocks,
        sewing=sewing,
        source_fingerprint=source_fingerprint,
        interaction_fingerprint=interaction_fingerprint,
        response_scope="literal_exact_double_commutator_oracle",
        static_hessian_authority="projected_signed_ab",
    )


def _exact_sector() -> tuple[
    TDHFGenericSignedQSector,
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    hamiltonian, plus_operators, minus_operators, one_body_energies = (
        _literal_hamiltonian()
    )
    a_plus, b_plus_minus, a_minus, b_minus_plus = (
        _literal_double_commutator_blocks(
            hamiltonian, plus_operators, minus_operators
        )
    )
    plus_pairs, minus_pairs = _pairs()
    blocks = TDHFSignedQBlocks(
        plus_pairs=plus_pairs,
        minus_pairs=minus_pairs,
        A_plus=a_plus,
        B_plus_minus=b_plus_minus,
        A_minus=a_minus,
        B_minus_plus=b_minus_plus,
    )
    sector = _sector_from_blocks(blocks)
    matrices = build_tdhf_signed_q_matrices(
        sector.blocks, sector.sewing, raise_on_structure_error=True
    )
    assert matrices.structure.ok
    return sector, hamiltonian, plus_operators, minus_operators, one_body_energies


def _orbital_tangent(particle: int, hole: int) -> np.ndarray:
    result = np.zeros((N_ORBITALS, N_ORBITALS), dtype=np.complex128)
    result[particle, hole] = 1.0
    return result


def _basis(sector: TDHFGenericSignedQSector) -> TDHFTransitionTangentBasis:
    p0 = np.diag([1.0] * N_PARTICLES + [0.0] * (N_ORBITALS - N_PARTICLES))
    return TDHFTransitionTangentBasis(
        source_projector=p0,
        plus_tangents=tuple(_orbital_tangent(*pair) for pair in PLUS),
        minus_tangents=tuple(_orbital_tangent(*pair) for pair in MINUS),
        source_fingerprint=sector.source_fingerprint,
        plus_pairs_fingerprint=fingerprint_tdhf_pairs(sector.blocks.plus_pairs),
        minus_pairs_fingerprint=fingerprint_tdhf_pairs(sector.blocks.minus_pairs),
    )


def _slater_vector(projector: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    occupied = eigenvectors[:, np.argsort(eigenvalues)[-N_PARTICLES:]]
    amplitudes = np.array(
        [
            np.linalg.det(occupied[list(combination), :])
            for combination in combinations(range(N_ORBITALS), N_PARTICLES)
        ],
        dtype=np.complex128,
    )
    return amplitudes / np.linalg.norm(amplitudes)


def _energy(projector: np.ndarray, hamiltonian: np.ndarray) -> float:
    assert not projector.flags.writeable
    state = _slater_vector(projector)
    return float(np.vdot(state, hamiltonian @ state).real)


def _energy_callback(hamiltonian: np.ndarray):
    def exact_slater_energy(projector: np.ndarray) -> float:
        return _energy(projector, hamiltonian)

    return exact_slater_energy


def _five_point_first_derivative(
    basis: TDHFTransitionTangentBasis,
    direction: TDHFPhysicalDirection,
    hamiltonian: np.ndarray,
    step: float,
) -> float:
    projectors = tuple(
        exact_tdhf_projector_path(basis, direction, multiplier * step)
        for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0)
    )
    for projector in projectors:
        projector.setflags(write=False)
    energies = tuple(_energy(projector, hamiltonian) for projector in projectors)
    fm2, fm1, _, fp1, fp2 = energies
    return (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * step)


def _many_body_generator(
    direction: TDHFPhysicalDirection,
    plus: tuple[np.ndarray, ...],
    minus: tuple[np.ndarray, ...],
) -> np.ndarray:
    z = sum(
        (coefficient * operator for coefficient, operator in zip(direction.x, plus)),
        start=np.zeros_like(plus[0]),
    )
    z += sum(
        (coefficient * operator for coefficient, operator in zip(direction.y, minus)),
        start=np.zeros_like(minus[0]),
    )
    return z - z.conj().T


def _directions() -> tuple[TDHFPhysicalDirection, ...]:
    real_mixed = np.array([0.48, -0.31, 0.57, 0.59], dtype=np.complex128)
    real_mixed /= np.linalg.norm(real_mixed)
    complex_mixed = np.array(
        [0.41 + 0.23j, -0.17 + 0.31j, 0.52 - 0.11j, -0.29 + 0.39j]
    )
    complex_mixed /= np.linalg.norm(complex_mixed)
    return (
        TDHFPhysicalDirection("x", np.array([1.0, 0.0]), np.zeros(2)),
        TDHFPhysicalDirection("y", np.zeros(2), np.array([0.0, 1.0j])),
        TDHFPhysicalDirection("mixed", real_mixed[:2], real_mixed[2:]),
        TDHFPhysicalDirection(
            "complex", complex_mixed[:2], complex_mixed[2:]
        ),
    )


def _ladder() -> TDHFScalarCurvatureStepLadder:
    return TDHFScalarCurvatureStepLadder(
        steps=(2.0e-2, 1.0e-2, 5.0e-3),
        tolerances=TDHFScalarCurvatureTolerances(
            stationarity_absolute=3.0e-8,
            stationarity_relative=1.0e-10,
            curvature_absolute=2.0e-8,
            curvature_relative=3.0e-7,
            roundoff_multiplier=256.0,
            projector_tolerance=8.0e-11,
        ),
        registration_label="literal-8-orbital-all-steps-v2",
    )


def _convention(
    denominator: float = 1.0,
) -> TDHFEnergyConvention:
    return TDHFEnergyConvention(
        normalization="total" if denominator == 1.0 else "per_area",
        denominator=denominator,
        energy_units="literal_eV",
        curvature_units=(
            "literal_eV" if denominator == 1.0 else "literal_eV_per_attested_area"
        ),
        denominator_source="caller-attested reporting conversion in exact unit test",
    )


def _certify(
    sector: TDHFGenericSignedQSector,
    basis: TDHFTransitionTangentBasis,
    hamiltonian: np.ndarray,
    *,
    directions: tuple[TDHFPhysicalDirection, ...] | None = None,
    denominator: float = 1.0,
) -> TDHFScalarCurvatureCertificate:
    registered = (
        canonical_tdhf_scalar_directions(
            len(basis.plus_tangents), len(basis.minus_tangents)
        )
        if directions is None
        else directions
    )
    callback = _energy_callback(hamiltonian)
    manifest = make_tdhf_scalar_functional_manifest(
        energy_callback=callback,
        source_functional_fingerprint=sha256(
            b"literal exact Slater expectation functional v1"
        ).hexdigest(),
        immutable_callback_input_fingerprint=fingerprint_tdhf_matrix(hamiltonian),
        provenance="Literal many-body Hamiltonian captured by the exact test oracle.",
    )
    approval = make_tdhf_scalar_curvature_approval(
        sector=sector,
        tangent_basis=basis,
        directions=registered,
        energy_callback=callback,
        functional_manifest=manifest,
        energy_convention=_convention(denominator),
        step_ladder=_ladder(),
        interaction_fingerprint=sector.interaction_fingerprint,
        provenance="Detached registration before exact-oracle callback evaluation.",
    )
    return certify_tdhf_scalar_curvature(
        approval=approval,
        sector=sector,
        tangent_basis=basis,
        energy_callback=callback,
        functional_manifest=manifest,
    )


def _certified_case(
    denominator: float = 1.0,
) -> tuple[
    TDHFScalarCurvatureCertificate,
    TDHFGenericSignedQSector,
    TDHFTransitionTangentBasis,
    np.ndarray,
]:
    sector, hamiltonian, _, _, _ = _exact_sector()
    basis = _basis(sector)
    return _certify(sector, basis, hamiltonian, denominator=denominator), sector, basis, hamiltonian


def _wrong_sector(
    sector: TDHFGenericSignedQSector,
    *,
    a_plus: np.ndarray | None = None,
    b_plus_minus: np.ndarray | None = None,
    a_minus: np.ndarray | None = None,
    b_minus_plus: np.ndarray | None = None,
) -> TDHFGenericSignedQSector:
    blocks = TDHFSignedQBlocks(
        plus_pairs=sector.blocks.plus_pairs,
        minus_pairs=sector.blocks.minus_pairs,
        A_plus=sector.blocks.A_plus if a_plus is None else a_plus,
        B_plus_minus=(
            sector.blocks.B_plus_minus
            if b_plus_minus is None
            else b_plus_minus
        ),
        A_minus=sector.blocks.A_minus if a_minus is None else a_minus,
        B_minus_plus=(
            sector.blocks.B_minus_plus
            if b_minus_plus is None
            else b_minus_plus
        ),
    )
    return _sector_from_blocks(blocks)


def test_public_factory_certifies_literal_double_commutator_all_ladders() -> None:
    sector, hamiltonian, plus, minus, _ = _exact_sector()
    basis = _basis(sector)
    certificate = _certify(sector, basis, hamiltonian)
    matrices = build_tdhf_signed_q_matrices(
        sector.blocks, sector.sewing, raise_on_structure_error=True
    )

    assert np.iscomplexobj(sector.blocks.B_plus_minus)
    assert np.max(np.abs(sector.blocks.B_plus_minus.imag)) > 1.0e-3
    assert not np.allclose(
        sector.blocks.B_plus_minus, sector.blocks.B_plus_minus.T
    )
    assert certificate.status.scalar_curvature_executed
    assert certificate.status.stationarity_complete_all_passed
    assert certificate.stationarity_complete_all_passed
    assert certificate.registered_direction_curvatures_match
    assert certificate.mathematical_scalar_hessian_match
    assert certificate.mathematical_scalar_curvature_match
    assert not certificate.static_hessian_authority_promoted
    assert not certificate.promotion_eligible
    assert not certificate.normalization_physics_certified
    assert "does not establish Nk" in certificate.normalization_statement
    assert certificate.sector_fingerprint == fingerprint_tdhf_sector(sector)
    assert certificate.sector_source_fingerprint == SOURCE_FINGERPRINT
    assert certificate.source_projector_fingerprint == fingerprint_tdhf_matrix(
        basis.source_projector
    )
    assert certificate.tangent_basis_fingerprint == basis.fingerprint
    assert certificate.plus_tangent_fingerprints == basis.plus_tangent_fingerprints
    assert certificate.minus_tangent_fingerprints == basis.minus_tangent_fingerprints
    assert certificate.plus_pairs_fingerprint == fingerprint_tdhf_pairs(
        sector.blocks.plus_pairs
    )
    assert certificate.minus_pairs_fingerprint == fingerprint_tdhf_pairs(
        sector.blocks.minus_pairs
    )
    assert certificate.h_plus_fingerprint == fingerprint_tdhf_matrix(
        matrices.H_plus
    )
    assert certificate.callback_provenance_fingerprint == certificate.callback.fingerprint
    assert certificate.functional_manifest_fingerprint == (
        certificate.functional_manifest.fingerprint
    )
    assert certificate.convention_fingerprint == certificate.convention.fingerprint
    assert len(certificate.fingerprint) == 64
    assert tuple(item.label for item in certificate.direction_evidence) == tuple(
        [f"diag[{index}]" for index in range(4)]
        + [
            label
            for left in range(4)
            for right in range(left + 1, 4)
            for label in (f"real[{left},{right}]", f"imag[{left},{right}]")
        ]
    )
    assert len(certificate.direction_evidence) == 16
    assert tuple(item.label for item in certificate.stationarity_evidence) == tuple(
        label
        for index in range(4)
        for label in (
            f"stationarity.real[{index}]",
            f"stationarity.imag[{index}]",
        )
    )
    assert len(certificate.stationarity_evidence) == 8
    assert all(
        direction.all_registered_steps_passed
        and len(direction.steps) == len(_ladder().steps)
        and all(step.stationarity_passed for step in direction.steps)
        for direction in certificate.stationarity_evidence
    )
    assert certificate.stationarity_direction_inventory_fingerprint == (
        certificate.approval.stationarity_direction_inventory_fingerprint
    )
    assert len(certificate.reconstruction_evidence) == len(_ladder().steps)
    reconstructed = certificate.reconstruction_evidence[-1]
    np.testing.assert_allclose(
        reconstructed.reconstructed_hessian, matrices.H_plus, atol=reconstructed.matrix_bound
    )
    assert certificate.reconstructed_hessian_fingerprint == (
        reconstructed.reconstructed_hessian_fingerprint
    )
    assert certificate.reconstructed_hessian_max_abs_residual == (
        reconstructed.max_abs_residual
    )

    reference = np.zeros(len(STATES), dtype=np.complex128)
    reference[REFERENCE] = 1.0
    for direction, evidence in zip(
        canonical_tdhf_scalar_directions(2, 2), certificate.direction_evidence
    ):
        generator = _many_body_generator(direction, plus, minus)
        first_commutator = _commutator(hamiltonian, generator)
        exact_second = float(
            np.vdot(
                reference,
                _commutator(first_commutator, generator) @ reference,
            ).real
        )
        assert evidence.raw_target_curvature == pytest.approx(
            exact_second, abs=2.0e-13
        )
        assert evidence.raw_target_curvature == pytest.approx(
            2.0 * evidence.target_quadratic_form
        )
        assert len(evidence.steps) == len(_ladder().steps)
        assert evidence.all_registered_steps_passed
        assert all(
            step.stationarity_passed
            and step.curvature_passed
            and step.stationarity_residual <= step.stationarity_bound
            and step.curvature_residual <= step.curvature_bound
            and step.stationarity_roundoff_allowance
            <= TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_STATIONARITY_ALLOWANCE_MAXIMUM
            and step.curvature_roundoff_allowance
            <= TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_CURVATURE_ALLOWANCE_MAXIMUM
            for step in evidence.steps
        )


def test_public_factory_rejects_redistributed_lane_cardinality_before_callback() -> None:
    sector, hamiltonian, _, _, _ = _exact_sector()
    basis = _basis(sector)
    redistributed_basis = TDHFTransitionTangentBasis(
        source_projector=basis.source_projector,
        plus_tangents=basis.plus_tangents + basis.minus_tangents[:1],
        minus_tangents=basis.minus_tangents[1:],
        source_fingerprint=basis.source_fingerprint,
        plus_pairs_fingerprint=basis.plus_pairs_fingerprint,
        minus_pairs_fingerprint=basis.minus_pairs_fingerprint,
    )
    assert (
        len(redistributed_basis.plus_tangents),
        len(redistributed_basis.minus_tangents),
    ) == (3, 1)
    assert len(redistributed_basis.plus_tangents) + len(
        redistributed_basis.minus_tangents
    ) == len(basis.plus_tangents) + len(basis.minus_tangents)
    assert redistributed_basis.plus_pairs_fingerprint == basis.plus_pairs_fingerprint
    assert redistributed_basis.minus_pairs_fingerprint == basis.minus_pairs_fingerprint

    callback_calls = 0

    def counted_energy(projector: np.ndarray) -> float:
        nonlocal callback_calls
        callback_calls += 1
        return _energy(projector, hamiltonian)

    manifest = make_tdhf_scalar_functional_manifest(
        energy_callback=counted_energy,
        source_functional_fingerprint=sha256(
            b"asymmetric redistributed tangent-lane cardinality P1"
        ).hexdigest(),
        immutable_callback_input_fingerprint=fingerprint_tdhf_matrix(hamiltonian),
        provenance="Same-total redistributed tangent-lane cardinality canary.",
    )
    with pytest.raises(ValueError, match="tangent lane cardinalities"):
        make_tdhf_scalar_curvature_approval(
            sector=sector,
            tangent_basis=redistributed_basis,
            directions=canonical_tdhf_scalar_directions(3, 1),
            energy_callback=counted_energy,
            functional_manifest=manifest,
            energy_convention=_convention(),
            step_ladder=_ladder(),
            interaction_fingerprint=sector.interaction_fingerprint,
            provenance="Reject redistributed lane counts before callback execution.",
        )
    assert callback_calls == 0


def test_imaginary_quadrature_source_canary_fails_stationarity_before_curvature() -> None:
    sector, hamiltonian, plus, _, _ = _exact_sector()
    basis = _basis(sector)
    dimension = len(basis.plus_tangents) + len(basis.minus_tangents)
    stationarity = canonical_tdhf_stationarity_directions(2, 2)

    assert len(stationarity) == 2 * dimension
    for index, direction in enumerate(stationarity):
        coordinate = index // 2
        expected = np.zeros(dimension, dtype=np.complex128)
        expected[coordinate] = 1.0 if index % 2 == 0 else 1.0j
        np.testing.assert_array_equal(direction.vector, expected)
    # The lower signed-Hessian lane is y*, so i e_2 requires y_0=-i.
    np.testing.assert_array_equal(stationarity[5].y, np.array([-1.0j, 0.0j]))

    source_strength = 3.0e-3
    t0 = plus[0]
    source = 1.0j * (t0 - t0.conj().T)
    np.testing.assert_allclose(source, source.conj().T, atol=0.0, rtol=0.0)
    perturbed_hamiltonian = hamiltonian + source_strength * source
    step = _ladder().steps[0]

    # P1 canary: the d^2 curvature inventory misses i e_0 as a first-order
    # probe, while the separate 2d real-tangent inventory detects only that
    # imaginary quadrature.
    for direction in canonical_tdhf_scalar_directions(2, 2):
        assert _five_point_first_derivative(
            basis, direction, perturbed_hamiltonian, step
        ) == pytest.approx(0.0, abs=2.0e-12)
    stationarity_gradients = tuple(
        _five_point_first_derivative(
            basis, direction, perturbed_hamiltonian, step
        )
        for direction in stationarity
    )
    assert stationarity_gradients[1] == pytest.approx(
        2.0 * source_strength, rel=2.0e-7, abs=2.0e-12
    )
    assert all(
        abs(value) < 2.0e-12
        for index, value in enumerate(stationarity_gradients)
        if index != 1
    )

    callback_calls = 0

    def imaginary_source_energy(projector: np.ndarray) -> float:
        nonlocal callback_calls
        callback_calls += 1
        return _energy(projector, perturbed_hamiltonian)

    manifest = make_tdhf_scalar_functional_manifest(
        energy_callback=imaginary_source_energy,
        source_functional_fingerprint=sha256(
            b"Hermitian i(T0-T0dagger) imaginary-gradient canary"
        ).hexdigest(),
        immutable_callback_input_fingerprint=fingerprint_tdhf_matrix(
            perturbed_hamiltonian
        ),
        provenance=(
            "Exact oracle with a Hermitian source whose gradient is confined to "
            "the i e_0 quadrature."
        ),
    )
    approval = make_tdhf_scalar_curvature_approval(
        sector=sector,
        tangent_basis=basis,
        directions=canonical_tdhf_scalar_directions(2, 2),
        energy_callback=imaginary_source_energy,
        functional_manifest=manifest,
        energy_convention=_convention(),
        step_ladder=_ladder(),
        interaction_fingerprint=sector.interaction_fingerprint,
        provenance="Detached approval for the final stationarity P1 canary.",
    )
    assert approval.stationarity_direction_fingerprints == tuple(
        item.fingerprint for item in stationarity
    )
    assert approval.canonical_stationarity_complete_inventory
    with pytest.raises(ValueError, match="stationarity direction inventory"):
        replace(approval, stationarity_direction_inventory_fingerprint="f" * 64)
    with pytest.raises(
        ValueError,
        match=r"scalar-stationarity certification failed.*stationarity\.imag\[0\]",
    ):
        certify_tdhf_scalar_curvature(
            approval=approval,
            sector=sector,
            tangent_basis=basis,
            energy_callback=imaginary_source_energy,
            functional_manifest=manifest,
        )
    # Every h along all 2d stationarity directions ran, then certification
    # aborted before any d^2 curvature stencil or certificate construction.
    assert callback_calls == 2 * dimension * len(_ladder().steps) * 5


def test_incomplete_inventory_certifies_only_registered_directions() -> None:
    sector, hamiltonian, _, _, _ = _exact_sector()
    basis = _basis(sector)
    subset = canonical_tdhf_scalar_directions(2, 2)[:3]
    certificate = _certify(
        sector, basis, hamiltonian, directions=subset
    )

    assert certificate.stationarity_complete_all_passed
    assert certificate.registered_direction_curvatures_match
    assert not certificate.mathematical_scalar_hessian_match
    assert not certificate.mathematical_scalar_curvature_match
    assert certificate.status.authority == "registered_raw_direction_curvatures_only"
    assert certificate.reconstruction_evidence == ()
    assert certificate.reconstructed_hessian_fingerprint is None
    assert certificate.reconstructed_hessian_max_abs_residual is None


def test_constant_energy_tiny_step_cannot_pass_incomplete_inventory() -> None:
    sector, hamiltonian, _, _, _ = _exact_sector()
    basis = _basis(sector)
    calls = 0

    def constant_energy(_projector: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 1.0

    manifest = make_tdhf_scalar_functional_manifest(
        energy_callback=constant_energy,
        source_functional_fingerprint=sha256(
            b"constant-energy tiny-step negative"
        ).hexdigest(),
        immutable_callback_input_fingerprint=fingerprint_tdhf_matrix(hamiltonian),
        provenance="Constant-energy tiny-step non-vacuity canary.",
    )
    approval = make_tdhf_scalar_curvature_approval(
        sector=sector,
        tangent_basis=basis,
        directions=(canonical_tdhf_scalar_directions(2, 2)[0],),
        energy_callback=constant_energy,
        functional_manifest=manifest,
        energy_convention=_convention(),
        step_ladder=_ladder(),
        interaction_fingerprint=sector.interaction_fingerprint,
        provenance="Incomplete direction inventory with a preregistered valid ladder.",
    )
    object.__setattr__(approval.step_ladder, "steps", (1.0e-8, 5.0e-9))
    with pytest.raises(ValueError, match="locked v1 dimensionless unitary-angle range"):
        certify_tdhf_scalar_curvature(
            approval=approval,
            sector=sector,
            tangent_basis=basis,
            energy_callback=constant_energy,
            functional_manifest=manifest,
        )
    assert calls == 0


def test_huge_energy_cancellation_roundoff_cannot_pass_incomplete_inventory() -> None:
    sector, hamiltonian, _, _, _ = _exact_sector()
    basis = _basis(sector)
    calls = 0

    def huge_offset_energy(projector: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return 1.0e12 + _energy(projector, hamiltonian)

    manifest = make_tdhf_scalar_functional_manifest(
        energy_callback=huge_offset_energy,
        source_functional_fingerprint=sha256(
            b"huge-offset cancellation negative"
        ).hexdigest(),
        immutable_callback_input_fingerprint=fingerprint_tdhf_matrix(hamiltonian),
        provenance="Huge energy-zero cancellation non-vacuity canary.",
    )
    approval = make_tdhf_scalar_curvature_approval(
        sector=sector,
        tangent_basis=basis,
        directions=(canonical_tdhf_scalar_directions(2, 2)[0],),
        energy_callback=huge_offset_energy,
        functional_manifest=manifest,
        energy_convention=_convention(),
        step_ladder=_ladder(),
        interaction_fingerprint=sector.interaction_fingerprint,
        provenance="Incomplete inventory must not inherit a huge roundoff allowance.",
    )
    with pytest.raises(ValueError, match="vacuous derived roundoff allowance"):
        certify_tdhf_scalar_curvature(
            approval=approval,
            sector=sector,
            tangent_basis=basis,
            energy_callback=huge_offset_energy,
            functional_manifest=manifest,
        )
    assert calls == 5


@pytest.mark.parametrize("roundoff_multiplier", (0.0, 1.0e-12, 128.0, 4096.0))
def test_v1_roundoff_multiplier_rejects_zero_small_and_other_values(
    roundoff_multiplier: float,
) -> None:
    assert TDHF_SCALAR_CURVATURE_V1_ROUNDOFF_MULTIPLIER == 256.0
    with pytest.raises(ValueError, match="roundoff_multiplier.*exact locked v1 value 256.0"):
        replace(_ladder().tolerances, roundoff_multiplier=roundoff_multiplier)

def test_huge_tolerance_approval_is_rejected_by_locked_v1_maxima() -> None:
    with pytest.raises(ValueError, match="locked v1|vacuous"):
        TDHFScalarCurvatureTolerances(
            stationarity_absolute=1.0e30,
            stationarity_relative=1.0e30,
            curvature_absolute=1.0e30,
            curvature_relative=1.0e30,
            roundoff_multiplier=1.0e30,
            projector_tolerance=1.0e30,
            matrix_absolute=1.0e30,
            matrix_relative=1.0e30,
        )


def test_functional_manifest_hamiltonian_hash_changes_approval_and_certificate() -> None:
    first, sector, basis, hamiltonian = _certified_case()
    shifted_hamiltonian = hamiltonian + 0.123 * np.eye(len(STATES))
    second = _certify(sector, basis, shifted_hamiltonian)

    assert first.h_plus_fingerprint == second.h_plus_fingerprint
    assert first.functional_manifest.implementation_fingerprint == (
        second.functional_manifest.implementation_fingerprint
    )
    assert first.functional_manifest.immutable_callback_input_fingerprint != (
        second.functional_manifest.immutable_callback_input_fingerprint
    )
    assert first.approval_fingerprint != second.approval_fingerprint
    assert first.fingerprint != second.fingerprint


def test_reporting_denominator_never_changes_raw_mathematical_gate() -> None:
    raw_certificate, sector, basis, hamiltonian = _certified_case(1.0)
    reported_certificate = _certify(
        sector, basis, hamiltonian, denominator=7.0
    )

    assert raw_certificate.convention_fingerprint != reported_certificate.convention_fingerprint
    assert not raw_certificate.normalization_physics_certified
    assert not reported_certificate.normalization_physics_certified
    for raw_direction, reported_direction in zip(
        raw_certificate.direction_evidence,
        reported_certificate.direction_evidence,
    ):
        assert raw_direction.raw_target_curvature == reported_direction.raw_target_curvature
        assert reported_direction.reported_target_curvature == pytest.approx(
            raw_direction.raw_target_curvature / 7.0
        )
        for raw_step, reported_step in zip(
            raw_direction.steps, reported_direction.steps
        ):
            assert raw_step.raw_second_derivative == reported_step.raw_second_derivative
            assert raw_step.curvature_residual == reported_step.curvature_residual
            assert raw_step.curvature_bound == reported_step.curvature_bound
            assert reported_step.reported_second_derivative == pytest.approx(
                raw_step.raw_second_derivative / 7.0
            )


def test_public_certification_is_gauge_covariant_with_consistent_blocks() -> None:
    _, sector, basis, hamiltonian = _certified_case()
    base_certificate = _certify(
        sector, basis, hamiltonian, directions=_directions()
    )
    assert base_certificate.registered_direction_curvatures_match
    assert not base_certificate.mathematical_scalar_hessian_match
    phases_plus = np.exp(1j * np.array([0.37, -0.51]))
    phases_minus = np.exp(1j * np.array([-0.29, 0.63]))
    p = np.diag(phases_plus)
    m = np.diag(phases_minus)
    blocks = TDHFSignedQBlocks(
        plus_pairs=sector.blocks.plus_pairs,
        minus_pairs=sector.blocks.minus_pairs,
        A_plus=p.conj() @ sector.blocks.A_plus @ p,
        B_plus_minus=p.conj() @ sector.blocks.B_plus_minus @ m.conj(),
        A_minus=m.conj() @ sector.blocks.A_minus @ m,
        B_minus_plus=m.conj() @ sector.blocks.B_minus_plus @ p.conj(),
    )
    rotated_sector = _sector_from_blocks(blocks)
    rotated_basis = TDHFTransitionTangentBasis(
        source_projector=basis.source_projector,
        plus_tangents=tuple(
            phase * tangent
            for phase, tangent in zip(phases_plus, basis.plus_tangents)
        ),
        minus_tangents=tuple(
            phase * tangent
            for phase, tangent in zip(phases_minus, basis.minus_tangents)
        ),
        source_fingerprint=SOURCE_FINGERPRINT,
        plus_pairs_fingerprint=basis.plus_pairs_fingerprint,
        minus_pairs_fingerprint=basis.minus_pairs_fingerprint,
    )
    rotated_directions = tuple(
        TDHFPhysicalDirection(
            direction.label,
            direction.x / phases_plus,
            direction.y / phases_minus,
        )
        for direction in _directions()
    )
    for original, rotated in zip(_directions(), rotated_directions):
        for parameter in (-0.07, 0.0, 0.09):
            np.testing.assert_allclose(
                exact_tdhf_projector_path(basis, original, parameter),
                exact_tdhf_projector_path(rotated_basis, rotated, parameter),
                atol=2.0e-13,
            )
    rotated_certificate = _certify(
        rotated_sector,
        rotated_basis,
        hamiltonian,
        directions=rotated_directions,
    )
    assert rotated_certificate.tangent_basis_fingerprint != base_certificate.tangent_basis_fingerprint
    assert rotated_certificate.h_plus_fingerprint != base_certificate.h_plus_fingerprint
    for original, rotated in zip(
        base_certificate.direction_evidence,
        rotated_certificate.direction_evidence,
    ):
        assert rotated.raw_target_curvature == pytest.approx(
            original.raw_target_curvature, abs=3.0e-13
        )
        np.testing.assert_allclose(
            rotated.raw_curvature_plateau,
            original.raw_curvature_plateau,
            atol=3.0e-12,
        )


def test_public_factory_rejects_nonstationarity_factor_two_a0_and_b_errors() -> None:
    sector, hamiltonian, plus, _, one_body = _exact_sector()
    basis = _basis(sector)

    nonstationary = hamiltonian + 0.07 * (plus[0] + plus[0].conj().T)
    with pytest.raises(ValueError, match="stationarity"):
        _certify(sector, basis, nonstationary, directions=(_directions()[0],))

    half = _wrong_sector(
        sector,
        a_plus=0.5 * sector.blocks.A_plus,
        b_plus_minus=0.5 * sector.blocks.B_plus_minus,
        a_minus=0.5 * sector.blocks.A_minus,
        b_minus_plus=0.5 * sector.blocks.B_minus_plus,
    )
    with pytest.raises(ValueError, match="raw curvature"):
        _certify(half, _basis(half), hamiltonian)

    plus_gaps = np.diag([one_body[p] - one_body[h] for p, h in PLUS])
    minus_gaps = np.diag([one_body[p] - one_body[h] for p, h in MINUS])
    missing_a0 = _wrong_sector(
        sector,
        a_plus=sector.blocks.A_plus - plus_gaps,
        a_minus=sector.blocks.A_minus - minus_gaps,
    )
    with pytest.raises(ValueError, match="raw curvature"):
        _certify(missing_a0, _basis(missing_a0), hamiltonian)

    # Supplying A(-q)* as A(-q) emulates the wrong y/y* lane convention;
    # structure remains Hermitian, so only the scalar oracle can reject it.
    wrong_y_convention = _wrong_sector(
        sector,
        a_minus=sector.blocks.A_minus.conj(),
    )
    with pytest.raises(ValueError, match="raw curvature"):
        _certify(wrong_y_convention, _basis(wrong_y_convention), hamiltonian)

    wrong_sign = _wrong_sector(
        sector,
        b_plus_minus=-sector.blocks.B_plus_minus,
        b_minus_plus=-sector.blocks.B_minus_plus,
    )
    with pytest.raises(ValueError, match="raw curvature"):
        _certify(wrong_sign, _basis(wrong_sign), hamiltonian)

    conjugated = sector.blocks.B_plus_minus.conj()
    wrong_conjugation = _wrong_sector(
        sector,
        b_plus_minus=conjugated,
        b_minus_plus=conjugated.T,
    )
    with pytest.raises(ValueError, match="raw curvature"):
        _certify(wrong_conjugation, _basis(wrong_conjugation), hamiltonian)


def test_public_factory_rejects_y_lane_source_pair_and_callback_mutations() -> None:
    sector, hamiltonian, _, _, _ = _exact_sector()
    basis = _basis(sector)
    lane_swapped = TDHFTransitionTangentBasis(
        source_projector=basis.source_projector,
        plus_tangents=basis.plus_tangents,
        minus_tangents=tuple(reversed(basis.minus_tangents)),
        source_fingerprint=basis.source_fingerprint,
        plus_pairs_fingerprint=basis.plus_pairs_fingerprint,
        minus_pairs_fingerprint=basis.minus_pairs_fingerprint,
    )
    with pytest.raises(ValueError, match="raw curvature"):
        _certify(sector, lane_swapped, hamiltonian)

    wrong_y = TDHFPhysicalDirection(
        "wrong-y-convention",
        _directions()[-1].x,
        _directions()[-1].y.conj(),
    )
    assert not np.allclose(
        exact_tdhf_projector_path(basis, _directions()[-1], 0.08),
        exact_tdhf_projector_path(basis, wrong_y, 0.08),
    )

    crossed_source = replace(sector, source_fingerprint="a" * 64)
    with pytest.raises(ValueError, match="source_fingerprint|sewing source"):
        _certify(crossed_source, basis, hamiltonian)

    stale_sewing = replace(sector.sewing, plus_pairs_fingerprint="b" * 64)
    with pytest.raises(ValueError, match=r"sewing \+q pair"):
        _certify(replace(sector, sewing=stale_sewing), basis, hamiltonian)

    wrong_source_basis = TDHFTransitionTangentBasis(
        source_projector=basis.source_projector,
        plus_tangents=basis.plus_tangents,
        minus_tangents=basis.minus_tangents,
        source_fingerprint="c" * 64,
        plus_pairs_fingerprint=basis.plus_pairs_fingerprint,
        minus_pairs_fingerprint=basis.minus_pairs_fingerprint,
    )
    with pytest.raises(ValueError, match="tangent basis source_fingerprint"):
        _certify(sector, wrong_source_basis, hamiltonian)

    def mutating_callback(projector: np.ndarray) -> float:
        projector[0, 0] += 1.0
        return 0.0

    manifest = make_tdhf_scalar_functional_manifest(
        energy_callback=mutating_callback,
        source_functional_fingerprint=sha256(b"mutating callback test").hexdigest(),
        immutable_callback_input_fingerprint=fingerprint_tdhf_matrix(hamiltonian),
        provenance="Deliberately mutating negative test callback.",
    )
    approval = make_tdhf_scalar_curvature_approval(
        sector=sector,
        tangent_basis=basis,
        directions=(_directions()[0],),
        energy_callback=mutating_callback,
        functional_manifest=manifest,
        energy_convention=_convention(),
        step_ladder=_ladder(),
        interaction_fingerprint=INTERACTION_FINGERPRINT,
        provenance="Detached mutation-test approval.",
    )
    with pytest.raises(ValueError, match="mutate|read-only"):
        certify_tdhf_scalar_curvature(
            approval=approval,
            sector=sector,
            tangent_basis=basis,
            energy_callback=mutating_callback,
            functional_manifest=manifest,
        )


def test_factory_status_and_certificate_cannot_be_publicly_constructed() -> None:
    with pytest.raises(TypeError, match="_factory_token"):
        TDHFScalarCurvatureFactoryStatus(  # type: ignore[call-arg]
            scalar_curvature_executed=True,
            stationarity_complete_all_passed=True,
            registered_direction_curvatures_match=True,
            mathematical_scalar_hessian_match=True,
            mathematical_scalar_curvature_match=True,
            static_hessian_authority_promoted=False,
            promotion_eligible=False,
            authority="raw_mathematical_scalar_hessian_match",
        )
    with pytest.raises(TypeError, match="private factory token"):
        TDHFScalarCurvatureFactoryStatus(
            _factory_token=object(),
            scalar_curvature_executed=True,
            stationarity_complete_all_passed=True,
            registered_direction_curvatures_match=True,
            mathematical_scalar_hessian_match=True,
            mathematical_scalar_curvature_match=True,
            static_hessian_authority_promoted=False,
            promotion_eligible=False,
            authority="raw_mathematical_scalar_hessian_match",
        )

    certificate, _, _, _ = _certified_case()
    kwargs = {item.name: getattr(certificate, item.name) for item in fields(certificate)}
    with pytest.raises(TypeError, match="private factory token"):
        TDHFScalarCurvatureCertificate(
            _factory_token=object(),
            **kwargs,
        )
