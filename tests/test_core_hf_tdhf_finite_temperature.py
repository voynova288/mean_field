from __future__ import annotations

import ast
from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import mean_field.core.hf.tdhf_finite_temperature as finite_tdhf
from mean_field.core.hf.tdhf_finite_temperature import (
    EnergyOrientedTransitionEndpoints,
    FiniteTemperatureTDHFManifest,
    FiniteTemperatureTDHFSelfConjugateAssembly,
    FiniteTemperatureTDHFSignedQAssembly,
    FiniteTemperatureTransitionWeights,
    assemble_finite_temperature_self_conjugate_tdhf,
    assemble_finite_temperature_signed_q_tdhf,
    build_finite_temperature_transition_weights,
    make_finite_temperature_tdhf_manifest,
)
from mean_field.core.hf.tdhf_signed import (
    TDHFGenericSignedQ,
    TDHFGenericSignedQSector,
    TDHFNambuSewing,
    TDHFSelfConjugateQ,
    TDHFSelfConjugateQSector,
    TDHFSignedQBlocks,
    TDHFTransitionLabelProtocol,
    build_standard_nambu_sewing,
    build_tdhf_signed_q_matrices,
    fingerprint_tdhf_pairs,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


SOURCE = _sha("same-parent-full-HF-eigenbasis")
INTERACTION = _sha("same-parent-interaction-derivative")
QUADRATURE = "source-column quadrature included exactly once in K"
RESPONSE_SCOPE = "collisionless-full-HF-density-response"
FIXED_INTERACTION_CONVENTION = (
    "physical_density_derivative_source_weight_once_v1"
)


def _manifest(
    *,
    source: str = SOURCE,
    interaction: str = INTERACTION,
    quadrature: str = QUADRATURE,
    response_scope: str = RESPONSE_SCOPE,
) -> FiniteTemperatureTDHFManifest:
    return make_finite_temperature_tdhf_manifest(
        source_eigenbasis_fingerprint=source,
        interaction_derivative_fingerprint=interaction,
        quadrature_convention=quadrature,
        response_scope=response_scope,
    )


def _endpoints(
    particle: int,
    hole: int,
    *,
    momentum_label: str | None = "q",
    with_flavor: bool = False,
) -> EnergyOrientedTransitionEndpoints:
    return EnergyOrientedTransitionEndpoints(
        particle_index=particle,
        hole_index=hole,
        particle_momentum=(momentum_label, "higher", particle),
        hole_momentum=(momentum_label, "lower", hole),
        particle_flavor=("full-HF", particle) if with_flavor else None,
        hole_flavor=("full-HF", hole) if with_flavor else None,
    )


def _weights(
    endpoints: tuple[EnergyOrientedTransitionEndpoints, ...],
    energies: np.ndarray,
    weights: np.ndarray,
    *,
    manifest: FiniteTemperatureTDHFManifest | None = None,
    scope: str = "+q/full-HF",
    mu: float = 0.0,
    thermal_energy: float = 0.7,
    supplied_occupations: np.ndarray | None = None,
) -> FiniteTemperatureTransitionWeights:
    return build_finite_temperature_transition_weights(
        endpoints,
        energies,
        mu=mu,
        thermal_energy=thermal_energy,
        transition_quadrature_weights=weights,
        manifest=_manifest() if manifest is None else manifest,
        transition_scope=scope,
        supplied_occupations=supplied_occupations,
    )


def _assemble_signed(
    plus: FiniteTemperatureTransitionWeights,
    minus: FiniteTemperatureTransitionWeights,
    K_A_plus: np.ndarray,
    K_B_plus_minus: np.ndarray,
    K_A_minus: np.ndarray,
    K_B_minus_plus: np.ndarray,
    *,
    raise_on_structure_error: bool = False,
) -> FiniteTemperatureTDHFSignedQAssembly:
    return assemble_finite_temperature_signed_q_tdhf(
        plus,
        minus,
        K_A_plus,
        K_B_plus_minus,
        K_A_minus,
        K_B_minus_plus,
        raise_on_structure_error=raise_on_structure_error,
    )


def test_manifest_is_strict_factory_built_and_binds_complete_contract() -> None:
    manifest = _manifest()
    assert manifest.interaction_block_convention == FIXED_INTERACTION_CONVENTION
    assert len(manifest.fingerprint) == 64
    assert manifest.fingerprint == _manifest().fingerprint

    variants = (
        _manifest(source=_sha("other source")),
        _manifest(interaction=_sha("other interaction")),
        _manifest(quadrature="other quadrature"),
        _manifest(response_scope="other response scope"),
    )
    assert all(item.fingerprint != manifest.fingerprint for item in variants)

    with pytest.raises(ValueError, match="interaction_block_convention"):
        make_finite_temperature_tdhf_manifest(
            source_eigenbasis_fingerprint=SOURCE,
            interaction_derivative_fingerprint=INTERACTION,
            quadrature_convention=QUADRATURE,
            response_scope=RESPONSE_SCOPE,
            interaction_block_convention="legacy-free-text",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="public factory"):
        FiniteTemperatureTDHFManifest(
            source_eigenbasis_fingerprint=SOURCE,
            interaction_derivative_fingerprint=INTERACTION,
            quadrature_convention=QUADRATURE,
            response_scope=RESPONSE_SCOPE,
            interaction_block_convention=FIXED_INTERACTION_CONVENTION,
            _factory_token=object(),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        ["mutable-list"],
        {"mutable": "mapping"},
        ("nested", ["mutable-list"]),
        np.asarray([1, 2]),
    ],
)
def test_endpoint_metadata_rejects_mutable_or_non_json_objects(metadata: object) -> None:
    with pytest.raises(TypeError, match="deeply immutable JSON-like"):
        EnergyOrientedTransitionEndpoints(
            particle_index=1,
            hole_index=0,
            particle_momentum=metadata,
        )


def test_energy_endpoints_implement_neutral_protocol_without_flavor_classifier() -> None:
    endpoint = EnergyOrientedTransitionEndpoints(
        particle_index=np.int64(3),
        hole_index=np.int32(1),
        particle_momentum=("k", np.int64(2), (True, 0.25)),
        hole_momentum=None,
    )
    assert isinstance(endpoint, TDHFTransitionLabelProtocol)
    assert endpoint.particle == 3
    assert endpoint.hole == 1
    assert endpoint.particle_flavor is None
    assert endpoint.hole_flavor is None
    assert endpoint.particle_momentum == ("k", 2, (True, 0.25))
    assert isinstance(endpoint.particle_momentum[1], int)


@pytest.mark.parametrize(
    ("keyword", "bad_value", "message"),
    [
        ("source_eigenbasis_fingerprint", "A" * 64, "64 lowercase hexadecimal"),
        ("interaction_derivative_fingerprint", "0" * 63, "64 lowercase hexadecimal"),
        ("quadrature_convention", "", "nonempty"),
        ("response_scope", "  ", "nonempty"),
    ],
)
def test_manifest_provenance_gates(
    keyword: str, bad_value: str, message: str
) -> None:
    arguments = {
        "source_eigenbasis_fingerprint": SOURCE,
        "interaction_derivative_fingerprint": INTERACTION,
        "quadrature_convention": QUADRATURE,
        "response_scope": RESPONSE_SCOPE,
    }
    arguments[keyword] = bad_value
    with pytest.raises(ValueError, match=message):
        make_finite_temperature_tdhf_manifest(**arguments)


def test_stable_fermi_difference_readonly_vectors_and_factory_only_weights() -> None:
    energies = np.asarray([-0.8, 0.35, 1.4])
    endpoints = (_endpoints(1, 0), _endpoints(2, 0))
    mu = 0.13
    thermal_energy = 0.41
    quadrature_weights = np.asarray([0.2, 1.7])
    layer = _weights(
        endpoints,
        energies,
        quadrature_weights,
        mu=mu,
        thermal_energy=thermal_energy,
    )

    occupations = 1.0 / (1.0 + np.exp((energies - mu) / thermal_energy))
    expected_d = np.asarray(
        [
            occupations[item.hole] - occupations[item.particle]
            for item in endpoints
        ]
    )
    np.testing.assert_allclose(layer.d, expected_d, rtol=3e-15, atol=0.0)
    np.testing.assert_allclose(layer.log_d, np.log(expected_d), rtol=3e-15, atol=0.0)
    np.testing.assert_allclose(layer.sqrt_d, np.sqrt(expected_d), rtol=3e-15, atol=0.0)
    assert layer.endpoints == endpoints
    assert len(layer.fingerprint) == 64
    assert not layer.d.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        layer.d[0] = 0.0

    with pytest.raises(TypeError, match="public factory"):
        FiniteTemperatureTransitionWeights(
            manifest=layer.manifest,
            transition_scope=layer.transition_scope,
            mu=layer.mu,
            thermal_energy=layer.thermal_energy,
            transitions=layer.transitions,
            _factory_token=object(),
        )


def test_every_factory_only_record_rejects_direct_init_and_dataclass_replace() -> None:
    manifest = _manifest()
    plus = _weights(
        (_endpoints(1, 0, momentum_label="+q"),),
        np.asarray([-1.0, 1.0]),
        np.ones(1),
        manifest=manifest,
        scope="+q/factory-only",
    )
    minus = _weights(
        (_endpoints(1, 0, momentum_label="-q"),),
        np.asarray([-1.0, 1.0]),
        np.ones(1),
        manifest=manifest,
        scope="-q/factory-only",
    )
    zeros = np.zeros((1, 1))
    self_assembly = assemble_finite_temperature_self_conjugate_tdhf(
        plus, zeros, zeros
    )
    signed_assembly = _assemble_signed(
        plus, minus, zeros, zeros, zeros, zeros
    )
    records = (
        manifest,
        plus.transitions[0],
        plus,
        self_assembly,
        signed_assembly,
    )
    assert isinstance(plus.transitions[0], finite_tdhf._DirectedThermalTransition)
    for record in records:
        assert type(record).__dataclass_params__.init is False
        with pytest.raises(TypeError, match="public factory"):
            type(record)()
        with pytest.raises(TypeError, match="public factory"):
            replace(record)
        replaceable = next(item for item in fields(record) if item.init)
        with pytest.raises(TypeError, match="public factory"):
            replace(
                record,
                **{replaceable.name: getattr(record, replaceable.name)},
            )


def test_underflow_mixed_rows_and_columns_assemble_from_log_scales() -> None:
    layer = _weights(
        (_endpoints(1, 0), _endpoints(3, 2)),
        np.asarray([-1.0, 1.0, 800.0, 801.0]),
        np.asarray([0.3, 1.7]),
        thermal_energy=1.0,
        scope="self-conjugate/mixed-underflow",
    )
    assert layer.d[0] > 0.0
    assert layer.d[1] == 0.0
    assert layer.sqrt_d[1] > 0.0

    K_A = np.asarray(
        [[0.2, -3.0e100], [1.0e100, -0.1]], dtype=np.complex128
    )
    K_B = np.zeros((2, 2), dtype=np.complex128)
    expected_interaction = (
        layer.assembly_left_scale()[:, None]
        * K_A
        * layer.assembly_right_scale()[None, :]
    )
    assembly = assemble_finite_temperature_self_conjugate_tdhf(layer, K_A, K_B)
    np.testing.assert_allclose(
        assembly.A,
        np.diag(layer.gaps) + expected_interaction,
        rtol=5e-15,
        atol=0.0,
    )
    assert assembly.A[0, 1] != 0.0
    assert assembly.A[1, 0] != 0.0
    assert np.all(np.isfinite(assembly.A))
    assert not assembly.A.flags.writeable

    saved_A = assembly.A.copy()
    K_A[1, 0] = 0.0
    np.testing.assert_array_equal(assembly.A, saved_A)

    with pytest.raises(ValueError, match="shape"):
        assemble_finite_temperature_self_conjugate_tdhf(
            layer, np.zeros((2, 1)), K_B
        )
    malformed = np.zeros((2, 2))
    malformed[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        assemble_finite_temperature_self_conjugate_tdhf(layer, malformed, K_B)
    with pytest.raises(TypeError, match="rectangular numeric matrix"):
        assemble_finite_temperature_self_conjugate_tdhf(
            layer, [[1.0], [1.0, 2.0]], K_B
        )


@pytest.mark.parametrize(
    "bad_value",
    [complex(np.nan, 0.0), complex(np.inf, 0.0), complex(0.0, -np.inf)],
)
@pytest.mark.parametrize("bad_block_index", range(4))
def test_signed_assembly_rejects_nonfinite_values_in_every_k_block(
    bad_value: complex,
    bad_block_index: int,
) -> None:
    manifest = _manifest()
    energies = np.asarray([-1.0, 1.0])
    plus = _weights(
        (_endpoints(1, 0, momentum_label="+q"),),
        energies,
        np.ones(1),
        manifest=manifest,
        scope="+q/nonfinite-K",
    )
    minus = _weights(
        (_endpoints(1, 0, momentum_label="-q"),),
        energies,
        np.ones(1),
        manifest=manifest,
        scope="-q/nonfinite-K",
    )
    derivatives = [np.zeros((1, 1), dtype=np.complex128) for _ in range(4)]
    derivatives[bad_block_index][0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        _assemble_signed(plus, minus, *derivatives)


def test_coordinate_roundtrip_uses_independent_sqrt_d() -> None:
    layer = _weights(
        (_endpoints(2, 0), _endpoints(3, 1)),
        np.asarray([-1.4, -0.2, 0.5, 1.1]),
        np.asarray([0.07, 2.3]),
        thermal_energy=0.6,
    )
    z = np.asarray([0.3 + 0.7j, -1.2 + 0.1j])
    x = layer.physical_to_thermal_euclidean(z)
    np.testing.assert_allclose(
        x,
        np.sqrt(layer.weights) / layer.sqrt_d * z,
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        layer.thermal_euclidean_to_physical(x),
        z,
        rtol=2e-15,
        atol=2e-15,
    )


def test_complete_signed_nambu_uses_neutral_endpoints_directly() -> None:
    manifest = _manifest()
    energies = np.asarray([-1.2, -0.3, 0.4, 1.7])
    plus = _weights(
        (_endpoints(2, 0, momentum_label="+q"), _endpoints(3, 1, momentum_label="+q")),
        energies,
        np.asarray([0.13, 2.1]),
        manifest=manifest,
        scope="+q/full-HF",
        thermal_energy=0.55,
    )
    minus = _weights(
        (_endpoints(2, 0, momentum_label="-q"), _endpoints(3, 1, momentum_label="-q")),
        energies,
        np.asarray([2.4, 0.11]),
        manifest=_manifest(),
        scope="-q/full-HF",
        thermal_energy=0.55,
    )

    C_plus = np.asarray([[0.4, -0.7 + 0.2j], [-0.7 - 0.2j, -0.2]])
    C_minus = np.asarray([[-0.1, 0.3 - 0.5j], [0.3 + 0.5j, 0.6]])
    target_B = np.asarray([[0.2 + 0.1j, -0.8j], [0.7 - 0.3j, -0.4 + 0.2j]])
    lp = plus.assembly_left_scale()
    rp = plus.assembly_right_scale()
    lm = minus.assembly_left_scale()
    rm = minus.assembly_right_scale()
    K_A_plus = C_plus / lp[:, None] / rp[None, :]
    K_B_plus_minus = target_B / lp[:, None] / rm[None, :]
    K_A_minus = C_minus / lm[:, None] / rm[None, :]
    K_B_minus_plus = target_B.T / lm[:, None] / rp[None, :]

    assembly = _assemble_signed(
        plus,
        minus,
        K_A_plus,
        K_B_plus_minus,
        K_A_minus,
        K_B_minus_plus,
        raise_on_structure_error=True,
    )
    assert assembly.manifest.fingerprint == manifest.fingerprint
    assert assembly.plus_endpoints == plus.endpoints
    assert assembly.minus_endpoints == minus.endpoints
    assert assembly.structure.ok
    assert len(assembly.assembly_fingerprint) == 64

    q = TDHFGenericSignedQ(
        plus_raw=(1, 0),
        minus_raw=(-1, 0),
        plus_canonical=(1, 0),
        minus_canonical=(9, 0),
        provenance="finite-T-test-grid",
    )
    sewing = build_standard_nambu_sewing(
        plus.endpoints,
        minus.endpoints,
        source_fingerprint=manifest.source_eigenbasis_fingerprint,
        construction="finite_T_energy_oriented_neutral_labels_v1",
    )
    sector = assembly.to_typed_generic_sector(q, sewing)
    assert isinstance(sector, TDHFGenericSignedQSector)
    assert sector.source_fingerprint == manifest.source_eigenbasis_fingerprint
    assert sector.interaction_fingerprint == assembly.assembly_fingerprint
    assert sector.interaction_fingerprint != manifest.interaction_derivative_fingerprint
    assert sector.static_hessian_authority == "not_established"
    response_scope = json.loads(sector.response_scope)
    assert response_scope == {
        "manifest_response_scope": manifest.response_scope,
        "transition_scopes": [
            {"lane": "+q", "scope": plus.transition_scope},
            {"lane": "-q", "scope": minus.transition_scope},
        ],
    }
    blocks = sector.blocks
    assert isinstance(blocks, TDHFSignedQBlocks)
    assert blocks.plus_pairs == plus.endpoints
    assert blocks.minus_pairs == minus.endpoints
    assert all(label.particle_flavor is None for label in blocks.plus_pairs)
    matrices = build_tdhf_signed_q_matrices(
        blocks,
        sewing,
        raise_on_structure_error=True,
    )

    expected_A_plus = np.diag(plus.gaps) + lp[:, None] * K_A_plus * rp[None, :]
    expected_B_plus = lp[:, None] * K_B_plus_minus * rm[None, :]
    expected_A_minus = np.diag(minus.gaps) + lm[:, None] * K_A_minus * rm[None, :]
    expected_B_minus = lm[:, None] * K_B_minus_plus * rp[None, :]
    expected_H_plus = np.block(
        [
            [expected_A_plus, expected_B_plus],
            [np.conj(expected_B_minus), np.conj(expected_A_minus)],
        ]
    )
    expected_H_minus = np.block(
        [
            [expected_A_minus, expected_B_minus],
            [np.conj(expected_B_plus), np.conj(expected_A_plus)],
        ]
    )
    np.testing.assert_allclose(matrices.H_plus, expected_H_plus)
    np.testing.assert_allclose(matrices.H_minus, expected_H_minus)

    z_plus = np.asarray([0.4 - 0.2j, -0.3 + 0.8j])
    z_minus_star = np.asarray([-0.5 + 0.1j, 0.6 + 0.4j])
    density_H_plus_action = np.concatenate(
        [
            plus.gaps * z_plus
            + plus.d * (K_A_plus @ z_plus + K_B_plus_minus @ z_minus_star),
            minus.gaps * z_minus_star
            + minus.d
            * (
                np.conj(K_B_minus_plus) @ z_plus
                + np.conj(K_A_minus) @ z_minus_star
            ),
        ]
    )
    euclidean_H_plus_action = np.concatenate(
        [
            plus.physical_to_thermal_euclidean(density_H_plus_action[:2]),
            minus.physical_to_thermal_euclidean(density_H_plus_action[2:]),
        ]
    )
    x_plus = np.concatenate(
        [
            plus.physical_to_thermal_euclidean(z_plus),
            minus.physical_to_thermal_euclidean(z_minus_star),
        ]
    )
    np.testing.assert_allclose(
        matrices.H_plus @ x_plus,
        euclidean_H_plus_action,
        rtol=3e-14,
        atol=3e-14,
    )

    y_minus = np.asarray([-0.2 + 0.9j, 0.7 - 0.1j])
    y_plus_star = np.asarray([0.6 + 0.3j, -0.4 - 0.8j])
    density_H_minus_action = np.concatenate(
        [
            minus.gaps * y_minus
            + minus.d * (K_A_minus @ y_minus + K_B_minus_plus @ y_plus_star),
            plus.gaps * y_plus_star
            + plus.d
            * (
                np.conj(K_B_plus_minus) @ y_minus
                + np.conj(K_A_plus) @ y_plus_star
            ),
        ]
    )
    euclidean_H_minus_action = np.concatenate(
        [
            minus.physical_to_thermal_euclidean(density_H_minus_action[:2]),
            plus.physical_to_thermal_euclidean(density_H_minus_action[2:]),
        ]
    )
    x_minus = np.concatenate(
        [
            minus.physical_to_thermal_euclidean(y_minus),
            plus.physical_to_thermal_euclidean(y_plus_star),
        ]
    )
    np.testing.assert_allclose(
        matrices.H_minus @ x_minus,
        euclidean_H_minus_action,
        rtol=3e-14,
        atol=3e-14,
    )

    expected_eta_plus = np.diag([1.0, 1.0, -1.0, -1.0])
    expected_eta_minus = np.diag([1.0, 1.0, -1.0, -1.0])
    expected_L_plus = expected_eta_plus @ expected_H_plus
    expected_L_minus = expected_eta_minus @ expected_H_minus
    np.testing.assert_array_equal(matrices.eta_plus, np.diag(expected_eta_plus))
    np.testing.assert_array_equal(matrices.eta_minus, np.diag(expected_eta_minus))
    np.testing.assert_allclose(matrices.L_plus, expected_L_plus)
    np.testing.assert_allclose(matrices.L_minus, expected_L_minus)
    np.testing.assert_allclose(
        matrices.L_plus @ x_plus,
        expected_eta_plus @ euclidean_H_plus_action,
    )
    np.testing.assert_allclose(
        matrices.L_minus @ x_minus,
        expected_eta_minus @ euclidean_H_minus_action,
    )

    identity = np.eye(2, dtype=np.complex128)
    zeros = np.zeros((2, 2), dtype=np.complex128)
    expected_plus_to_minus = np.block([[zeros, identity], [identity, zeros]])
    expected_minus_to_plus = np.conj(expected_plus_to_minus.T)
    np.testing.assert_array_equal(sewing.plus_to_minus, expected_plus_to_minus)
    np.testing.assert_array_equal(sewing.minus_to_plus, expected_minus_to_plus)
    np.testing.assert_allclose(
        expected_L_minus @ expected_plus_to_minus
        + expected_plus_to_minus @ np.conj(expected_L_plus),
        np.zeros((4, 4)),
        atol=3e-14,
    )
    np.testing.assert_allclose(
        expected_L_plus @ expected_minus_to_plus
        + expected_minus_to_plus @ np.conj(expected_L_minus),
        np.zeros((4, 4)),
        atol=3e-14,
    )


def test_typed_signed_conversion_rejects_unbound_sewing_and_binds_k_bytes() -> None:
    manifest = _manifest()
    energies = np.asarray([-1.0, 1.0])
    plus = _weights(
        (_endpoints(1, 0, momentum_label="+q"),),
        energies,
        np.ones(1),
        manifest=manifest,
        scope="+q/binding",
    )
    minus = _weights(
        (_endpoints(1, 0, momentum_label="-q"),),
        energies,
        np.ones(1),
        manifest=manifest,
        scope="-q/binding",
    )
    zero = np.zeros((1, 1))
    assembly = _assemble_signed(plus, minus, zero, zero, zero, zero)
    changed = _assemble_signed(
        plus,
        minus,
        np.asarray([[1.0e-6]]),
        zero,
        zero,
        zero,
    )
    assert changed.assembly_fingerprint != assembly.assembly_fingerprint
    q = TDHFGenericSignedQ(
        plus_raw=(1, 0),
        minus_raw=(-1, 0),
        plus_canonical=(1, 0),
        minus_canonical=(9, 0),
        provenance="binding-test",
    )
    sewing = build_standard_nambu_sewing(
        plus.endpoints,
        minus.endpoints,
        source_fingerprint=manifest.source_eigenbasis_fingerprint,
    )
    assert (
        changed.to_typed_generic_sector(q, sewing).interaction_fingerprint
        == changed.assembly_fingerprint
    )
    wrong_source = replace(sewing, source_fingerprint=_sha("wrong source"))
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        assembly.to_typed_generic_sector(q, wrong_source)
    wrong_pairs = replace(sewing, plus_pairs_fingerprint=_sha("wrong pairs"))
    with pytest.raises(ValueError, match="pair fingerprints mismatch"):
        assembly.to_typed_generic_sector(q, wrong_pairs)
    with pytest.raises(ValueError, match="static_hessian_authority"):
        assembly.to_typed_generic_sector(
            q,
            sewing,
            static_hessian_authority="invalid",  # type: ignore[arg-type]
        )


def test_manifest_mismatch_and_direct_assembly_forgery_are_rejected() -> None:
    energies = np.asarray([-1.0, 1.0])
    plus = _weights(
        (_endpoints(1, 0, momentum_label="+q"),),
        energies,
        np.ones(1),
        scope="+q",
    )
    minus = _weights(
        (_endpoints(1, 0, momentum_label="-q"),),
        energies,
        np.ones(1),
        manifest=_manifest(interaction=_sha("different interaction")),
        scope="-q",
    )
    zeros = np.zeros((1, 1))
    with pytest.raises(ValueError, match="exact same.*manifest fingerprint"):
        _assemble_signed(plus, minus, zeros, zeros, zeros, zeros)

    with pytest.raises(TypeError, match="public factory"):
        FiniteTemperatureTDHFSignedQAssembly(
            plus_transition_weights=plus,
            minus_transition_weights=plus,
            _K_A_plus=zeros,
            _K_B_plus_minus=zeros,
            _K_A_minus=zeros,
            _K_B_minus_plus=zeros,
            _factory_token=object(),
        )
    with pytest.raises(TypeError, match="public factory"):
        FiniteTemperatureTDHFSelfConjugateAssembly(
            transition_weights=plus,
            _K_A=zeros,
            _K_B=zeros,
            _factory_token=object(),
        )

    assert "interaction_derivative_fingerprint" not in inspect.signature(
        assemble_finite_temperature_self_conjugate_tdhf
    ).parameters
    with pytest.raises(TypeError, match="unexpected keyword"):
        assemble_finite_temperature_self_conjugate_tdhf(
            plus,
            zeros,
            zeros,
            interaction_derivative_fingerprint=INTERACTION,  # type: ignore[call-arg]
        )


def test_self_conjugate_structure_gate_and_low_temperature_limit() -> None:
    layer = _weights(
        (_endpoints(2, 0), _endpoints(3, 1)),
        np.asarray([-2.0, -1.0, 1.0, 2.0]),
        np.asarray([0.2, 2.5]),
        thermal_energy=1.0e-3,
        scope="self-conjugate",
    )
    np.testing.assert_array_equal(layer.d, np.ones(2))
    left = np.sqrt(layer.weights)
    right = 1.0 / np.sqrt(layer.weights)
    target_C = np.asarray([[0.2, -0.3j], [0.3j, -0.5]])
    target_B = np.asarray([[0.1j, 0.7], [0.7, 0.4]])
    K_A = target_C / left[:, None] / right[None, :]
    K_B = target_B / left[:, None] / right[None, :]
    assembly = assemble_finite_temperature_self_conjugate_tdhf(
        layer,
        K_A,
        K_B,
        raise_on_structure_error=True,
    )
    np.testing.assert_allclose(
        assembly.A,
        np.diag(layer.gaps) + left[:, None] * K_A * right[None, :],
    )
    np.testing.assert_allclose(
        assembly.B,
        left[:, None] * K_B * right[None, :],
    )
    q = TDHFSelfConjugateQ(
        plus_raw=(0, 0),
        minus_raw=(0, 0),
        canonical=(0, 0),
        provenance="finite-T-self-conjugate-test",
    )
    sector = assembly.to_typed_self_conjugate_sector(
        q,
        "canonical finite-T q=0 sewing",
    )
    assert isinstance(sector, TDHFSelfConjugateQSector)
    assert sector.canonical_pairs == layer.endpoints
    assert sector.source_fingerprint == layer.manifest.source_eigenbasis_fingerprint
    assert sector.interaction_fingerprint == assembly.assembly_fingerprint
    assert sector.static_hessian_authority == "not_established"
    assert json.loads(sector.response_scope) == {
        "manifest_response_scope": layer.manifest.response_scope,
        "transition_scopes": [
            {"lane": "self_conjugate", "scope": layer.transition_scope}
        ],
    }

    with pytest.raises(ValueError, match="structure gate failed"):
        assemble_finite_temperature_self_conjugate_tdhf(
            layer,
            K_A,
            K_B + np.asarray([[0.0, 0.7], [0.0, 0.0]]),
            raise_on_structure_error=True,
        )


def test_complex_inputs_gap_uniqueness_and_occupation_gates_fail_closed() -> None:
    endpoint = _endpoints(1, 0)
    base = np.asarray([-1.0, 1.0])
    with pytest.raises(ValueError, match="imaginary parts are forbidden"):
        _weights((endpoint,), base.astype(complex) + [0.0j, 1.0e-30j], np.ones(1))
    with pytest.raises(ValueError, match="imaginary parts are forbidden"):
        _weights((endpoint,), base, np.asarray([1.0 + 1.0e-30j]))
    with pytest.raises(ValueError, match="eps_particle > eps_hole"):
        _weights((_endpoints(0, 1),), base, np.ones(1))
    with pytest.raises(ValueError, match="duplicate"):
        _weights((endpoint, endpoint), base, np.ones(2))
    with pytest.raises(IndexError, match="full HF eigensystem"):
        _weights((_endpoints(3, 0),), base, np.ones(1))

    occupations = 1.0 / (1.0 + np.exp(base / 0.5))
    _weights(
        (endpoint,),
        base,
        np.ones(1),
        thermal_energy=0.5,
        supplied_occupations=occupations,
    )
    with pytest.raises(ValueError, match="supplied occupation mismatch"):
        _weights(
            (endpoint,),
            base,
            np.ones(1),
            thermal_energy=0.5,
            supplied_occupations=np.asarray([1.0, 0.0]),
        )


def test_module_is_system_agnostic_and_public_surface_is_minimal() -> None:
    module_path = (
        Path(__file__).parents[1]
        / "src"
        / "mean_field"
        / "core"
        / "hf"
        / "tdhf_finite_temperature.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any(
        module == "mean_field.systems"
        or module.startswith("mean_field.systems.")
        or module == "systems"
        or module.startswith("systems.")
        for module in imported_modules
    )

    expected = {
        "EnergyOrientedTransitionEndpoints",
        "FiniteTemperatureSelfConjugateStructureResiduals",
        "FiniteTemperatureSignedQStructureResiduals",
        "FiniteTemperatureTDHFManifest",
        "FiniteTemperatureTDHFSelfConjugateAssembly",
        "FiniteTemperatureTDHFSignedQAssembly",
        "FiniteTemperatureTransitionWeights",
        "assemble_finite_temperature_self_conjugate_tdhf",
        "assemble_finite_temperature_signed_q_tdhf",
        "build_finite_temperature_transition_weights",
        "make_finite_temperature_tdhf_manifest",
    }
    module = __import__(
        "mean_field.core.hf.tdhf_finite_temperature", fromlist=["__all__"]
    )
    assert set(module.__all__) == expected

    forbidden = {
        "DirectedThermalTransition",
        "energy_oriented_endpoints_to_particle_hole_pairs",
        "ThermalTransitionWeights",
        "assemble_finite_temperature_signed_q",
    }
    assert forbidden.isdisjoint(module.__all__)
