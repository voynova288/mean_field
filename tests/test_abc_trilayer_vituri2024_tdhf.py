"""Tiny reduced-faithful tests for diagnostic Vituri signed-q A/B assembly."""

from dataclasses import replace
import inspect

import numpy as np
import pytest

from mean_field.core.hf import (
    TDHFGenericSignedQ,
    analyze_tdhf_typed_sector,
    build_tdhf_signed_q_matrices,
    fingerprint_tdhf_matrix,
    fingerprint_tdhf_pairs,
    fingerprint_tdhf_sector,
)
import mean_field.systems.abc_trilayer as abc_module
import mean_field.systems.abc_trilayer.vituri2024_tdhf as tdhf_module
from mean_field.systems.abc_trilayer import (
    SM_TEX_SHA256,
    VITURI2024_TDHF_NO_GO_LIMITS,
    VITURI2024_TDHF_Q_PROVENANCE,
    VITURI2024_TDHF_RESPONSE_SCOPE,
    Vituri2024DiagonalHFTransitionReceipt,
    Vituri2024FiniteAreaReceipt,
    Vituri2024Flavor,
    Vituri2024InteractionChoiceReceipt,
    Vituri2024Orbital,
    Vituri2024SignedQTransitionInventoryPair,
    Vituri2024TDHFAssemblyContext,
    Vituri2024TransitionInventory,
    assemble_vituri2024_tdhf_signed_q,
    bind_vituri2024_interaction,
    vituri2024_rpa_a_element,
    vituri2024_rpa_b_element,
    vituri2024_tdhf_interaction_fingerprint,
)

_Q = (1.0 / 128.0, 2.0 / 128.0)
_SOURCE_SHA = "a" * 64
_SOURCE_TEXT = "Tiny immutable diagonal source; HF stationarity is not certified."
_KINEMATICS_SHA = "b" * 64
_KINEMATICS_TEXT = "Exact local continuum quartet; no torus or carry authority."
_AREA_SHA = "c" * 64
_DELTA1 = 0.028


def _orbital(momentum: tuple[float, float]) -> Vituri2024Orbital:
    return Vituri2024Orbital(
        flavor=Vituri2024Flavor(valley=1, spin=1),
        momentum_inverse_angstrom=momentum,
    )


def _transition(
    particle_momentum: tuple[float, float],
    hole_momentum: tuple[float, float],
    *,
    particle_energy: float,
    hole_energy: float,
    source_sha: str = _SOURCE_SHA,
) -> Vituri2024DiagonalHFTransitionReceipt:
    return Vituri2024DiagonalHFTransitionReceipt(
        particle=_orbital(particle_momentum),
        hole=_orbital(hole_momentum),
        particle_energy_ev=particle_energy,
        hole_energy_ev=hole_energy,
        source_artifact_sha256=source_sha,
        source_text=_SOURCE_TEXT,
    )


def _plus_transition() -> Vituri2024DiagonalHFTransitionReceipt:
    return _transition(_Q, (0.0, 0.0), particle_energy=0.43, hole_energy=-0.17)


def _minus_transition(
    *, source_sha: str = _SOURCE_SHA,
) -> Vituri2024DiagonalHFTransitionReceipt:
    return _transition(
        (2.0 / 128.0, -3.0 / 128.0),
        (3.0 / 128.0, -1.0 / 128.0),
        particle_energy=0.61,
        hole_energy=-0.31,
        source_sha=source_sha,
    )


def _area(value: float = 53.0) -> Vituri2024FiniteAreaReceipt:
    return Vituri2024FiniteAreaReceipt(
        area_angstrom_squared=value,
        provider_sha256=_AREA_SHA,
        source_text="Tiny caller-attested area; no real mesh or quadrature authority.",
    )


def _interaction() -> Vituri2024InteractionChoiceReceipt:
    return Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=250.0,
        coulomb_e2_ev_angstrom=14.3996454784255,
        q0_evaluation="analytic_kernel_limit_only",
        provider_sha256="d" * 64,
        source_sha256=SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text="Tiny interaction choice; not an HF q=0 background.",
    )


def _context(
    *,
    area_value: float = 53.0,
    bound_interaction: bool = False,
    kinematics_sha: str = _KINEMATICS_SHA,
    kinematics_text: str = _KINEMATICS_TEXT,
) -> Vituri2024TDHFAssemblyContext:
    interaction = _interaction()
    return Vituri2024TDHFAssemblyContext(
        area=_area(area_value),
        Delta1=_DELTA1,
        interaction=(
            bind_vituri2024_interaction(interaction)
            if bound_interaction
            else interaction
        ),
        kinematics_provider_sha256=kinematics_sha,
        kinematics_source_text=kinematics_text,
    )


def _signed_pair(
    *,
    minus_source_sha: str = _SOURCE_SHA,
    minus_area: float = 53.0,
    bound_interaction: bool = False,
) -> Vituri2024SignedQTransitionInventoryPair:
    return Vituri2024SignedQTransitionInventoryPair(
        plus_inventory=Vituri2024TransitionInventory(
            q_inverse_angstrom=_Q,
            transitions=(_plus_transition(),),
        ),
        minus_inventory=Vituri2024TransitionInventory(
            q_inverse_angstrom=(-_Q[0], -_Q[1]),
            transitions=(_minus_transition(source_sha=minus_source_sha),),
        ),
        plus_context=_context(bound_interaction=bound_interaction),
        minus_context=_context(
            area_value=minus_area, bound_interaction=bound_interaction
        ),
    )


def _direct_a(left, right):
    context = _context()
    return vituri2024_rpa_a_element(
        left,
        right,
        context.area,
        context.Delta1,
        context.interaction,
        kinematics_provider_sha256=context.kinematics_provider_sha256,
        kinematics_source_text=context.kinematics_source_text,
    )


def _direct_b(left, right):
    context = _context()
    return vituri2024_rpa_b_element(
        left,
        right,
        context.area,
        context.Delta1,
        context.interaction,
        kinematics_provider_sha256=context.kinematics_provider_sha256,
        kinematics_source_text=context.kinematics_source_text,
    )


def test_vituri_signed_q_assembles_all_four_local_lanes_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_a = tdhf_module.vituri2024_rpa_a_element
    original_b = tdhf_module.vituri2024_rpa_b_element
    calls: list[tuple[str, tuple[float, float], tuple[float, float]]] = []

    def record_a(left, right, *args, **kwargs):
        calls.append(("A", tdhf_module._transfer(left), tdhf_module._transfer(right)))
        return original_a(left, right, *args, **kwargs)

    def record_b(left, right, *args, **kwargs):
        calls.append(("B", tdhf_module._transfer(left), tdhf_module._transfer(right)))
        return original_b(left, right, *args, **kwargs)

    monkeypatch.setattr(tdhf_module, "vituri2024_rpa_a_element", record_a)
    monkeypatch.setattr(tdhf_module, "vituri2024_rpa_b_element", record_b)
    pair = _signed_pair()
    result = assemble_vituri2024_tdhf_signed_q(pair)

    q_minus = (-_Q[0], -_Q[1])
    assert calls == [
        ("A", _Q, _Q),
        ("B", _Q, q_minus),
        ("A", q_minus, q_minus),
        ("B", q_minus, _Q),
    ]
    plus = pair.plus_inventory.transitions[0]
    minus = pair.minus_inventory.transitions[0]
    assert result.blocks.A_plus[0, 0] == _direct_a(plus, plus).value
    assert result.blocks.B_plus_minus[0, 0] == _direct_b(plus, minus).value
    assert result.blocks.A_minus[0, 0] == _direct_a(minus, minus).value
    assert result.blocks.B_minus_plus[0, 0] == _direct_b(minus, plus).value

    element_fingerprints = (
        result.A_plus_element_fingerprints
        + result.B_plus_minus_element_fingerprints
        + result.A_minus_element_fingerprints
        + result.B_minus_plus_element_fingerprints
    )
    assert len(set(element_fingerprints)) == 4
    assert len(
        {
            result.A_plus_elements[0].kinematics_fingerprint,
            result.B_plus_minus_elements[0].kinematics_fingerprint,
            result.A_minus_elements[0].kinematics_fingerprint,
            result.B_minus_plus_elements[0].kinematics_fingerprint,
        }
    ) == 4
    assert result.A_plus_matrix_fingerprint == fingerprint_tdhf_matrix(
        result.blocks.A_plus
    )
    assert result.B_plus_minus_matrix_fingerprint == fingerprint_tdhf_matrix(
        result.blocks.B_plus_minus
    )
    assert result.A_minus_matrix_fingerprint == fingerprint_tdhf_matrix(
        result.blocks.A_minus
    )
    assert result.B_minus_plus_matrix_fingerprint == fingerprint_tdhf_matrix(
        result.blocks.B_minus_plus
    )


def test_vituri_signed_q_structure_pair_order_labels_and_diagnostic_scope() -> None:
    result = assemble_vituri2024_tdhf_signed_q(_signed_pair())
    blocks = result.blocks
    np.testing.assert_allclose(blocks.A_plus, blocks.A_plus.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(blocks.A_minus, blocks.A_minus.conj().T, atol=1.0e-12)
    np.testing.assert_allclose(
        blocks.B_plus_minus, blocks.B_minus_plus.T, atol=1.0e-12
    )
    matrices = build_tdhf_signed_q_matrices(
        blocks,
        result.sector.sewing,
        raise_on_structure_error=True,
    )
    assert matrices.structure.ok
    assert matrices.structure.sewing_closure == 0.0

    assert isinstance(result.sector.q, TDHFGenericSignedQ)
    assert result.sector.q.plus_raw == result.sector.q.plus_canonical == _Q
    assert result.sector.q.minus_raw == result.sector.q.minus_canonical == (
        -_Q[0],
        -_Q[1],
    )
    assert VITURI2024_TDHF_Q_PROVENANCE in result.sector.q.provenance
    assert "no_torus_or_carry" in result.sector.sewing.construction
    assert result.sector.sewing.plus_pairs_fingerprint == fingerprint_tdhf_pairs(
        blocks.plus_pairs
    )
    assert result.sector.sewing.minus_pairs_fingerprint == fingerprint_tdhf_pairs(
        blocks.minus_pairs
    )
    assert result.plus_pairs_fingerprint == fingerprint_tdhf_pairs(blocks.plus_pairs)
    assert result.minus_pairs_fingerprint == fingerprint_tdhf_pairs(blocks.minus_pairs)
    assert blocks.plus_pairs[0].particle_momentum == _Q
    assert blocks.plus_pairs[0].hole_momentum == (0.0, 0.0)
    assert blocks.plus_pairs[0].particle_flavor == Vituri2024Flavor(1, 1)
    assert blocks.plus_pairs[0].hole_flavor == Vituri2024Flavor(1, 1)
    assert [index for index, _ in result.orbital_id_map] == list(
        range(len(result.orbital_id_map))
    )

    assert result.sector.static_hessian_authority == "projected_signed_ab"
    assert result.sector.response_scope == VITURI2024_TDHF_RESPONSE_SCOPE
    analysis = analyze_tdhf_typed_sector(result.sector)
    assert analysis.static.kind == "not_established"
    assert result.source_fingerprint == result.signed_pair.source_fingerprint
    assert result.context_fingerprint == result.signed_pair.context_fingerprint
    assert (
        result.assembly_context_fingerprint
        == result.signed_pair.assembly_context_fingerprint
        == result.signed_pair.plus_context.assembly_context_fingerprint
    )
    assert result.tdhf_eigensolver_called is False
    assert result.post_symmetrized is False
    assert result.post_hermitized is False
    assert result.hf_stationarity_certified is False
    assert result.real_mesh_area_authority is False
    assert result.q0_background_authority is False
    assert result.uv_domain_convergence_authority is False
    assert result.cdw_source_authority is False
    assert result.paper_numerical_parity is False
    assert result.production_ready is False
    assert result.executable_ready is False
    assert result.no_go_limits == VITURI2024_TDHF_NO_GO_LIMITS
    assert result.full_assembly_compatibility_keys == (
        "source_fingerprint",
        "context_fingerprint",
        "assembly_context_fingerprint",
    )
    assert result.interaction_fingerprint == vituri2024_tdhf_interaction_fingerprint(
        result.signed_pair.plus_context
    )
    assert result.interaction_fingerprint == (
        abc_module.vituri2024_tdhf_interaction_fingerprint(
            result.signed_pair.plus_context
        )
    )
    assert "vituri2024_tdhf_interaction_fingerprint" in tdhf_module.__all__
    bound = assemble_vituri2024_tdhf_signed_q(
        _signed_pair(bound_interaction=True)
    )
    assert bound.interaction_fingerprint != result.interaction_fingerprint
    assert bound.signed_pair.plus_context.interaction_binding_fingerprint is not None
    assert all(
        element.interaction_binding_fingerprint is not None
        for element in (
            bound.A_plus_elements
            + bound.B_plus_minus_elements
            + bound.A_minus_elements
            + bound.B_minus_plus_elements
        )
    )
    assert len(result.sewing_fingerprint) == 64
    assert len(result.sector_fingerprint) == 64
    assert len(result.fingerprint) == 64


def test_vituri_inventory_order_and_fail_closed_inputs() -> None:
    plus = _plus_transition()
    second = _transition(
        (-2.0 / 128.0, 3.0 / 128.0),
        (-3.0 / 128.0, 1.0 / 128.0),
        particle_energy=0.57,
        hole_energy=-0.29,
    )
    forward = Vituri2024TransitionInventory(
        q_inverse_angstrom=_Q,
        transitions=(plus, second),
    )
    reverse = Vituri2024TransitionInventory(
        q_inverse_angstrom=_Q,
        transitions=(second, plus),
    )
    assert forward.fingerprint != reverse.fingerprint
    assert forward.ordered_transition_fingerprints == (
        plus.fingerprint,
        second.fingerprint,
    )
    assert forward.reciprocal_torus_authority is False
    assert forward.reciprocal_carry_authority is False
    assert forward.canonicalization_authority is False

    with pytest.raises(ValueError, match="nonzero local q"):
        Vituri2024TransitionInventory(
            q_inverse_angstrom=(0.0, 0.0),
            transitions=(plus,),
        )
    wrong_q_transition = _transition(
        (1.0 / 64.0, 0.0),
        (0.0, 0.0),
        particle_energy=0.4,
        hole_energy=-0.2,
    )
    wrong_minus = Vituri2024TransitionInventory(
        q_inverse_angstrom=(1.0 / 64.0, 0.0),
        transitions=(wrong_q_transition,),
    )
    with pytest.raises(ValueError, match="exact q_minus=-q_plus"):
        Vituri2024SignedQTransitionInventoryPair(
            plus_inventory=Vituri2024TransitionInventory(_Q, (plus,)),
            minus_inventory=wrong_minus,
            plus_context=_context(),
            minus_context=_context(),
        )
    with pytest.raises(ValueError, match="source_fingerprint mismatch"):
        _signed_pair(minus_source_sha="e" * 64)
    with pytest.raises(ValueError, match="context_fingerprint mismatch"):
        _signed_pair(minus_area=54.0)

    shared = _orbital(_Q)
    occupation_conflict = (
        plus,
        Vituri2024DiagonalHFTransitionReceipt(
            particle=_orbital((2.0 * _Q[0], 2.0 * _Q[1])),
            hole=shared,
            particle_energy_ev=0.72,
            hole_energy_ev=plus.particle_energy_ev,
            source_artifact_sha256=_SOURCE_SHA,
            source_text=_SOURCE_TEXT,
        ),
    )
    with pytest.raises(ValueError, match="both particle and hole"):
        Vituri2024TransitionInventory(_Q, occupation_conflict)

    with pytest.raises(ValueError, match="duplicate transition fingerprints"):
        Vituri2024TransitionInventory(_Q, (plus, plus))

    inconsistent_duplicate = replace(
        plus,
        particle_energy_ev=plus.particle_energy_ev + 0.1,
    )
    assert inconsistent_duplicate.fingerprint != plus.fingerprint
    with pytest.raises(ValueError, match=r"duplicate physical \(particle, hole\) pairs"):
        Vituri2024TransitionInventory(_Q, (plus, inconsistent_duplicate))


def test_vituri_crossed_valid_element_rejects_different_kinematics_context() -> None:
    result = assemble_vituri2024_tdhf_signed_q(_signed_pair())
    plus = result.signed_pair.plus_inventory.transitions[0]
    crossed_context = _context(
        kinematics_sha="e" * 64,
        kinematics_text=(
            "Independent valid local quartet provider; intentionally crossed."
        ),
    )
    crossed = vituri2024_rpa_a_element(
        plus,
        plus,
        crossed_context.area,
        crossed_context.Delta1,
        crossed_context.interaction,
        kinematics_provider_sha256=crossed_context.kinematics_provider_sha256,
        kinematics_source_text=crossed_context.kinematics_source_text,
    )
    assert crossed.context_fingerprint == result.context_fingerprint
    assert crossed.value_ev == result.A_plus_elements[0].value_ev
    assert (
        crossed.kinematics.provider_sha256,
        crossed.kinematics.source_text,
    ) != (
        result.A_plus_elements[0].kinematics.provider_sha256,
        result.A_plus_elements[0].kinematics.source_text,
    )

    crossed_matrix = np.array([[crossed.value_ev]], dtype=np.complex128)
    crossed_matrix.setflags(write=False)
    crossed_blocks = replace(result.blocks, A_plus=crossed_matrix)
    crossed_sector = replace(result.sector, blocks=crossed_blocks)
    with pytest.raises(
        ValueError,
        match="A_plus assembly_context_fingerprint compatibility failed",
    ):
        replace(
            result,
            blocks=crossed_blocks,
            sector=crossed_sector,
            A_plus_elements=(crossed,),
            A_plus_element_fingerprints=(crossed.fingerprint,),
            A_plus_matrix_fingerprint=fingerprint_tdhf_matrix(crossed_matrix),
            sector_fingerprint=fingerprint_tdhf_sector(crossed_sector),
        )


def test_vituri_assembly_tampering_exports_and_no_solver_surface() -> None:
    result = assemble_vituri2024_tdhf_signed_q(_signed_pair())
    result.blocks.A_plus.setflags(write=True)
    result.blocks.A_plus[0, 0] += 0.25
    with pytest.raises(ValueError, match="matrix no longer equals|matrix fingerprint"):
        _ = result.fingerprint

    for name in (
        "Vituri2024TransitionInventory",
        "Vituri2024TDHFAssemblyContext",
        "Vituri2024SignedQTransitionInventoryPair",
        "Vituri2024TDHFSignedQAssemblyReceipt",
        "assemble_vituri2024_tdhf_signed_q",
    ):
        assert hasattr(abc_module, name)
    assert not hasattr(tdhf_module, "solve_vituri2024_tdhf")
    assert not hasattr(tdhf_module, "run_vituri2024_tdhf")
    source = inspect.getsource(tdhf_module.assemble_vituri2024_tdhf_signed_q)
    assert "solve_tdhf" not in source
    module_doc = tdhf_module.__doc__ or ""
    for text in (
        "four signed lanes",
        "no copying",
        "HF stationarity",
        "q=0 background",
        "UV/domain",
        "CDW source",
        "production/executable",
        "does not invoke a TDHF/RPA eigenmode solver",
    ):
        assert text in module_doc
