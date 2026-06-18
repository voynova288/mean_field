from __future__ import annotations

import numpy as np
import pytest

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    ReferenceDensity as ContractReferenceDensity,
    assert_density_state_consistent,
)
from mean_field.core.hf.contracts_bridge import (
    density_state_from_delta,
    density_state_from_projector,
    make_contract_reference_density,
    normalize_contract_reference_scheme,
)
from mean_field.core.hf.density import ReferenceDensity as HFDensityReference


def _complex_projector_field() -> np.ndarray:
    vector = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    projector = np.outer(vector, vector.conjugate())[:, :, None]
    return np.repeat(projector, 2, axis=2)


def test_density_state_from_delta_roundtrips_stored_projector() -> None:
    projector = np.zeros((2, 2, 2), dtype=np.complex128)
    projector[0, 0, :] = 1.0
    reference = 0.5 * np.eye(2, dtype=np.complex128)[:, :, None] * np.ones((1, 1, 2), dtype=np.complex128)

    state = density_state_from_delta(
        projector - reference,
        reference,
        reference_scheme="average",
        filling=0.0,
        n_occupied_total=2,
        metadata={"source": "toy_delta"},
    )

    assert isinstance(state, ContractDensityState)
    assert state.reference.scheme == "average"
    assert state.metadata == {"source": "toy_delta"}
    np.testing.assert_allclose(state.projector, projector)
    np.testing.assert_allclose(state.density_delta, projector - reference)
    assert_density_state_consistent(state)


def test_density_state_from_projector_preserves_stored_orientation() -> None:
    projector = _complex_projector_field()
    reference = np.zeros_like(projector)

    state = density_state_from_projector(
        projector,
        reference,
        reference_scheme="custom",
        filling=0.0,
        n_occupied_total=2,
    )

    np.testing.assert_allclose(state.projector, projector)
    np.testing.assert_allclose(state.density_delta, projector)
    assert_density_state_consistent(state)
    assert not np.allclose(state.projector, np.swapaxes(projector, 0, 1))


def test_make_contract_reference_density_converts_legacy_reference_with_metadata() -> None:
    legacy = HFDensityReference.average(2, 1, value=0.25)

    reference = make_contract_reference_density(
        legacy,
        scheme="cn",
        metadata={"system": "toy"},
    )

    assert isinstance(reference, ContractReferenceDensity)
    assert reference.scheme == "CN"
    np.testing.assert_allclose(reference.reference, legacy.data)
    assert reference.metadata["hf_density_reference_convention"] == "average:0.25"
    assert reference.metadata["hf_density_axis_order"] == "abk"
    assert reference.metadata["system"] == "toy"


def test_contract_reference_conversion_requires_explicit_scheme_for_arrays() -> None:
    reference = np.zeros((2, 2, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="scheme is required"):
        make_contract_reference_density(reference)

    with pytest.raises(ValueError, match="Unsupported reference scheme"):
        normalize_contract_reference_scheme("charge-neutrality")


def test_existing_contract_reference_is_preserved_without_name_collision() -> None:
    reference = ContractReferenceDensity(
        scheme="central_average",
        reference=np.zeros((2, 2, 1), dtype=np.complex128),
        metadata={"already": "contract"},
    )

    converted = make_contract_reference_density(reference)

    assert converted.scheme == "central_average"
    assert converted.metadata == {"already": "contract"}
    assert isinstance(converted, ContractReferenceDensity)
    assert not isinstance(converted, HFDensityReference)

    with pytest.raises(ValueError, match="does not match existing"):
        make_contract_reference_density(reference, scheme="custom")


def test_mixed_density_can_be_wrapped_without_projector_claim() -> None:
    mixed_projector = 0.5 * np.eye(2, dtype=np.complex128)[:, :, None]
    reference = np.zeros_like(mixed_projector)

    state = density_state_from_projector(
        mixed_projector,
        reference,
        reference_scheme="custom",
        filling=0.0,
        n_occupied_total=1,
    )

    assert_density_state_consistent(state, require_projector=False)
    with pytest.raises(ValueError, match="idempotent"):
        assert_density_state_consistent(state, require_projector=True)
