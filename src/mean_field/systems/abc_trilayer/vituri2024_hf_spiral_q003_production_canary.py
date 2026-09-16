"""Source-bound production/scalar implementation canary for Vituri q003.

The public qualifier accepts only an identity-registered source record emitted
by the existing two-source q003 candidate-proof factory.  It executes the
actual production transition embedding on every signed lane, including empty
partners, and compares one coordinate-wise hash-derived vector per canonical
orbit to an independent exact-integer scalar action.

This is registered-vector implementation evidence.  It is not a complete-basis
sweep, an exact-unitary curvature calculation, an operator theorem, or an
authorization for eigensolvers, stability, production, or paper claims.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from hashlib import sha256
import inspect
import json
import marshal
import math
from pathlib import Path
from typing import Final

import numpy as np

from . import vituri2024_hf_spiral_full_hessian as _hessian
from . import vituri2024_hf_spiral_q003_same_functional_bridge as _bridge
from . import vituri2024_hf_spiral_q003_same_functional_proof as _proof
from .vituri2024_hf_spiral_full_stability import Vituri2024HFSpiralFullSectorKey

Array = np.ndarray

VITURI2024_Q003_PRODUCTION_CANARY_API_VERSION: Final[str] = (
    "vituri2024_q003_production_embedding_action_canary.v2"
)
VITURI2024_Q003_PRODUCTION_CANARY_AUTHORITY: Final[str] = (
    "source_record_bound_whole_inventory_registered_vector_implementation_canary_"
    "no_orientation_implementation_operator_theorem_scalar_hessian_eigensolver_"
    "stability_production_or_paper_authority"
)
VITURI2024_Q003_PRODUCTION_CANARY_ABSOLUTE_TOLERANCE_EV: Final[float] = 2.0e-10
VITURI2024_Q003_PRODUCTION_CANARY_RELATIVE_TOLERANCE: Final[float] = 2.0e-10
VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM: Final[float] = 1.0e-14

_EXPECTED_Q003_NONEMPTY_LANES = 38_259
_EXPECTED_Q003_CANONICAL_ORBITS = 21_170
_EXPECTED_Q003_COMPLEX_DIMENSION = 15_052_040
_EXPECTED_Q003_REAL_DIMENSION = 30_104_080


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _stable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": value.shape, "sha256": _array_sha256(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _stable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, np.generic):
        return _stable(value.item())
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported q003 canary fingerprint type {type(value).__name__}")


def _fingerprint(value: object) -> str:
    return sha256(json.dumps(_stable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be explicit nonempty text")
    return value


def _max_abs(value: object) -> float:
    result = float(np.max(np.abs(np.asarray(value)), initial=0.0))
    if not math.isfinite(result):
        raise ValueError("nonfinite q003 canary value")
    return result


def _function_fingerprint(function: object) -> str:
    if not inspect.isfunction(function):
        raise TypeError("q003 canary dependency must be a Python function")
    return _fingerprint({
        "module": function.__module__,
        "qualname": function.__qualname__,
        "source_sha256": sha256(inspect.getsource(function).encode()).hexdigest(),
        "code_sha256": sha256(marshal.dumps(function.__code__)).hexdigest(),
    })


def _module_sha256() -> str:
    path_value = inspect.getsourcefile(_expected_support)
    if path_value is None:
        raise ValueError("q003 canary module has no source file")
    return sha256(Path(path_value).read_bytes()).hexdigest()


def _expected_support(labels: Array, displacement_x: int, displacement_y: int) -> tuple[Array, Array]:
    """Construct finite-square no-wrap support from exact integer labels."""

    size = int(round(math.sqrt(labels.shape[0])))
    if size * size != labels.shape[0] or size % 2 != 1:
        raise ValueError("q003 canary labels do not form an odd centered square")
    half = size // 2
    x_low = max(-half, -half - displacement_x)
    x_high = min(half, half - displacement_x)
    y_low = max(-half, -half - displacement_y)
    y_high = min(half, half - displacement_y)
    if x_low > x_high or y_low > y_high:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    base_x = np.tile(np.arange(x_low, x_high + 1, dtype=np.int64), y_high - y_low + 1)
    base_y = np.repeat(np.arange(y_low, y_high + 1, dtype=np.int64), x_high - x_low + 1)
    bases = (base_y + half) * size + base_x + half
    target_x = base_x + displacement_x
    target_y = base_y + displacement_y
    targets = (target_y + half) * size + target_x + half
    if (
        not np.array_equal(labels[bases, 0], base_x)
        or not np.array_equal(labels[bases, 1], base_y)
        or not np.array_equal(labels[targets, 0], target_x)
        or not np.array_equal(labels[targets, 1], target_y)
    ):
        raise ValueError("q003 canary centered-square support arithmetic drifted")
    return bases, targets


def _expected_embedding(
    *, key: Vituri2024HFSpiralFullSectorKey, labels: Array, occupations: Array, nk: int
) -> tuple[Array, Array, tuple[Array, ...]]:
    bases, targets = _expected_support(labels, key.displacement_x, key.displacement_y)
    particle_k: list[int] = []
    hole_k: list[int] = []
    particle_slot: list[int] = []
    hole_slot: list[int] = []
    for particle_valley, hole_valley in _proof._production_pair_order(key.valley_charge):
        for base, target in zip(bases.tolist(), targets.tolist(), strict=True):
            if occupations[hole_valley, base] and not occupations[particle_valley, target]:
                particle_k.append(target)
                hole_k.append(base)
                particle_slot.append(particle_valley)
                hole_slot.append(hole_valley)
    pk = np.asarray(particle_k, dtype=np.int64)
    hk = np.asarray(hole_k, dtype=np.int64)
    ps = np.asarray(particle_slot, dtype=np.int64)
    hs = np.asarray(hole_slot, dtype=np.int64)
    return bases, targets, (ps, pk, hs, hk, ps * nk + pk, hs * nk + hk)


def _hash_coordinate(seed: bytes, index: int) -> complex:
    digest = sha256(seed + int(index).to_bytes(8, "big", signed=False)).digest()
    real = (int.from_bytes(digest[:8], "big") + 0.5) / float(2**64) - 0.5
    imaginary = (int.from_bytes(digest[8:16], "big") + 0.5) / float(2**64) - 0.5
    return complex(real, imaginary)


def _hash_vector(
    *, source_record_fingerprint: str, first_key: Vituri2024HFSpiralFullSectorKey,
    first_dimension: int, second_dimension: int,
) -> tuple[Array, Array, str]:
    """Hash every coordinate independently and normalize the resulting vector."""

    total = first_dimension + second_dimension
    if total <= 0:
        raise ValueError("q003 canary cannot build a zero-dimensional vector")
    seed = (
        source_record_fingerprint
        + f":{first_key.displacement_x}:{first_key.displacement_y}:"
        + f"{first_key.valley_charge}:{first_dimension}:{second_dimension}"
    ).encode()
    values = np.fromiter((_hash_coordinate(seed, index) for index in range(total)), dtype=np.complex128, count=total)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("q003 canary hash vector is vacuous")
    values /= norm
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) == 0.0):
        raise RuntimeError("q003 canary hash vector has zero or nonfinite coordinates")
    return values[:first_dimension], values[first_dimension:], _array_sha256(values)


def _manual_pack(first: Array, second: Array, *, factor: float) -> Array:
    joined = factor * np.concatenate((np.asarray(first), np.asarray(second)))
    return np.concatenate((joined.real, joined.imag))


def _residual(actual: Array, expected: Array) -> tuple[float, float, float]:
    absolute = _max_abs(np.asarray(actual) - np.asarray(expected))
    scale = max(1.0, _max_abs(actual), _max_abs(expected))
    return (
        absolute,
        absolute / scale,
        VITURI2024_Q003_PRODUCTION_CANARY_ABSOLUTE_TOLERANCE_EV
        + VITURI2024_Q003_PRODUCTION_CANARY_RELATIVE_TOLERANCE * scale,
    )


def _scalar_pair_action(
    prepared: object,
    first: Array,
    second: Array,
    *,
    first_expected: tuple[Array, Array, tuple[Array, ...]],
    second_expected: tuple[Array, Array, tuple[Array, ...]],
    component: str,
) -> tuple[Array, Array]:
    first_bases, first_targets, first_arrays = first_expected
    second_bases, second_targets, second_arrays = second_expected
    fps, _fpk, fhs, fhk, _fpid, _fhid = first_arrays
    sps, spk, shs, _shk, _spid, _shid = second_arrays
    block = np.zeros((2, 2, prepared.context.nk), dtype=np.complex128)
    block[fps, fhs, fhk] = first
    if second.size:
        indices = (shs, sps, spk)
        if np.any(block[indices] != 0.0):
            raise RuntimeError("q003 canary independent signed-lane injection collided")
        block[indices] = second.conj()
    first_result = _proof._scalar_component_action(
        prepared.context.response, prepared.orbit.first, block,
        first_bases, first_targets, component,
    )
    partner_block = np.zeros_like(block)
    partner_block[:, :, first_targets] = block[:, :, first_bases].swapaxes(0, 1).conj()
    second_result = _proof._scalar_component_action(
        prepared.context.response, prepared.orbit.second, partner_block,
        second_bases, second_targets, component,
    )
    return (
        np.asarray(first_result[fps, fhs, fhk]),
        np.asarray(second_result[sps, shs, second_arrays[3]]),
    )


def _production_direct_cached(prepared: object, first: Array, second: Array) -> tuple[Array, Array]:
    callback = prepared.hessian._interaction_response
    if type(callback) is not _hessian._VituriPairedInteractionCallback:
        raise TypeError("q003 canary production callback type drifted")
    component_callback = _hessian._VituriPairedInteractionCallback(
        response=prepared.context.response,
        first_embedding=prepared.first_embedding,
        second_embedding=prepared.second_embedding,
        first_action=_proof._ReducedComponentAction(
            prepared.context.response, prepared.orbit.first,
            callback.first_action.bases, callback.first_action.targets, "direct",
        ),
        second_action=_proof._ReducedComponentAction(
            prepared.context.response, prepared.orbit.second,
            callback.second_action.bases, callback.second_action.targets, "direct",
        ),
        expected_response_fingerprint=prepared.context.response.response_fingerprint,
    )
    return component_callback(first, second)


def _bind_source_records(
    context: object,
    proof_source_record: object,
    bridge_receipt: object,
    independent_source_fock: object,
) -> tuple[Array, _bridge.Vituri2024Q003SameFunctionalSourceRecord]:
    if type(proof_source_record) is not _proof.Vituri2024Q003SameFunctionalProofSourceRecord:
        raise TypeError("q003 production canary requires an exact candidate-proof source record")
    if type(bridge_receipt) is not _bridge.Vituri2024Q003SameFunctionalStructuralReceipt:
        raise TypeError("q003 production canary requires an exact same-functional bridge receipt")
    proof_source_record.validate_live_state()
    bridge_receipt.validate_live_state()
    matching_sources = tuple(
        source
        for source in bridge_receipt.sources
        if source.group_label == proof_source_record.group_label
        and source.context_fingerprint == context.context_fingerprint
    )
    if len(matching_sources) != 1:
        raise ValueError("q003 production canary bridge receipt has no unique matching source")
    bridge_source_record = matching_sources[0]
    inventory = context.inventory
    response = context.response
    plan = response.fft_plan
    if (
        not isinstance(independent_source_fock, np.ndarray)
        or independent_source_fock.dtype != np.dtype(np.complex128)
        or independent_source_fock.shape != (4, 4, context.nk)
        or not np.all(np.isfinite(independent_source_fock))
    ):
        raise ValueError("q003 canary independent source Fock must be finite complex128 (4,4,Nk)")
    clean_fock = np.frombuffer(
        independent_source_fock.tobytes(order="C"), dtype=np.complex128
    ).reshape(independent_source_fock.shape)
    clean_fock.setflags(write=False)
    common = (
        proof_source_record.group_label == bridge_source_record.group_label,
        proof_source_record.context_fingerprint == bridge_source_record.context_fingerprint == context.context_fingerprint,
        proof_source_record.inventory_fingerprint == bridge_source_record.inventory_fingerprint == inventory.inventory_fingerprint,
        proof_source_record.response_fingerprint == bridge_source_record.response_fingerprint == response.response_fingerprint,
        bridge_source_record.independent_source_fock_sha256 == _array_sha256(clean_fock),
        proof_source_record.selected_spinors_sha256 == _array_sha256(response.selected_spinors),
        proof_source_record.integer_mesh_labels_sha256 == _array_sha256(inventory.integer_mesh_labels),
        proof_source_record.selected_occupations_sha256 == _array_sha256(inventory.selected_occupations),
        proof_source_record.fft_plan_fingerprint == plan.fingerprint,
        proof_source_record.kernel_sha256 == _array_sha256(plan.kernel_by_signed_displacement),
        proof_source_record.area_angstrom_squared == float(response.area_angstrom_squared),
        proof_source_record.nonempty_signed_lane_count == bridge_source_record.nonempty_sector_count == _EXPECTED_Q003_NONEMPTY_LANES,
        proof_source_record.canonical_orbit_count == bridge_source_record.canonical_orbit_count == _EXPECTED_Q003_CANONICAL_ORBITS,
        proof_source_record.complex_dimension == bridge_source_record.complex_dimension == _EXPECTED_Q003_COMPLEX_DIMENSION,
        proof_source_record.real_dimension == bridge_source_record.real_dimension == _EXPECTED_Q003_REAL_DIMENSION,
    )
    if not all(common):
        raise ValueError("q003 production canary context/source-record binding mismatch")
    return clean_fock, bridge_source_record


@dataclass(frozen=True, slots=True)
class _CanaryComputation:
    orbit_start: int
    orbit_stop: int
    expected_embedding_stream_sha256: str
    production_embedding_stream_sha256: str
    vector_stream_sha256: str
    production_output_stream_sha256: str
    scalar_output_stream_sha256: str
    signed_lane_keys_checked: int
    nonempty_signed_lanes_checked: int
    empty_signed_lanes_checked: int
    canonical_orbits_checked: int
    one_empty_orbits_checked: int
    complex_dimensions_touched: int
    support_mismatch_count: int
    flavor_block_mismatch_count: int
    embedding_mismatch_count: int
    direct_applicable_orbits: int
    direct_informative_orbits: int
    exchange_informative_orbits: int
    total_informative_orbits: int
    one_body_informative_orbits: int
    full_action_informative_orbits: int
    quadratic_form_informative_orbits: int
    minimum_direct_output_norm_ev: float
    minimum_exchange_output_norm_ev: float
    minimum_total_output_norm_ev: float
    minimum_one_body_output_norm_ev: float
    minimum_full_action_output_norm_ev: float
    minimum_quadratic_form_magnitude_ev: float
    maximum_one_body_absolute_residual_ev: float
    maximum_one_body_relative_residual: float
    maximum_direct_absolute_residual_ev: float
    maximum_direct_relative_residual: float
    maximum_exchange_absolute_residual_ev: float
    maximum_exchange_relative_residual: float
    maximum_total_absolute_residual_ev: float
    maximum_total_relative_residual: float
    maximum_quadratic_form_absolute_residual_ev: float
    maximum_quadratic_form_relative_residual: float


def _compute_canary(
    *, context: _hessian.Vituri2024HFSpiralFullHessianContext,
    source_record_fingerprint: str,
    independent_source_fock: Array | None = None,
    orbit_start: int = 0,
    orbit_stop: int | None = None,
) -> _CanaryComputation:
    labels, occupations, _size, nk = _proof._strict_centered_inputs(context.inventory)
    total_orbits = context.inventory.nonempty_conjugate_orbit_count
    if type(orbit_start) is not int or orbit_start < 0:
        raise ValueError("q003 canary orbit_start must be a nonnegative exact int")
    resolved_stop = total_orbits if orbit_stop is None else orbit_stop
    if type(resolved_stop) is not int or not orbit_start < resolved_stop <= total_orbits:
        raise ValueError("q003 canary orbit range must be nonempty and inside inventory")
    if independent_source_fock is None:
        independent_selected_fock = context.selected_fock_diagonal_ev
    else:
        selected = np.asarray(
            context.inventory.restricted_preparation.receipt.selected_flavor_indices,
            dtype=np.int64,
        )
        momenta = np.arange(nk, dtype=np.int64)
        selected_full = independent_source_fock[np.ix_(selected, selected, momenta)]
        off_diagonal = np.array(selected_full, copy=True)
        off_diagonal[0, 0] = 0.0
        off_diagonal[1, 1] = 0.0
        if _max_abs(off_diagonal) > _bridge.VITURI2024_Q003_ONE_BODY_BASIS_TOLERANCE_EV:
            raise ValueError("q003 canary independent source Fock is not diagonal in the tangent basis")
        independent_selected_fock = np.stack(
            (selected_full[0, 0].real, selected_full[1, 1].real)
        )
    expected_hash = sha256()
    production_hash = sha256()
    vector_hash = sha256()
    production_output_hash = sha256()
    scalar_output_hash = sha256()
    counts = {name: 0 for name in (
        "keys", "nonempty", "empty", "orbits", "one_empty", "dimensions",
        "support_bad", "flavor_bad", "embedding_bad", "direct_applicable",
        "direct_informative", "exchange_informative", "total_informative",
        "one_body_informative", "full_informative", "quadratic_informative",
    )}
    maxima = {name: 0.0 for name in (
        "one_body_abs", "one_body_rel", "direct_abs", "direct_rel",
        "exchange_abs", "exchange_rel", "total_abs", "total_rel",
        "quadratic_abs", "quadratic_rel",
    )}
    minimum_direct = math.inf
    minimum_exchange = math.inf
    minimum_total = math.inf
    minimum_one_body = math.inf
    minimum_full = math.inf
    minimum_quadratic = math.inf

    for orbit_index, orbit in enumerate(
        context.inventory.iter_conjugate_orbits(include_zero_dimension=False)
    ):
        if orbit_index < orbit_start:
            continue
        if orbit_index >= resolved_stop:
            break
        prepared = context.build_orbit_hessian(orbit.first)
        callback = prepared.hessian._interaction_response
        if type(callback) is not _hessian._VituriPairedInteractionCallback:
            raise TypeError("q003 canary production callback type drifted")
        expected_by_key: dict[Vituri2024HFSpiralFullSectorKey, tuple[Array, Array, tuple[Array, ...]]] = {}
        dimensions_pair = (
            prepared.first_embedding.complex_dimension,
            prepared.second_embedding.complex_dimension,
        )
        if (dimensions_pair[0] == 0) != (dimensions_pair[1] == 0):
            counts["one_empty"] += 1

        for key, embedding, action in (
            (prepared.orbit.first, prepared.first_embedding, callback.first_action),
            (prepared.orbit.second, prepared.second_embedding, callback.second_action),
        ):
            expected = _expected_embedding(key=key, labels=labels, occupations=occupations, nk=nk)
            expected_by_key[key] = expected
            expected_bases, expected_targets, expected_arrays = expected
            counts["keys"] += 1
            dimension = embedding.complex_dimension
            counts["nonempty" if dimension else "empty"] += 1
            counts["dimensions"] += dimension
            counts["support_bad"] += int(
                not np.array_equal(action.bases, expected_bases)
                or not np.array_equal(action.targets, expected_targets)
            )
            counts["flavor_bad"] += int(
                _proof._production_pair_order(key.valley_charge)
                != _proof._scalar_module._allowed_blocks(key.valley_charge)
            )
            actual_arrays = (
                embedding.particle_valley_slots, embedding.particle_k_indices,
                embedding.hole_valley_slots, embedding.hole_k_indices,
                embedding.particle_orbital_ids, embedding.hole_orbital_ids,
            )
            counts["embedding_bad"] += int(
                dimension != expected_arrays[0].size
                or any(not np.array_equal(actual, wanted) for actual, wanted in zip(actual_arrays, expected_arrays, strict=True))
            )
            header = np.asarray([key.displacement_x, key.displacement_y, key.valley_charge, dimension], dtype="<i8")
            expected_hash.update(header.tobytes())
            production_hash.update(header.tobytes())
            for array in (expected_bases, expected_targets, *expected_arrays):
                expected_hash.update(np.asarray(array, dtype="<i8").tobytes())
            for array in (action.bases, action.targets, *actual_arrays):
                production_hash.update(np.asarray(array, dtype="<i8").tobytes())

        if counts["support_bad"] or counts["flavor_bad"] or counts["embedding_bad"]:
            raise RuntimeError(f"q003 production canary geometry mismatch at orbit {prepared.orbit.first}")

        first, second, vector_sha = _hash_vector(
            source_record_fingerprint=source_record_fingerprint,
            first_key=prepared.orbit.first,
            first_dimension=dimensions_pair[0],
            second_dimension=dimensions_pair[1],
        )
        vector_hash.update(bytes.fromhex(vector_sha))
        real_vector = _manual_pack(first, second, factor=1.0)
        # Execute the actual production unpack/action/factor-two matvec path once.
        production_full = prepared.hessian.matvec(real_vector)
        scalar_total_complex = _scalar_pair_action(
            prepared, first, second,
            first_expected=expected_by_key[prepared.orbit.first],
            second_expected=expected_by_key[prepared.orbit.second],
            component="total",
        )
        production_direct_complex = _production_direct_cached(prepared, first, second)
        scalar_direct_complex = _scalar_pair_action(
            prepared, first, second,
            first_expected=expected_by_key[prepared.orbit.first],
            second_expected=expected_by_key[prepared.orbit.second],
            component="direct",
        )
        scalar_total = _manual_pack(*scalar_total_complex, factor=2.0)
        production_direct = prepared.hessian.frame.pack_complex(
            *production_direct_complex, factor=2.0
        )
        scalar_direct = _manual_pack(*scalar_direct_complex, factor=2.0)

        first_expected_arrays = expected_by_key[prepared.orbit.first][2]
        second_expected_arrays = expected_by_key[prepared.orbit.second][2]
        independent_gaps = (
            independent_selected_fock[first_expected_arrays[0], first_expected_arrays[1]]
            - independent_selected_fock[first_expected_arrays[2], first_expected_arrays[3]],
            independent_selected_fock[second_expected_arrays[0], second_expected_arrays[1]]
            - independent_selected_fock[second_expected_arrays[2], second_expected_arrays[3]],
        )
        scalar_one_body_complex = (
            independent_gaps[0] * first,
            independent_gaps[1] * second,
        )
        scalar_one_body = _manual_pack(*scalar_one_body_complex, factor=2.0)
        production_one_body_complex = (
            prepared.hessian.frame.first.one_body_gaps_ev * first,
            prepared.hessian.frame.second.one_body_gaps_ev * second,
        )
        production_one_body = prepared.hessian.frame.pack_complex(
            *production_one_body_complex, factor=2.0
        )
        production_total = production_full - production_one_body
        production_exchange = production_total - production_direct
        scalar_exchange = scalar_total - scalar_direct
        scalar_full = scalar_one_body + scalar_total

        for label, actual, wanted in (
            ("one_body", production_one_body, scalar_one_body),
            ("direct", production_direct, scalar_direct),
            ("exchange", production_exchange, scalar_exchange),
            ("total", production_total, scalar_total),
        ):
            absolute, relative, tolerance = _residual(actual, wanted)
            maxima[f"{label}_abs"] = max(maxima[f"{label}_abs"], absolute)
            maxima[f"{label}_rel"] = max(maxima[f"{label}_rel"], relative)
            if absolute > tolerance:
                raise RuntimeError(
                    f"q003 production/scalar {label} mismatch at "
                    f"orbit {prepared.orbit.first}: {absolute:.6e}"
                )

        direct_norm = float(np.linalg.norm(scalar_direct))
        exchange_norm = float(np.linalg.norm(scalar_exchange))
        total_norm = float(np.linalg.norm(scalar_total))
        one_body_norm = float(np.linalg.norm(scalar_one_body))
        full_norm = float(np.linalg.norm(scalar_full))
        if prepared.orbit.first.valley_charge == 0 or prepared.orbit.second.valley_charge == 0:
            counts["direct_applicable"] += 1
            minimum_direct = min(minimum_direct, direct_norm)
            counts["direct_informative"] += int(
                direct_norm > VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM
            )
        minimum_exchange = min(minimum_exchange, exchange_norm)
        minimum_total = min(minimum_total, total_norm)
        minimum_one_body = min(minimum_one_body, one_body_norm)
        minimum_full = min(minimum_full, full_norm)
        counts["exchange_informative"] += int(
            exchange_norm > VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM
        )
        counts["total_informative"] += int(
            total_norm > VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM
        )
        counts["one_body_informative"] += int(
            one_body_norm > VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM
        )
        counts["full_informative"] += int(
            full_norm > VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM
        )

        production_quadratic = float(real_vector @ production_full)
        scalar_quadratic = float(real_vector @ scalar_full)
        quadratic_abs = abs(production_quadratic - scalar_quadratic)
        quadratic_scale = max(1.0, abs(production_quadratic), abs(scalar_quadratic))
        quadratic_rel = quadratic_abs / quadratic_scale
        quadratic_tol = VITURI2024_Q003_PRODUCTION_CANARY_ABSOLUTE_TOLERANCE_EV + VITURI2024_Q003_PRODUCTION_CANARY_RELATIVE_TOLERANCE * quadratic_scale
        quadratic_magnitude = abs(scalar_quadratic)
        minimum_quadratic = min(minimum_quadratic, quadratic_magnitude)
        counts["quadratic_informative"] += int(
            quadratic_magnitude > VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM
        )
        maxima["quadratic_abs"] = max(maxima["quadratic_abs"], quadratic_abs)
        maxima["quadratic_rel"] = max(maxima["quadratic_rel"], quadratic_rel)
        if quadratic_abs > quadratic_tol:
            raise RuntimeError(f"q003 registered-vector quadratic-form mismatch at orbit {prepared.orbit.first}: {quadratic_abs:.6e}")
        production_output_hash.update(np.asarray(production_full, dtype="<f8").tobytes())
        scalar_output_hash.update(np.asarray(scalar_full, dtype="<f8").tobytes())
        counts["orbits"] += 1

    if counts["orbits"] == 0 or counts["dimensions"] == 0:
        raise RuntimeError("q003 production canary is vacuous")
    informative_coverage = (
        counts["direct_informative"] == counts["direct_applicable"],
        counts["exchange_informative"] == counts["orbits"],
        counts["total_informative"] == counts["orbits"],
        counts["one_body_informative"] == counts["orbits"],
        counts["full_informative"] == counts["orbits"],
        counts["quadratic_informative"] == counts["orbits"],
    )
    if not all(informative_coverage):
        raise RuntimeError(
            "q003 production canary registered-vector coverage is vacuous: "
            f"direct={counts['direct_informative']}/{counts['direct_applicable']}, "
            f"exchange={counts['exchange_informative']}/{counts['orbits']}, "
            f"total={counts['total_informative']}/{counts['orbits']}, "
            f"one_body={counts['one_body_informative']}/{counts['orbits']}, "
            f"full={counts['full_informative']}/{counts['orbits']}, "
            f"quadratic={counts['quadratic_informative']}/{counts['orbits']}"
        )
    if counts["orbits"] != resolved_stop - orbit_start:
        raise RuntimeError("q003 production canary orbit range coverage drifted")
    return _CanaryComputation(
        orbit_start=orbit_start,
        orbit_stop=resolved_stop,
        expected_embedding_stream_sha256=expected_hash.hexdigest(),
        production_embedding_stream_sha256=production_hash.hexdigest(),
        vector_stream_sha256=vector_hash.hexdigest(),
        production_output_stream_sha256=production_output_hash.hexdigest(),
        scalar_output_stream_sha256=scalar_output_hash.hexdigest(),
        signed_lane_keys_checked=counts["keys"], nonempty_signed_lanes_checked=counts["nonempty"],
        empty_signed_lanes_checked=counts["empty"], canonical_orbits_checked=counts["orbits"],
        one_empty_orbits_checked=counts["one_empty"], complex_dimensions_touched=counts["dimensions"],
        support_mismatch_count=counts["support_bad"], flavor_block_mismatch_count=counts["flavor_bad"],
        embedding_mismatch_count=counts["embedding_bad"], direct_applicable_orbits=counts["direct_applicable"],
        direct_informative_orbits=counts["direct_informative"], exchange_informative_orbits=counts["exchange_informative"],
        total_informative_orbits=counts["total_informative"],
        one_body_informative_orbits=counts["one_body_informative"],
        full_action_informative_orbits=counts["full_informative"],
        quadratic_form_informative_orbits=counts["quadratic_informative"],
        minimum_direct_output_norm_ev=float(minimum_direct),
        minimum_exchange_output_norm_ev=float(minimum_exchange), minimum_total_output_norm_ev=float(minimum_total),
        minimum_one_body_output_norm_ev=float(minimum_one_body),
        minimum_full_action_output_norm_ev=float(minimum_full),
        minimum_quadratic_form_magnitude_ev=float(minimum_quadratic),
        maximum_one_body_absolute_residual_ev=float(maxima["one_body_abs"]), maximum_one_body_relative_residual=float(maxima["one_body_rel"]),
        maximum_direct_absolute_residual_ev=float(maxima["direct_abs"]), maximum_direct_relative_residual=float(maxima["direct_rel"]),
        maximum_exchange_absolute_residual_ev=float(maxima["exchange_abs"]), maximum_exchange_relative_residual=float(maxima["exchange_rel"]),
        maximum_total_absolute_residual_ev=float(maxima["total_abs"]), maximum_total_relative_residual=float(maxima["total_rel"]),
        maximum_quadratic_form_absolute_residual_ev=float(maxima["quadratic_abs"]), maximum_quadratic_form_relative_residual=float(maxima["quadratic_rel"]),
    )


def _make_receipt_boundary():
    token = object()
    registry: dict[int, tuple[object, str]] = {}

    @dataclass(frozen=True, slots=True)
    class _Receipt:
        _factory_token: InitVar[object]
        source_group_label: str
        proof_source_record_fingerprint: str
        bridge_receipt_fingerprint: str
        bridge_source_record_fingerprint: str
        independent_source_fock_sha256: str
        context_fingerprint: str
        inventory_fingerprint: str
        response_fingerprint: str
        implementation_fingerprint: str
        candidate_orientation_derivation_sha256: str
        computation: _CanaryComputation
        provenance: str
        receipt_fingerprint: str = field(init=False)
        api_version: str = field(default=VITURI2024_Q003_PRODUCTION_CANARY_API_VERSION, init=False)
        authority: str = field(default=VITURI2024_Q003_PRODUCTION_CANARY_AUTHORITY, init=False)
        actual_production_embedding_executed: bool = field(default=True, init=False)
        full_q003_signed_lane_keys_enumerated: bool = field(default=True, init=False)
        one_coordinate_hash_vector_per_orbit: bool = field(default=True, init=False)
        candidate_orientation_derivation_recorded: bool = field(default=True, init=False)
        asymmetric_orientation_implementation_canary_passed: bool = field(default=False, init=False)
        registered_vector_component_parity_passed: bool = field(default=True, init=False)
        complete_real_basis_executed: bool = field(default=False, init=False)
        production_matvec_executed: bool = field(default=True, init=False)
        exact_unitary_curvature_executed: bool = field(default=False, init=False)
        whole_operator_identity_established: bool = field(default=False, init=False)
        production_j_jdagger_theorem_established: bool = field(default=False, init=False)
        formula_correspondence_established: bool = field(default=False, init=False)
        scalar_hessian_authority_promoted: bool = field(default=False, init=False)
        hermitian_eigensolver_authorized: bool = field(default=False, init=False)
        local_stability_established: bool = field(default=False, init=False)
        production_ready: bool = field(default=False, init=False)
        paper_reproduction_verified: bool = field(default=False, init=False)

        def __post_init__(self, _factory_token: object) -> None:
            if _factory_token is not token:
                raise TypeError("q003 production canary receipt is factory-only")
            _text(self.source_group_label, "q003 canary source group")
            _text(self.provenance, "q003 canary provenance")
            for name in (
                "proof_source_record_fingerprint", "bridge_receipt_fingerprint",
                "bridge_source_record_fingerprint", "independent_source_fock_sha256", "context_fingerprint", "inventory_fingerprint",
                "response_fingerprint", "implementation_fingerprint", "candidate_orientation_derivation_sha256",
            ):
                _sha256(getattr(self, name), f"q003 canary {name}")
            if type(self.computation) is not _CanaryComputation:
                raise TypeError("q003 canary receipt requires an exact computation")
            expected = (
                self.computation.orbit_start == 0,
                self.computation.orbit_stop == _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.nonempty_signed_lanes_checked == _EXPECTED_Q003_NONEMPTY_LANES,
                self.computation.canonical_orbits_checked == _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.complex_dimensions_touched == _EXPECTED_Q003_COMPLEX_DIMENSION,
                self.computation.signed_lane_keys_checked == 2 * _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.empty_signed_lanes_checked == 4_081,
                self.computation.one_empty_orbits_checked == 4_081,
                self.computation.direct_informative_orbits == self.computation.direct_applicable_orbits,
                self.computation.exchange_informative_orbits == _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.total_informative_orbits == _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.one_body_informative_orbits == _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.full_action_informative_orbits == _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.quadratic_form_informative_orbits == _EXPECTED_Q003_CANONICAL_ORBITS,
                self.computation.expected_embedding_stream_sha256 == self.computation.production_embedding_stream_sha256,
                self.computation.support_mismatch_count == 0,
                self.computation.flavor_block_mismatch_count == 0,
                self.computation.embedding_mismatch_count == 0,
                self.actual_production_embedding_executed is True,
                self.full_q003_signed_lane_keys_enumerated is True,
                self.one_coordinate_hash_vector_per_orbit is True,
                self.candidate_orientation_derivation_recorded is True,
                self.asymmetric_orientation_implementation_canary_passed is False,
                self.registered_vector_component_parity_passed is True,
                self.complete_real_basis_executed is False,
                self.production_matvec_executed is True,
                self.exact_unitary_curvature_executed is False,
                self.whole_operator_identity_established is False,
                self.production_j_jdagger_theorem_established is False,
                self.formula_correspondence_established is False,
                self.scalar_hessian_authority_promoted is False,
                self.hermitian_eigensolver_authorized is False,
                self.local_stability_established is False,
                self.production_ready is False,
                self.paper_reproduction_verified is False,
            )
            if not all(expected):
                raise ValueError("q003 production canary scope or authority drifted")
            payload = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "receipt_fingerprint"}
            object.__setattr__(self, "receipt_fingerprint", _fingerprint(payload))

        def validate_live_state(self) -> None:
            _validate_bindings()
            registration = registry.get(id(self))
            if (
                registration is None
                or registration[0] is not self
                or registration[1] != self.receipt_fingerprint
            ):
                raise ValueError("q003 production canary receipt is not an identity-registered factory product")
            if self.implementation_fingerprint != vituri2024_q003_production_canary_implementation_fingerprint():
                raise ValueError("q003 production canary implementation drifted")
            payload = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "receipt_fingerprint"}
            if self.receipt_fingerprint != _fingerprint(payload):
                raise ValueError("q003 production canary receipt drifted")

        @property
        def fingerprint(self) -> str:
            self.validate_live_state()
            return self.receipt_fingerprint

    def create(**kwargs):
        receipt = _Receipt(_factory_token=token, **kwargs)
        registry[id(receipt)] = (receipt, receipt.receipt_fingerprint)
        return receipt

    return create


_make_receipt = _make_receipt_boundary()
del _make_receipt_boundary


def vituri2024_q003_production_canary_implementation_fingerprint() -> str:
    dependencies = (
        _hessian.Vituri2024HFSpiralFullHessianContext._build_embedding,
        _hessian.Vituri2024HFSpiralFullHessianContext.build_orbit_hessian,
        _proof._scalar_component_action,
        _proof._recorded_asymmetric_kernel_orientation_derivation,
        _expected_support, _expected_embedding, _hash_coordinate, _hash_vector,
        _manual_pack, _scalar_pair_action, _production_direct_cached,
        _bind_source_records, _compute_canary,
        _qualify_vituri2024_q003_production_canary_impl,
        qualify_vituri2024_q003_production_canary,
    )
    return _fingerprint({
        "api_version": VITURI2024_Q003_PRODUCTION_CANARY_API_VERSION,
        "module_sha256": _module_sha256(),
        "proof_implementation_fingerprint": _proof.vituri2024_q003_same_functional_proof_implementation_fingerprint(),
        "dependencies": tuple(_function_fingerprint(item) for item in dependencies),
        "tolerances": (
            VITURI2024_Q003_PRODUCTION_CANARY_ABSOLUTE_TOLERANCE_EV,
            VITURI2024_Q003_PRODUCTION_CANARY_RELATIVE_TOLERANCE,
            VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM,
        ),
    })


def _qualify_vituri2024_q003_production_canary_impl(
    *,
    context: _hessian.Vituri2024HFSpiralFullHessianContext,
    proof_source_record: _proof.Vituri2024Q003SameFunctionalProofSourceRecord,
    bridge_receipt: _bridge.Vituri2024Q003SameFunctionalStructuralReceipt,
    independent_source_fock: Array,
    provenance: str,
    _issuer: object,
) -> object:
    """Execute source-bound registered-vector parity on the exact q003 inventory."""

    _validate_bindings()
    if type(context) is not _hessian.Vituri2024HFSpiralFullHessianContext:
        raise TypeError("q003 production canary requires an exact Hessian context")
    _text(provenance, "q003 canary provenance")
    context.validate_live_state()
    clean_independent_fock, bridge_source_record = _bind_source_records(
        context,
        proof_source_record,
        bridge_receipt,
        independent_source_fock,
    )
    inventory = context.inventory
    exact_counts = (
        inventory.nonempty_sector_count,
        inventory.nonempty_conjugate_orbit_count,
        inventory.complex_dimension,
        inventory.real_dimension,
    )
    if exact_counts != (
        _EXPECTED_Q003_NONEMPTY_LANES, _EXPECTED_Q003_CANONICAL_ORBITS,
        _EXPECTED_Q003_COMPLEX_DIMENSION, _EXPECTED_Q003_REAL_DIMENSION,
    ):
        raise ValueError("production canary input is not the exact q003 inventory")
    computation = _compute_canary(
        context=context,
        source_record_fingerprint=proof_source_record.fingerprint,
        independent_source_fock=clean_independent_fock,
    )
    orientation = _proof._recorded_asymmetric_kernel_orientation_derivation()
    return _issuer(
        source_group_label=proof_source_record.group_label,
        proof_source_record_fingerprint=proof_source_record.fingerprint,
        bridge_receipt_fingerprint=bridge_receipt.receipt_fingerprint,
        bridge_source_record_fingerprint=bridge_source_record.fingerprint,
        independent_source_fock_sha256=_array_sha256(clean_independent_fock),
        context_fingerprint=context.context_fingerprint,
        inventory_fingerprint=inventory.inventory_fingerprint,
        response_fingerprint=context.response.response_fingerprint,
        implementation_fingerprint=vituri2024_q003_production_canary_implementation_fingerprint(),
        candidate_orientation_derivation_sha256=_fingerprint(orientation),
        computation=computation,
        provenance=provenance,
    )


def _make_public_qualifier(issuer: object, implementation: object):
    def qualify_vituri2024_q003_production_canary(
        *,
        context: _hessian.Vituri2024HFSpiralFullHessianContext,
        proof_source_record: _proof.Vituri2024Q003SameFunctionalProofSourceRecord,
        bridge_receipt: _bridge.Vituri2024Q003SameFunctionalStructuralReceipt,
        independent_source_fock: Array,
        provenance: str,
    ) -> object:
        return implementation(
            context=context,
            proof_source_record=proof_source_record,
            bridge_receipt=bridge_receipt,
            independent_source_fock=independent_source_fock,
            provenance=provenance,
            _issuer=issuer,
        )

    return qualify_vituri2024_q003_production_canary


qualify_vituri2024_q003_production_canary = _make_public_qualifier(
    _make_receipt,
    _qualify_vituri2024_q003_production_canary_impl,
)
del _make_public_qualifier
del _make_receipt


# Private reduced helper: returns no authoritative receipt and cannot disable the
# exact-q003 gate on the public factory.
def _qualify_reduced_production_canary(
    *,
    context: _hessian.Vituri2024HFSpiralFullHessianContext,
    source_seed_sha256: str,
    orbit_start: int = 0,
    orbit_stop: int | None = None,
) -> _CanaryComputation:
    _validate_bindings()
    if type(context) is not _hessian.Vituri2024HFSpiralFullHessianContext:
        raise TypeError("reduced canary helper requires an exact Hessian context")
    _sha256(source_seed_sha256, "reduced canary seed")
    context.validate_live_state()
    return _compute_canary(
        context=context,
        source_record_fingerprint=source_seed_sha256,
        orbit_start=orbit_start,
        orbit_stop=orbit_stop,
    )


_IMPORT_BINDINGS = {
    "prod_embedding": _hessian.Vituri2024HFSpiralFullHessianContext._build_embedding,
    "prod_orbit_builder": _hessian.Vituri2024HFSpiralFullHessianContext.build_orbit_hessian,
    "scalar_action": _proof._scalar_component_action,
    "expected_support": _expected_support,
    "expected_embedding": _expected_embedding,
    "hash_vector": _hash_vector,
    "compute_canary": _compute_canary,
    "bind_source_records": _bind_source_records,
    "scalar_pair_action": _scalar_pair_action,
    "production_direct_cached": _production_direct_cached,
    "manual_pack": _manual_pack,
    "residual": _residual,
    "qualifier_impl": _qualify_vituri2024_q003_production_canary_impl,
    "public_qualifier": qualify_vituri2024_q003_production_canary,
}


def _validate_bindings(
    _expected: tuple[tuple[str, object], ...] = tuple(_IMPORT_BINDINGS.items()),
) -> None:
    current = {
        "prod_embedding": _hessian.Vituri2024HFSpiralFullHessianContext._build_embedding,
        "prod_orbit_builder": _hessian.Vituri2024HFSpiralFullHessianContext.build_orbit_hessian,
        "scalar_action": _proof._scalar_component_action,
        "expected_support": _expected_support,
        "expected_embedding": _expected_embedding,
        "hash_vector": _hash_vector,
        "compute_canary": _compute_canary,
        "bind_source_records": _bind_source_records,
        "scalar_pair_action": _scalar_pair_action,
        "production_direct_cached": _production_direct_cached,
        "manual_pack": _manual_pack,
        "residual": _residual,
        "qualifier_impl": _qualify_vituri2024_q003_production_canary_impl,
        "public_qualifier": qualify_vituri2024_q003_production_canary,
    }
    for label, expected in _expected:
        if current[label] is not expected:
            raise RuntimeError(f"q003 production canary binding drifted: {label}")
    if (
        VITURI2024_Q003_PRODUCTION_CANARY_ABSOLUTE_TOLERANCE_EV != 2.0e-10
        or VITURI2024_Q003_PRODUCTION_CANARY_RELATIVE_TOLERANCE != 2.0e-10
        or VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM != 1.0e-14
    ):
        raise RuntimeError("q003 production canary tolerance drifted")
    _proof._validate_bindings()


del _IMPORT_BINDINGS


__all__ = [
    "VITURI2024_Q003_PRODUCTION_CANARY_ABSOLUTE_TOLERANCE_EV",
    "VITURI2024_Q003_PRODUCTION_CANARY_API_VERSION",
    "VITURI2024_Q003_PRODUCTION_CANARY_AUTHORITY",
    "VITURI2024_Q003_PRODUCTION_CANARY_RELATIVE_TOLERANCE",
    "VITURI2024_Q003_PRODUCTION_CANARY_SIGNAL_MINIMUM",
    "qualify_vituri2024_q003_production_canary",
    "vituri2024_q003_production_canary_implementation_fingerprint",
]
