"""Typed physical-Q and half-open reciprocal-carry contracts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .source import OpenMXSource


def _float3(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain three components")
    return array


def _int3(values: Iterable[int], name: str) -> np.ndarray:
    raw = np.asarray(tuple(values))
    if raw.shape != (3,):
        raise ValueError(f"{name} must contain three components")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype")
    return np.asarray(raw, dtype=np.int64)


@dataclass(frozen=True)
class PhysicalTransfer:
    source_id: str
    reciprocal_bohr: np.ndarray
    k_target_fractional: np.ndarray
    k_source_fractional: np.ndarray
    folding_carry: np.ndarray
    local_field: np.ndarray
    integer_shift: np.ndarray
    q_bohr_inv: np.ndarray

    @classmethod
    def create(
        cls,
        source: OpenMXSource,
        k_target_fractional: Iterable[float],
        k_source_fractional: Iterable[float],
        folding_carry: Iterable[int],
        local_field: Iterable[int],
    ) -> "PhysicalTransfer":
        target = _float3(k_target_fractional, "k_target_fractional")
        source_k = _float3(k_source_fractional, "k_source_fractional")
        carry = _int3(folding_carry, "folding_carry")
        local = _int3(local_field, "local_field")
        shift = carry + local
        q = (target - source_k + shift) @ source.reciprocal_bohr
        return cls(
            source_id=source.input_sha256,
            reciprocal_bohr=np.asarray(source.reciprocal_bohr),
            k_target_fractional=target,
            k_source_fractional=source_k,
            folding_carry=carry,
            local_field=local,
            integer_shift=shift,
            q_bohr_inv=q,
        )


@dataclass(frozen=True)
class PhysicalTransferPair:
    forward: PhysicalTransfer
    reverse: PhysicalTransfer

    @classmethod
    def create(
        cls,
        source: OpenMXSource,
        k_target_fractional: Iterable[float],
        k_source_fractional: Iterable[float],
        forward_folding_carry: Iterable[int],
        reverse_folding_carry: Iterable[int],
        forward_local_field: Iterable[int],
        *,
        tolerance: float = 1.0e-13,
    ) -> "PhysicalTransferPair":
        forward_carry = _int3(forward_folding_carry, "forward_folding_carry")
        reverse_carry = _int3(reverse_folding_carry, "reverse_folding_carry")
        forward_local = _int3(forward_local_field, "forward_local_field")
        reverse_local = -forward_local - forward_carry - reverse_carry
        forward = PhysicalTransfer.create(
            source,
            k_target_fractional,
            k_source_fractional,
            forward_carry,
            forward_local,
        )
        reverse = PhysicalTransfer.create(
            source,
            k_source_fractional,
            k_target_fractional,
            reverse_carry,
            reverse_local,
        )
        if not np.array_equal(reverse.integer_shift, -forward.integer_shift):
            raise ValueError("reverse integer reciprocal shift is not the forward negative")
        if not np.allclose(
            reverse.q_bohr_inv,
            -forward.q_bohr_inv,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("reverse physical Q is not the forward negative")
        return cls(forward=forward, reverse=reverse)


@dataclass(frozen=True)
class OrientedTransferRef:
    record_index: int
    orientation: int

    def __post_init__(self) -> None:
        if self.record_index < 0:
            raise ValueError("record_index must be nonnegative")
        if self.orientation not in (-1, 1):
            raise ValueError("orientation must be +1 or -1")


@dataclass(frozen=True)
class FullKMeshPhysicalQChart:
    """Pair-dependent half-open transfers with finite reverse/C3 closure."""

    mesh: int
    k_fractional: np.ndarray
    reduced_steps: np.ndarray
    folding_carries: np.ndarray
    local_fields: np.ndarray
    admitted_mask: np.ndarray
    physical_q_numerators: np.ndarray
    physical_q_bohr_inv: np.ndarray
    channel_q_indices: np.ndarray
    reverse_local_field_indices: np.ndarray
    c3_pair_indices: np.ndarray
    c3_local_field_indices: np.ndarray
    c3_rotation_fractional: np.ndarray
    source_id: str

    @property
    def nk(self) -> int:
        return int(self.mesh * self.mesh)

    @property
    def admitted_channel_count(self) -> int:
        return int(np.count_nonzero(self.admitted_mask))


@dataclass(frozen=True)
class KMeshTransferRecord:
    record_index: int
    direction_index: int
    source_index: int
    target_index: int
    mesh_step: np.ndarray
    pair: PhysicalTransferPair


@dataclass(frozen=True)
class KMeshTransferChart:
    mesh: int
    directions: np.ndarray
    records: tuple[KMeshTransferRecord, ...]
    c3_images: tuple[OrientedTransferRef, ...]
    source_id: str


def _integer_unimodular_inverse(matrix: np.ndarray) -> np.ndarray:
    integer = np.asarray(matrix, dtype=np.int64)
    if integer.shape != (3, 3):
        raise ValueError("integer symmetry matrix must have shape (3, 3)")
    determinant = int(round(float(np.linalg.det(integer))))
    if abs(determinant) != 1:
        raise ValueError("symmetry matrix must be unimodular")
    inverse = np.rint(np.linalg.inv(integer)).astype(np.int64)
    identity = np.eye(3, dtype=np.int64)
    if not np.array_equal(integer @ inverse, identity) or not np.array_equal(
        inverse @ integer, identity
    ):
        raise ValueError("failed to construct exact integer symmetry inverse")
    return inverse


def build_full_half_open_physical_q_chart(
    source: OpenMXSource,
    mesh: int,
    *,
    c3_rotation_fractional: np.ndarray,
) -> FullKMeshPhysicalQChart:
    """Close all ordered-pair ``g=0`` transfers under reverse and C3.

    Reduced mesh steps use the componentwise centered half-open interval
    ``[-mesh/2, mesh/2)`` for even meshes.  Local-field labels are admitted
    pair by pair; the global label inventory is only their deterministic
    union and must not be interpreted as a Cartesian product with all pairs.
    """

    mesh = int(mesh)
    if mesh < 2:
        raise ValueError("mesh must be at least two")
    rotation = np.asarray(c3_rotation_fractional)
    if rotation.shape != (3, 3) or not np.issubdtype(rotation.dtype, np.integer):
        raise TypeError("C3 rotation must be one integer 3x3 matrix")
    rotation = np.asarray(rotation, dtype=np.int64)
    rotation_inverse = _integer_unimodular_inverse(rotation)
    if not np.array_equal(
        rotation_inverse @ rotation_inverse @ rotation_inverse,
        np.eye(3, dtype=np.int64),
    ):
        raise ValueError("declared C3 rotation does not have order three")

    nk = mesh * mesh
    coordinates = np.asarray(
        [(i, j, 0) for i in range(mesh) for j in range(mesh)],
        dtype=np.int64,
    )
    k_fractional = coordinates.astype(np.float64) / float(mesh)
    reduced_steps = np.empty((nk, nk, 3), dtype=np.int64)
    folding_carries = np.empty((nk, nk, 3), dtype=np.int64)
    half = mesh // 2
    for target_index, target in enumerate(coordinates):
        for source_index, source_k in enumerate(coordinates):
            raw = target - source_k
            reduced = (raw + half) % mesh - half
            carry_numerator = reduced - raw
            if np.any(carry_numerator % mesh != 0):
                raise RuntimeError("nonintegral reciprocal folding carry")
            reduced_steps[target_index, source_index] = reduced
            folding_carries[target_index, source_index] = (
                carry_numerator // mesh
            )

    c3_pair_indices = np.empty((nk, nk), dtype=np.int64)
    rotated_indices = np.empty(nk, dtype=np.int64)
    for index, coordinate in enumerate(coordinates):
        rotated = coordinate @ rotation_inverse
        rotated %= mesh
        rotated_indices[index] = int(rotated[0] * mesh + rotated[1])
    for target_index in range(nk):
        for source_index in range(nk):
            c3_pair_indices[target_index, source_index] = int(
                rotated_indices[target_index] * nk
                + rotated_indices[source_index]
            )

    states: set[tuple[int, int, int, int, int]] = {
        (target, source_k, 0, 0, 0)
        for target in range(nk)
        for source_k in range(nk)
    }
    maximum_states = nk * nk * 64
    for _ in range(16):
        expanded = set(states)
        for target, source_k, gx, gy, gz in states:
            local = np.array([gx, gy, gz], dtype=np.int64)
            reverse_local = (
                -local
                - folding_carries[target, source_k]
                - folding_carries[source_k, target]
            )
            expanded.add(
                (
                    source_k,
                    target,
                    int(reverse_local[0]),
                    int(reverse_local[1]),
                    int(reverse_local[2]),
                )
            )

            mapped_target = int(rotated_indices[target])
            mapped_source = int(rotated_indices[source_k])
            q_numerator = reduced_steps[target, source_k] + mesh * local
            mapped_q_numerator = q_numerator @ rotation_inverse
            mapped_reduced = reduced_steps[mapped_target, mapped_source]
            local_numerator = mapped_q_numerator - mapped_reduced
            if np.any(local_numerator % mesh != 0):
                raise RuntimeError("C3 image has nonintegral local-field label")
            mapped_local = local_numerator // mesh
            expanded.add(
                (
                    mapped_target,
                    mapped_source,
                    int(mapped_local[0]),
                    int(mapped_local[1]),
                    int(mapped_local[2]),
                )
            )
        if expanded == states:
            break
        if len(expanded) > maximum_states:
            raise RuntimeError("physical-Q closure exceeded its finite safety bound")
        states = expanded
    else:
        raise RuntimeError("physical-Q reverse/C3 closure did not converge")

    local_fields = np.asarray(
        sorted({state[2:] for state in states}), dtype=np.int64
    )
    local_lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(local_fields)
    }
    n_local = len(local_fields)
    admitted_mask = np.zeros((n_local, nk, nk), dtype=bool)
    for target, source_k, gx, gy, gz in states:
        admitted_mask[local_lookup[(gx, gy, gz)], target, source_k] = True

    q_numerator_set = {
        tuple(
            int(value)
            for value in (
                reduced_steps[target, source_k]
                + mesh * np.array([gx, gy, gz], dtype=np.int64)
            )
        )
        for target, source_k, gx, gy, gz in states
    }
    q_numerators = np.asarray(
        sorted(
            q_numerator_set,
            key=lambda item: (
                item[0] * item[0] + item[1] * item[1] + item[2] * item[2],
                item,
            ),
        ),
        dtype=np.int64,
    )
    q_lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(q_numerators)
    }
    q_bohr_inv = (
        q_numerators.astype(np.float64) / float(mesh)
    ) @ source.reciprocal_bohr
    channel_q_indices = np.full((n_local, nk, nk), -1, dtype=np.int64)
    reverse_local_indices = np.full_like(channel_q_indices, -1)
    c3_local_indices = np.full_like(channel_q_indices, -1)

    state_lookup = set(states)
    for local_index, local in enumerate(local_fields):
        for target in range(nk):
            for source_k in range(nk):
                if not admitted_mask[local_index, target, source_k]:
                    continue
                q_numerator = reduced_steps[target, source_k] + mesh * local
                channel_q_indices[local_index, target, source_k] = q_lookup[
                    tuple(int(value) for value in q_numerator)
                ]
                reverse_local = (
                    -local
                    - folding_carries[target, source_k]
                    - folding_carries[source_k, target]
                )
                reverse_state = (
                    source_k,
                    target,
                    int(reverse_local[0]),
                    int(reverse_local[1]),
                    int(reverse_local[2]),
                )
                if reverse_state not in state_lookup:
                    raise RuntimeError("reverse channel is absent from closure")
                reverse_local_indices[local_index, target, source_k] = local_lookup[
                    tuple(int(value) for value in reverse_local)
                ]

                mapped_pair = int(c3_pair_indices[target, source_k])
                mapped_target, mapped_source = divmod(mapped_pair, nk)
                mapped_q_numerator = q_numerator @ rotation_inverse
                mapped_reduced = reduced_steps[mapped_target, mapped_source]
                mapped_local_numerator = mapped_q_numerator - mapped_reduced
                if np.any(mapped_local_numerator % mesh != 0):
                    raise RuntimeError("C3 channel label is nonintegral")
                mapped_local = mapped_local_numerator // mesh
                mapped_state = (
                    mapped_target,
                    mapped_source,
                    int(mapped_local[0]),
                    int(mapped_local[1]),
                    int(mapped_local[2]),
                )
                if mapped_state not in state_lookup:
                    raise RuntimeError("C3 channel is absent from closure")
                c3_local_indices[local_index, target, source_k] = local_lookup[
                    tuple(int(value) for value in mapped_local)
                ]

    for local_index in range(n_local):
        for target in range(nk):
            for source_k in range(nk):
                if not admitted_mask[local_index, target, source_k]:
                    continue
                reverse_local = int(
                    reverse_local_indices[local_index, target, source_k]
                )
                if (
                    int(reverse_local_indices[reverse_local, source_k, target])
                    != local_index
                ):
                    raise RuntimeError("reverse channel map is not an involution")
                current_local = local_index
                current_target = target
                current_source = source_k
                for _ in range(3):
                    next_local = int(
                        c3_local_indices[
                            current_local, current_target, current_source
                        ]
                    )
                    mapped_pair = int(
                        c3_pair_indices[current_target, current_source]
                    )
                    current_target, current_source = divmod(mapped_pair, nk)
                    current_local = next_local
                if (current_local, current_target, current_source) != (
                    local_index,
                    target,
                    source_k,
                ):
                    raise RuntimeError("C3 channel map does not close")

    q_counts = np.bincount(
        channel_q_indices[channel_q_indices >= 0], minlength=len(q_numerators)
    )
    if not np.all(q_counts == nk):
        raise RuntimeError("each physical Q must have exactly Nk pair representatives")

    arrays = (
        k_fractional,
        reduced_steps,
        folding_carries,
        local_fields,
        admitted_mask,
        q_numerators,
        q_bohr_inv,
        channel_q_indices,
        reverse_local_indices,
        c3_pair_indices,
        c3_local_indices,
        rotation,
    )
    for array in arrays:
        array.setflags(write=False)
    return FullKMeshPhysicalQChart(
        mesh=mesh,
        k_fractional=k_fractional,
        reduced_steps=reduced_steps,
        folding_carries=folding_carries,
        local_fields=local_fields,
        admitted_mask=admitted_mask,
        physical_q_numerators=q_numerators,
        physical_q_bohr_inv=q_bohr_inv,
        channel_q_indices=channel_q_indices,
        reverse_local_field_indices=reverse_local_indices,
        c3_pair_indices=c3_pair_indices,
        c3_local_field_indices=c3_local_indices,
        c3_rotation_fractional=rotation,
        source_id=source.input_sha256,
    )


def write_full_physical_q_chart_npz(
    path: str | Path,
    chart: FullKMeshPhysicalQChart,
) -> None:
    """Atomically write a pickle-free full physical-Q chart artifact."""

    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    arrays = {
        "schema": np.asarray("ptse2_full_physical_q_chart/v1"),
        "mesh": np.asarray(chart.mesh, dtype=np.int64),
        "source_id": np.asarray(chart.source_id),
        "k_fractional": chart.k_fractional,
        "reduced_steps": chart.reduced_steps,
        "folding_carries": chart.folding_carries,
        "local_fields": chart.local_fields,
        "admitted_mask": chart.admitted_mask,
        "physical_q_numerators": chart.physical_q_numerators,
        "physical_q_bohr_inv": chart.physical_q_bohr_inv,
        "channel_q_indices": chart.channel_q_indices,
        "reverse_local_field_indices": chart.reverse_local_field_indices,
        "c3_pair_indices": chart.c3_pair_indices,
        "c3_local_field_indices": chart.c3_local_field_indices,
        "c3_rotation_fractional": chart.c3_rotation_fractional,
    }
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def write_physical_q_vectors(
    path: str | Path,
    chart: FullKMeshPhysicalQChart,
) -> None:
    """Atomically write Cartesian physical-Q vectors in oracle units."""

    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write("# qx qy qz [bohr^-1], ordered by exact integer Q numerator\n")
        for vector in chart.physical_q_bohr_inv:
            handle.write(" ".join(f"{float(value):.17e}" for value in vector) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def build_half_open_nearest_transfer_chart(
    source: OpenMXSource,
    mesh: int,
    *,
    directions: Iterable[Iterable[int]] = (
        (1, 0, 0),
        (0, 1, 0),
        (-1, 1, 0),
    ),
    c3_rotation_fractional: np.ndarray | None = None,
) -> KMeshTransferChart:
    """Build all nearest-neighbor transfers and their oriented C3 images."""

    if mesh < 2:
        raise ValueError("mesh must be at least two")
    direction_array = np.asarray(tuple(tuple(row) for row in directions))
    if direction_array.ndim != 2 or direction_array.shape[1] != 3:
        raise ValueError("directions must have shape (n_direction, 3)")
    if not np.issubdtype(direction_array.dtype, np.integer):
        raise TypeError("directions must use an integer dtype")
    direction_array = np.asarray(direction_array, dtype=np.int64)
    if len({tuple(row) for row in direction_array}) != len(direction_array):
        raise ValueError("directions must be unique")

    records: list[KMeshTransferRecord] = []
    oriented_lookup: dict[
        tuple[int, int, tuple[int, int, int]], OrientedTransferRef
    ] = {}
    for direction_index, step in enumerate(direction_array):
        for source_index in range(mesh * mesh):
            i, j = divmod(source_index, mesh)
            source_k = np.array([i / mesh, j / mesh, 0.0])
            target_unfolded = source_k + step / mesh
            target_k = np.mod(target_unfolded, 1.0)
            carry = np.rint(target_unfolded - target_k).astype(np.int64)
            target_grid = np.rint(target_k[:2] * mesh).astype(np.int64) % mesh
            target_index = int(target_grid[0] * mesh + target_grid[1])
            pair = PhysicalTransferPair.create(
                source,
                target_k,
                source_k,
                carry,
                -carry,
                [0, 0, 0],
            )
            record_index = len(records)
            record = KMeshTransferRecord(
                record_index=record_index,
                direction_index=direction_index,
                source_index=source_index,
                target_index=target_index,
                mesh_step=step.copy(),
                pair=pair,
            )
            records.append(record)
            forward_key = (target_index, source_index, tuple(int(v) for v in step))
            reverse_key = (source_index, target_index, tuple(int(v) for v in -step))
            if forward_key in oriented_lookup or reverse_key in oriented_lookup:
                raise RuntimeError("duplicate oriented transfer in chart")
            oriented_lookup[forward_key] = OrientedTransferRef(record_index, 1)
            oriented_lookup[reverse_key] = OrientedTransferRef(record_index, -1)

    if c3_rotation_fractional is None:
        c3_images = tuple(OrientedTransferRef(index, 1) for index in range(len(records)))
    else:
        rotation = np.asarray(c3_rotation_fractional, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("C3 rotation must have shape (3, 3)")
        rotation_integer = np.rint(rotation).astype(np.int64)
        if not np.allclose(rotation, rotation_integer, atol=1.0e-12, rtol=0.0):
            raise ValueError("C3 rotation must be integer-valued")
        rotation_inverse = np.rint(np.linalg.inv(rotation_integer)).astype(np.int64)

        def rotate_index(index: int) -> int:
            i, j = divmod(index, mesh)
            rotated = np.array([i, j, 0], dtype=np.int64) @ rotation_inverse
            rotated %= mesh
            return int(rotated[0] * mesh + rotated[1])

        images: list[OrientedTransferRef] = []
        for record in records:
            rotated_step = tuple(
                int(value) for value in record.mesh_step @ rotation_inverse
            )
            key = (
                rotate_index(record.target_index),
                rotate_index(record.source_index),
                rotated_step,
            )
            try:
                images.append(oriented_lookup[key])
            except KeyError as exc:
                raise ValueError(
                    f"C3 image is outside the oriented transfer chart: {key}"
                ) from exc
        c3_images = tuple(images)
        for index in range(len(records)):
            current = OrientedTransferRef(index, 1)
            for _ in range(3):
                image = c3_images[current.record_index]
                current = OrientedTransferRef(
                    image.record_index,
                    current.orientation * image.orientation,
                )
            if current != OrientedTransferRef(index, 1):
                raise RuntimeError("C3 transfer map does not close after three steps")

    return KMeshTransferChart(
        mesh=int(mesh),
        directions=direction_array,
        records=tuple(records),
        c3_images=c3_images,
        source_id=source.input_sha256,
    )


def find_q_index(
    q_inventory_bohr_inv: np.ndarray,
    transfer: PhysicalTransfer,
    *,
    tolerance: float = 1.0e-12,
) -> int:
    inventory = np.asarray(q_inventory_bohr_inv, dtype=np.float64)
    if inventory.ndim != 2 or inventory.shape[1] != 3:
        raise ValueError("Q inventory must have shape (n_q, 3)")
    residual = np.linalg.norm(inventory - transfer.q_bohr_inv[None, :], axis=1)
    matches = np.flatnonzero(residual <= tolerance)
    if matches.size != 1:
        raise ValueError(
            f"expected exactly one oracle Q match, found {matches.size}; "
            f"minimum residual={float(np.min(residual, initial=np.inf))}"
        )
    return int(matches[0])
