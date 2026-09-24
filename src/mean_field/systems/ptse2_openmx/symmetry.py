"""PAO-space symmetry transports and super-gauge sewing for PtSe2 OpenMX data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse

from .source import OpenMXSource


@dataclass(frozen=True)
class PAOSymmetryTransport:
    """One unitary spatial operation in original OpenMX spinor-PAO order."""

    operation_index: int
    rotation_fractional: np.ndarray
    translation_fractional: np.ndarray
    atom_mappings: np.ndarray
    image_shifts: np.ndarray
    local_spinor_blocks: tuple[np.ndarray, ...]
    source_json_sha256: str
    maximum_mapping_residual: float
    maximum_local_unitarity_residual: float


@dataclass(frozen=True)
class PAOSymmetrySewing:
    """Sparse map from source-k PAOs to folded target-k PAOs."""

    matrix: scipy.sparse.csr_matrix
    source_k_fractional: np.ndarray
    target_k_unfolded_fractional: np.ndarray
    target_k_folded_fractional: np.ndarray
    target_reciprocal_carry: np.ndarray


def load_pao_symmetry_transport_json(
    path: str | Path,
    source: OpenMXSource,
    *,
    operation_index: int,
    expected_sha256: str | None = None,
) -> PAOSymmetryTransport:
    """Load and validate one operation from a replayed symmetry-transport JSON."""

    transport_path = Path(path).resolve()
    payload_bytes = transport_path.read_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if expected_sha256 is not None and payload_sha256 != expected_sha256:
        raise ValueError("symmetry-transport JSON hash mismatch")
    payload = json.loads(payload_bytes)
    if int(payload.get("natoms", -1)) != len(source.atoms):
        raise ValueError("symmetry transport/source atom-count mismatch")
    if int(payload.get("nwann_spinless", -1)) != source.scalar_pao_count:
        raise ValueError("symmetry transport/source scalar-PAO mismatch")
    if int(payload.get("nwann", -1)) != source.spinor_pao_count:
        raise ValueError("symmetry transport/source spinor-PAO mismatch")

    expected_offsets = []
    offset = 0
    for atom in source.atoms:
        expected_offsets.append(offset)
        offset += source.basis_by_species[atom.species].scalar_orbital_count
    if payload.get("atom_offsets") != expected_offsets:
        raise ValueError("symmetry transport/source atom offsets differ")

    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("symmetry transport has no operation inventory")
    try:
        operation = next(
            item for item in operations if int(item["index"]) == operation_index
        )
    except StopIteration as exc:
        raise ValueError(f"missing symmetry operation {operation_index}") from exc
    if bool(operation.get("antiunitary", False)):
        raise ValueError("antiunitary PAO transport is not supported here")

    rotation = np.asarray(operation["rotation_frac"], dtype=np.float64)
    translation = np.asarray(operation["translation_frac"], dtype=np.float64)
    mappings = np.asarray(operation["atom_mappings"], dtype=np.int64)
    image_shifts = np.asarray(operation["image_shifts"], dtype=np.int64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("invalid symmetry rotation/translation shape")
    rotation_integer = np.rint(rotation).astype(np.int64)
    if not np.allclose(rotation, rotation_integer, atol=1.0e-12, rtol=0.0):
        raise ValueError("fractional symmetry rotation is not integer-valued")
    if mappings.shape != (len(source.atoms),) or sorted(mappings.tolist()) != list(
        range(len(source.atoms))
    ):
        raise ValueError("atom mappings are not one complete permutation")
    if image_shifts.shape != (len(source.atoms), 3):
        raise ValueError("invalid atom-image-shift table")

    raw_blocks = operation.get("orbital_blocks")
    if not isinstance(raw_blocks, dict):
        raise ValueError("symmetry operation has no local orbital blocks")
    blocks: list[np.ndarray] = []
    maximum_unitarity = 0.0
    for atom in source.atoms:
        scalar_count = source.basis_by_species[
            atom.species
        ].scalar_orbital_count
        raw = np.asarray(raw_blocks[str(atom.index)], dtype=np.float64)
        if raw.shape != (2 * scalar_count, 2 * scalar_count, 2):
            raise ValueError(
                f"invalid local spinor block shape for atom {atom.index}"
            )
        block = np.asarray(raw[..., 0] + 1j * raw[..., 1], dtype=np.complex128)
        residual = block.conj().T @ block - np.eye(block.shape[0])
        maximum_unitarity = max(
            maximum_unitarity,
            float(np.max(np.abs(residual), initial=0.0)),
        )
        blocks.append(block)

    positions_fractional = np.vstack(
        [atom.position_angstrom for atom in source.atoms]
    ) @ np.linalg.inv(source.lattice_angstrom)
    maximum_mapping = 0.0
    for atom_index, target_index in enumerate(mappings):
        transformed = positions_fractional[atom_index] @ rotation.T + translation
        expected = positions_fractional[target_index] + image_shifts[atom_index]
        maximum_mapping = max(
            maximum_mapping, float(np.linalg.norm(transformed - expected))
        )
    reported_mapping = float(operation.get("max_mapping_residual", np.nan))
    if not np.isfinite(reported_mapping) or not np.isclose(
        maximum_mapping, reported_mapping, atol=1.0e-12, rtol=1.0e-6
    ):
        raise ValueError("recomputed atom-mapping residual differs from JSON")
    if maximum_unitarity > 1.0e-10:
        raise ValueError(
            "local PAO symmetry block is not unitary: "
            f"max_abs={maximum_unitarity:.16e}"
        )

    return PAOSymmetryTransport(
        operation_index=int(operation_index),
        rotation_fractional=rotation_integer,
        translation_fractional=translation,
        atom_mappings=mappings,
        image_shifts=image_shifts,
        local_spinor_blocks=tuple(blocks),
        source_json_sha256=payload_sha256,
        maximum_mapping_residual=maximum_mapping,
        maximum_local_unitarity_residual=maximum_unitarity,
    )


def write_pao_symmetry_transport_npz(
    path: str | Path,
    transport: PAOSymmetryTransport,
) -> None:
    """Write one validated transport as a compact, pickle-free NPZ."""

    arrays: dict[str, np.ndarray] = {
        "operation_index": np.asarray(transport.operation_index, dtype=np.int64),
        "rotation_fractional": transport.rotation_fractional,
        "translation_fractional": transport.translation_fractional,
        "atom_mappings": transport.atom_mappings,
        "image_shifts": transport.image_shifts,
        "source_json_sha256": np.asarray(transport.source_json_sha256),
        "maximum_mapping_residual": np.asarray(
            transport.maximum_mapping_residual, dtype=np.float64
        ),
        "maximum_local_unitarity_residual": np.asarray(
            transport.maximum_local_unitarity_residual, dtype=np.float64
        ),
        "block_count": np.asarray(
            len(transport.local_spinor_blocks), dtype=np.int64
        ),
    }
    for index, block in enumerate(transport.local_spinor_blocks):
        arrays[f"block_{index:04d}"] = block
    output = Path(path)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def load_pao_symmetry_transport_npz(
    path: str | Path,
    source: OpenMXSource,
    *,
    expected_json_sha256: str | None = None,
) -> PAOSymmetryTransport:
    """Load a compact transport and revalidate dimensions and local unitarity."""

    with np.load(Path(path), allow_pickle=False) as payload:
        block_count = int(payload["block_count"])
        if block_count != len(source.atoms):
            raise ValueError("compact symmetry/source atom-count mismatch")
        blocks = tuple(
            np.asarray(payload[f"block_{index:04d}"], dtype=np.complex128)
            for index in range(block_count)
        )
        transport = PAOSymmetryTransport(
            operation_index=int(payload["operation_index"]),
            rotation_fractional=np.asarray(
                payload["rotation_fractional"], dtype=np.int64
            ),
            translation_fractional=np.asarray(
                payload["translation_fractional"], dtype=np.float64
            ),
            atom_mappings=np.asarray(payload["atom_mappings"], dtype=np.int64),
            image_shifts=np.asarray(payload["image_shifts"], dtype=np.int64),
            local_spinor_blocks=blocks,
            source_json_sha256=str(payload["source_json_sha256"]),
            maximum_mapping_residual=float(payload["maximum_mapping_residual"]),
            maximum_local_unitarity_residual=float(
                payload["maximum_local_unitarity_residual"]
            ),
        )
    if (
        expected_json_sha256 is not None
        and transport.source_json_sha256 != expected_json_sha256
    ):
        raise ValueError("compact transport/source-JSON hash mismatch")
    if transport.rotation_fractional.shape != (3, 3):
        raise ValueError("invalid compact fractional rotation")
    if transport.translation_fractional.shape != (3,):
        raise ValueError("invalid compact fractional translation")
    if transport.atom_mappings.shape != (len(source.atoms),) or sorted(
        transport.atom_mappings.tolist()
    ) != list(range(len(source.atoms))):
        raise ValueError("invalid compact atom mapping")
    if transport.image_shifts.shape != (len(source.atoms), 3):
        raise ValueError("invalid compact image-shift table")
    positions_fractional = np.vstack(
        [atom.position_angstrom for atom in source.atoms]
    ) @ np.linalg.inv(source.lattice_angstrom)
    maximum_mapping = 0.0
    maximum_unitarity = 0.0
    for atom, block in zip(source.atoms, blocks, strict=True):
        scalar_count = source.basis_by_species[
            atom.species
        ].scalar_orbital_count
        if block.shape != (2 * scalar_count, 2 * scalar_count):
            raise ValueError("invalid compact local block shape")
        residual = block.conj().T @ block - np.eye(block.shape[0])
        maximum_unitarity = max(
            maximum_unitarity,
            float(np.max(np.abs(residual), initial=0.0)),
        )
        source_index = atom.index
        target_index = int(transport.atom_mappings[source_index])
        transformed = (
            positions_fractional[source_index]
            @ transport.rotation_fractional.T
            + transport.translation_fractional
        )
        expected = (
            positions_fractional[target_index]
            + transport.image_shifts[source_index]
        )
        maximum_mapping = max(
            maximum_mapping, float(np.linalg.norm(transformed - expected))
        )
    if maximum_unitarity > 1.0e-10 or not np.isclose(
        maximum_unitarity,
        transport.maximum_local_unitarity_residual,
        atol=1.0e-13,
        rtol=1.0e-6,
    ):
        raise ValueError("compact local-block unitarity receipt mismatch")
    if not np.isclose(
        maximum_mapping,
        transport.maximum_mapping_residual,
        atol=1.0e-12,
        rtol=1.0e-6,
    ):
        raise ValueError("compact atom-mapping receipt mismatch")
    return transport


def assemble_super_gauge_symmetry_sewing(
    source: OpenMXSource,
    transport: PAOSymmetryTransport,
    k_fractional: np.ndarray,
) -> PAOSymmetrySewing:
    """Construct the folded PAO sewing for ``k -> k W^{-1}``.

    In the super gauge, the unfolded source-atom phase is

    ``exp[2πi (k·tau_a - k'·(tau_p+L_a))]``.

    If ``k'=kbar+n``, ``chi(k')=chi(kbar) exp(+2πi n·tau)``;
    the target rows therefore acquire the corresponding positive phase.
    """

    k_source = np.asarray(k_fractional, dtype=np.float64)
    if k_source.shape != (3,):
        raise ValueError("k_fractional must have shape (3,)")
    rotation_inverse = np.linalg.inv(transport.rotation_fractional)
    k_unfolded = k_source @ rotation_inverse
    k_folded = np.mod(k_unfolded, 1.0)
    k_folded[np.isclose(k_folded, 1.0, atol=1.0e-12, rtol=0.0)] = 0.0
    carry = np.rint(k_unfolded - k_folded).astype(np.int64)
    if not np.allclose(k_unfolded, k_folded + carry, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("failed to fold symmetry-related momentum")

    positions_fractional = np.vstack(
        [atom.position_angstrom for atom in source.atoms]
    ) @ np.linalg.inv(source.lattice_angstrom)
    scalar_offsets = np.empty(len(source.atoms), dtype=np.int64)
    offset = 0
    for atom in source.atoms:
        scalar_offsets[atom.index] = offset
        offset += source.basis_by_species[atom.species].scalar_orbital_count
    if offset != source.scalar_pao_count:
        raise RuntimeError("scalar PAO offset construction failed")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    scalar_total = source.scalar_pao_count
    for source_atom in source.atoms:
        source_index = source_atom.index
        target_index = int(transport.atom_mappings[source_index])
        target_atom = source.atoms[target_index]
        source_count = source.basis_by_species[
            source_atom.species
        ].scalar_orbital_count
        target_count = source.basis_by_species[
            target_atom.species
        ].scalar_orbital_count
        if target_count != source_count:
            raise ValueError("symmetry maps atoms with different PAO counts")
        block = transport.local_spinor_blocks[source_index]
        source_local = np.concatenate(
            [
                scalar_offsets[source_index] + np.arange(source_count),
                scalar_total
                + scalar_offsets[source_index]
                + np.arange(source_count),
            ]
        )
        target_local = np.concatenate(
            [
                scalar_offsets[target_index] + np.arange(target_count),
                scalar_total
                + scalar_offsets[target_index]
                + np.arange(target_count),
            ]
        )
        tau_source = positions_fractional[source_index]
        tau_target = positions_fractional[target_index]
        image = transport.image_shifts[source_index]
        unfolded_phase = np.exp(
            2j
            * np.pi
            * (
                float(k_source @ tau_source)
                - float(k_unfolded @ (tau_target + image))
            )
        )
        folded_phase = np.exp(2j * np.pi * float(carry @ tau_target))
        values_block = unfolded_phase * folded_phase * block
        row_grid, col_grid = np.meshgrid(
            target_local, source_local, indexing="ij"
        )
        rows.append(row_grid.reshape(-1))
        cols.append(col_grid.reshape(-1))
        values.append(values_block.reshape(-1))

    matrix = scipy.sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(source.spinor_pao_count, source.spinor_pao_count),
        dtype=np.complex128,
    ).tocsr()
    matrix.eliminate_zeros()
    return PAOSymmetrySewing(
        matrix=matrix,
        source_k_fractional=k_source,
        target_k_unfolded_fractional=k_unfolded,
        target_k_folded_fractional=k_folded,
        target_reciprocal_carry=carry,
    )
