"""Reader and Bloch assembler for the private OpenMX PAO-grid vertex oracle."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import scipy.sparse

from .source import OpenMXSource, assemble_super_gauge_transfer_matrix
from .transfers import PhysicalTransferPair, find_q_index

_MAGIC = b"OMXPAOVERTEX1\x00\x00\x00"
_VERSION = 1


@dataclass(frozen=True)
class PAOVertexOracleMetadata:
    shard_paths: tuple[Path, ...]
    rank_count: int
    atom_count: int
    q_vectors_bohr_inv: np.ndarray
    local_block_counts: tuple[int, ...]
    binding_verified: bool = False
    source_id: str | None = None
    source_input_sha256: str | None = None
    source_structure_digest: str | None = None
    source_class: str | None = None
    anchor_overlap_sha256: str | None = None
    operator_convention: str | None = None


@dataclass(frozen=True)
class PAOVertexBlock:
    central_atom: int
    neighbor_atom: int
    cell: tuple[int, int, int]
    values: np.ndarray


@dataclass(frozen=True)
class ActiveVertexProjectionRequest:
    k_target_fractional: np.ndarray
    k_source_fractional: np.ndarray
    target_coefficients: np.ndarray
    source_coefficients: np.ndarray


@dataclass(frozen=True)
class ReversePairedVertices:
    """Raw control-variate pair plus its declared reverse-Hermitian projection."""

    raw_forward: scipy.sparse.csr_matrix
    raw_reverse: scipy.sparse.csr_matrix
    projected_forward: scipy.sparse.csr_matrix
    projected_reverse: scipy.sparse.csr_matrix


def read_oracle_metadata(shard_paths: Sequence[str | Path]) -> PAOVertexOracleMetadata:
    paths = tuple(Path(path).resolve() for path in shard_paths)
    if not paths:
        raise ValueError("at least one OpenMX vertex-oracle shard is required")
    records = [_read_header(path) for path in paths]
    ranks = [record[0] for record in records]
    rank_count = records[0][1]
    atom_count = records[0][2]
    q_vectors = records[0][4]
    if rank_count != len(paths) or sorted(ranks) != list(range(rank_count)):
        raise ValueError("vertex-oracle shards do not form one complete MPI rank set")
    for _, candidate_rank_count, candidate_atoms, _, candidate_q in records[1:]:
        if candidate_rank_count != rank_count or candidate_atoms != atom_count:
            raise ValueError("inconsistent vertex-oracle shard headers")
        if not np.array_equal(candidate_q, q_vectors):
            raise ValueError("vertex-oracle Q vectors differ between shards")
    ordered = sorted(zip(ranks, paths, records, strict=True))
    return PAOVertexOracleMetadata(
        shard_paths=tuple(item[1] for item in ordered),
        rank_count=rank_count,
        atom_count=atom_count,
        q_vectors_bohr_inv=q_vectors,
        local_block_counts=tuple(item[2][3] for item in ordered),
    )


def read_bound_oracle_metadata(
    manifest_path: str | Path,
    source: OpenMXSource,
    anchor_overlap_path: str | Path,
    *,
    expected_source_class: str,
) -> PAOVertexOracleMetadata:
    """Load oracle shards through a hash- and source-bound sidecar."""

    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema") != "ptse2_openmx_pao_vertex_oracle/v1":
        raise ValueError("unsupported PAO vertex-oracle manifest schema")
    if payload.get("source_class") != expected_source_class:
        raise ValueError("oracle source class does not match the requested lane")
    if payload.get("source_input_sha256") != source.input_sha256:
        raise ValueError("oracle OpenMX input hash does not match source")
    if payload.get("source_structure_digest") != source.structure_digest:
        raise ValueError("oracle ordered structure digest does not match source")
    if payload.get("operator_convention") != "absolute_exp_plus_i_Q_dot_r":
        raise ValueError("unsupported oracle operator/sign convention")
    if payload.get("q_units") != "bohr^-1_cartesian":
        raise ValueError("unsupported oracle Q units")

    anchor_path = Path(anchor_overlap_path).resolve()
    anchor_hash = _sha256_file(anchor_path)
    if payload.get("anchor_overlap_sha256") != anchor_hash:
        raise ValueError("oracle anchor-overlap hash mismatch")

    shard_paths: list[Path] = []
    for item in payload["shards"]:
        shard = (path.parent / item["file"]).resolve()
        if _sha256_file(shard) != item["sha256"]:
            raise ValueError(f"oracle shard hash mismatch: {shard}")
        shard_paths.append(shard)
    metadata = read_oracle_metadata(shard_paths)
    manifest_q = np.asarray(payload["q_vectors_bohr_inv"], dtype=np.float64)
    if not np.array_equal(metadata.q_vectors_bohr_inv, manifest_q):
        raise ValueError("oracle shard and manifest Q inventories differ")
    return replace(
        metadata,
        binding_verified=True,
        source_id=str(payload["source_id"]),
        source_input_sha256=source.input_sha256,
        source_structure_digest=source.structure_digest,
        source_class=expected_source_class,
        anchor_overlap_sha256=anchor_hash,
        operator_convention=str(payload["operator_convention"]),
    )


def iter_oracle_blocks(
    metadata: PAOVertexOracleMetadata,
    q_index: int,
) -> Iterator[PAOVertexBlock]:
    if q_index < 0 or q_index >= metadata.q_vectors_bohr_inv.shape[0]:
        raise IndexError("q_index is outside the oracle Q inventory")
    for path, local_block_count in zip(
        metadata.shard_paths, metadata.local_block_counts, strict=True
    ):
        with path.open("rb") as handle:
            _skip_header(handle)
            for current_q in range(metadata.q_vectors_bohr_inv.shape[0]):
                marker = _read_exact_array(handle, np.dtype("<i4"), 1)
                if int(marker[0]) != current_q:
                    raise ValueError(f"invalid Q marker in {path}")
                for _ in range(local_block_count):
                    record = _read_exact_array(handle, np.dtype("<i4"), 7)
                    central, neighbor, r1, r2, r3, nrow, ncol = (
                        int(value) for value in record
                    )
                    raw = _read_exact_array(
                        handle, np.dtype("<f8"), 2 * nrow * ncol
                    )
                    if current_q == q_index:
                        values = raw.reshape(nrow, ncol, 2)
                        yield PAOVertexBlock(
                            central_atom=central - 1,
                            neighbor_atom=neighbor - 1,
                            cell=(r1, r2, r3),
                            values=values[..., 0] + 1j * values[..., 1],
                        )
            if handle.read(1):
                raise ValueError(f"trailing bytes in vertex-oracle shard {path}")


def iter_oracle_shard_q_blocks(
    metadata: PAOVertexOracleMetadata,
    shard_rank: int,
) -> Iterator[tuple[int, tuple[PAOVertexBlock, ...]]]:
    """Stream every Q channel from one shard with a single sequential read."""

    if shard_rank < 0 or shard_rank >= metadata.rank_count:
        raise IndexError("shard_rank is outside the oracle shard inventory")
    path = metadata.shard_paths[shard_rank]
    local_block_count = metadata.local_block_counts[shard_rank]
    with path.open("rb") as handle:
        _skip_header(handle)
        for current_q in range(metadata.q_vectors_bohr_inv.shape[0]):
            marker = _read_exact_array(handle, np.dtype("<i4"), 1)
            if int(marker[0]) != current_q:
                raise ValueError(f"invalid Q marker in {path}")
            blocks: list[PAOVertexBlock] = []
            for _ in range(local_block_count):
                record = _read_exact_array(handle, np.dtype("<i4"), 7)
                central, neighbor, r1, r2, r3, nrow, ncol = (
                    int(value) for value in record
                )
                raw = _read_exact_array(
                    handle, np.dtype("<f8"), 2 * nrow * ncol
                )
                values = raw.reshape(nrow, ncol, 2)
                blocks.append(
                    PAOVertexBlock(
                        central_atom=central - 1,
                        neighbor_atom=neighbor - 1,
                        cell=(r1, r2, r3),
                        values=values[..., 0] + 1j * values[..., 1],
                    )
                )
            yield current_q, tuple(blocks)
        if handle.read(1):
            raise ValueError(f"trailing bytes in vertex-oracle shard {path}")


def read_oracle_shard_blocks(
    metadata: PAOVertexOracleMetadata,
    q_index: int,
    shard_rank: int,
) -> tuple[PAOVertexBlock, ...]:
    """Read one Q channel from exactly one oracle shard."""

    if q_index < 0 or q_index >= metadata.q_vectors_bohr_inv.shape[0]:
        raise IndexError("q_index is outside the oracle Q inventory")
    if shard_rank < 0 or shard_rank >= metadata.rank_count:
        raise IndexError("shard_rank is outside the oracle shard inventory")
    for current_q, blocks in iter_oracle_shard_q_blocks(metadata, shard_rank):
        if current_q == q_index:
            return blocks
    raise RuntimeError("oracle Q channel was not found during shard scan")


def realspace_overlap_to_scalar_blocks(
    overlap_blocks: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    source: OpenMXSource,
    *,
    shard_rank: int = 0,
    shard_count: int = 1,
    tolerance: float = 1.0e-12,
) -> tuple[PAOVertexBlock, ...]:
    """Convert spin-diagonal sparse S(R) into distributed scalar atom blocks."""

    if shard_count < 1 or shard_rank < 0 or shard_rank >= shard_count:
        raise ValueError("invalid overlap-block shard rank/count")
    scalar_count = source.scalar_pao_count
    dimension = source.spinor_pao_count
    atom_by_scalar = np.empty(scalar_count, dtype=np.int64)
    local_by_scalar = np.empty(scalar_count, dtype=np.int64)
    offset = 0
    for atom in source.atoms:
        count = source.basis_by_species[atom.species].scalar_orbital_count
        atom_by_scalar[offset : offset + count] = atom.index
        local_by_scalar[offset : offset + count] = np.arange(count)
        offset += count
    if offset != scalar_count:
        raise RuntimeError("scalar PAO map construction failed")

    records: list[PAOVertexBlock] = []
    record_index = 0
    for cell in sorted(overlap_blocks):
        row, col, value = overlap_blocks[cell]
        row = np.asarray(row, dtype=np.int64)
        col = np.asarray(col, dtype=np.int64)
        value = np.asarray(value, dtype=np.complex128)
        if row.shape != col.shape or row.shape != value.shape:
            raise ValueError(f"inconsistent sparse overlap arrays for cell {cell}")
        if row.size and (
            row.min() < 0
            or col.min() < 0
            or row.max() >= dimension
            or col.max() >= dimension
        ):
            raise ValueError(f"overlap index outside spinor dimension for cell {cell}")
        spin_row = row >= scalar_count
        spin_col = col >= scalar_count
        off_spin = spin_row != spin_col
        if np.max(np.abs(value[off_spin]), initial=0.0) > tolerance:
            raise ValueError("overlap contains a nonzero spin-off-diagonal block")
        scalar_row = row[~spin_row & ~spin_col]
        scalar_col = col[~spin_row & ~spin_col]
        scalar_value = value[~spin_row & ~spin_col]
        down_row = row[spin_row & spin_col] - scalar_count
        down_col = col[spin_row & spin_col] - scalar_count
        down_value = value[spin_row & spin_col]
        up_matrix = scipy.sparse.coo_matrix(
            (scalar_value, (scalar_row, scalar_col)),
            shape=(scalar_count, scalar_count),
        ).tocsr()
        down_matrix = scipy.sparse.coo_matrix(
            (down_value, (down_row, down_col)),
            shape=(scalar_count, scalar_count),
        ).tocsr()
        spin_difference = up_matrix - down_matrix
        if np.max(np.abs(spin_difference.data), initial=0.0) > tolerance:
            raise ValueError("spin-up/down overlap blocks differ")

        grouped: dict[tuple[int, int], np.ndarray] = {}
        for scalar_i, scalar_j, element in zip(
            scalar_row, scalar_col, scalar_value, strict=True
        ):
            atom_i = int(atom_by_scalar[scalar_i])
            atom_j = int(atom_by_scalar[scalar_j])
            key = (atom_i, atom_j)
            if key not in grouped:
                nrow = source.basis_by_species[
                    source.atoms[atom_i].species
                ].scalar_orbital_count
                ncol = source.basis_by_species[
                    source.atoms[atom_j].species
                ].scalar_orbital_count
                grouped[key] = np.zeros((nrow, ncol), dtype=np.complex128)
            grouped[key][
                int(local_by_scalar[scalar_i]), int(local_by_scalar[scalar_j])
            ] += element
        for (atom_i, atom_j), matrix in sorted(grouped.items()):
            if record_index % shard_count == shard_rank:
                records.append(
                    PAOVertexBlock(
                        central_atom=atom_i,
                        neighbor_atom=atom_j,
                        cell=cell,
                        values=matrix,
                    )
                )
            record_index += 1
    return tuple(records)


def project_oracle_blocks_active_batch(
    blocks: Sequence[PAOVertexBlock],
    source: OpenMXSource,
    requests: Sequence[ActiveVertexProjectionRequest],
) -> np.ndarray:
    """Project one local PAO-block shard for a batch of target/source states.

    The result is a partial shard contribution. Distributed callers must sum
    it over the complete bound oracle shard inventory.
    """

    if not requests:
        raise ValueError("at least one active projection request is required")
    scalar_count = source.scalar_pao_count
    target_band_count = int(requests[0].target_coefficients.shape[1])
    source_band_count = int(requests[0].source_coefficients.shape[1])
    target_cartesian = []
    source_cartesian = []
    for request in requests:
        target_k = np.asarray(request.k_target_fractional, dtype=np.float64)
        source_k = np.asarray(request.k_source_fractional, dtype=np.float64)
        target_coefficients = np.asarray(
            request.target_coefficients, dtype=np.complex128
        )
        source_coefficients = np.asarray(
            request.source_coefficients, dtype=np.complex128
        )
        if target_k.shape != (3,) or source_k.shape != (3,):
            raise ValueError("projection k points must have shape (3,)")
        if target_coefficients.shape != (
            source.spinor_pao_count,
            target_band_count,
        ):
            raise ValueError("inconsistent target active coefficient shape")
        if source_coefficients.shape != (
            source.spinor_pao_count,
            source_band_count,
        ):
            raise ValueError("inconsistent source active coefficient shape")
        target_cartesian.append(target_k @ source.reciprocal_angstrom)
        source_cartesian.append(source_k @ source.reciprocal_angstrom)
    target_cartesian_array = np.asarray(target_cartesian)
    source_cartesian_array = np.asarray(source_cartesian)

    scalar_offsets = np.empty(len(source.atoms), dtype=np.int64)
    offset = 0
    for atom in source.atoms:
        scalar_offsets[atom.index] = offset
        offset += source.basis_by_species[atom.species].scalar_orbital_count
    if offset != scalar_count:
        raise RuntimeError("scalar PAO offset construction failed")

    result = np.zeros(
        (len(requests), target_band_count, source_band_count),
        dtype=np.complex128,
    )
    for block in blocks:
        central_atom = source.atoms[block.central_atom]
        neighbor_atom = source.atoms[block.neighbor_atom]
        nrow, ncol = block.values.shape
        expected_rows = source.basis_by_species[
            central_atom.species
        ].scalar_orbital_count
        expected_cols = source.basis_by_species[
            neighbor_atom.species
        ].scalar_orbital_count
        if (nrow, ncol) != (expected_rows, expected_cols):
            raise ValueError("oracle block shape does not match PAO source")
        row = scalar_offsets[central_atom.index] + np.arange(nrow)
        col = scalar_offsets[neighbor_atom.index] + np.arange(ncol)
        target_blocks = np.stack(
            [
                np.stack(
                    [
                        request.target_coefficients[row],
                        request.target_coefficients[scalar_count + row],
                    ],
                    axis=0,
                )
                for request in requests
            ],
            axis=0,
        )
        source_blocks = np.stack(
            [
                np.stack(
                    [
                        request.source_coefficients[col],
                        request.source_coefficients[scalar_count + col],
                    ],
                    axis=0,
                )
                for request in requests
            ],
            axis=0,
        )
        applied = np.einsum(
            "ij,bsjn->bsin", block.values, source_blocks, optimize=True
        )
        contribution = np.einsum(
            "bsim,bsin->bmn", target_blocks.conj(), applied, optimize=True
        )
        r_cart = np.asarray(block.cell, dtype=np.float64) @ source.lattice_angstrom
        tau_a = np.asarray(central_atom.position_angstrom)
        tau_b = np.asarray(neighbor_atom.position_angstrom)
        phases = np.exp(
            1j * (source_cartesian_array @ r_cart)
            - 1j * (target_cartesian_array @ tau_a)
            + 1j * (source_cartesian_array @ tau_b)
        )
        result += phases[:, None, None] * contribution
    return result


def assemble_scalar_super_gauge_vertex(
    metadata: PAOVertexOracleMetadata,
    source: OpenMXSource,
    q_index: int,
    k_target_fractional: Iterable[float],
    k_source_fractional: Iterable[float],
    *,
    allow_unbound: bool = False,
) -> scipy.sparse.csr_matrix:
    """Assemble scalar PAO M_Q in the source/target super gauges.

    Oracle blocks already include the absolute ``exp(+i Q.r)`` factor.  The
    remaining Bloch phase is

    ``exp(+i k_source.R - i k_target.tau_a + i k_source.tau_b)``.
    """

    _validate_binding(metadata, source, allow_unbound=allow_unbound)
    if metadata.atom_count != len(source.atoms):
        raise ValueError("oracle/source atom counts differ")
    k_target = np.asarray(tuple(k_target_fractional), dtype=np.float64)
    k_source = np.asarray(tuple(k_source_fractional), dtype=np.float64)
    if k_target.shape != (3,) or k_source.shape != (3,):
        raise ValueError("target/source k points must each have three components")
    target_cart = k_target @ source.reciprocal_angstrom
    source_cart = k_source @ source.reciprocal_angstrom

    atom_offsets = np.empty(len(source.atoms), dtype=np.int64)
    offset = 0
    for atom in source.atoms:
        atom_offsets[atom.index] = offset
        offset += source.basis_by_species[atom.species].scalar_orbital_count
    if offset != source.scalar_pao_count:
        raise RuntimeError("scalar PAO offset construction failed")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for block in iter_oracle_blocks(metadata, q_index):
        central_atom = source.atoms[block.central_atom]
        neighbor_atom = source.atoms[block.neighbor_atom]
        expected_rows = source.basis_by_species[
            central_atom.species
        ].scalar_orbital_count
        expected_cols = source.basis_by_species[
            neighbor_atom.species
        ].scalar_orbital_count
        if block.values.shape != (expected_rows, expected_cols):
            raise ValueError("oracle block shape does not match source PAO basis")

        r_cart = np.asarray(block.cell, dtype=np.float64) @ source.lattice_angstrom
        tau_a = np.asarray(central_atom.position_angstrom)
        tau_b = np.asarray(neighbor_atom.position_angstrom)
        phase = np.exp(
            1j * float(source_cart @ r_cart)
            - 1j * float(target_cart @ tau_a)
            + 1j * float(source_cart @ tau_b)
        )
        row_indices = atom_offsets[central_atom.index] + np.arange(expected_rows)
        col_indices = atom_offsets[neighbor_atom.index] + np.arange(expected_cols)
        rows.append(np.repeat(row_indices, expected_cols))
        cols.append(np.tile(col_indices, expected_rows))
        values.append((phase * block.values).reshape(-1))

    if not rows:
        raise ValueError("oracle Q channel contains no PAO blocks")
    return scipy.sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(source.scalar_pao_count, source.scalar_pao_count),
        dtype=np.complex128,
    ).tocsr()


def assemble_reference_subtracted_scalar_vertex(
    metadata: PAOVertexOracleMetadata,
    source: OpenMXSource,
    q_index: int,
    zero_q_index: int,
    overlap_blocks: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    k_target_fractional: Iterable[float],
    k_source_fractional: Iterable[float],
    *,
    allow_unbound: bool = False,
) -> scipy.sparse.csr_matrix:
    """Known-zero-mode control variate ``S + I_grid(Q) - I_grid(0)``.

    This is a new, explicit quadrature definition.  It is not evidence that
    the raw native-grid Q=0 oracle passed the overlap identity gate.
    """

    if not np.array_equal(
        metadata.q_vectors_bohr_inv[zero_q_index], np.zeros(3, dtype=np.float64)
    ):
        raise ValueError("zero_q_index does not identify the exact Q=0 channel")
    grid_q = assemble_scalar_super_gauge_vertex(
        metadata,
        source,
        q_index,
        k_target_fractional,
        k_source_fractional,
        allow_unbound=allow_unbound,
    )
    grid_zero = assemble_scalar_super_gauge_vertex(
        metadata,
        source,
        zero_q_index,
        k_target_fractional,
        k_source_fractional,
        allow_unbound=allow_unbound,
    )
    anchor_spinor = assemble_super_gauge_transfer_matrix(
        overlap_blocks,
        source,
        k_target_fractional,
        k_source_fractional,
    )
    scalar_count = source.scalar_pao_count
    anchor = anchor_spinor[:scalar_count, :scalar_count]
    corrected = (anchor + grid_q - grid_zero).tocsr()
    corrected.eliminate_zeros()
    return corrected


def assemble_reference_subtracted_reverse_pair(
    metadata: PAOVertexOracleMetadata,
    source: OpenMXSource,
    transfer_pair: PhysicalTransferPair,
    zero_q_index: int,
    overlap_blocks: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    *,
    allow_unbound: bool = False,
) -> ReversePairedVertices:
    """Build and explicitly reverse-project one physical Q/-Q pair.

    Both unprojected matrices are retained so projection cannot be reported as
    independent evidence of reverse-Q accuracy.
    """

    if transfer_pair.forward.source_id != source.input_sha256:
        raise ValueError("physical transfer/source identity mismatch")
    if transfer_pair.reverse.source_id != source.input_sha256:
        raise ValueError("reverse physical transfer/source identity mismatch")
    forward_q_index = find_q_index(
        metadata.q_vectors_bohr_inv, transfer_pair.forward
    )
    reverse_q_index = find_q_index(
        metadata.q_vectors_bohr_inv, transfer_pair.reverse
    )
    target = transfer_pair.forward.k_target_fractional
    source_k = transfer_pair.forward.k_source_fractional
    raw_forward = assemble_reference_subtracted_scalar_vertex(
        metadata,
        source,
        forward_q_index,
        zero_q_index,
        overlap_blocks,
        target,
        source_k,
        allow_unbound=allow_unbound,
    )
    raw_reverse = assemble_reference_subtracted_scalar_vertex(
        metadata,
        source,
        reverse_q_index,
        zero_q_index,
        overlap_blocks,
        transfer_pair.reverse.k_target_fractional,
        transfer_pair.reverse.k_source_fractional,
        allow_unbound=allow_unbound,
    )
    projected_forward = ((raw_forward + raw_reverse.getH()) * 0.5).tocsr()
    projected_reverse = projected_forward.getH().tocsr()
    return ReversePairedVertices(
        raw_forward=raw_forward,
        raw_reverse=raw_reverse,
        projected_forward=projected_forward,
        projected_reverse=projected_reverse,
    )


def assemble_reference_subtracted_spinor_vertex(
    metadata: PAOVertexOracleMetadata,
    source: OpenMXSource,
    q_index: int,
    zero_q_index: int,
    overlap_blocks: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    k_target_fractional: Iterable[float],
    k_source_fractional: Iterable[float],
    *,
    allow_unbound: bool = False,
) -> scipy.sparse.csr_matrix:
    scalar = assemble_reference_subtracted_scalar_vertex(
        metadata,
        source,
        q_index,
        zero_q_index,
        overlap_blocks,
        k_target_fractional,
        k_source_fractional,
        allow_unbound=allow_unbound,
    )
    return scipy.sparse.block_diag((scalar, scalar), format="csr")


def assemble_spinor_super_gauge_vertex(
    metadata: PAOVertexOracleMetadata,
    source: OpenMXSource,
    q_index: int,
    k_target_fractional: Iterable[float],
    k_source_fractional: Iterable[float],
    *,
    allow_unbound: bool = False,
) -> scipy.sparse.csr_matrix:
    scalar = assemble_scalar_super_gauge_vertex(
        metadata,
        source,
        q_index,
        k_target_fractional,
        k_source_fractional,
        allow_unbound=allow_unbound,
    )
    return scipy.sparse.block_diag((scalar, scalar), format="csr")


def _validate_binding(
    metadata: PAOVertexOracleMetadata,
    source: OpenMXSource,
    *,
    allow_unbound: bool,
) -> None:
    if not metadata.binding_verified:
        if allow_unbound:
            return
        raise ValueError(
            "unbound vertex-oracle metadata; use read_bound_oracle_metadata or "
            "set allow_unbound=True only in synthetic tests"
        )
    if metadata.source_input_sha256 != source.input_sha256:
        raise ValueError("bound oracle/source input hash mismatch")
    if metadata.source_structure_digest != source.structure_digest:
        raise ValueError("bound oracle/source ordered structure mismatch")


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_header(path: Path) -> tuple[int, int, int, int, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16)
        if magic != _MAGIC:
            raise ValueError(f"invalid vertex-oracle magic in {path}")
        header = _read_exact_array(handle, np.dtype("<i4"), 6)
        version, rank, rank_count, atom_count, q_count, block_count = (
            int(value) for value in header
        )
        if version != _VERSION:
            raise ValueError(f"unsupported vertex-oracle version {version}")
        q_vectors = _read_exact_array(handle, np.dtype("<f8"), 3 * q_count).reshape(
            q_count, 3
        )
        return rank, rank_count, atom_count, block_count, q_vectors


def _skip_header(handle) -> None:
    magic = handle.read(16)
    if magic != _MAGIC:
        raise ValueError("invalid vertex-oracle magic")
    header = _read_exact_array(handle, np.dtype("<i4"), 6)
    q_count = int(header[4])
    _read_exact_array(handle, np.dtype("<f8"), 3 * q_count)


def _read_exact_array(handle, dtype: np.dtype, count: int) -> np.ndarray:
    size = dtype.itemsize * count
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError("truncated vertex-oracle shard")
    return np.frombuffer(payload, dtype=dtype, count=count)
