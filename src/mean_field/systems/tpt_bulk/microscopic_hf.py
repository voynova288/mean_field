"""Fail-closed microscopic Hartree--Fock inputs for accepted TPT bulk.

This module never constructs a density vertex from ``wannier90_hr.dat``,
Wannier centres, point charges, or neighbour-overlap data.  Production input is
an immutable same-parent bundle containing wavefunction-gauge information and
precomputed microscopic matrix elements

    rho_mn(k,Q) = <psi_m,k+q | exp(i Q.r) | psi_n,k>,  Q = q + G.

Hartree and Fock use the same rho inventory but carry separate, explicitly
supplied quadrature weights.  This matters at the uniform mode: the total-charge
Hartree mode is removed by the periodic background convention, while the Fock
self-cell weight remains an independent physical/numerical input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mean_field.core.hf.density_vertex import (
    DensityVertexInteractionSpec,
    DensityVertexReferenceSpec,
    PreparedDensityVertexFamily,
    SparseDensityVertex,
    hartree_fock_action,
    prepare_density_vertex_family,
)
from mean_field.core.hf.engine import (
    DensityUpdateResult,
    InteractionEnergyResult,
)
from mean_field.core.hf.occupations import global_canonical_occupations
from mean_field.core.hf.problem import HartreeFockKernel, HartreeFockProblem

from .source import (
    TPT_BULK_HR_SHA256,
    TPT_BULK_NUM_WANN,
    TPT_BULK_SOURCE_MESH,
    TPTBulkActiveEigensystem,
    TPTBulkSource,
    build_tpt_bulk_active_eigensystem,
)
from .wavefunctions import (
    TPT_BULK_POSCAR_SHA256,
    TPT_BULK_POTCAR_SHA256,
    TPT_BULK_WANNIER90_CHK_SHA256,
    TPT_BULK_WANNIER90_EIG_SHA256,
    TPT_BULK_WANNIER90_MMN_SHA256,
    TPT_BULK_WAVECAR_SHA256,
)

Array = np.ndarray

TPT_BULK_MICROSCOPIC_FORMAT = "tpt-bulk-microscopic-hf-v2"
TPT_BULK_MICROSCOPIC_MANIFEST = "MICROSCOPIC_HF_MANIFEST.json"
TPT_BULK_MICROSCOPIC_DATA = "microscopic_hf_data.npz"
TPT_BULK_ACTIVE_RANKS_1BASED = (233, 248)
TPT_BULK_ACTIVE_DIMENSION = 16
TPT_BULK_MAX_CLOSURE_TOLERANCE = 1.0e-10
TPT_BULK_MIN_INSULATING_GAP_EV = 1.0e-6
# Noncircular trust root.  Production remains disabled until a separately
# reviewed manifest digest is deliberately added in source control.
TPT_BULK_APPROVED_MICROSCOPIC_MANIFEST_SHA256: frozenset[str] = frozenset()
TPT_BULK_DENSITY_VERTEX_DEFINITION = (
    "rho_mn(k,q+G)=<psi_m,k+q|exp(i(q+G).r)|psi_n,k>"
)
_LOADER_AUTHORITY = object()

_ALLOWED_VERTEX_CONSTRUCTIONS = frozenset(
    {
        "vasp_paw_augmented_plane_wave_contraction",
        "all_electron_real_space_bloch_wavefunction_integral",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an explicit string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be explicit and nonempty")
    return text


def _require_exact_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an exact integer")
    return int(value)


def _immutable_copy(value: Array, *, dtype: np.dtype[Any]) -> Array:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    digest = _require_text(value, name=name).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return digest


def _resolve_bundle_file(root: Path, relative_path: object, *, name: str) -> Path:
    raw = Path(_require_text(relative_path, name=name))
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{name} must be a bundle-relative path")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the microscopic bundle") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing source-bound authority file: {resolved}")
    return resolved


def _read_hash_bound_bundle_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Path, bytes, str]:
    path = _resolve_bundle_file(root, record.get("path"), name=f"{name} path")
    expected = _require_sha256(record.get("sha256"), name=f"{name} sha256")
    with path.open("rb") as handle:
        data = handle.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"{name} hash mismatch")
    return path, data, actual


def _read_hash_bound_json(
    root: Path,
    record: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Mapping[str, Any], str]:
    _path, data, digest = _read_hash_bound_bundle_file(root, record, name=name)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    return _require_mapping(value, name=name), digest


def _verify_bundle_hash_map(
    root: Path,
    records: Mapping[str, Any],
    *,
    name: str,
) -> tuple[str, ...]:
    if not records:
        raise ValueError(f"{name} must be nonempty")
    hashes: list[str] = []
    for relative_path, expected_raw in records.items():
        path = _resolve_bundle_file(root, relative_path, name=f"{name} path")
        expected = _require_sha256(expected_raw, name=f"{name}[{relative_path}]")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"authority hash mismatch for {relative_path}")
        hashes.append(actual)
    return tuple(hashes)


@dataclass(frozen=True)
class TPTBulkPhysicalContract:
    """Resolved physical choices required before microscopic HF is usable."""

    construction_method: str
    density_vertex_definition: str
    wavefunction_source_kind: str
    wavefunction_source_hashes: tuple[str, ...]
    vertex_generator_sha256: str
    wavefunction_provenance: str
    paw_augmentation_provenance: str
    local_field_provenance: str
    hartree_screening: str
    fock_screening: str
    periodic_coulomb_convention: str
    fock_self_cell_convention: str
    normalization_provenance: str
    filling_provenance: str
    reference_definition: str
    double_counting_convention: str
    ensemble: str
    occupation_boundary_status: str
    total_occupied_states: int

    def __post_init__(self) -> None:
        for name in (
            "construction_method",
            "density_vertex_definition",
            "wavefunction_source_kind",
            "vertex_generator_sha256",
            "wavefunction_provenance",
            "paw_augmentation_provenance",
            "local_field_provenance",
            "hartree_screening",
            "fock_screening",
            "periodic_coulomb_convention",
            "fock_self_cell_convention",
            "normalization_provenance",
            "filling_provenance",
            "reference_definition",
            "double_counting_convention",
            "ensemble",
            "occupation_boundary_status",
        ):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), name=name),
            )
        if self.density_vertex_definition != TPT_BULK_DENSITY_VERTEX_DEFINITION:
            raise ValueError("density_vertex_definition does not match the implemented rho convention")
        if self.construction_method not in _ALLOWED_VERTEX_CONSTRUCTIONS:
            raise ValueError(
                "unsupported density-vertex construction; HR eigenvector overlaps, "
                "point-centre phases, identity form factors, and .mmn neighbour "
                "overlaps are not microscopic rho(k,q+G)"
            )
        allowed_source_kinds = {
            "vasp_ncl_wavecar_potcar_paw",
            "spinor_all_electron_real_space_bloch_wavefunctions",
        }
        if self.wavefunction_source_kind not in allowed_source_kinds:
            raise ValueError(
                "wavefunction source is not a PAW-complete/all-electron psi authority"
            )
        expected_source_kind = {
            "vasp_paw_augmented_plane_wave_contraction": (
                "vasp_ncl_wavecar_potcar_paw"
            ),
            "all_electron_real_space_bloch_wavefunction_integral": (
                "spinor_all_electron_real_space_bloch_wavefunctions"
            ),
        }[self.construction_method]
        if self.wavefunction_source_kind != expected_source_kind:
            raise ValueError("density-vertex construction and psi source kind disagree")
        hashes = tuple(
            _require_sha256(value, name="wavefunction source hash")
            for value in self.wavefunction_source_hashes
        )
        if not hashes:
            raise ValueError("wavefunction_source_hashes must contain SHA-256 digests")
        generator_hash = _require_sha256(
            self.vertex_generator_sha256, name="vertex_generator_sha256"
        )
        object.__setattr__(self, "wavefunction_source_hashes", hashes)
        object.__setattr__(self, "vertex_generator_sha256", generator_hash)
        if self.reference_definition != "accepted_hr_neutral_fixed_rank_projector":
            raise ValueError(
                "this adapter requires the accepted-HR neutral fixed-rank projector"
            )
        if self.ensemble != "canonical_zero_temperature_fixed_rank":
            raise ValueError(
                "this adapter currently supports only an explicitly authorized "
                "canonical zero-temperature fixed-rank ensemble"
            )
        if self.occupation_boundary_status != "insulating":
            raise ValueError(
                "fixed-rank TPT bulk HF requires a source-authorized insulating "
                "occupation boundary; metallic/fractional occupations need a "
                "separate finite-temperature adapter"
            )
        occupied = _require_exact_int(
            self.total_occupied_states, name="total_occupied_states"
        )
        if occupied <= 0:
            raise ValueError("total_occupied_states must be positive")
        object.__setattr__(self, "total_occupied_states", occupied)


@dataclass(frozen=True)
class TPTBulkInteractionModel:
    """Validated rho inventory and separate Hartree/Fock quadratures."""

    mesh: tuple[int, int, int]
    dimension: int
    channel_labels: Array
    q_mesh_shifts: Array
    g_integer: Array
    physical_q_numerators: Array
    hartree_vertices: tuple[SparseDensityVertex, ...]
    fock_vertices: tuple[SparseDensityVertex, ...]
    prepared_hartree: PreparedDensityVertexFamily
    prepared_fock: PreparedDensityVertexFamily
    hartree_spec: DensityVertexInteractionSpec
    fock_spec: DensityVertexInteractionSpec
    references: DensityVertexReferenceSpec
    reference_projector_ket: Array
    contract: TPTBulkPhysicalContract
    provenance: str
    _authority_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority_token not in (None, _LOADER_AUTHORITY):
            raise ValueError("invalid TPT bulk interaction authority token")

    @property
    def nk(self) -> int:
        return int(np.prod(self.mesh))


@dataclass(frozen=True)
class TPTBulkInteractionAction:
    """Separated microscopic Hartree and Fock self-energies."""

    hartree: Array
    fock: Array
    total: Array


@dataclass(frozen=True)
class TPTBulkInteractionEnergyEvaluation:
    """One action and its matching per-cell scalar functional."""

    action: TPTBulkInteractionAction
    energy_ev_per_cell: float


@dataclass(frozen=True)
class TPTBulkMicroscopicHFInputs:
    """Accepted-parent input bundle ready for action/energy evaluation."""

    source: TPTBulkSource
    eigensystem: TPTBulkActiveEigensystem
    psi_wannier: Array
    h0_ket_ev: Array
    interaction: TPTBulkInteractionModel
    manifest: Mapping[str, Any]
    manifest_path: Path
    data_path: Path
    data_sha256: str
    max_psi_orthonormality_residual: float
    min_parent_subspace_singular_value: float
    occupation_boundary_gap_ev: float
    _authority_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority_token is not _LOADER_AUTHORITY:
            raise ValueError("TPT bulk microscopic inputs must be issued by the loader")
        if self.interaction._authority_token is not _LOADER_AUTHORITY:
            raise ValueError("TPT bulk interaction lacks loader authority")


@dataclass
class TPTBulkHFState:
    """Repository stored-orientation state for the generic HF engine."""

    h0: Array
    density: Array
    hamiltonian: Array
    energies: Array
    mu: float
    precision: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def from_inputs(
        cls,
        inputs: TPTBulkMicroscopicHFInputs,
        *,
        precision: float,
    ) -> "TPTBulkHFState":
        if inputs._authority_token is not _LOADER_AUTHORITY:
            raise ValueError("TPT bulk HF state requires loader-issued inputs")
        value = float(precision)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("precision must be finite and positive")
        h0 = _ket_hamiltonian_to_stored(inputs.h0_ket_ev)
        nb, _, nk = h0.shape
        return cls(
            h0=h0,
            density=np.zeros_like(h0),
            hamiltonian=h0.copy(),
            energies=np.zeros((nb, nk), dtype=float),
            mu=0.0,
            precision=value,
        )


def _mesh_tuple(mesh: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(_require_exact_int(item, name="mesh component") for item in mesh)
    if len(values) != 3 or any(item <= 0 for item in values):
        raise ValueError("mesh must contain three positive exact integers")
    return values


def _target_indices_for_shift(
    mesh: tuple[int, int, int], q_shift: Array
) -> Array:
    nx, ny, nz = mesh
    shift = np.asarray(q_shift, dtype=np.int64)
    out = np.empty(nx * ny * nz, dtype=np.int64)
    source = 0
    # Match the accepted Wannier90 k-point list: ix varies fastest.
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                tx = (ix + int(shift[0])) % nx
                ty = (iy + int(shift[1])) % ny
                tz = (iz + int(shift[2])) % nz
                out[source] = (tz * ny + ty) * nx + tx
                source += 1
    return out


def _validate_reference_projector(
    reference: Array,
    *,
    nk: int,
    dimension: int,
    total_occupied_states: int,
    tolerance: float,
) -> Array:
    projector = np.asarray(reference, dtype=np.complex128)
    if projector.shape != (nk, dimension, dimension):
        raise ValueError(
            "reference_projector_ket must have shape "
            f"{(nk, dimension, dimension)}, got {projector.shape}"
        )
    if not np.all(np.isfinite(projector)):
        raise ValueError("reference_projector_ket must be finite")
    hermiticity = float(
        np.max(np.abs(projector - projector.conj().transpose(0, 2, 1)))
    )
    if hermiticity > tolerance:
        raise ValueError("reference_projector_ket is not Hermitian")
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (projector + projector.conj().transpose(0, 2, 1))
    )
    if float(np.min(eigenvalues)) < -tolerance or float(np.max(eigenvalues)) > 1 + tolerance:
        raise ValueError("reference_projector_ket spectrum lies outside [0,1]")
    particle_number = float(
        np.trace(projector, axis1=1, axis2=2).real.sum()
    )
    if abs(particle_number - total_occupied_states) > tolerance * max(1, nk):
        raise ValueError(
            "reference_projector_ket trace does not match total_occupied_states"
        )
    idempotency = float(np.max(np.abs(projector @ projector - projector)))
    if idempotency > tolerance:
        raise ValueError(
            "zero-temperature fixed-rank reference_projector_ket is not idempotent"
        )
    return _immutable_copy(projector, dtype=np.dtype(np.complex128))


def _dense_vertex(
    *,
    label: int,
    target_by_source: Array,
    rho_channel: Array,
    weight: float,
    provenance: str,
) -> SparseDensityVertex:
    nk, dimension, _ = rho_channel.shape
    entries_per_k = dimension * dimension
    source_k = np.repeat(np.arange(nk, dtype=np.int64), entries_per_k)
    target_k = np.repeat(target_by_source, entries_per_k)
    rows_one = np.repeat(np.arange(dimension, dtype=np.int64), dimension)
    columns_one = np.tile(np.arange(dimension, dtype=np.int64), dimension)
    rows = np.tile(rows_one, nk)
    columns = np.tile(columns_one, nk)
    values = np.asarray(rho_channel, dtype=np.complex128).reshape(-1)
    # SparseDensityVertex uses structural absence for exact zeros.  Retaining
    # dense zero entries would make Gamma0 differ structurally from identity.
    nonzero = values != 0.0
    if not np.any(nonzero):
        raise ValueError(f"density channel {label} is identically zero")
    return SparseDensityVertex(
        label=(int(label), 0),
        target_k=target_k[nonzero],
        source_k=source_k[nonzero],
        rows=rows[nonzero],
        columns=columns[nonzero],
        values=values[nonzero],
        weight=float(weight),
        provenance=provenance,
    )


def _build_tpt_bulk_interaction_model(
    *,
    mesh: Sequence[int],
    rho: Array,
    channel_labels: Array,
    q_mesh_shifts: Array,
    g_integer: Array,
    hartree_weight_ev: Array,
    fock_weight_ev: Array,
    reference_projector_ket: Array,
    contract: TPTBulkPhysicalContract,
    closure_tolerance: float,
    provenance: str,
    require_complete_q_mesh: bool,
    _authority_token: object | None = None,
) -> TPTBulkInteractionModel:
    """Validate physical-Q rho data and prepare separate H/F contractions.

    This constructor is also the manufactured-test entry point.  Production
    authority additionally requires :func:`load_tpt_bulk_microscopic_hf_inputs`,
    which binds the inventory to the accepted HR hash and supplied ``psi`` gauge.
    """

    resolved_mesh = _mesh_tuple(mesh)
    tolerance = float(closure_tolerance)
    if (
        not math.isfinite(tolerance)
        or tolerance < 0.0
        or tolerance > TPT_BULK_MAX_CLOSURE_TOLERANCE
    ):
        raise ValueError(
            "closure_tolerance must be nonnegative and no larger than the "
            "code-owned fail-closed maximum"
        )
    provenance_text = _require_text(provenance, name="provenance")
    if type(require_complete_q_mesh) is not bool:
        raise TypeError("require_complete_q_mesh must be an explicit bool")

    labels_raw = np.asarray(channel_labels)
    q_raw = np.asarray(q_mesh_shifts)
    g_raw = np.asarray(g_integer)
    if labels_raw.dtype.kind not in "iu" or labels_raw.dtype.kind == "b":
        raise TypeError("channel_labels must be an exact integer array")
    if q_raw.dtype.kind not in "iu" or q_raw.dtype.kind == "b":
        raise TypeError("q_mesh_shifts must be an exact integer array")
    if g_raw.dtype.kind not in "iu" or g_raw.dtype.kind == "b":
        raise TypeError("g_integer must be an exact integer array")
    labels = np.asarray(labels_raw, dtype=np.int64).reshape(-1)
    q_shifts = np.asarray(q_raw, dtype=np.int64)
    g_vectors = np.asarray(g_raw, dtype=np.int64)
    n_channels = labels.size
    if q_shifts.shape != (n_channels, 3) or g_vectors.shape != (n_channels, 3):
        raise ValueError("q_mesh_shifts and g_integer must have shape (n_channel,3)")
    if np.unique(labels).size != n_channels or 0 not in labels:
        raise ValueError("channel_labels must be unique and contain exact label 0")
    by_label = {int(label): index for index, label in enumerate(labels)}
    if any(-int(label) not in by_label for label in labels):
        raise ValueError("channel_labels must be closed under exact sign reversal")

    density_vertices = np.asarray(rho, dtype=np.complex128)
    nk = int(np.prod(resolved_mesh))
    if density_vertices.ndim != 4 or density_vertices.shape[0] != n_channels:
        raise ValueError("rho must have shape (n_channel,nk,nb,nb)")
    if density_vertices.shape[1] != nk or density_vertices.shape[2] != density_vertices.shape[3]:
        raise ValueError("rho mesh or square active-basis dimensions are inconsistent")
    dimension = int(density_vertices.shape[2])
    if dimension <= 0 or not np.all(np.isfinite(density_vertices)):
        raise ValueError("rho must be finite with positive active dimension")

    h_weight_raw = np.asarray(hartree_weight_ev)
    f_weight_raw = np.asarray(fock_weight_ev)
    if h_weight_raw.dtype.kind not in "fiu" or h_weight_raw.dtype.kind == "b":
        raise TypeError("Hartree weights must be a real numeric array")
    if f_weight_raw.dtype.kind not in "fiu" or f_weight_raw.dtype.kind == "b":
        raise TypeError("Fock weights must be a real numeric array")
    h_weight = np.asarray(h_weight_raw, dtype=float).reshape(-1)
    f_weight = np.asarray(f_weight_raw, dtype=float).reshape(-1)
    if h_weight.shape != (n_channels,) or f_weight.shape != (n_channels,):
        raise ValueError("Hartree/Fock weight arrays must have shape (n_channel,)")
    if (
        not np.all(np.isfinite(h_weight))
        or not np.all(np.isfinite(f_weight))
        or np.any(h_weight < 0.0)
        or np.any(f_weight < 0.0)
    ):
        raise ValueError("Hartree/Fock Coulomb weights must be finite and nonnegative")

    mesh_array = np.asarray(resolved_mesh, dtype=np.int64)
    physical_q = q_shifts + g_vectors * mesh_array[None, :]
    if np.unique(physical_q, axis=0).shape[0] != n_channels:
        raise ValueError("physical q+G channels must be unique")
    zero_channels = np.flatnonzero(np.all(physical_q == 0, axis=1))
    if zero_channels.size != 1 or int(labels[zero_channels[0]]) != 0:
        raise ValueError("label 0 must be the unique physical q+G=0 channel")
    zero_index = int(zero_channels[0])
    if h_weight[zero_index] != 0.0:
        raise ValueError(
            "periodic background removal must set the uniform Hartree weight to zero"
        )
    nonuniform = np.ones(n_channels, dtype=bool)
    nonuniform[zero_index] = False
    if np.any(h_weight[nonuniform] <= 0.0):
        raise ValueError("only the uniform Hartree channel may have zero weight")
    if np.any(f_weight <= 0.0):
        raise ValueError("every Fock channel requires an explicit positive cell weight")

    expected_identity = np.broadcast_to(
        np.eye(dimension, dtype=np.complex128), (nk, dimension, dimension)
    )
    if not np.allclose(
        density_vertices[zero_index], expected_identity, atol=tolerance, rtol=0.0
    ):
        raise ValueError("rho(k,Q=0) must be the exact identity in the declared psi gauge")

    q_residues = np.mod(q_shifts, mesh_array[None, :])
    if require_complete_q_mesh:
        expected_residues = {
            (ix, iy, iz)
            for ix in range(resolved_mesh[0])
            for iy in range(resolved_mesh[1])
            for iz in range(resolved_mesh[2])
        }
        if {tuple(int(v) for v in row) for row in q_residues} != expected_residues:
            raise ValueError("rho inventory does not cover every source-mesh q transfer")

    target_maps = tuple(
        _target_indices_for_shift(resolved_mesh, q_shifts[index])
        for index in range(n_channels)
    )
    for index, label_value in enumerate(labels):
        label = int(label_value)
        reverse_index = by_label[-label]
        if not np.array_equal(physical_q[reverse_index], -physical_q[index]):
            raise ValueError("reverse channel does not carry exact -(q+G)")
        if abs(h_weight[index] - h_weight[reverse_index]) > tolerance:
            raise ValueError("reverse Hartree weights differ")
        if abs(f_weight[index] - f_weight[reverse_index]) > tolerance:
            raise ValueError("reverse Fock weights differ")
        target = target_maps[index]
        reverse_target = target_maps[reverse_index]
        for source_k, target_k in enumerate(target):
            if int(reverse_target[int(target_k)]) != source_k:
                raise ValueError("reverse q channel does not invert the K mapping")
            expected = density_vertices[index, source_k].conj().T
            actual = density_vertices[reverse_index, int(target_k)]
            if not np.allclose(actual, expected, atol=tolerance, rtol=0.0):
                raise ValueError("rho_-Q(k+q) is not rho_Q(k)^dagger")

    reference = _validate_reference_projector(
        reference_projector_ket,
        nk=nk,
        dimension=dimension,
        total_occupied_states=contract.total_occupied_states,
        tolerance=tolerance,
    )
    if contract.total_occupied_states >= nk * dimension:
        raise ValueError("total_occupied_states must be below active-space capacity")

    hartree_vertices = tuple(
        _dense_vertex(
            label=int(labels[index]),
            target_by_source=target_maps[index],
            rho_channel=density_vertices[index],
            weight=float(h_weight[index]),
            provenance=f"{provenance_text}; Hartree channel {int(labels[index])}",
        )
        for index in range(n_channels)
    )
    fock_vertices = tuple(
        _dense_vertex(
            label=int(labels[index]),
            target_by_source=target_maps[index],
            rho_channel=density_vertices[index],
            weight=float(f_weight[index]),
            provenance=f"{provenance_text}; Fock channel {int(labels[index])}",
        )
        for index in range(n_channels)
    )
    hartree_spec = DensityVertexInteractionSpec(
        include_hartree=True,
        include_fock=False,
        hartree_reference_policy="subtract_explicit",
        fock_reference_policy="subtract_explicit",
        zero_mode_policy="background_removed",
        closure_tolerance=tolerance,
        normalization_provenance=contract.normalization_provenance,
        unresolved_physical_choices=(),
    )
    fock_spec = DensityVertexInteractionSpec(
        include_hartree=False,
        include_fock=True,
        hartree_reference_policy="subtract_explicit",
        fock_reference_policy="subtract_explicit",
        zero_mode_policy="include_supplied",
        closure_tolerance=tolerance,
        normalization_provenance=contract.normalization_provenance,
        unresolved_physical_choices=(),
    )
    references = DensityVertexReferenceSpec(
        hartree=reference,
        fock=reference,
        provenance=(
            f"{contract.filling_provenance}; {contract.double_counting_convention}"
        ),
    )
    prepared_hartree = prepare_density_vertex_family(
        hartree_vertices,
        nk=nk,
        dimension=dimension,
        closure_tolerance=tolerance,
        zero_mode_policy="background_removed",
    )
    prepared_fock = prepare_density_vertex_family(
        fock_vertices,
        nk=nk,
        dimension=dimension,
        closure_tolerance=tolerance,
        zero_mode_policy="include_supplied",
    )
    return TPTBulkInteractionModel(
        mesh=resolved_mesh,
        dimension=dimension,
        channel_labels=_immutable_copy(labels, dtype=np.dtype(np.int64)),
        q_mesh_shifts=_immutable_copy(q_shifts, dtype=np.dtype(np.int64)),
        g_integer=_immutable_copy(g_vectors, dtype=np.dtype(np.int64)),
        physical_q_numerators=_immutable_copy(physical_q, dtype=np.dtype(np.int64)),
        hartree_vertices=hartree_vertices,
        fock_vertices=fock_vertices,
        prepared_hartree=prepared_hartree,
        prepared_fock=prepared_fock,
        hartree_spec=hartree_spec,
        fock_spec=fock_spec,
        references=references,
        reference_projector_ket=reference,
        contract=contract,
        provenance=provenance_text,
        _authority_token=_authority_token,
    )


def _evaluate_tpt_bulk_interaction(
    projector_ket: Array,
    h0_ket_ev: Array,
    model: TPTBulkInteractionModel,
    *,
    target_workers: int = 1,
    fock_tile_rows: int = 256,
) -> TPTBulkInteractionEnergyEvaluation:
    """Return microscopic H/F actions and matching per-cell energy."""

    projector = np.asarray(projector_ket, dtype=np.complex128)
    h0 = np.asarray(h0_ket_ev, dtype=np.complex128)
    expected_shape = (model.nk, model.dimension, model.dimension)
    if projector.shape != expected_shape or h0.shape != expected_shape:
        raise ValueError(f"projector and h0 must both have shape {expected_shape}")
    if not np.all(np.isfinite(projector)) or not np.all(np.isfinite(h0)):
        raise ValueError("projector and h0 must be finite")

    hartree_result = hartree_fock_action(
        projector,
        model.prepared_hartree,
        model.hartree_spec,
        model.references,
        target_workers=target_workers,
        fock_tile_rows=fock_tile_rows,
    )
    fock_result = hartree_fock_action(
        projector,
        model.prepared_fock,
        model.fock_spec,
        model.references,
        target_workers=target_workers,
        fock_tile_rows=fock_tile_rows,
    )
    hartree = hartree_result.hartree
    fock = fock_result.fock
    total = hartree + fock
    delta = projector - model.reference_projector_ket
    value = np.einsum("kab,kba->", h0, projector, optimize=True)
    value += 0.5 * np.einsum("kab,kba->", hartree, delta, optimize=True)
    value += 0.5 * np.einsum("kab,kba->", fock, delta, optimize=True)
    value /= model.nk
    imaginary_bound = 256.0 * np.finfo(float).eps * max(1.0, abs(value))
    if abs(value.imag) > imaginary_bound:
        raise RuntimeError("TPT bulk HF energy has a non-roundoff imaginary part")
    return TPTBulkInteractionEnergyEvaluation(
        action=TPTBulkInteractionAction(
            hartree=hartree,
            fock=fock,
            total=total,
        ),
        energy_ev_per_cell=float(value.real),
    )


def _tpt_bulk_interaction_action(
    projector_ket: Array,
    model: TPTBulkInteractionModel,
    *,
    target_workers: int = 1,
    fock_tile_rows: int = 256,
) -> TPTBulkInteractionAction:
    zero_h0 = np.zeros(
        (model.nk, model.dimension, model.dimension), dtype=np.complex128
    )
    return _evaluate_tpt_bulk_interaction(
        projector_ket,
        zero_h0,
        model,
        target_workers=target_workers,
        fock_tile_rows=fock_tile_rows,
    ).action


def _parse_contract(physics: Mapping[str, Any]) -> TPTBulkPhysicalContract:
    screening = _require_mapping(physics.get("screening"), name="physics.screening")
    wavefunction_sources = _require_mapping(
        physics.get("wavefunction_sources"), name="physics.wavefunction_sources"
    )
    source_hashes = _require_mapping(
        wavefunction_sources.get("sha256"),
        name="physics.wavefunction_sources.sha256",
    )
    if not source_hashes:
        raise ValueError("physics.wavefunction_sources.sha256 must be nonempty")
    generator = _require_mapping(
        physics.get("vertex_generator"), name="physics.vertex_generator"
    )
    return TPTBulkPhysicalContract(
        construction_method=physics.get("construction_method", ""),
        density_vertex_definition=physics.get("density_vertex_definition", ""),
        wavefunction_source_kind=wavefunction_sources.get("source_kind", ""),
        wavefunction_source_hashes=tuple(source_hashes.values()),
        vertex_generator_sha256=generator.get("sha256", ""),
        wavefunction_provenance=physics.get("wavefunction_provenance", ""),
        paw_augmentation_provenance=physics.get(
            "paw_augmentation_provenance", ""
        ),
        local_field_provenance=physics.get("local_field_provenance", ""),
        hartree_screening=screening.get("hartree_screening", ""),
        fock_screening=screening.get("fock_screening", ""),
        periodic_coulomb_convention=physics.get("periodic_coulomb_convention", ""),
        fock_self_cell_convention=physics.get("fock_self_cell_convention", ""),
        normalization_provenance=physics.get("normalization_provenance", ""),
        filling_provenance=physics.get("filling_provenance", ""),
        reference_definition=physics.get("reference_definition", ""),
        double_counting_convention=physics.get("double_counting_convention", ""),
        ensemble=physics.get("ensemble", ""),
        occupation_boundary_status=physics.get("occupation_boundary_status", ""),
        total_occupied_states=_require_exact_int(
            physics.get("total_occupied_states"), name="total_occupied_states"
        ),
    )


def _validate_parent_manifest(parent: Mapping[str, Any]) -> None:
    if parent.get("hr_sha256") != TPT_BULK_HR_SHA256:
        raise ValueError("microscopic bundle is not bound to the accepted HR hash")
    if _require_exact_int(parent.get("num_wann"), name="parent.num_wann") != TPT_BULK_NUM_WANN:
        raise ValueError("microscopic bundle num_wann is not 320")
    if _mesh_tuple(parent.get("source_mesh", ())) != TPT_BULK_SOURCE_MESH:
        raise ValueError("microscopic bundle must use the accepted 12x4x4 source mesh")
    ranks = tuple(
        _require_exact_int(value, name="active rank")
        for value in parent.get("active_ranks_1based", ())
    )
    if ranks != TPT_BULK_ACTIVE_RANKS_1BASED:
        raise ValueError("microscopic bundle active ranks do not match 233--248")
    gauge = _require_text(parent.get("gauge"), name="parent.gauge")
    if gauge != "psi_wannier_columns_in_accepted_parent_active_basis":
        raise ValueError("unsupported or ambiguous parent/gauge declaration")
    if parent.get("k_ordering") != "wannier90_x_fastest_signed_fractional":
        raise ValueError("microscopic bundle K ordering is not source-authorized")


def _validate_active_space_authority(record: Mapping[str, Any]) -> None:
    if record.get("schema") != "tpt-bulk-active-space-authority-v1":
        raise ValueError("unsupported active-space authority schema")
    if record.get("status") != "validated_for_microscopic_hf":
        raise ValueError("active ranks lack an external microscopic-HF validation")
    if record.get("parent_hr_sha256") != TPT_BULK_HR_SHA256:
        raise ValueError("active-space authority is bound to a different parent")
    ranks = tuple(
        _require_exact_int(value, name="active-space authority rank")
        for value in record.get("active_ranks_1based", ())
    )
    if ranks != TPT_BULK_ACTIVE_RANKS_1BASED:
        raise ValueError("active-space authority validates different ranks")
    if record.get("gauge") != "psi_wannier_columns_in_accepted_parent_active_basis":
        raise ValueError("active-space authority validates a different gauge")


def _validate_filling_authority(
    record: Mapping[str, Any],
    contract: TPTBulkPhysicalContract,
) -> None:
    if record.get("schema") != "tpt-bulk-neutral-filling-authority-v1":
        raise ValueError("unsupported neutral-filling authority schema")
    if record.get("status") != "neutral_insulating_fixed_rank":
        raise ValueError("filling authority is not neutral and insulating")
    if record.get("parent_hr_sha256") != TPT_BULK_HR_SHA256:
        raise ValueError("filling authority is bound to a different parent")
    if _mesh_tuple(record.get("source_mesh", ())) != TPT_BULK_SOURCE_MESH:
        raise ValueError("filling authority uses a different source mesh")
    if record.get("ensemble") != contract.ensemble:
        raise ValueError("filling authority and interaction ensemble disagree")
    if record.get("reference_definition") != contract.reference_definition:
        raise ValueError("filling authority and reference definition disagree")
    occupied = _require_exact_int(
        record.get("total_occupied_states"),
        name="filling authority total_occupied_states",
    )
    if occupied != contract.total_occupied_states:
        raise ValueError("manifest occupied count is not filling-authority bound")
    if _require_text(record.get("provenance"), name="filling authority provenance") != (
        contract.filling_provenance
    ):
        raise ValueError("manifest filling provenance is not authority-bound")


def _validate_vertex_receipt(
    record: Mapping[str, Any],
    *,
    contract: TPTBulkPhysicalContract,
    payload_sha256: str,
    wavefunction_hashes: Mapping[str, Any],
) -> None:
    if record.get("schema") != "tpt-bulk-density-vertex-receipt-v2":
        raise ValueError("unsupported density-vertex receipt schema")
    if record.get("status") != "complete":
        raise ValueError("density-vertex generation receipt is not complete")
    if record.get("parent_hr_sha256") != TPT_BULK_HR_SHA256:
        raise ValueError("density-vertex receipt is bound to a different parent")
    if record.get("payload_sha256") != payload_sha256:
        raise ValueError("density-vertex receipt does not bind the NPZ payload")
    if record.get("generator_sha256") != contract.vertex_generator_sha256:
        raise ValueError("density-vertex receipt does not bind the generator")
    receipt_sources = _require_mapping(
        record.get("wavefunction_source_sha256"),
        name="density-vertex receipt wavefunction_source_sha256",
    )
    expected_sources = {
        str(path): _require_sha256(digest, name=f"wavefunction hash {path}")
        for path, digest in wavefunction_hashes.items()
    }
    if dict(receipt_sources) != expected_sources:
        raise ValueError("density-vertex receipt does not bind the psi source files")
    if record.get("density_vertex_definition") != contract.density_vertex_definition:
        raise ValueError("density-vertex receipt uses a different rho definition")
    if record.get("construction_method") != contract.construction_method:
        raise ValueError("density-vertex receipt uses a different construction")
    if record.get("paw_augmentation_provenance") != (
        contract.paw_augmentation_provenance
    ):
        raise ValueError(
            "density-vertex receipt uses different PAW/all-electron provenance"
        )
    if _mesh_tuple(record.get("source_mesh", ())) != TPT_BULK_SOURCE_MESH:
        raise ValueError("density-vertex receipt uses a different source mesh")
    ranks = tuple(
        _require_exact_int(value, name="density-vertex receipt active rank")
        for value in record.get("active_ranks_1based", ())
    )
    if ranks != TPT_BULK_ACTIVE_RANKS_1BASED:
        raise ValueError("density-vertex receipt uses a different active space")
    if record.get("k_ordering") != "wannier90_x_fastest_signed_fractional":
        raise ValueError("density-vertex receipt uses a different K ordering")


def load_tpt_bulk_microscopic_hf_inputs(
    bundle_root: str | Path,
    *,
    source: TPTBulkSource,
) -> TPTBulkMicroscopicHFInputs:
    """Load a hash-bound same-parent ``psi``/``rho`` bundle.

    The accepted HR-only source directory does not satisfy this API.  Missing
    wavefunction-level data, microscopic rho, screening, filling, or reference
    metadata raises before any Hartree/Fock action can be evaluated.
    """

    root = Path(bundle_root).expanduser().resolve()
    manifest_path = root / TPT_BULK_MICROSCOPIC_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing fail-closed microscopic manifest: {manifest_path}"
        )
    with manifest_path.open("rb") as manifest_handle:
        manifest_bytes = manifest_handle.read()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 not in TPT_BULK_APPROVED_MICROSCOPIC_MANIFEST_SHA256:
        raise ValueError(
            "microscopic manifest lacks a code-reviewed external trust root"
        )
    try:
        manifest_raw = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("microscopic manifest is not valid UTF-8 JSON") from exc
    manifest = _require_mapping(manifest_raw, name="manifest")
    if manifest.get("format_version") != TPT_BULK_MICROSCOPIC_FORMAT:
        raise ValueError("unsupported TPT bulk microscopic bundle format")
    if source.hr_sha256 != TPT_BULK_HR_SHA256:
        raise ValueError("source is not the accepted TPT bulk HR parent")
    parent = _require_mapping(manifest.get("parent"), name="manifest.parent")
    _validate_parent_manifest(parent)
    active_authority_ref = _require_mapping(
        parent.get("active_space_authority"),
        name="manifest.parent.active_space_authority",
    )
    active_authority, _active_authority_hash = _read_hash_bound_json(
        root,
        active_authority_ref,
        name="active-space authority",
    )
    _validate_active_space_authority(active_authority)
    physics = _require_mapping(manifest.get("physics"), name="manifest.physics")
    contract = _parse_contract(physics)
    filling_authority_ref = _require_mapping(
        physics.get("filling_authority"), name="physics.filling_authority"
    )
    filling_authority, _filling_authority_hash = _read_hash_bound_json(
        root,
        filling_authority_ref,
        name="neutral-filling authority",
    )
    _validate_filling_authority(filling_authority, contract)
    wavefunction_sources = _require_mapping(
        physics.get("wavefunction_sources"), name="physics.wavefunction_sources"
    )
    wavefunction_hash_map = _require_mapping(
        wavefunction_sources.get("sha256"),
        name="physics.wavefunction_sources.sha256",
    )
    if contract.construction_method == "vasp_paw_augmented_plane_wave_contraction":
        expected_parent_hashes = {
            "WAVECAR": TPT_BULK_WAVECAR_SHA256,
            "POTCAR": TPT_BULK_POTCAR_SHA256,
            "POSCAR": TPT_BULK_POSCAR_SHA256,
            "wannier90.chk": TPT_BULK_WANNIER90_CHK_SHA256,
            "wannier90.eig": TPT_BULK_WANNIER90_EIG_SHA256,
            "wannier90.mmn": TPT_BULK_WANNIER90_MMN_SHA256,
        }
        provided_by_basename: dict[str, str] = {}
        for relative_path, digest in wavefunction_hash_map.items():
            basename = Path(str(relative_path)).name
            if basename in provided_by_basename:
                raise ValueError(f"duplicate PAW parent source basename {basename}")
            provided_by_basename[basename] = _require_sha256(
                digest, name=f"wavefunction hash {relative_path}"
            )
        for basename, expected_hash in expected_parent_hashes.items():
            if provided_by_basename.get(basename) != expected_hash:
                raise ValueError(
                    f"PAW density vertices require accepted-parent {basename}"
                )
    verified_wavefunction_hashes = _verify_bundle_hash_map(
        root,
        wavefunction_hash_map,
        name="wavefunction sources",
    )
    if tuple(contract.wavefunction_source_hashes) != verified_wavefunction_hashes:
        raise ValueError("wavefunction source hash order changed during verification")
    generator = _require_mapping(
        physics.get("vertex_generator"), name="physics.vertex_generator"
    )
    _generator_path, _generator_bytes, verified_generator_hash = (
        _read_hash_bound_bundle_file(root, generator, name="vertex generator")
    )
    if verified_generator_hash != contract.vertex_generator_sha256:
        raise ValueError("vertex generator hash changed during verification")
    if physics.get("hartree_reference_policy") != "subtract_explicit":
        raise ValueError("Hartree must use the explicit neutral reference projector")
    if physics.get("fock_reference_policy") != "subtract_explicit":
        raise ValueError("Fock must use the same explicit double-counting reference")
    if physics.get("zero_mode_policy") != "background_removed_hartree_only":
        raise ValueError("uniform-mode policy must remove only the Hartree charge mode")
    if physics.get("local_field_inventory_complete") is not True:
        raise ValueError("local-field inventory is not declared complete")
    if physics.get("q_mesh_inventory_complete") is not True:
        raise ValueError("q-mesh inventory is not declared complete")

    files = _require_mapping(manifest.get("files"), name="manifest.files")
    data_record = _require_mapping(
        files.get(TPT_BULK_MICROSCOPIC_DATA),
        name=f"manifest.files.{TPT_BULK_MICROSCOPIC_DATA}",
    )
    expected_data_hash = _require_sha256(
        data_record.get("sha256"), name="microscopic data sha256"
    )
    data_path = _resolve_bundle_file(
        root,
        TPT_BULK_MICROSCOPIC_DATA,
        name="microscopic data payload path",
    )
    required_arrays = {
        "k_fractional",
        "psi_wannier",
        "rho",
        "channel_labels",
        "q_mesh_shifts",
        "g_integer",
        "hartree_weight_ev",
        "fock_weight_ev",
        "reference_projector_ket",
    }
    with data_path.open("rb") as data_handle:
        data_digest = hashlib.sha256()
        for block in iter(lambda: data_handle.read(1024 * 1024), b""):
            data_digest.update(block)
        actual_data_hash = data_digest.hexdigest()
        if actual_data_hash != expected_data_hash:
            raise ValueError("microscopic data payload hash mismatch")
        data_handle.seek(0)
        with np.load(data_handle, allow_pickle=False) as payload:
            missing = required_arrays.difference(payload.files)
            if missing:
                raise ValueError(
                    "microscopic payload lacks required arrays: "
                    + ", ".join(sorted(missing))
                )
            arrays = {
                name: np.array(payload[name], copy=True) for name in required_arrays
            }
    vertex_receipt_ref = _require_mapping(
        physics.get("vertex_receipt"), name="physics.vertex_receipt"
    )
    vertex_receipt, _vertex_receipt_hash = _read_hash_bound_json(
        root,
        vertex_receipt_ref,
        name="density-vertex receipt",
    )
    _validate_vertex_receipt(
        vertex_receipt,
        contract=contract,
        payload_sha256=actual_data_hash,
        wavefunction_hashes=wavefunction_hash_map,
    )
    if not source.hr_path.is_file():
        raise FileNotFoundError("accepted TPT bulk source HR is missing")
    actual_parent_hash = _sha256(source.hr_path)
    if actual_parent_hash != TPT_BULK_HR_SHA256 or actual_parent_hash != source.hr_sha256:
        raise ValueError("live source HR hash differs from the accepted parent")

    eigensystem = build_tpt_bulk_active_eigensystem(
        source,
        mesh=TPT_BULK_SOURCE_MESH,
        active_ranks_1based=TPT_BULK_ACTIVE_RANKS_1BASED,
    )
    k_fractional = np.asarray(arrays["k_fractional"], dtype=float)
    if k_fractional.shape != eigensystem.k_fractional.shape:
        raise ValueError("microscopic payload K shape differs from accepted source mesh")
    periodic_k_delta = k_fractional - eigensystem.k_fractional
    periodic_k_delta -= np.rint(periodic_k_delta)
    if not np.allclose(periodic_k_delta, 0.0, atol=1.0e-14, rtol=0.0):
        raise ValueError("microscopic payload K ordering differs from accepted source mesh")

    psi = np.asarray(arrays["psi_wannier"], dtype=np.complex128)
    expected_psi_shape = (
        eigensystem.nk,
        TPT_BULK_NUM_WANN,
        TPT_BULK_ACTIVE_DIMENSION,
    )
    if psi.shape != expected_psi_shape or not np.all(np.isfinite(psi)):
        raise ValueError(f"psi_wannier must be finite with shape {expected_psi_shape}")
    gram = np.einsum("kwi,kwj->kij", psi.conj(), psi, optimize=True)
    identity = np.eye(TPT_BULK_ACTIVE_DIMENSION, dtype=np.complex128)[None, :, :]
    orthonormality = float(np.max(np.abs(gram - identity)))
    closure_tolerance = float(physics.get("closure_tolerance"))
    if (
        not math.isfinite(closure_tolerance)
        or closure_tolerance <= 0.0
        or closure_tolerance > TPT_BULK_MAX_CLOSURE_TOLERANCE
    ):
        raise ValueError(
            "physics.closure_tolerance exceeds the code-owned fail-closed maximum"
        )
    if orthonormality > closure_tolerance:
        raise ValueError("psi_wannier columns are not orthonormal")

    local_psi = np.transpose(eigensystem.active_eigenvectors, (2, 0, 1))
    gauge_overlap = np.einsum("kwi,kwj->kij", local_psi.conj(), psi, optimize=True)
    singular_values = np.linalg.svd(gauge_overlap, compute_uv=False)
    min_singular = float(np.min(singular_values))
    if 1.0 - min_singular > closure_tolerance:
        raise ValueError("psi_wannier does not span the accepted-parent active subspace")
    source_energies = eigensystem.active_energies_ev.T
    if contract.total_occupied_states % eigensystem.nk != 0:
        raise ValueError(
            "insulating fixed-rank authority requires an integer occupied rank per K"
        )
    occupied_per_k = contract.total_occupied_states // eigensystem.nk
    if not 0 < occupied_per_k < TPT_BULK_ACTIVE_DIMENSION:
        raise ValueError("occupied rank lies outside the declared active space")
    valence_max = float(np.max(source_energies[:, occupied_per_k - 1]))
    conduction_min = float(np.min(source_energies[:, occupied_per_k]))
    occupation_gap = conduction_min - valence_max
    gap_floor = TPT_BULK_MIN_INSULATING_GAP_EV + 64.0 * np.finfo(float).eps
    if occupation_gap <= gap_floor:
        raise ValueError(
            "declared fixed-rank boundary does not exceed the code-owned "
            "minimum insulating gap"
        )
    h0_ket = np.einsum(
        "kia,ki,kib->kab",
        gauge_overlap.conj(),
        source_energies,
        gauge_overlap,
        optimize=True,
    )
    h0_hermiticity = float(
        np.max(np.abs(h0_ket - h0_ket.conj().transpose(0, 2, 1)))
    )
    if h0_hermiticity > closure_tolerance:
        raise ValueError("parent/gauge transformed H0 is not Hermitian")
    expected_reference = np.einsum(
        "kia,kib->kab",
        gauge_overlap[:, :occupied_per_k, :].conj(),
        gauge_overlap[:, :occupied_per_k, :],
        optimize=True,
    )
    supplied_reference = np.asarray(
        arrays["reference_projector_ket"], dtype=np.complex128
    )
    if supplied_reference.shape != expected_reference.shape or not np.allclose(
        supplied_reference,
        expected_reference,
        atol=closure_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "reference_projector_ket is not the accepted-HR neutral occupied projector"
        )

    model = _build_tpt_bulk_interaction_model(
        mesh=TPT_BULK_SOURCE_MESH,
        rho=arrays["rho"],
        channel_labels=arrays["channel_labels"],
        q_mesh_shifts=arrays["q_mesh_shifts"],
        g_integer=arrays["g_integer"],
        hartree_weight_ev=arrays["hartree_weight_ev"],
        fock_weight_ev=arrays["fock_weight_ev"],
        reference_projector_ket=arrays["reference_projector_ket"],
        contract=contract,
        closure_tolerance=closure_tolerance,
        provenance=(
            f"{manifest_path}; manifest sha256={manifest_sha256}; "
            f"payload sha256={actual_data_hash}; parent HR sha256={source.hr_sha256}; "
            "gauge bound by psi_wannier"
        ),
        require_complete_q_mesh=True,
        _authority_token=_LOADER_AUTHORITY,
    )
    if model.dimension != TPT_BULK_ACTIVE_DIMENSION:
        raise ValueError("microscopic rho active dimension is not 16")
    return TPTBulkMicroscopicHFInputs(
        source=source,
        eigensystem=eigensystem,
        psi_wannier=_immutable_copy(psi, dtype=np.dtype(np.complex128)),
        h0_ket_ev=_immutable_copy(h0_ket, dtype=np.dtype(np.complex128)),
        interaction=model,
        manifest=dict(manifest),
        manifest_path=manifest_path,
        data_path=data_path,
        data_sha256=actual_data_hash,
        max_psi_orthonormality_residual=orthonormality,
        min_parent_subspace_singular_value=min_singular,
        occupation_boundary_gap_ev=occupation_gap,
        _authority_token=_LOADER_AUTHORITY,
    )


def evaluate_tpt_bulk_microscopic_hf(
    projector_ket: Array,
    inputs: TPTBulkMicroscopicHFInputs,
    *,
    target_workers: int = 1,
    fock_tile_rows: int = 256,
) -> TPTBulkInteractionEnergyEvaluation:
    """Evaluate only loader-authorized accepted-parent TPT bulk inputs."""

    if inputs._authority_token is not _LOADER_AUTHORITY:
        raise ValueError("TPT bulk microscopic HF inputs lack loader authority")
    if inputs.interaction._authority_token is not _LOADER_AUTHORITY:
        raise ValueError("TPT bulk interaction lacks loader authority")
    return _evaluate_tpt_bulk_interaction(
        projector_ket,
        inputs.h0_ket_ev,
        inputs.interaction,
        target_workers=target_workers,
        fock_tile_rows=fock_tile_rows,
    )


def _ket_projector_to_stored(projector_ket: Array) -> Array:
    projector = np.asarray(projector_ket, dtype=np.complex128)
    if projector.ndim != 3 or projector.shape[1] != projector.shape[2]:
        raise ValueError("ket projector must have shape (nk,nb,nb)")
    return np.transpose(projector, (2, 1, 0)).copy()


def _stored_projector_to_ket(stored: Array) -> Array:
    projector = np.asarray(stored, dtype=np.complex128)
    if projector.ndim != 3 or projector.shape[0] != projector.shape[1]:
        raise ValueError("stored projector must have shape (nb,nb,nk)")
    return np.transpose(projector, (2, 1, 0)).copy()


def _ket_hamiltonian_to_stored(hamiltonian_ket: Array) -> Array:
    hamiltonian = np.asarray(hamiltonian_ket, dtype=np.complex128)
    if hamiltonian.ndim != 3 or hamiltonian.shape[1] != hamiltonian.shape[2]:
        raise ValueError("ket Hamiltonian must have shape (nk,nb,nb)")
    return np.transpose(hamiltonian, (1, 2, 0)).copy()


def _stored_hamiltonian_to_ket(hamiltonian: Array) -> Array:
    stored = np.asarray(hamiltonian, dtype=np.complex128)
    if stored.ndim != 3 or stored.shape[0] != stored.shape[1]:
        raise ValueError("stored Hamiltonian must have shape (nb,nb,nk)")
    return np.transpose(stored, (2, 0, 1)).copy()


def build_tpt_bulk_hf_problem(
    inputs: TPTBulkMicroscopicHFInputs,
    *,
    mixing: float,
    fermi_degeneracy_tolerance: float,
    interaction_workers: int = 1,
    eigensolver_workers: int = 1,
    fock_tile_rows: int = 256,
) -> HartreeFockProblem:
    """Bind fully resolved zero-temperature inputs to the generic SCF engine."""

    mix = float(mixing)
    if not math.isfinite(mix) or not (0.0 < mix <= 1.0):
        raise ValueError("mixing must lie in (0,1]")
    fermi_tolerance = float(fermi_degeneracy_tolerance)
    if not math.isfinite(fermi_tolerance) or fermi_tolerance < 0.0:
        raise ValueError("fermi_degeneracy_tolerance must be finite and nonnegative")
    for name, value in (
        ("interaction_workers", interaction_workers),
        ("eigensolver_workers", eigensolver_workers),
        ("fock_tile_rows", fock_tile_rows),
    ):
        if type(value) is not int or value <= 0:
            raise TypeError(f"{name} must be an exact positive integer")

    if inputs._authority_token is not _LOADER_AUTHORITY:
        raise ValueError("TPT bulk HF requires loader-issued microscopic inputs")
    model = inputs.interaction
    if model._authority_token is not _LOADER_AUTHORITY:
        raise ValueError("TPT bulk HF requires loader-validated source authority")
    h0_expected = np.asarray(inputs.h0_ket_ev, dtype=np.complex128)

    def initialize(state: TPTBulkHFState, *, init_mode: str, seed: int) -> None:
        del seed
        if not np.array_equal(_stored_hamiltonian_to_ket(state.h0), h0_expected):
            raise ValueError("state H0 differs from source-bound microscopic inputs")
        if init_mode == "h0_global":
            occupied = global_canonical_occupations(
                h0_expected,
                total_occupied=model.contract.total_occupied_states,
                fermi_degeneracy_tolerance=fermi_tolerance,
                eigensolver_workers=eigensolver_workers,
            )
            state.density[:, :, :] = _ket_projector_to_stored(
                occupied.projector_ket
            )
            state.energies[:, :] = occupied.energies
            state.mu = occupied.chemical_potential
        elif init_mode == "reference":
            state.density[:, :, :] = _ket_projector_to_stored(
                model.reference_projector_ket
            )
            values = np.linalg.eigvalsh(h0_expected).T
            state.energies[:, :] = values
            state.mu = float(
                np.sort(values.reshape(-1, order="F"))[
                    model.contract.total_occupied_states - 1
                ]
            )
        elif init_mode == "preserve":
            projector = _stored_projector_to_ket(state.density)
            particle_number = float(
                np.trace(projector, axis1=1, axis2=2).real.sum()
            )
            if abs(particle_number - model.contract.total_occupied_states) > 1.0e-10:
                raise ValueError("preserved density has the wrong particle number")
        else:
            raise ValueError("init_mode must be 'h0_global', 'reference', or 'preserve'")
        state.hamiltonian[:, :, :] = state.h0

    def evaluate(stored_density: Array, stored_h0: Array) -> TPTBulkInteractionEnergyEvaluation:
        return _evaluate_tpt_bulk_interaction(
            _stored_projector_to_ket(stored_density),
            _stored_hamiltonian_to_ket(stored_h0),
            model,
            target_workers=interaction_workers,
            fock_tile_rows=fock_tile_rows,
        )

    def interaction_builder(stored_density: Array) -> Array:
        zero_h0 = np.zeros_like(inputs.h0_ket_ev)
        result = _evaluate_tpt_bulk_interaction(
            _stored_projector_to_ket(stored_density),
            zero_h0,
            model,
            target_workers=interaction_workers,
            fock_tile_rows=fock_tile_rows,
        )
        return _ket_hamiltonian_to_stored(result.action.total)

    def interaction_energy_builder(
        stored_density: Array, stored_h0: Array
    ) -> InteractionEnergyResult:
        result = evaluate(stored_density, stored_h0)
        return InteractionEnergyResult(
            interaction_h=_ket_hamiltonian_to_stored(result.action.total),
            energy=result.energy_ev_per_cell,
        )

    def energy_functional(
        _stored_interaction: Array,
        stored_h0: Array,
        stored_density: Array,
    ) -> float:
        return evaluate(stored_density, stored_h0).energy_ev_per_cell

    def density_builder(stored_hamiltonian: Array) -> DensityUpdateResult:
        occupied = global_canonical_occupations(
            _stored_hamiltonian_to_ket(stored_hamiltonian),
            total_occupied=model.contract.total_occupied_states,
            fermi_degeneracy_tolerance=fermi_tolerance,
            eigensolver_workers=eigensolver_workers,
        )
        return DensityUpdateResult(
            density=_ket_projector_to_stored(occupied.projector_ket),
            energies=occupied.energies,
            mu=occupied.chemical_potential,
            observables={"fermi_gap": occupied.fermi_gap},
        )

    return HartreeFockProblem(
        initializer=initialize,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_builder,
            density_builder=density_builder,
            energy_functional=energy_functional,
            interaction_energy_builder=interaction_energy_builder,
            oda_parameterizer=lambda _state, _delta: mix,
            convergence_rule="raw",
        ),
    )


__all__ = [
    "TPT_BULK_DENSITY_VERTEX_DEFINITION",
    "TPT_BULK_MICROSCOPIC_DATA",
    "TPT_BULK_MICROSCOPIC_FORMAT",
    "TPT_BULK_MICROSCOPIC_MANIFEST",
    "TPTBulkHFState",
    "TPTBulkInteractionEnergyEvaluation",
    "TPTBulkMicroscopicHFInputs",
    "build_tpt_bulk_hf_problem",
    "evaluate_tpt_bulk_microscopic_hf",
    "load_tpt_bulk_microscopic_hf_inputs",
]
