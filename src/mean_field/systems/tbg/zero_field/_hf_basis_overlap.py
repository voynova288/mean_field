from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar
import hashlib
import json
from types import MappingProxyType
from typing import Any, Literal

from ._hf_shared import *  # noqa: F401,F403

from .interaction import (
    TBGZeroFieldInteractionSpec,
    TBG_ZERO_FIELD_COMPANION_CUTOFF_PARITY,
    TBG_ZERO_FIELD_REFERENCE_SCHEME,
    TBG_ZERO_FIELD_TRANSFER_CUTOFF_POLICY,
)
from .model import (
    BMSolution,
    TBGZeroFieldTorusMesh,
    tbg_zero_field_bm_generation_fingerprint,
)
from ..params import TBGParameters


_compute_density_overlap_trace_from_diagonal = compute_density_overlap_trace_from_diagonal


TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA = "mean_field.tbg.zero_field.hf_source_receipt"
TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA_VERSION = 5
TBG_ZERO_FIELD_SCREENED_BLOCK_BUNDLE_SCHEMA = "mean_field.tbg.zero_field.screened_block_bundle"
TBG_ZERO_FIELD_SCREENED_BLOCK_BUNDLE_SCHEMA_VERSION = 2
TBG_ZERO_FIELD_CENTERED_REFERENCE_CONVENTION = (
    "D_stored[a,b]=<c_a† c_b>-0.5*delta_ab"
)
TBG_ZERO_FIELD_REFERENCE_PROJECTOR_CONVENTION = (
    "central_average_active_two_band:R=0.5*I_(spin,valley,band,k)"
)
TBG_ZERO_FIELD_ACTIVE_G_SHELL = 3
TBG_ZERO_FIELD_PRIMITIVE_CELL_NU_INTEGER_ATOL = 1.0e-12
TBG_ZERO_FIELD_HF_RUN_PROVENANCE_SCHEMA = (
    "mean_field.tbg.zero_field.hf_run_provenance"
)
TBG_ZERO_FIELD_HF_RUN_PROVENANCE_SCHEMA_VERSION = 2
_TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER_LABEL = (
    "TBGZeroFieldFullRestrictedRunner/v2"
)
_TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER = object()

def validate_tbg_zero_field_seed(value: object) -> int:
    """Return an exact Python seed without accepting bools or truncating floats."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a non-bool integer")
    return int(value)

def validate_tbg_zero_field_typed_max_iter(value: object) -> int:
    """Return a typed solver limit without accepting bools or truncating floats."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("max_iter must be a non-negative integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError("max_iter must be a non-negative integer")
    return resolved

def validate_tbg_zero_field_typed_overlap_lg(value: object) -> int:
    """Validate the odd overlap grid needed by the canonical shell-3 inventory."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("overlap_lg must be a positive odd integer")
    resolved = int(value)
    minimum = 2 * TBG_ZERO_FIELD_ACTIVE_G_SHELL + 1
    if resolved <= 0 or resolved % 2 == 0:
        raise ValueError("overlap_lg must be a positive odd integer")
    if resolved < minimum:
        raise ValueError(
            "overlap_lg is insufficient for the typed canonical G-label shell; "
            f"expected overlap_lg >= {minimum}, got {resolved}"
        )
    return resolved

def validate_tbg_zero_field_typed_bm_lg(value: object) -> int:
    """Require a BM plane-wave source that resolves the canonical shell-3 inventory."""

    minimum = 2 * TBG_ZERO_FIELD_ACTIVE_G_SHELL + 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            "Typed TBG primitive-cell shell-3 HF requires BM solution.lg to be "
            f"an odd integer >= {minimum}; other BM sources are diagnostic-only"
        )
    resolved = int(value)
    if resolved < minimum or resolved % 2 == 0:
        raise ValueError(
            "Typed TBG primitive-cell shell-3 HF requires BM solution.lg to be "
            f"an odd integer >= {minimum}; got {resolved}. Undersized or even "
            "BM sources are diagnostic-only"
        )
    return resolved

def validate_tbg_zero_field_primitive_cell_nu(value: object) -> float:
    """Normalize the integer filling accepted by typed primitive-cell TBG HF."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(float(value))
    ):
        raise ValueError("Typed TBG primitive-cell HF nu must be a finite real integer")
    resolved = float(value)
    rounded = int(round(resolved))
    if abs(resolved - float(rounded)) > TBG_ZERO_FIELD_PRIMITIVE_CELL_NU_INTEGER_ATOL:
        raise ValueError(
            "Typed TBG primitive-cell HF requires integer nu; fractional fillings "
            "require a separate supercell workflow and are refused"
        )
    return float(rounded)


def conventional_projector_to_stored_density(projector: np.ndarray) -> np.ndarray:
    """Return ``D_stored[a,b,k]=<c_a†c_b>-0.5*delta_ab``.

    ``projector`` is the conventional ket-bra matrix ``U_occ U_occ†``. This
    boundary conversion transposes it into the existing core-HF stored
    orientation before subtracting the central-average reference.
    """

    conventional = np.asarray(projector, dtype=np.complex128)
    if conventional.ndim != 3 or conventional.shape[0] != conventional.shape[1]:
        raise ValueError(
            "Expected conventional projector shape (nt, nt, nk), got "
            f"{conventional.shape}"
        )
    stored = conventional_projector_to_stored(conventional)
    stored -= 0.5 * np.eye(conventional.shape[0], dtype=np.complex128)[:, :, None]
    return stored

def stored_density_to_conventional_projector(stored_density: np.ndarray) -> np.ndarray:
    """Invert :func:`conventional_projector_to_stored_density` exactly."""

    stored = np.asarray(stored_density, dtype=np.complex128)
    if stored.ndim != 3 or stored.shape[0] != stored.shape[1]:
        raise ValueError(
            "Expected stored density shape (nt, nt, nk), got "
            f"{stored.shape}"
        )
    stored_projector = stored + 0.5 * np.eye(stored.shape[0], dtype=np.complex128)[:, :, None]
    return stored_projector_to_conventional(stored_projector)

def _validate_initial_density_override_policy(
    initial_density: np.ndarray | None,
    *,
    legacy_untyped: bool,
    hf_mode: Literal["full", "restricted"],
) -> None:
    if initial_density is not None and not legacy_untyped:
        raise ValueError(
            f"Typed {hf_mode} HF rejects arbitrary initial_density overrides; "
            "the typed resume trajectory is not implemented and remains fail-closed"
        )

def _screening_values_match(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=1.0e-13, atol=0.0))


def _resolve_tbg_zero_field_screening(
    params: TBGParameters,
    *,
    interaction_spec: TBGZeroFieldInteractionSpec | None,
    legacy_untyped: bool,
    relative_permittivity: float | None,
    screening_lm: float | None,
    finite_zero_limit: bool | None,
    zero_cutoff: float | None,
) -> tuple[float, float, bool, float]:
    """Resolve exact kernel inputs without confusing physical nm and ``lm``."""

    if interaction_spec is not None:
        if not isinstance(interaction_spec, TBGZeroFieldInteractionSpec):
            raise TypeError(
                "interaction_spec must be TBGZeroFieldInteractionSpec, got "
                f"{type(interaction_spec).__name__}"
            )
        if legacy_untyped:
            raise ValueError("interaction_spec and legacy_untyped=True are mutually exclusive")
        if screening_lm is not None:
            raise ValueError(
                "Raw screening_lm is rejected in the typed TBG zero-field path; "
                "supply physical dsc_nm through TBGZeroFieldInteractionSpec"
            )
        if interaction_spec.transfer_cutoff_policy != TBG_ZERO_FIELD_TRANSFER_CUTOFF_POLICY:
            raise ValueError(
                "The screened B0 block builder cannot execute transfer cutoff policy "
                f"{interaction_spec.transfer_cutoff_policy!r}"
            )
        expected = (
            interaction_spec.epsr,
            interaction_spec.screening_lm,
            interaction_spec.finite_zero_limit,
            interaction_spec.zero_cutoff,
        )
        supplied = (
            relative_permittivity,
            screening_lm,
            finite_zero_limit,
            zero_cutoff,
        )
        names = (
            "relative_permittivity",
            "screening_lm",
            "finite_zero_limit",
            "zero_cutoff",
        )
        for name, raw_value, expected_value in zip(names, supplied, expected, strict=True):
            if raw_value is None:
                continue
            if name == "finite_zero_limit":
                matches = isinstance(raw_value, (bool, np.bool_)) and bool(raw_value) == bool(expected_value)
            else:
                matches = _screening_values_match(float(raw_value), float(expected_value))
            if not matches:
                raise ValueError(
                    f"Raw {name}={raw_value!r} conflicts with typed interaction_spec "
                    f"value {expected_value!r}"
                )
        return (
            float(interaction_spec.epsr),
            float(interaction_spec.screening_lm),
            bool(interaction_spec.finite_zero_limit),
            float(interaction_spec.zero_cutoff),
        )

    if not legacy_untyped:
        raise ValueError(
            "TBG zero-field screening requires interaction_spec=TBGZeroFieldInteractionSpec; "
            "raw dimensionless screening parameters are available only with "
            "legacy_untyped=True for diagnostic compatibility"
        )
    resolved_epsr = 15.0 if relative_permittivity is None else float(relative_permittivity)
    resolved_lm = (
        float(np.sqrt(abs(params.a1) * abs(params.a2)))
        if screening_lm is None
        else float(screening_lm)
    )
    resolved_zero_limit = False if finite_zero_limit is None else bool(finite_zero_limit)
    resolved_cutoff = 1.0e-6 if zero_cutoff is None else float(zero_cutoff)
    if resolved_epsr <= 0.0 or resolved_lm <= 0.0 or resolved_cutoff <= 0.0:
        raise ValueError("Legacy raw screening values must be positive")
    return resolved_epsr, resolved_lm, resolved_zero_limit, resolved_cutoff


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return digest


def _update_fingerprint_bytes(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, byteorder="little", signed=False))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def _update_fingerprint_array(
    digest: Any,
    label: str,
    values: np.ndarray,
    *,
    dtype: str,
) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    _update_fingerprint_bytes(
        digest,
        f"{label}.shape",
        np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"),
    )
    _update_fingerprint_bytes(digest, f"{label}.values", array.tobytes(order="C"))


def tbg_zero_field_lattice_kvec_sha256(values: np.ndarray) -> str:
    lattice = np.ascontiguousarray(
        np.asarray(values, dtype=np.dtype("<c16")).reshape(-1)
    )
    return hashlib.sha256(lattice.tobytes(order="C")).hexdigest()


def tbg_zero_field_active_shift_inventory(
    shell: int = TBG_ZERO_FIELD_ACTIVE_G_SHELL,
) -> tuple[tuple[int, int], ...]:
    """Return the canonical G-label hex shell, independently of overlap_lg."""

    resolved_shell = int(shell)
    if resolved_shell < 0:
        raise ValueError(f"shell must be non-negative, got {shell}")
    return tuple(
        (m, n)
        for n in range(-resolved_shell, resolved_shell + 1)
        for m in range(-resolved_shell, resolved_shell + 1)
        if max(abs(m), abs(n), abs(m + n)) <= resolved_shell
    )

def tbg_zero_field_active_shift_inventory_sha256(
    shifts: tuple[tuple[int, int], ...],
) -> str:
    normalized = tuple((int(m), int(n)) for m, n in shifts)
    payload = json.dumps(normalized, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def tbg_zero_field_central_average_reference_projector(solution: BMSolution) -> np.ndarray:
    if int(solution.nb) != 2:
        raise ValueError(
            "Typed TBG zero-field HF requires the central active two-band BM window (n_band=2)"
        )
    reference = np.repeat(
        (0.5 * np.eye(int(solution.nt), dtype=np.complex128))[:, :, None],
        int(solution.nk),
        axis=2,
    )
    reference.setflags(write=False)
    return reference

def tbg_zero_field_reference_projector_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_fingerprint_array(
        digest,
        "central_average_reference_projector",
        np.asarray(values, dtype=np.complex128),
        dtype="<c16",
    )
    return digest.hexdigest()

def _tbg_zero_field_hf_history_sha256(values: np.ndarray, *, name: str) -> str:
    history = np.asarray(values, dtype=float)
    if history.ndim != 1 or not np.all(np.isfinite(history)):
        raise ValueError(f"{name} must be a finite one-dimensional solver history")
    digest = hashlib.sha256()
    _update_fingerprint_bytes(digest, "domain", b"TBGZeroFieldHFSolverHistory/v1")
    _update_fingerprint_array(digest, name, history, dtype="<f8")
    return digest.hexdigest()

_TBG_ZERO_FIELD_HF_REQUIRED_FINAL_DIAGNOSTICS = frozenset(
    {
        "hf_energy",
        "final_raw_norm",
        "filling",
        "offdiag_flavor_norm",
        "restricted_gap",
        "occupied_sigma_mean",
    }
)


def _tbg_zero_field_hf_finite_numeric_diagnostics(
    diagnostics: Any,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(diagnostics, Mapping):
        raise ValueError("Typed HF final-state diagnostics must be a mapping")
    missing = sorted(
        _TBG_ZERO_FIELD_HF_REQUIRED_FINAL_DIAGNOSTICS - set(diagnostics)
    )
    if missing:
        raise ValueError(
            "Typed HF final-state diagnostics are incomplete; "
            f"missing={missing}"
        )

    finite: list[tuple[str, float]] = []
    for name, value in diagnostics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "Typed HF final-state diagnostic names must be non-empty strings"
            )
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            finite.append((name, numeric_value))
    return tuple(sorted(finite, key=lambda item: item[0]))


def _tbg_zero_field_hf_state_source_sha256(state: Any) -> str:
    receipt = state.hf_source_receipt
    if not isinstance(receipt, TBGZeroFieldHFSourceReceipt):
        raise ValueError("Typed HF final-state hashing requires its typed source receipt")
    interaction_spec = state.interaction_spec
    if not isinstance(interaction_spec, TBGZeroFieldInteractionSpec):
        raise ValueError("Typed HF final-state hashing requires TBGZeroFieldInteractionSpec")

    digest = hashlib.sha256()
    _update_fingerprint_bytes(digest, "domain", b"TBGZeroFieldHFFinalState/v2")
    for name, values, dtype in (
        ("h0", state.h0, "<c16"),
        ("sigma_z", state.sigma_z, "<c16"),
        ("density", state.density, "<c16"),
        ("hamiltonian", state.hamiltonian, "<c16"),
        ("energies", state.energies, "<f8"),
        ("sigma_ztauz", state.sigma_ztauz, "<f8"),
    ):
        _update_fingerprint_array(digest, name, values, dtype=dtype)

    for name in ("mu", "nu", "v0", "precision"):
        value = getattr(state, name)
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"Typed HF final-state {name} must be a finite real scalar")
        _update_fingerprint_array(
            digest,
            name,
            np.asarray([float(value)], dtype=np.float64),
            dtype="<f8",
        )

    dimensions: list[int] = []
    for name in ("n_spin", "n_eta", "n_band"):
        value = getattr(state, name)
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(f"Typed HF final-state {name} must be a positive integer")
        dimensions.append(int(value))
    _update_fingerprint_array(
        digest,
        "state_dimensions",
        np.asarray(dimensions, dtype=np.int64),
        dtype="<i8",
    )

    diagnostics = _tbg_zero_field_hf_finite_numeric_diagnostics(state.diagnostics)
    _update_fingerprint_array(
        digest,
        "diagnostics.count",
        np.asarray([len(diagnostics)], dtype=np.int64),
        dtype="<i8",
    )
    for index, (name, value) in enumerate(diagnostics):
        _update_fingerprint_bytes(
            digest,
            f"diagnostics.{index}.name",
            name.encode("utf-8"),
        )
        _update_fingerprint_array(
            digest,
            f"diagnostics.{index}.value",
            np.asarray([value], dtype=np.float64),
            dtype="<f8",
        )

    _update_fingerprint_bytes(
        digest,
        "typed_receipt_fingerprint",
        receipt.fingerprint.encode("ascii"),
    )
    _update_fingerprint_bytes(
        digest,
        "interaction_spec_fingerprint",
        interaction_spec.fingerprint.encode("ascii"),
    )
    return digest.hexdigest()

def _require_torus_mesh(solution: BMSolution) -> TBGZeroFieldTorusMesh:
    mesh = solution.torus_mesh
    if not isinstance(mesh, TBGZeroFieldTorusMesh):
        raise ValueError(
            "Typed TBG zero-field HF requires BMSolution.torus_mesh from "
            "solve_bm_model_on_torus; endpoint-inclusive B0 grids are legacy diagnostics only"
        )
    if not np.array_equal(
        np.asarray(solution.lattice_kvec, dtype=np.complex128).reshape(-1),
        mesh.kvec,
    ):
        raise ValueError("BMSolution lattice_kvec no longer matches its carried torus mesh")
    return mesh

def tbg_zero_field_bm_solution_fingerprint(solution: BMSolution) -> str:
    """Hash the complete live, solver-attested BM source for typed HF."""

    mesh = _require_torus_mesh(solution)
    solution.validate_source_attestation(require_torus=True)
    source_attestation = solution.source_attestation
    if source_attestation is None:  # guarded above; keeps type narrowing explicit
        raise RuntimeError("Validated BM source unexpectedly lacks an attestation")
    digest = hashlib.sha256()
    _update_fingerprint_bytes(digest, "domain", b"TBGZeroFieldBMSolution/v3")
    _update_fingerprint_bytes(digest, "mesh_fingerprint", mesh.fingerprint.encode("ascii"))
    _update_fingerprint_bytes(
        digest,
        "source_attestation_fingerprint",
        source_attestation.fingerprint.encode("ascii"),
    )
    _update_fingerprint_bytes(
        digest,
        "bm_generation_fingerprint",
        solution.generation_fingerprint.encode("ascii"),
    )
    return digest.hexdigest()

def _freeze_array(values: np.ndarray, *, dtype: str) -> np.ndarray:
    frozen = np.array(values, dtype=np.dtype(dtype), order="C", copy=True)
    frozen.setflags(write=False)
    return frozen

def _freeze_overlap_blocks(blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    shifts = tuple((int(m), int(n)) for m, n in blocks.shifts)
    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=_freeze_array(blocks.gvecs, dtype="<c16"),
        overlaps=MappingProxyType(
            {shift: _freeze_array(blocks.overlaps[shift], dtype="<c16") for shift in shifts}
        ),
        diagonal_overlaps=MappingProxyType(
            {
                shift: _freeze_array(values, dtype="<c16")
                for shift, values in blocks.diagonal_overlaps.items()
            }
        ),
        hartree_screening=MappingProxyType(
            {shift: float(value) for shift, value in blocks.hartree_screening.items()}
        ),
        fock_screening=MappingProxyType(
            {
                shift: _freeze_array(values, dtype="<f8")
                for shift, values in blocks.fock_screening.items()
            }
        ),
    )

def tbg_zero_field_overlap_kernel_inventory_fingerprint(
    blocks: HFOverlapBlockSet,
) -> str:
    """Hash the exact ordered overlap and independently active kernel inventory."""

    shifts = tuple((int(shift[0]), int(shift[1])) for shift in blocks.shifts)
    if len(set(shifts)) != len(shifts):
        raise ValueError("Overlap shifts must be unique for source fingerprinting")
    gvecs = np.asarray(blocks.gvecs, dtype=np.complex128)
    if gvecs.shape != (len(shifts),):
        raise ValueError(
            "Overlap shift and reciprocal-vector inventories differ for source fingerprinting"
        )
    shift_set = set(shifts)
    if set(blocks.overlaps) != shift_set:
        raise ValueError("Overlap block keys must exactly match the ordered shift inventory")
    for label, mapping in (
        ("diagonal-overlap", blocks.diagonal_overlaps),
        ("Hartree", blocks.hartree_screening),
        ("Fock", blocks.fock_screening),
    ):
        if not set(mapping) <= shift_set:
            raise ValueError(f"{label} inventory contains shifts absent from overlaps")

    digest = hashlib.sha256()
    _update_fingerprint_bytes(
        digest,
        "domain",
        b"TBGZeroFieldHFOverlapKernelInventory/v1",
    )
    _update_fingerprint_bytes(
        digest,
        "shift_count",
        int(len(shifts)).to_bytes(8, byteorder="little", signed=False),
    )
    for index, shift in enumerate(shifts):
        prefix = f"shift[{index}]"
        _update_fingerprint_array(
            digest,
            f"{prefix}.label",
            np.asarray(shift),
            dtype="<i8",
        )
        _update_fingerprint_array(
            digest,
            f"{prefix}.gvec",
            np.asarray(gvecs[index]),
            dtype="<c16",
        )
        _update_fingerprint_array(
            digest,
            f"{prefix}.overlap",
            blocks.overlaps[shift],
            dtype="<c16",
        )

        has_diagonal = shift in blocks.diagonal_overlaps
        _update_fingerprint_bytes(
            digest,
            f"{prefix}.diagonal.present",
            bytes((int(has_diagonal),)),
        )
        if has_diagonal:
            _update_fingerprint_array(
                digest,
                f"{prefix}.diagonal",
                blocks.diagonal_overlaps[shift],
                dtype="<c16",
            )

        has_hartree = shift in blocks.hartree_screening
        _update_fingerprint_bytes(
            digest,
            f"{prefix}.hartree.present",
            bytes((int(has_hartree),)),
        )
        if has_hartree:
            _update_fingerprint_array(
                digest,
                f"{prefix}.hartree",
                np.asarray(float(blocks.hartree_screening[shift])),
                dtype="<f8",
            )

        has_fock = shift in blocks.fock_screening
        _update_fingerprint_bytes(
            digest,
            f"{prefix}.fock.present",
            bytes((int(has_fock),)),
        )
        if has_fock:
            _update_fingerprint_array(
                digest,
                f"{prefix}.fock",
                blocks.fock_screening[shift],
                dtype="<c16",
            )
    return digest.hexdigest()


class _ScreenedBlockBundleAttestation:
    """Private identity token binding a factory-produced immutable block object."""

    __slots__ = (
        "screened_blocks",
        "bm_solution_sha256",
        "bm_generation_fingerprint",
        "mesh_fingerprint",
        "interaction_spec_fingerprint",
        "overlap_lg",
    )

    def __init__(
        self,
        screened_blocks: HFOverlapBlockSet,
        *,
        bm_solution_sha256: str,
        bm_generation_fingerprint: str,
        mesh_fingerprint: str,
        interaction_spec_fingerprint: str,
        overlap_lg: int,
    ) -> None:
        self.screened_blocks = screened_blocks
        self.bm_solution_sha256 = bm_solution_sha256
        self.bm_generation_fingerprint = bm_generation_fingerprint
        self.mesh_fingerprint = mesh_fingerprint
        self.interaction_spec_fingerprint = interaction_spec_fingerprint
        self.overlap_lg = int(overlap_lg)

@dataclass(frozen=True)
class TBGZeroFieldScreenedBlockBundle:
    """Builder-attested immutable screened blocks for one BM torus source."""

    screened_blocks: HFOverlapBlockSet
    interaction_spec: TBGZeroFieldInteractionSpec
    overlap_lg: int
    active_shifts: tuple[tuple[int, int], ...]
    active_shift_inventory_sha256: str
    overlap_kernel_inventory_sha256: str
    bm_solution_sha256: str
    bm_generation_fingerprint: str
    mesh_fingerprint: str
    n_band: int
    reference_projector_convention: str
    reference_projector_dimensions: tuple[int, int, int]
    reference_projector_sha256: str
    companion_circular_total_q_cutoff_parity: str
    _attestation: object = field(repr=False, compare=False)
    schema: str = TBG_ZERO_FIELD_SCREENED_BLOCK_BUNDLE_SCHEMA
    schema_version: int = TBG_ZERO_FIELD_SCREENED_BLOCK_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != TBG_ZERO_FIELD_SCREENED_BLOCK_BUNDLE_SCHEMA:
            raise ValueError(f"Unsupported screened-block bundle schema {self.schema!r}")
        if int(self.schema_version) != TBG_ZERO_FIELD_SCREENED_BLOCK_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported screened-block bundle schema version {self.schema_version!r}"
            )
        if not isinstance(self.interaction_spec, TBGZeroFieldInteractionSpec):
            raise TypeError("interaction_spec must be TBGZeroFieldInteractionSpec")
        if not isinstance(self.screened_blocks, HFOverlapBlockSet):
            raise TypeError("screened_blocks must be HFOverlapBlockSet")
        overlap_lg = validate_tbg_zero_field_typed_overlap_lg(self.overlap_lg)
        active_shifts = tuple((int(m), int(n)) for m, n in self.active_shifts)
        expected_active = tbg_zero_field_active_shift_inventory()
        if active_shifts != expected_active:
            raise ValueError(
                "Typed screened-block active shift inventory must exactly equal the canonical "
                "G-label hex shell"
            )
        shift_set = set(self.screened_blocks.shifts)
        if not set(expected_active) <= shift_set:
            missing = sorted(set(expected_active) - shift_set)
            raise ValueError(
                "Typed overlap inventory does not fully cover the canonical G-label hex shell; "
                f"missing={missing}"
            )
        for label, mapping in (
            ("diagonal", self.screened_blocks.diagonal_overlaps),
            ("Hartree", self.screened_blocks.hartree_screening),
            ("Fock", self.screened_blocks.fock_screening),
        ):
            if set(mapping) != set(expected_active):
                raise ValueError(
                    f"Typed {label} screened inventory must exactly equal the canonical active shifts"
                )
        expected_active_hash = tbg_zero_field_active_shift_inventory_sha256(active_shifts)
        if _validate_sha256(
            self.active_shift_inventory_sha256,
            name="active_shift_inventory_sha256",
        ) != expected_active_hash:
            raise ValueError("Active shift inventory hash does not match active_shifts")
        expected_kernel_hash = tbg_zero_field_overlap_kernel_inventory_fingerprint(
            self.screened_blocks
        )
        if _validate_sha256(
            self.overlap_kernel_inventory_sha256,
            name="overlap_kernel_inventory_sha256",
        ) != expected_kernel_hash:
            raise ValueError("Screened overlap/kernel inventory hash does not match blocks")
        if int(self.n_band) != 2:
            raise ValueError("Typed TBG zero-field screened blocks require n_band=2")
        dimensions = tuple(int(value) for value in self.reference_projector_dimensions)
        if len(dimensions) != 3 or dimensions[0] != dimensions[1] or dimensions[2] <= 0:
            raise ValueError(
                "reference_projector_dimensions must be square (nt,nt,nk) dimensions"
            )
        if self.reference_projector_convention != TBG_ZERO_FIELD_REFERENCE_PROJECTOR_CONVENTION:
            raise ValueError("Unsupported central-average reference projector convention")
        if (
            self.companion_circular_total_q_cutoff_parity
            != TBG_ZERO_FIELD_COMPANION_CUTOFF_PARITY
        ):
            raise ValueError(
                "Companion circular total-Q cutoff parity must be recorded as not established"
            )
        for name in (
            "bm_solution_sha256",
            "bm_generation_fingerprint",
            "mesh_fingerprint",
            "reference_projector_sha256",
        ):
            _validate_sha256(getattr(self, name), name=name)
        attestation = self._attestation
        if not isinstance(attestation, _ScreenedBlockBundleAttestation):
            raise ValueError(
                "TBGZeroFieldScreenedBlockBundle may be created only by "
                "build_tbg_zero_field_screened_block_bundle"
            )
        if (
            attestation.screened_blocks is not self.screened_blocks
            or attestation.bm_solution_sha256 != self.bm_solution_sha256
            or attestation.bm_generation_fingerprint != self.bm_generation_fingerprint
            or attestation.mesh_fingerprint != self.mesh_fingerprint
            or attestation.interaction_spec_fingerprint != self.interaction_spec.fingerprint
            or attestation.overlap_lg != overlap_lg
        ):
            raise ValueError("Screened-block bundle builder attestation does not match its payload")
        if np.asarray(self.screened_blocks.gvecs).flags.writeable:
            raise ValueError("Typed screened-block gvecs must be read-only")
        for mapping in (
            self.screened_blocks.overlaps,
            self.screened_blocks.diagonal_overlaps,
            self.screened_blocks.fock_screening,
        ):
            if any(np.asarray(values).flags.writeable for values in mapping.values()):
                raise ValueError("Typed screened-block arrays must be read-only")
        object.__setattr__(self, "overlap_lg", overlap_lg)
        object.__setattr__(self, "active_shifts", active_shifts)
        object.__setattr__(self, "reference_projector_dimensions", dimensions)
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def _payload(self) -> dict[str, object]:
        return {
            "active_shift_inventory": [list(shift) for shift in self.active_shifts],
            "active_shift_inventory_sha256": self.active_shift_inventory_sha256,
            "bm_generation_fingerprint": self.bm_generation_fingerprint,
            "bm_solution_sha256": self.bm_solution_sha256,
            "companion_circular_total_q_cutoff_parity": self.companion_circular_total_q_cutoff_parity,
            "interaction_spec_fingerprint": self.interaction_spec.fingerprint,
            "mesh_fingerprint": self.mesh_fingerprint,
            "n_band": self.n_band,
            "overlap_kernel_inventory_sha256": self.overlap_kernel_inventory_sha256,
            "overlap_lg": self.overlap_lg,
            "reference_projector_convention": self.reference_projector_convention,
            "reference_projector_dimensions": list(self.reference_projector_dimensions),
            "reference_projector_sha256": self.reference_projector_sha256,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "transfer_cutoff_policy": self.interaction_spec.transfer_cutoff_policy,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def validate_for_solution(self, solution: BMSolution) -> None:
        validate_tbg_zero_field_typed_bm_lg(solution.lg)
        mesh = _require_torus_mesh(solution)
        if mesh.fingerprint != self.mesh_fingerprint:
            raise ValueError("Screened-block bundle mesh does not match BMSolution.torus_mesh")
        if solution.generation_fingerprint != self.bm_generation_fingerprint:
            raise ValueError("Screened-block bundle BM generation fingerprint does not match BMSolution")
        if tbg_zero_field_bm_solution_fingerprint(solution) != self.bm_solution_sha256:
            raise ValueError("Screened-block bundle was generated from a different or stale BMSolution")
        if int(solution.nb) != self.n_band:
            raise ValueError("Screened-block bundle n_band does not match BMSolution")
        reference = tbg_zero_field_central_average_reference_projector(solution)
        if tuple(reference.shape) != self.reference_projector_dimensions:
            raise ValueError("Screened-block bundle reference dimensions do not match BMSolution")
        if tbg_zero_field_reference_projector_sha256(reference) != self.reference_projector_sha256:
            raise ValueError("Screened-block bundle central-average reference hash mismatch")
        if (
            tbg_zero_field_overlap_kernel_inventory_fingerprint(self.screened_blocks)
            != self.overlap_kernel_inventory_sha256
        ):
            raise ValueError("Screened-block bundle contents changed after construction")

@dataclass(frozen=True)
class TBGZeroFieldHFSourceReceipt:
    """Immutable identity of the exact zero-field HF functional source."""

    hf_mode: Literal["full", "restricted"]
    beta: float
    v0: float
    lattice_kvec_sha256: str
    overlap_kernel_inventory_sha256: str
    interaction_contract: Literal["typed", "legacy_untyped_diagnostic"]
    interaction_spec_fingerprint: str | None
    screened_block_bundle_sha256: str | None = None
    bm_solution_sha256: str | None = None
    bm_generation_fingerprint: str | None = None
    mesh_fingerprint: str | None = None
    overlap_lg: int | None = None
    active_shift_inventory: tuple[tuple[int, int], ...] = ()
    active_shift_inventory_sha256: str | None = None
    n_band: int | None = None
    reference_projector_convention: str | None = None
    reference_projector_dimensions: tuple[int, int, int] | None = None
    reference_projector_sha256: str | None = None
    companion_circular_total_q_cutoff_parity: str | None = None
    schema: str = TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA
    schema_version: int = TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA_VERSION
    centered_reference_convention: str = TBG_ZERO_FIELD_CENTERED_REFERENCE_CONVENTION

    def __post_init__(self) -> None:
        if self.schema != TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA:
            raise ValueError(f"Unsupported TBG zero-field HF receipt schema {self.schema!r}")
        if (
            not isinstance(self.schema_version, (int, np.integer))
            or isinstance(self.schema_version, (bool, np.bool_))
            or int(self.schema_version) != TBG_ZERO_FIELD_HF_SOURCE_RECEIPT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported TBG zero-field HF receipt schema version {self.schema_version!r}"
            )
        if self.hf_mode not in ("full", "restricted"):
            raise ValueError(f"Unsupported TBG zero-field hf_mode {self.hf_mode!r}")
        for name in ("beta", "v0"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, (bool, np.bool_))
                or not np.isfinite(float(value))
            ):
                raise ValueError(f"TBG zero-field HF receipt {name} must be a finite real scalar")
            object.__setattr__(self, name, float(value))
        if self.centered_reference_convention != TBG_ZERO_FIELD_CENTERED_REFERENCE_CONVENTION:
            raise ValueError("Unsupported TBG zero-field centered-reference convention")
        object.__setattr__(
            self,
            "lattice_kvec_sha256",
            _validate_sha256(self.lattice_kvec_sha256, name="lattice_kvec_sha256"),
        )
        object.__setattr__(
            self,
            "overlap_kernel_inventory_sha256",
            _validate_sha256(
                self.overlap_kernel_inventory_sha256,
                name="overlap_kernel_inventory_sha256",
            ),
        )
        active_shifts = tuple((int(m), int(n)) for m, n in self.active_shift_inventory)
        object.__setattr__(self, "active_shift_inventory", active_shifts)
        if self.interaction_contract == "typed":
            for name in (
                "interaction_spec_fingerprint",
                "screened_block_bundle_sha256",
                "bm_solution_sha256",
                "bm_generation_fingerprint",
                "mesh_fingerprint",
                "active_shift_inventory_sha256",
                "reference_projector_sha256",
            ):
                object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))
            overlap_lg = validate_tbg_zero_field_typed_overlap_lg(self.overlap_lg)
            if active_shifts != tbg_zero_field_active_shift_inventory():
                raise ValueError("Typed HF receipt active shifts do not equal the canonical G-label shell")
            if (
                tbg_zero_field_active_shift_inventory_sha256(active_shifts)
                != self.active_shift_inventory_sha256
            ):
                raise ValueError("Typed HF receipt active shift hash mismatch")
            if int(self.n_band or 0) != 2:
                raise ValueError("Typed HF receipt requires n_band=2")
            dimensions = tuple(int(value) for value in (self.reference_projector_dimensions or ()))
            if len(dimensions) != 3 or dimensions[0] != dimensions[1]:
                raise ValueError("Typed HF receipt reference dimensions must be (nt,nt,nk)")
            if self.reference_projector_convention != TBG_ZERO_FIELD_REFERENCE_PROJECTOR_CONVENTION:
                raise ValueError("Typed HF receipt central-average reference convention mismatch")
            if (
                self.companion_circular_total_q_cutoff_parity
                != TBG_ZERO_FIELD_COMPANION_CUTOFF_PARITY
            ):
                raise ValueError(
                    "Typed HF receipt must state companion circular total-Q cutoff parity is not established"
                )
            object.__setattr__(self, "overlap_lg", overlap_lg)
            object.__setattr__(self, "n_band", int(self.n_band))
            object.__setattr__(self, "reference_projector_dimensions", dimensions)
        elif self.interaction_contract == "legacy_untyped_diagnostic":
            typed_only = (
                self.interaction_spec_fingerprint,
                self.screened_block_bundle_sha256,
                self.bm_solution_sha256,
                self.bm_generation_fingerprint,
                self.mesh_fingerprint,
                self.overlap_lg,
                self.active_shift_inventory_sha256,
                self.n_band,
                self.reference_projector_convention,
                self.reference_projector_dimensions,
                self.reference_projector_sha256,
                self.companion_circular_total_q_cutoff_parity,
            )
            if any(value is not None for value in typed_only) or active_shifts:
                raise ValueError("Diagnostic HF receipts cannot claim typed bundle/reference fields")
        else:
            raise ValueError(
                f"Unsupported TBG zero-field interaction_contract {self.interaction_contract!r}"
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def _payload(self) -> dict[str, object]:
        return {
            "active_shift_inventory": [list(shift) for shift in self.active_shift_inventory],
            "active_shift_inventory_sha256": self.active_shift_inventory_sha256,
            "beta": self.beta,
            "bm_generation_fingerprint": self.bm_generation_fingerprint,
            "bm_solution_sha256": self.bm_solution_sha256,
            "centered_reference_convention": self.centered_reference_convention,
            "companion_circular_total_q_cutoff_parity": self.companion_circular_total_q_cutoff_parity,
            "hf_mode": self.hf_mode,
            "interaction_contract": self.interaction_contract,
            "interaction_spec_fingerprint": self.interaction_spec_fingerprint,
            "lattice_kvec_sha256": self.lattice_kvec_sha256,
            "mesh_fingerprint": self.mesh_fingerprint,
            "n_band": self.n_band,
            "overlap_kernel_inventory_sha256": self.overlap_kernel_inventory_sha256,
            "overlap_lg": self.overlap_lg,
            "reference_projector_convention": self.reference_projector_convention,
            "reference_projector_dimensions": (
                None
                if self.reference_projector_dimensions is None
                else list(self.reference_projector_dimensions)
            ),
            "reference_projector_sha256": self.reference_projector_sha256,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "screened_block_bundle_sha256": self.screened_block_bundle_sha256,
            "v0": self.v0,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "TBGZeroFieldHFSourceReceipt":
        payload = dict(metadata)
        expected_keys = {
            "active_shift_inventory",
            "active_shift_inventory_sha256",
            "beta",
            "bm_generation_fingerprint",
            "bm_solution_sha256",
            "centered_reference_convention",
            "companion_circular_total_q_cutoff_parity",
            "fingerprint",
            "hf_mode",
            "interaction_contract",
            "interaction_spec_fingerprint",
            "lattice_kvec_sha256",
            "mesh_fingerprint",
            "n_band",
            "overlap_kernel_inventory_sha256",
            "overlap_lg",
            "reference_projector_convention",
            "reference_projector_dimensions",
            "reference_projector_sha256",
            "schema",
            "schema_version",
            "screened_block_bundle_sha256",
            "v0",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "TBG zero-field HF receipt metadata keys differ from the supported schema: "
                f"expected={sorted(expected_keys)}, got={sorted(payload)}"
            )
        active_raw = payload["active_shift_inventory"]
        if not isinstance(active_raw, list):
            raise ValueError("active_shift_inventory metadata must be a list")
        dimensions_raw = payload["reference_projector_dimensions"]
        receipt = cls(
            hf_mode=payload["hf_mode"],  # type: ignore[arg-type]
            beta=payload["beta"],  # type: ignore[arg-type]
            v0=payload["v0"],  # type: ignore[arg-type]
            lattice_kvec_sha256=payload["lattice_kvec_sha256"],  # type: ignore[arg-type]
            overlap_kernel_inventory_sha256=payload[
                "overlap_kernel_inventory_sha256"
            ],  # type: ignore[arg-type]
            interaction_contract=payload["interaction_contract"],  # type: ignore[arg-type]
            interaction_spec_fingerprint=payload["interaction_spec_fingerprint"],  # type: ignore[arg-type]
            screened_block_bundle_sha256=payload["screened_block_bundle_sha256"],  # type: ignore[arg-type]
            bm_solution_sha256=payload["bm_solution_sha256"],  # type: ignore[arg-type]
            bm_generation_fingerprint=payload["bm_generation_fingerprint"],  # type: ignore[arg-type]
            mesh_fingerprint=payload["mesh_fingerprint"],  # type: ignore[arg-type]
            overlap_lg=payload["overlap_lg"],  # type: ignore[arg-type]
            active_shift_inventory=tuple(tuple(value) for value in active_raw),  # type: ignore[arg-type]
            active_shift_inventory_sha256=payload["active_shift_inventory_sha256"],  # type: ignore[arg-type]
            n_band=payload["n_band"],  # type: ignore[arg-type]
            reference_projector_convention=payload["reference_projector_convention"],  # type: ignore[arg-type]
            reference_projector_dimensions=(
                None if dimensions_raw is None else tuple(dimensions_raw)
            ),  # type: ignore[arg-type]
            reference_projector_sha256=payload["reference_projector_sha256"],  # type: ignore[arg-type]
            companion_circular_total_q_cutoff_parity=payload[
                "companion_circular_total_q_cutoff_parity"
            ],  # type: ignore[arg-type]
            schema=payload["schema"],  # type: ignore[arg-type]
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            centered_reference_convention=payload[
                "centered_reference_convention"
            ],  # type: ignore[arg-type]
        )
        expected_fingerprint = _validate_sha256(
            payload["fingerprint"],
            name="fingerprint",
        )
        if receipt.fingerprint != expected_fingerprint:
            raise ValueError("TBG zero-field HF receipt metadata fingerprint does not match its fields")
        return receipt


def build_tbg_zero_field_hf_source_receipt(
    *,
    hf_mode: Literal["full", "restricted"],
    beta: float,
    v0: float,
    solution: BMSolution,
    screened_block_bundle: TBGZeroFieldScreenedBlockBundle,
) -> TBGZeroFieldHFSourceReceipt:
    """Issue a typed receipt only from the matching builder-attested bundle."""

    if not isinstance(screened_block_bundle, TBGZeroFieldScreenedBlockBundle):
        raise TypeError(
            "screened_block_bundle must be TBGZeroFieldScreenedBlockBundle; arbitrary "
            "HFOverlapBlockSet objects cannot be relabelled as typed"
        )
    screened_block_bundle.validate_for_solution(solution)
    expected_v0 = coulomb_unit(solution.params)
    if float(v0) != expected_v0:
        raise ValueError(
            "Typed HF receipt v0 must equal coulomb_unit(solution.params) exactly; "
            f"got {float(v0)!r}, expected {expected_v0!r}"
        )
    bundle = screened_block_bundle
    if bundle.interaction_spec.reference_scheme != TBG_ZERO_FIELD_REFERENCE_SCHEME:
        raise ValueError("Typed screened bundle does not use the central-average reference scheme")
    return TBGZeroFieldHFSourceReceipt(
        hf_mode=hf_mode,
        beta=float(beta),
        v0=float(v0),
        lattice_kvec_sha256=tbg_zero_field_lattice_kvec_sha256(solution.lattice_kvec),
        overlap_kernel_inventory_sha256=bundle.overlap_kernel_inventory_sha256,
        interaction_contract="typed",
        interaction_spec_fingerprint=bundle.interaction_spec.fingerprint,
        screened_block_bundle_sha256=bundle.fingerprint,
        bm_solution_sha256=bundle.bm_solution_sha256,
        bm_generation_fingerprint=bundle.bm_generation_fingerprint,
        mesh_fingerprint=bundle.mesh_fingerprint,
        overlap_lg=bundle.overlap_lg,
        active_shift_inventory=bundle.active_shifts,
        active_shift_inventory_sha256=bundle.active_shift_inventory_sha256,
        n_band=bundle.n_band,
        reference_projector_convention=bundle.reference_projector_convention,
        reference_projector_dimensions=bundle.reference_projector_dimensions,
        reference_projector_sha256=bundle.reference_projector_sha256,
        companion_circular_total_q_cutoff_parity=(
            bundle.companion_circular_total_q_cutoff_parity
        ),
    )

def build_tbg_zero_field_diagnostic_hf_source_receipt(
    *,
    hf_mode: Literal["full", "restricted"],
    beta: float,
    v0: float,
    lattice_kvec: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
) -> TBGZeroFieldHFSourceReceipt:
    """Issue an explicitly synthetic/legacy diagnostic receipt, never a typed one."""

    return TBGZeroFieldHFSourceReceipt(
        hf_mode=hf_mode,
        beta=float(beta),
        v0=float(v0),
        lattice_kvec_sha256=tbg_zero_field_lattice_kvec_sha256(lattice_kvec),
        overlap_kernel_inventory_sha256=(
            tbg_zero_field_overlap_kernel_inventory_fingerprint(overlap_blocks)
        ),
        interaction_contract="legacy_untyped_diagnostic",
        interaction_spec_fingerprint=None,
    )


@dataclass(frozen=True)
class TBGZeroFieldHFRunProvenance:
    """Solver-issued identity for one typed zero-field HF run.

    Direct construction and ``dataclasses.replace`` are intentionally refused:
    only the full/restricted runner factory owns the in-process issuer token.
    """

    hf_mode: Literal["full", "restricted"]
    beta: float
    nu: float
    precision: float
    oda_stall_threshold: float
    requested_max_iterations: int
    seed: int
    normalized_init_mode: str
    typed_receipt_fingerprint: str
    interaction_spec_fingerprint: str
    bm_generation_fingerprint: str
    mesh_fingerprint: str
    iter_energy_sha256: str
    iter_err_sha256: str
    iter_oda_sha256: str
    state_source_sha256: str
    converged: bool
    exit_reason: str
    _issuer: InitVar[object | None] = None
    _issuer_identity: object = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuer: object | None) -> None:
        if _issuer is not _TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER:
            raise ValueError(
                "TBGZeroFieldHFRunProvenance is solver-issued; direct/manual "
                "construction is diagnostic and cannot authorize production"
            )
        object.__setattr__(
            self,
            "_issuer_identity",
            _TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER,
        )
        if self.hf_mode not in ("full", "restricted"):
            raise ValueError(f"Unsupported TBG zero-field hf_mode {self.hf_mode!r}")
        if (
            not isinstance(self.beta, (int, float, np.integer, np.floating))
            or isinstance(self.beta, (bool, np.bool_))
            or not np.isfinite(float(self.beta))
        ):
            raise ValueError("TBG zero-field HF run provenance beta must be finite")
        object.__setattr__(
            self,
            "nu",
            validate_tbg_zero_field_primitive_cell_nu(self.nu),
        )
        for name in ("precision", "oda_stall_threshold"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float, np.integer, np.floating))
                or isinstance(value, (bool, np.bool_))
                or not np.isfinite(float(value))
            ):
                raise ValueError(
                    f"TBG zero-field HF run provenance {name} must be a finite real scalar"
                )
            object.__setattr__(self, name, float(value))
        if self.precision <= 0.0:
            raise ValueError("TBG zero-field HF run provenance precision must be positive")
        if self.oda_stall_threshold <= 0.0:
            raise ValueError(
                "TBG zero-field HF run provenance oda_stall_threshold must be positive"
            )
        if (
            not isinstance(self.requested_max_iterations, (int, np.integer))
            or isinstance(self.requested_max_iterations, (bool, np.bool_))
            or int(self.requested_max_iterations) < 0
        ):
            raise ValueError("requested_max_iterations must be a non-negative integer")
        if not isinstance(self.seed, (int, np.integer)) or isinstance(self.seed, (bool, np.bool_)):
            raise ValueError("TBG zero-field HF run provenance seed must be an integer")
        if not isinstance(self.converged, (bool, np.bool_)):
            raise ValueError("TBG zero-field HF run provenance converged must be bool")
        normalized_init_mode = str(self.normalized_init_mode)
        if not normalized_init_mode or normalized_init_mode != normalized_init_mode.strip().lower():
            raise ValueError("normalized_init_mode must be a non-empty normalized lower-case string")
        exit_reason = str(self.exit_reason).strip()
        if not exit_reason:
            raise ValueError("TBG zero-field HF run provenance exit_reason must be non-empty")
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "requested_max_iterations", int(self.requested_max_iterations))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "normalized_init_mode", normalized_init_mode)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "exit_reason", exit_reason)
        for name in (
            "typed_receipt_fingerprint",
            "interaction_spec_fingerprint",
            "bm_generation_fingerprint",
            "mesh_fingerprint",
            "iter_energy_sha256",
            "iter_err_sha256",
            "iter_oda_sha256",
            "state_source_sha256",
        ):
            object.__setattr__(self, name, _validate_sha256(getattr(self, name), name=name))

    @property
    def filling(self) -> float:
        """Public filling alias for the TBG ``nu`` convention."""

        return self.nu

    @property
    def max_iterations(self) -> int:
        """Requested iteration limit (not the observed iteration count)."""

        return self.requested_max_iterations

    @property
    def init_mode(self) -> str:
        return self.normalized_init_mode

    @property
    def receipt_fingerprint(self) -> str:
        return self.typed_receipt_fingerprint

    def _payload(self) -> dict[str, object]:
        return {
            "schema": TBG_ZERO_FIELD_HF_RUN_PROVENANCE_SCHEMA,
            "schema_version": TBG_ZERO_FIELD_HF_RUN_PROVENANCE_SCHEMA_VERSION,
            "issuer": _TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER_LABEL,
            "hf_mode": self.hf_mode,
            "beta": self.beta,
            "nu": self.nu,
            "filling": self.nu,
            "precision": self.precision,
            "oda_stall_threshold": self.oda_stall_threshold,
            "requested_max_iterations": self.requested_max_iterations,
            "seed": self.seed,
            "normalized_init_mode": self.normalized_init_mode,
            "typed_receipt_fingerprint": self.typed_receipt_fingerprint,
            "interaction_spec_fingerprint": self.interaction_spec_fingerprint,
            "bm_generation_fingerprint": self.bm_generation_fingerprint,
            "mesh_fingerprint": self.mesh_fingerprint,
            "iter_energy_sha256": self.iter_energy_sha256,
            "iter_err_sha256": self.iter_err_sha256,
            "iter_oda_sha256": self.iter_oda_sha256,
            "state_source_sha256": self.state_source_sha256,
            "converged": self.converged,
            "exit_reason": self.exit_reason,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def _validate_solver_issued_live_run(self, run: Any) -> None:
        if (
            self._issuer_identity is not _TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER
            or getattr(run, "_production_issuer_identity", None)
            is not _TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER
            or getattr(run, "provenance", None) is not self
        ):
            raise ValueError(
                "Typed HF run provenance does not have the live solver issuer identity"
            )
        live_hashes = (
            (
                "iter_energy",
                self.iter_energy_sha256,
                _tbg_zero_field_hf_history_sha256(
                    run.iter_energy,
                    name="iter_energy",
                ),
            ),
            (
                "iter_err",
                self.iter_err_sha256,
                _tbg_zero_field_hf_history_sha256(run.iter_err, name="iter_err"),
            ),
            (
                "iter_oda",
                self.iter_oda_sha256,
                _tbg_zero_field_hf_history_sha256(run.iter_oda, name="iter_oda"),
            ),
            (
                "final state",
                self.state_source_sha256,
                _tbg_zero_field_hf_state_source_sha256(run.state),
            ),
        )
        for name, expected, actual in live_hashes:
            if actual != expected:
                raise ValueError(
                    f"Typed HF run provenance {name} hash does not match the live run"
                )
        if bool(run.converged) != self.converged:
            raise ValueError(
                "Typed HF run provenance converged status does not match the live run"
            )
        if str(run.exit_reason) != self.exit_reason:
            raise ValueError(
                "Typed HF run provenance exit_reason does not match the live run"
            )

@dataclass(frozen=True)
class RestrictedHartreeFockRun(HartreeFockRun):
    state: "RestrictedHartreeFockState"
    overlap_blocks: HFOverlapBlockSet
    screened_block_bundle: TBGZeroFieldScreenedBlockBundle | None = None
    provenance: TBGZeroFieldHFRunProvenance | None = None
    _production_issuer: InitVar[object | None] = None
    _production_issuer_identity: object | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def interaction_spec(self) -> TBGZeroFieldInteractionSpec | None:
        """Expose the exact interaction specification carried by the HF state."""

        return self.state.interaction_spec

    @property
    def hf_source_receipt(self) -> TBGZeroFieldHFSourceReceipt | None:
        """Expose the exact source receipt carried by the HF state."""

        return self.state.hf_source_receipt

    def __post_init__(self, _production_issuer: object | None) -> None:
        if self.provenance is None:
            if _production_issuer is not None:
                raise ValueError(
                    "Diagnostic/manual HF runs cannot carry a production issuer token"
                )
            object.__setattr__(self, "_production_issuer_identity", None)
            return
        if _production_issuer is not _TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER:
            raise ValueError(
                "Manually constructed HF runs cannot carry production provenance"
            )
        object.__setattr__(
            self,
            "_production_issuer_identity",
            _TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER,
        )
        self.provenance._validate_solver_issued_live_run(self)


def _issue_tbg_zero_field_typed_hf_run(
    *,
    hf_mode: Literal["full", "restricted"],
    state: "RestrictedHartreeFockState",
    overlap_blocks: HFOverlapBlockSet,
    screened_block_bundle: TBGZeroFieldScreenedBlockBundle,
    base_run: HartreeFockRun,
    beta: float,
    oda_stall_threshold: float,
    requested_max_iterations: int,
) -> RestrictedHartreeFockRun:
    """Issue the only production run/provenance pair after the base solver returns."""

    if base_run.state is not state:
        raise ValueError("Typed HF provenance factory requires the exact solver state")
    receipt = state.hf_source_receipt
    if not isinstance(receipt, TBGZeroFieldHFSourceReceipt):
        raise ValueError("Typed HF provenance factory requires the state source receipt")
    if receipt.hf_mode != hf_mode:
        raise ValueError("Typed HF provenance factory mode does not match the receipt")
    if overlap_blocks is not screened_block_bundle.screened_blocks:
        raise ValueError("Typed HF provenance factory requires the exact screened blocks")
    if receipt.screened_block_bundle_sha256 != screened_block_bundle.fingerprint:
        raise ValueError("Typed HF provenance factory bundle receipt mismatch")
    if receipt.interaction_spec_fingerprint != screened_block_bundle.interaction_spec.fingerprint:
        raise ValueError("Typed HF provenance factory interaction receipt mismatch")
    observed_iterations = max(
        len(base_run.iter_energy),
        len(base_run.iter_err),
        len(base_run.iter_oda),
    )
    if observed_iterations > int(requested_max_iterations):
        raise ValueError("Typed HF solver histories exceed requested_max_iterations")

    # The base solver returns only after its final-state callback has populated
    # the final observables and diagnostics. Snapshot that complete state here,
    # immediately before issuing provenance, never from a pre-final SCF state.
    final_state_sha256 = _tbg_zero_field_hf_state_source_sha256(state)
    provenance = TBGZeroFieldHFRunProvenance(
        hf_mode=hf_mode,
        beta=float(beta),
        nu=float(state.nu),
        precision=float(state.precision),
        oda_stall_threshold=float(oda_stall_threshold),
        requested_max_iterations=int(requested_max_iterations),
        seed=int(base_run.seed),
        normalized_init_mode=str(base_run.init_mode),
        typed_receipt_fingerprint=receipt.fingerprint,
        interaction_spec_fingerprint=screened_block_bundle.interaction_spec.fingerprint,
        bm_generation_fingerprint=screened_block_bundle.bm_generation_fingerprint,
        mesh_fingerprint=screened_block_bundle.mesh_fingerprint,
        iter_energy_sha256=_tbg_zero_field_hf_history_sha256(
            base_run.iter_energy,
            name="iter_energy",
        ),
        iter_err_sha256=_tbg_zero_field_hf_history_sha256(
            base_run.iter_err,
            name="iter_err",
        ),
        iter_oda_sha256=_tbg_zero_field_hf_history_sha256(
            base_run.iter_oda,
            name="iter_oda",
        ),
        state_source_sha256=final_state_sha256,
        converged=bool(base_run.converged),
        exit_reason=str(base_run.exit_reason),
        _issuer=_TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER,
    )
    return RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=overlap_blocks,
        screened_block_bundle=screened_block_bundle,
        provenance=provenance,
        iter_energy=base_run.iter_energy,
        iter_err=base_run.iter_err,
        iter_oda=base_run.iter_oda,
        init_mode=base_run.init_mode,
        seed=base_run.seed,
        converged=base_run.converged,
        exit_reason=base_run.exit_reason,
        _production_issuer=_TBG_ZERO_FIELD_HF_RUN_PROVENANCE_ISSUER,
    )

@dataclass
class RestrictedHartreeFockState:
    h0: np.ndarray
    sigma_z: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    sigma_ztauz: np.ndarray
    nu: float
    v0: float
    mu: float = float("nan")
    precision: float = 1e-5
    n_spin: int = 2
    n_eta: int = 2
    n_band: int = 2
    diagnostics: dict[str, float] = field(default_factory=dict)
    hf_source_receipt: "TBGZeroFieldHFSourceReceipt | None" = None
    interaction_spec: TBGZeroFieldInteractionSpec | None = None

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @classmethod
    def from_bm_solution(
        cls,
        solution: BMSolution,
        *,
        nu: float,
        precision: float = 1e-5,
    ) -> "RestrictedHartreeFockState":
        h0 = build_h0_from_bm(solution)
        nt, nk = h0.shape[0], h0.shape[2]
        return cls(
            h0=h0,
            sigma_z=np.asarray(solution.sigma_z, dtype=np.complex128).copy(),
            density=np.zeros((nt, nt, nk), dtype=np.complex128),
            hamiltonian=h0.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            sigma_ztauz=np.zeros((nt, nk), dtype=float),
            nu=float(nu),
            v0=coulomb_unit(solution.params),
            precision=float(precision),
            n_spin=int(solution.n_spin),
            n_eta=int(solution.n_eta),
            n_band=int(solution.nb),
        )


def validate_tbg_zero_field_typed_hf_source(
    state: RestrictedHartreeFockState,
    solution: BMSolution,
    screened_block_bundle: TBGZeroFieldScreenedBlockBundle,
    *,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
) -> None:
    """Validate that a typed runner uses one unchanged BM/bundle source."""

    screened_block_bundle.validate_for_solution(solution)
    if overlap_blocks is not screened_block_bundle.screened_blocks:
        raise ValueError(
            "Typed HF must receive screened_block_bundle.screened_blocks exactly; arbitrary "
            "or stale blocks cannot be relabelled typed"
        )
    if params is not solution.params:
        raise ValueError("Typed HF params must be the exact params object carried by BMSolution")
    if not np.array_equal(
        np.asarray(lattice_kvec, dtype=np.complex128).reshape(-1),
        np.asarray(solution.lattice_kvec, dtype=np.complex128).reshape(-1),
    ):
        raise ValueError("Typed HF lattice_kvec does not match the source BMSolution")
    expected_h0 = build_h0_from_bm(solution)
    if not np.array_equal(np.asarray(state.h0, dtype=np.complex128), expected_h0):
        raise ValueError("Typed HF state.h0 was not built from the source BMSolution")
    if not np.array_equal(
        np.asarray(state.sigma_z, dtype=np.complex128),
        np.asarray(solution.sigma_z, dtype=np.complex128),
    ):
        raise ValueError("Typed HF state.sigma_z was not built from the source BMSolution")
    expected_v0 = coulomb_unit(solution.params)
    if float(state.v0) != expected_v0:
        raise ValueError(
            "Typed HF state.v0 must equal coulomb_unit(solution.params) exactly; "
            f"got {float(state.v0)!r}, expected {expected_v0!r}"
        )
    if (
        int(state.n_spin),
        int(state.n_eta),
        int(state.n_band),
        int(state.nk),
    ) != (
        int(solution.n_spin),
        int(solution.n_eta),
        int(solution.nb),
        int(solution.nk),
    ):
        raise ValueError("Typed HF state dimensions do not match the source BMSolution")

def coulomb_unit(params: TBGParameters) -> float:
    electron_charge = 1.6e-19
    vacuum_permittivity = 8.8541878128e-12
    graphene_lattice_constant = 2.46e-10
    area_moire = abs((params.a1.conjugate() * params.a2).imag)
    return float(electron_charge / (4.0 * np.pi * vacuum_permittivity * area_moire * graphene_lattice_constant) * 1e3)


def screened_coulomb(
    q: complex,
    lm: float,
    *,
    relative_permittivity: float = 15.0,
    zero_cutoff: float = 1e-6,
    finite_zero_limit: bool = False,
) -> float:
    """Low-level legacy/untyped dimensionless kernel.

    Production zero-field HF must reach this function through
    :class:`TBGZeroFieldInteractionSpec`; direct calls are diagnostic-only and
    ``lm`` is dimensionless, not a gate distance in nm.
    """

    q_abs = abs(q)
    if q_abs < zero_cutoff:
        return float(2.0 * np.pi * 2.0 * lm / relative_permittivity) if finite_zero_limit else 0.0
    return float(2.0 * np.pi / (relative_permittivity * q_abs) * np.tanh(q_abs * 4.0 * lm / 2.0))


def build_h0_from_bm(solution: BMSolution) -> np.ndarray:
    nt = solution.nt
    nk = solution.nk
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    flattened = solution.flattened_energies()
    for ik in range(nk):
        np.fill_diagonal(h0[:, :, ik], flattened[:, ik])
    return h0


def reciprocal_shift_labels(lg: int) -> tuple[int, ...]:
    if lg <= 0 or lg % 2 == 0:
        raise ValueError(f"Expected a positive odd lg, got {lg}")
    half_width = (lg - 1) // 2
    return tuple(range(-half_width, half_width + 1))


def build_overlap_block_set(
    target_solution: BMSolution,
    source_solution: BMSolution | None = None,
    *,
    lg: int | None = None,
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
    relative_permittivity: float | None = None,
    screening_lm: float | None = None,
    finite_zero_limit: bool | None = None,
    zero_cutoff: float | None = None,
) -> HFOverlapBlockSet:
    from .overlap import calculate_overlap_between

    resolved_epsr, resolved_lm, resolved_zero_limit, resolved_zero_cutoff = (
        _resolve_tbg_zero_field_screening(
            target_solution.params,
            interaction_spec=interaction_spec,
            legacy_untyped=legacy_untyped,
            relative_permittivity=relative_permittivity,
            screening_lm=screening_lm,
            finite_zero_limit=finite_zero_limit,
            zero_cutoff=zero_cutoff,
        )
    )
    source_solution = target_solution if source_solution is None else source_solution
    lG = target_solution.lg if lg is None else int(lg)
    labels = reciprocal_shift_labels(lG)
    shifts = tuple((m, n) for n in labels for m in labels)
    typed_active_shifts = (
        None
        if interaction_spec is None
        else frozenset(tbg_zero_field_active_shift_inventory())
    )
    if typed_active_shifts is not None and not typed_active_shifts <= set(shifts):
        missing = sorted(typed_active_shifts - set(shifts))
        raise ValueError(
            "Typed overlap_lg inventory does not fully cover the canonical G-label hex shell; "
            f"overlap_lg={lG}, missing={missing}"
        )
    gvecs = np.asarray([m * target_solution.params.g1 + n * target_solution.params.g2 for m, n in shifts], dtype=np.complex128)
    overlaps = {shift: calculate_overlap_between(target_solution, source_solution, shift[0], shift[1]) for shift in shifts}
    diagonal_overlaps, hartree_screening, fock_screening = _precompute_overlap_screening(
        shifts,
        gvecs,
        overlaps,
        params=target_solution.params,
        target_kvec=np.asarray(target_solution.lattice_kvec, dtype=np.complex128),
        source_kvec=np.asarray(source_solution.lattice_kvec, dtype=np.complex128),
        relative_permittivity=resolved_epsr,
        screening_lm=resolved_lm,
        finite_zero_limit=resolved_zero_limit,
        zero_cutoff=resolved_zero_cutoff,
        active_shifts=typed_active_shifts,
    )
    return HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_tbg_zero_field_screened_block_bundle(
    solution: BMSolution,
    *,
    interaction_spec: TBGZeroFieldInteractionSpec,
    overlap_lg: int | None = None,
) -> TBGZeroFieldScreenedBlockBundle:
    """Build and freeze the sole block source accepted by typed HF receipts."""

    validate_tbg_zero_field_typed_bm_lg(solution.lg)
    mesh = _require_torus_mesh(solution)
    solution.validate_source_attestation(require_torus=True)
    if not isinstance(interaction_spec, TBGZeroFieldInteractionSpec):
        raise TypeError("interaction_spec must be TBGZeroFieldInteractionSpec")
    if interaction_spec.transfer_cutoff_policy != TBG_ZERO_FIELD_TRANSFER_CUTOFF_POLICY:
        raise ValueError("Typed screened-block builder received an unsupported cutoff policy")
    if interaction_spec.reference_scheme != TBG_ZERO_FIELD_REFERENCE_SCHEME:
        raise ValueError("Typed screened-block builder requires the central-average reference")
    resolved_overlap_lg = validate_tbg_zero_field_typed_overlap_lg(
        solution.lg if overlap_lg is None else overlap_lg
    )
    expected_active = tbg_zero_field_active_shift_inventory()
    available_labels = reciprocal_shift_labels(resolved_overlap_lg)
    available_shifts = {(m, n) for n in available_labels for m in available_labels}
    if not set(expected_active) <= available_shifts:
        missing = sorted(set(expected_active) - available_shifts)
        raise ValueError(
            "Typed overlap_lg inventory does not fully cover the canonical G-label hex shell; "
            f"overlap_lg={resolved_overlap_lg}, missing={missing}"
        )
    raw_blocks = build_overlap_block_set(
        solution,
        lg=resolved_overlap_lg,
        interaction_spec=interaction_spec,
    )
    screened_blocks = _freeze_overlap_blocks(raw_blocks)
    reference = tbg_zero_field_central_average_reference_projector(solution)
    generation_fingerprint = tbg_zero_field_bm_generation_fingerprint(
        solution.params,
        lg=solution.lg,
        periodic_g_grid=solution.periodic_g_grid,
        sigma_rotation=solution.sigma_rotation,
        calculate_chern_operator=solution.calculate_chern_operator,
        torus_mesh_fingerprint=mesh.fingerprint,
    )
    solution_hash = tbg_zero_field_bm_solution_fingerprint(solution)
    active_hash = tbg_zero_field_active_shift_inventory_sha256(expected_active)
    kernel_hash = tbg_zero_field_overlap_kernel_inventory_fingerprint(screened_blocks)
    reference_hash = tbg_zero_field_reference_projector_sha256(reference)
    attestation = _ScreenedBlockBundleAttestation(
        screened_blocks,
        bm_solution_sha256=solution_hash,
        bm_generation_fingerprint=generation_fingerprint,
        mesh_fingerprint=mesh.fingerprint,
        interaction_spec_fingerprint=interaction_spec.fingerprint,
        overlap_lg=resolved_overlap_lg,
    )
    return TBGZeroFieldScreenedBlockBundle(
        screened_blocks=screened_blocks,
        interaction_spec=interaction_spec,
        overlap_lg=resolved_overlap_lg,
        active_shifts=expected_active,
        active_shift_inventory_sha256=active_hash,
        overlap_kernel_inventory_sha256=kernel_hash,
        bm_solution_sha256=solution_hash,
        bm_generation_fingerprint=generation_fingerprint,
        mesh_fingerprint=mesh.fingerprint,
        n_band=int(solution.nb),
        reference_projector_convention=TBG_ZERO_FIELD_REFERENCE_PROJECTOR_CONVENTION,
        reference_projector_dimensions=tuple(reference.shape),
        reference_projector_sha256=reference_hash,
        companion_circular_total_q_cutoff_parity=(
            interaction_spec.companion_circular_total_q_cutoff_parity
        ),
        _attestation=attestation,
    )

def normalize_restricted_init_mode(init_mode: str) -> str:
    normalized = init_mode.strip().lower()
    aliases = {
        "bm": "bm",
        "random": "random",
        "educated": "educated",
        "vp": "vp",
        "kspinpair": "kspinpair",
        "spindown": "spindown",
        "downpair": "downpair",
        # These two names appear in the packaged B0 benchmark manifest.
        "sp": "spindown",
        "chern": "vp",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported restricted init mode: {init_mode}. "
            "Supported modes: bm, random, educated, vp, kspinpair, spindown, downpair, sp, chern"
        )
    return aliases[normalized]


def canonical_fig6_flavor_sequence(init_mode: str) -> tuple[tuple[int, int], ...]:
    init_mode = normalize_restricted_init_mode(init_mode)
    if init_mode in ("educated", "vp", "kspinpair"):
        return ((1, 0), (0, 0), (1, 1), (0, 1))
    if init_mode in ("spindown", "downpair"):
        return ((1, 0), (1, 1), (0, 0), (0, 1))
    raise ValueError(f"Unsupported canonical restricted init mode: {init_mode}")


def is_canonical_restricted_init(init_mode: str) -> bool:
    try:
        normalized = normalize_restricted_init_mode(init_mode)
    except ValueError:
        return False
    return normalized in ("educated", "vp", "kspinpair", "spindown", "downpair")


def restricted_occupied_state_count(nu: float, nt: int, nk: int) -> int:
    raw = (nu + 4.0) / 8.0 * nt * nk
    rounded = int(round(float(raw)))
    if abs(float(raw) - rounded) > 1e-9:
        raise ValueError(
            f"Filling nu={nu} gives non-integer occupied-state count {raw} "
            f"for nt={nt}, nk={nk}."
        )
    if rounded < 0 or rounded > nt * nk:
        raise ValueError(f"Filling nu={nu} gives occupied-state count {rounded} outside [0, {nt * nk}].")
    return rounded


def restricted_occupied_bands_per_k(nu: float, nt: int) -> int:
    raw = (nu + 4.0) / 8.0 * nt
    rounded = int(round(float(raw)))
    if abs(float(raw) - rounded) > 1e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer per-k occupation {raw} for nt={nt}.")
    if rounded < 0 or rounded > nt:
        raise ValueError(f"Filling nu={nu} gives per-k occupation {rounded} outside [0, {nt}].")
    return rounded


def restricted_filling(density: np.ndarray) -> float:
    nt = density.shape[0]
    nk = density.shape[2]
    total = float(np.trace(density, axis1=0, axis2=1).real.sum() + 0.5 * nt * nk)
    return float(8.0 * total / (nk * nt) - 4.0)


def _screened_coulomb_matrix(
    qvals: np.ndarray,
    lm: float,
    *,
    relative_permittivity: float = 15.0,
    zero_cutoff: float = 1e-6,
    finite_zero_limit: bool = False,
) -> np.ndarray:
    q_abs = np.abs(np.asarray(qvals, dtype=np.complex128))
    values = np.zeros_like(q_abs, dtype=float)
    if finite_zero_limit:
        values[q_abs < zero_cutoff] = 2.0 * np.pi * 2.0 * lm / relative_permittivity
    mask = q_abs >= zero_cutoff
    if np.any(mask):
        values[mask] = 2.0 * np.pi / (relative_permittivity * q_abs[mask]) * np.tanh(q_abs[mask] * 2.0 * lm)
    return values


def _hex_shell_contains(params: TBGParameters, gvec: complex) -> bool:
    g0 = abs(3.0 * params.g1 + 3.0 * params.g2) * 1.00001
    angle_mod = np.mod(np.angle(gvec), np.pi / 3.0) - np.pi / 6.0
    denominator = abs(np.cos(angle_mod))
    if denominator < 1e-15:
        return False
    shell_radius = g0 * np.cos(np.pi / 6.0) / denominator
    return abs(gvec) < shell_radius


def _precompute_overlap_screening(
    shifts: tuple[tuple[int, int], ...],
    gvecs: np.ndarray,
    overlaps: dict[tuple[int, int], np.ndarray],
    *,
    params: TBGParameters,
    target_kvec: np.ndarray,
    source_kvec: np.ndarray,
    relative_permittivity: float = 15.0,
    screening_lm: float | None = None,
    finite_zero_limit: bool = False,
    zero_cutoff: float = 1e-6,
    active_shifts: frozenset[tuple[int, int]] | None = None,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float], dict[tuple[int, int], np.ndarray]]:
    lm = float(np.sqrt(abs(params.a1) * abs(params.a2)) if screening_lm is None else screening_lm)
    diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(shifts, gvecs, strict=True):
        if active_shifts is None:
            is_active = _hex_shell_contains(params, complex(gvec))
        else:
            is_active = shift in active_shifts
        if not is_active:
            continue
        overlap = overlaps[shift]
        diagonal_overlaps[shift] = np.diagonal(overlap, axis1=1, axis2=3)
        hartree_screening[shift] = screened_coulomb(
            complex(gvec),
            lm,
            relative_permittivity=relative_permittivity,
            zero_cutoff=zero_cutoff,
            finite_zero_limit=finite_zero_limit,
        )
        fock_screening[shift] = _screened_coulomb_matrix(
            source_kvec[None, :] - target_kvec[:, None] + complex(gvec),
            lm,
            relative_permittivity=relative_permittivity,
            zero_cutoff=zero_cutoff,
            finite_zero_limit=finite_zero_limit,
        )
    return diagonal_overlaps, hartree_screening, fock_screening


def _with_tbg_overlap_screening(
    overlap_blocks: HFOverlapBlockSet,
    *,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
    relative_permittivity: float | None = None,
    screening_lm: float | None = None,
    finite_zero_limit: bool | None = None,
    zero_cutoff: float | None = None,
) -> HFOverlapBlockSet:
    resolved_epsr, resolved_lm, resolved_zero_limit, resolved_zero_cutoff = (
        _resolve_tbg_zero_field_screening(
            params,
            interaction_spec=interaction_spec,
            legacy_untyped=legacy_untyped,
            relative_permittivity=relative_permittivity,
            screening_lm=screening_lm,
            finite_zero_limit=finite_zero_limit,
            zero_cutoff=zero_cutoff,
        )
    )
    # A typed source is rebuilt from overlaps so stale raw kernels cannot be
    # relabeled with a different physical interaction fingerprint.  The legacy
    # diagnostic path preserves pre-populated kernels for benchmark parity.
    if interaction_spec is None:
        diagonal_overlaps = dict(overlap_blocks.diagonal_overlaps)
        hartree_screening = dict(overlap_blocks.hartree_screening)
        fock_screening = dict(overlap_blocks.fock_screening)
    else:
        diagonal_overlaps = {}
        hartree_screening = {}
        fock_screening = {}
    typed_active_shifts = (
        None
        if interaction_spec is None
        else frozenset(tbg_zero_field_active_shift_inventory())
    )
    if typed_active_shifts is not None and not typed_active_shifts <= set(overlap_blocks.shifts):
        missing = sorted(typed_active_shifts - set(overlap_blocks.shifts))
        raise ValueError(
            "Typed overlap inventory does not fully cover the canonical G-label hex shell; "
            f"missing={missing}"
        )
    for shift, gvec in zip(overlap_blocks.shifts, overlap_blocks.gvecs, strict=True):
        if typed_active_shifts is None:
            is_active = _hex_shell_contains(params, complex(gvec))
        else:
            is_active = shift in typed_active_shifts
        if not is_active:
            continue
        overlap = overlap_blocks.overlaps[shift]
        diagonal = np.diagonal(overlap, axis1=1, axis2=3)
        hartree = screened_coulomb(
            complex(gvec),
            resolved_lm,
            relative_permittivity=resolved_epsr,
            zero_cutoff=resolved_zero_cutoff,
            finite_zero_limit=resolved_zero_limit,
        )
        fock = _screened_coulomb_matrix(
            lattice_kvec[None, :] - lattice_kvec[:, None] + complex(gvec),
            resolved_lm,
            relative_permittivity=resolved_epsr,
            zero_cutoff=resolved_zero_cutoff,
            finite_zero_limit=resolved_zero_limit,
        )
        if interaction_spec is None:
            diagonal_overlaps.setdefault(shift, diagonal)
            hartree_screening.setdefault(shift, hartree)
            fock_screening.setdefault(shift, fock)
        else:
            diagonal_overlaps[shift] = diagonal
            hartree_screening[shift] = hartree
            fock_screening[shift] = fock
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=diagonal_overlaps,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )


def build_interaction_hamiltonian(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    lattice_kvec: np.ndarray,
    params: TBGParameters,
    v0: float,
    *,
    beta: float = 1.0,
    interaction_spec: TBGZeroFieldInteractionSpec | None = None,
    legacy_untyped: bool = False,
    relative_permittivity: float | None = None,
    screening_lm: float | None = None,
    finite_zero_limit: bool | None = None,
    zero_cutoff: float | None = None,
) -> np.ndarray:
    lattice_kvec = np.asarray(lattice_kvec, dtype=np.complex128)
    if lattice_kvec.size != density.shape[2]:
        raise ValueError(f"Expected {density.shape[2]} k-points, got {lattice_kvec.size}")
    screened_overlap_blocks = _with_tbg_overlap_screening(
        overlap_blocks,
        lattice_kvec=lattice_kvec,
        params=params,
        interaction_spec=interaction_spec,
        legacy_untyped=legacy_untyped,
        relative_permittivity=relative_permittivity,
        screening_lm=screening_lm,
        finite_zero_limit=finite_zero_limit,
        zero_cutoff=zero_cutoff,
    )
    return build_projected_interaction_hamiltonian(
        density,
        screened_overlap_blocks,
        v0=v0,
        beta=beta,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
