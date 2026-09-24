"""Prepare and validate source-sealed restart-only VASP PAW-overlap oracles.

Preparation never runs VASP.  It validates the accepted 40-atom TPT bulk
parent, makes an independent WAVECAR clone, writes a controlled
``ALGO=Nothing`` input, and seals the exact source/input hashes.  The clone
mode is explicit: copy-on-write is preferred, while a full byte copy is the
safe fallback on filesystems such as Lustre that do not support reflinks.
The resulting
capsule is diagnostic-only: neither shell preparation nor a VASP overlap file
can authorize Hartree--Fock without the remaining screening, filling/reference,
and double-counting gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from itertools import combinations, product, zip_longest
from typing import Any

import numpy as np

from mean_field.core.io import (
    publish_staged_file_noreplace,
    read_json_artifact,
    unique_staging_path,
    write_json_artifact,
)
from mean_field.runtime import ensure_not_running_compute_on_login_node
from mean_field.systems.tpt_bulk.source import (
    TPT_BULK_SOURCE_MESH,
    build_tpt_bulk_active_eigensystem,
    load_tpt_bulk_source,
)
from mean_field.systems.tpt_bulk.wavefunctions import (
    TPT_BULK_POSCAR_SHA256,
    TPT_BULK_POTCAR_SHA256,
    TPT_BULK_WANNIER90_CHK_SHA256,
    TPT_BULK_WANNIER90_EIG_SHA256,
    TPT_BULK_WANNIER90_MMN_SHA256,
    TPT_BULK_WAVECAR_SHA256,
    Wannier90CheckpointReader,
    iter_wannier90_mmn_blocks,
    read_wannier90_eig,
    write_projected_spinor_wavecar,
)

TPT_BULK_PARENT_INCAR_SHA256 = (
    "ea49077245f14e0de877381fc6aa9b75d367328398345d72378ad96cd899c9e6"
)
TPT_BULK_PARENT_KPOINTS_SHA256 = (
    "ee765b517958b85e1a26f4e290fb40b572c1e358bd228cd1e8cd8b37941e7e59"
)
TPT_BULK_PARENT_WANNIER90_WIN_SHA256 = (
    "79457a99ed1a9fcdeda9fa0c5312a02aeedf6bccba87927ecb4d4ed725fe143a"
)
TPT_BULK_VASP_NCL_EXECUTABLE = Path("/data/home/apps/vasp/6.4.1/build/ncl/vasp")
TPT_BULK_VASP_MODULES = (
    "oneapi22.3",
    "vasp/vasp6.4.1_wannier90-3.1_cpu",
)
TPT_BULK_VASP_KPAR = 4
TPT_BULK_VASP_NCORE = 1
TPT_BULK_VASP_MPI_RANKS = 32
TPT_BULK_VASP_NPAR = 8
TPT_BULK_NATIVE_MMN_MAX_ABS_TOLERANCE = float(np.finfo(np.float32).eps)
TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV = 1.0e-10
# The full oracle and parent coefficients originate from complex64 WAVECAR
# records.  Their replay authority therefore cannot be stronger than one
# binary32 epsilon, even though the transformed WAVECAR is stored as complex128
# to avoid adding a second binary32 quantization.  This tolerance is local to
# same-parent oracle replay and never relaxes the separate 1e-10 HF closure.
TPT_BULK_ACTIVE_MMN_PROJECTION_TOLERANCE = float(np.finfo(np.float32).eps)
TPT_BULK_ACTIVE_MMN_TEXT_ROUNDING_BOUND = 5.0e-10
TPT_BULK_PURE_G_REVERSE_CLOSURE_TOLERANCE = 2.0 * float(np.finfo(np.float32).eps)
# Populated only after the immutable full-band mixed-q+G certificate exists.
# Active validation fails closed while these external trust anchors are empty.
TPT_BULK_MIXED_QG_REFERENCE_JOB_ID: int | None = None
TPT_BULK_MIXED_QG_REFERENCE_CERTIFICATE_NAME = ""
TPT_BULK_MIXED_QG_REFERENCE_CERTIFICATE_SHA256 = ""
TPT_BULK_MIXED_QG_PREFLIGHT_PATH = Path(
    "/data/work/ziyuzhu/tpt_bulk_w90_shell_preflight_v12/"
    "b1_shell_1_2_6529/PREFLIGHT.json"
)
TPT_BULK_MIXED_QG_PREFLIGHT_SHA256 = (
    "961b34043d376fdc88b3cab4f2ad2b9e214b6e4161e63de20d2a57dfe8df8381"
)
_ACTIVE_ORACLE_LAYOUTS = {
    "regular64": {
        "partition": "regular256,regular128,regular6430",
        "allocated_cpus": 64,
        "mpi_ranks": 64,
        "ranks_per_node": 64,
        "kpar": 4,
        "ncore": 1,
        "npar": 16,
    },
    # This profile matches the accepted 600-band oracle's KPAR=4, NPAR=8
    # reduction tree.  The 56-CPU test node remains exclusive because no
    # larger rank count preserves both that decomposition and NBANDS=16.
    "test32_reference_matched": {
        "partition": "test",
        "allocated_cpus": 56,
        "physical_node_cpus": 56,
        "mpi_ranks": 32,
        "ranks_per_node": 32,
        "kpar": 4,
        "ncore": 1,
        "npar": 8,
    },
    # test001 has 56 CPUs.  Forty-eight ranks with KPAR=3 is the largest
    # near-full-node layout for which KPAR divides 192 and NPAR divides 16.
    "test48": {
        "partition": "test",
        "allocated_cpus": 56,
        "physical_node_cpus": 56,
        "mpi_ranks": 48,
        "ranks_per_node": 48,
        "kpar": 3,
        "ncore": 1,
        "npar": 16,
    },
}

_PARENT_HASHES = {
    "INCAR": TPT_BULK_PARENT_INCAR_SHA256,
    "KPOINTS": TPT_BULK_PARENT_KPOINTS_SHA256,
    "POSCAR": TPT_BULK_POSCAR_SHA256,
    "POTCAR": TPT_BULK_POTCAR_SHA256,
    "WAVECAR": TPT_BULK_WAVECAR_SHA256,
    "wannier90.win": TPT_BULK_PARENT_WANNIER90_WIN_SHA256,
}
PAW_ORACLE_SELECTED_PROVENANCE_VERSION = "paw_oracle_selected_provenance_v2"
PAW_ORACLE_SELECTED_PROVENANCE_SCOPE = (
    "paw_oracle_v2_selected_set_not_complete_python_import_closure"
)
_PAW_ORACLE_LEGACY_PROVENANCE_FIELDS = frozenset(
    {
        "runtime_source_hashes",
        "runtime_source_hashes_at_preparation",
        "runtime_source_hashes_at_validation_start",
        "runtime_source_drift",
    }
)
_RUNTIME_SOURCE_RELATIVE_PATHS = (
    "scripts/mean_field_tools.py",
    "scripts/submit_mean_field.sbatch",
    "src/mean_field/runtime.py",
    "src/mean_field/core/io/__init__.py",
    "src/mean_field/core/io/artifacts.py",
    "src/mean_field/devtools/tpt_bulk_paw_oracle.py",
    "src/mean_field/systems/tpt_bulk/source.py",
    "src/mean_field/systems/tpt_bulk/wavefunctions.py",
)


@dataclass(frozen=True)
class OracleCase:
    name: str
    kmesh_tolerance: str
    search_shells: int
    shell_list: tuple[int, ...]
    authority: str


ORACLE_CASES = {
    "native": OracleCase(
        name="native",
        kmesh_tolerance="1.0e-4",
        search_shells=12,
        shell_list=(1, 2, 4),
        authority="accepted_native_neighbour_replay_canary",
    ),
    "pure_g": OracleCase(
        name="pure_g",
        kmesh_tolerance="1.0e-7",
        search_shells=2200,
        shell_list=(28, 60, 2195),
        authority="diagnostic_finite_g_paw_oracle_canary",
    ),
    "mixed_qg": OracleCase(
        name="mixed_qg",
        kmesh_tolerance="1.0e-7",
        search_shells=6529,
        shell_list=(1, 2, 6529),
        authority="diagnostic_general_finite_q_plus_G_orientation_canary_only",
    ),
}


_TRANSFER_KMESH_TOLERANCE = 1.0e-7
_TRANSFER_CHUNK_CHANNELS = 32
_WANNIER90_KMESH_NSUPCELL = 5


def _read_json_object(path: Path, *, domain: str) -> dict[str, Any]:
    """Read one strict JSON artifact and require an object at the top level."""

    try:
        payload = read_json_artifact(path)
    except ValueError as exc:
        raise ValueError(f"{domain} is not valid strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{domain} top-level JSON value must be a dictionary")
    return payload


def _selected_provenance_preparation_fields(
    hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "selected_provenance_version": PAW_ORACLE_SELECTED_PROVENANCE_VERSION,
        "selected_provenance_scope": PAW_ORACLE_SELECTED_PROVENANCE_SCOPE,
        "selected_provenance_hashes_at_preparation": hashes,
    }


def _selected_provenance_report_fields(
    preparation: dict[str, str],
    validation: dict[str, str],
    drift: dict[str, dict[str, str]],
) -> dict[str, object]:
    return {
        "selected_provenance_version": PAW_ORACLE_SELECTED_PROVENANCE_VERSION,
        "selected_provenance_scope": PAW_ORACLE_SELECTED_PROVENANCE_SCOPE,
        "selected_provenance_hashes_at_preparation": preparation,
        "selected_provenance_hashes_at_validation": validation,
        "selected_provenance_drift": drift,
    }


def _selected_provenance_inventory(
    owner: object,
    *,
    domain: str,
    repository_root: Path | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]]]:
    """Validate the exact PAW-v2 selected set before drift authorization.

    This inventory is a selected file set, not a complete Python import closure.
    Legacy seals fail closed and cannot be upgraded from their old fields.
    """

    if not isinstance(owner, dict):
        raise ValueError(f"{domain} selected provenance container must be a dictionary")
    legacy = sorted(_PAW_ORACLE_LEGACY_PROVENANCE_FIELDS.intersection(owner))
    if legacy:
        raise ValueError(
            f"{domain} uses legacy selected-provenance fields {legacy}; legacy "
            "artifacts are not accepted and cannot be automatically upgraded"
        )
    version = owner.get("selected_provenance_version")
    if version is None:
        raise ValueError(
            f"{domain} is missing selected_provenance_version; legacy artifacts "
            "cannot be automatically upgraded"
        )
    if not isinstance(version, str):
        raise ValueError(f"{domain} selected_provenance_version must be a string")
    if version != PAW_ORACLE_SELECTED_PROVENANCE_VERSION:
        raise ValueError(
            f"{domain} has unknown selected_provenance_version {version!r}; expected "
            f"{PAW_ORACLE_SELECTED_PROVENANCE_VERSION!r}"
        )
    scope = owner.get("selected_provenance_scope")
    if scope is None:
        raise ValueError(
            f"{domain} is missing selected_provenance_scope; expected "
            f"{PAW_ORACLE_SELECTED_PROVENANCE_SCOPE!r}"
        )
    if not isinstance(scope, str):
        raise ValueError(f"{domain} selected_provenance_scope must be a string")
    if scope != PAW_ORACLE_SELECTED_PROVENANCE_SCOPE:
        raise ValueError(
            f"{domain} has unknown selected_provenance_scope {scope!r}; expected "
            f"{PAW_ORACLE_SELECTED_PROVENANCE_SCOPE!r}"
        )
    hashes_key = "selected_provenance_hashes_at_preparation"
    preparation = owner.get(hashes_key)
    if preparation is None:
        raise ValueError(
            f"{domain} is missing {hashes_key}; legacy artifacts cannot be "
            "automatically upgraded"
        )
    if not isinstance(preparation, dict):
        raise ValueError(f"{domain} {hashes_key} must be a dictionary")
    if any(not isinstance(key, str) for key in preparation):
        raise ValueError(f"{domain} {hashes_key} keys must all be strings")

    expected = set(_RUNTIME_SOURCE_RELATIVE_PATHS)
    actual = set(preparation)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{domain} selected provenance keyset mismatch: "
            f"missing={missing}; extra={extra}"
        )
    invalid_hashes = sorted(
        key
        for key, value in preparation.items()
        if not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    )
    if invalid_hashes:
        raise ValueError(
            f"{domain} selected provenance hashes must be lowercase 64-hex strings; "
            f"invalid={invalid_hashes}"
        )

    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.expanduser().resolve()
    )
    current_paths = {relative: root / relative for relative in expected}
    current_missing = sorted(
        relative for relative, path in current_paths.items() if not path.is_file()
    )
    if current_missing:
        raise ValueError(
            f"{domain} current selected provenance keyset mismatch: "
            f"missing={current_missing}; extra=[]"
        )
    try:
        current = {
            relative: _sha256(current_paths[relative])
            for relative in _RUNTIME_SOURCE_RELATIVE_PATHS
        }
    except OSError as exc:
        raise ValueError(
            f"{domain} could not hash current selected provenance: {exc}"
        ) from exc
    content_changed = {
        relative: {
            "at_preparation": preparation[relative],
            "at_validation": current[relative],
        }
        for relative in _RUNTIME_SOURCE_RELATIVE_PATHS
        if preparation[relative] != current[relative]
    }
    return dict(preparation), current, content_changed


def _validate_postflight_provenance(
    *,
    owner: object,
    domain: str,
    output_root: Path,
    job_id: int,
    source_seal_sha256: str,
    submission_receipt_sha256: str,
    revalidation_seal_path: Path | None,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, str]],
    dict[str, Any] | None,
]:
    """Validate PAW-v2 provenance and authorize only same-keyset drift."""

    preparation, validation, drift = _selected_provenance_inventory(
        owner,
        domain=domain,
    )
    authorization: dict[str, Any] | None = None
    if drift:
        if revalidation_seal_path is None:
            raise ValueError(
                f"{domain} selected provenance content drift requires external "
                "artifact revalidation authorization"
            )
        authorization = _load_external_revalidation_authorization(
            path=revalidation_seal_path,
            output_root=output_root,
            job_id=job_id,
            source_seal_sha256=source_seal_sha256,
            submission_receipt_sha256=submission_receipt_sha256,
            source_drift=drift,
        )
    elif revalidation_seal_path is not None:
        raise ValueError(
            f"{domain} artifact revalidation authorization supplied without "
            "selected provenance content drift"
        )
    return preparation, validation, drift, authorization


def _half_open_mesh_axis(size: int) -> range:
    if size <= 0 or size % 2:
        raise ValueError("TPT transfer planning requires positive even mesh sizes")
    # Match the accepted VASP/Wannier90 Gamma grid: the Nyquist point is +1/2,
    # not -1/2 (e.g. x numerators -5,...,+6 for N=12).
    return range(-(size // 2) + 1, size // 2 + 1)


def _canonical_q_and_g(
    physical_numerator: tuple[int, int, int],
    mesh: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    q_values: list[int] = []
    g_values: list[int] = []
    for numerator, size in zip(physical_numerator, mesh, strict=True):
        residue = numerator % size
        q_value = residue - size if 2 * residue > size else residue
        delta = numerator - q_value
        if delta % size:
            raise RuntimeError("physical transfer does not admit an integer G label")
        q_values.append(q_value)
        g_values.append(delta // size)
    return tuple(q_values), tuple(g_values)


def _sign_orbit(values: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    axes = [(-value, value) if value else (0,) for value in values]
    return tuple(sorted(set(product(*axes))))


def _positive_transfer(values: tuple[int, int, int]) -> bool:
    return next(value for value in values if value) > 0


def _shell_column(
    absolute_numerator: tuple[int, int, int],
    step_vectors: np.ndarray,
) -> np.ndarray:
    moment = np.zeros((3, 3), dtype=np.float64)
    for signed in _sign_orbit(absolute_numerator):
        vector = np.asarray(signed, dtype=np.float64) @ step_vectors
        moment += np.outer(vector, vector)
    return np.asarray(
        (
            moment[0, 0],
            moment[1, 1],
            moment[2, 2],
            moment[0, 1],
            moment[1, 2],
            moment[2, 0],
        ),
        dtype=np.float64,
    )


def _build_wannier90_shell_records(
    *,
    reciprocal_rows: np.ndarray,
    mesh: tuple[int, int, int],
    target_absolute_numerators: set[tuple[int, int, int]],
    kmesh_tolerance: float,
) -> list[dict[str, Any]]:
    """Reproduce the fixed-shell index ordering used by Wannier90 3.1."""

    reciprocal = np.asarray(reciprocal_rows, dtype=np.float64)
    if reciprocal.shape != (3, 3):
        raise ValueError("reciprocal lattice must have shape (3,3)")
    gram = reciprocal @ reciprocal.T
    if float(np.max(np.abs(gram - np.diag(np.diag(gram))))) > 1.0e-12:
        raise ValueError("stock shell planner is qualified only for the orthorhombic cell")
    if not target_absolute_numerators:
        return []
    tolerance = float(kmesh_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("kmesh_tolerance must be finite and positive")
    mesh_array = np.asarray(mesh, dtype=np.int64)
    step_vectors = reciprocal / mesh_array[:, None]

    def distance(values: tuple[int, int, int]) -> float:
        return float(np.linalg.norm(np.asarray(values, dtype=np.float64) @ step_vectors))

    max_target_distance = max(distance(values) for values in target_absolute_numerators)
    # Wannier90 3.1 kmesh.F90 uses a compile-time nsupcell=5 and enumerates
    # every accepted k-grid point plus all lmn translations in [-5,5]^3.
    # Reproducing that finite candidate box is essential: an infinite-lattice
    # distance sort gives wrong high shell indices even when low indices agree.
    candidate_axes = []
    for size in mesh:
        candidate_axes.append(
            sorted(
                {
                    q_value + size * image
                    for q_value in _half_open_mesh_axis(size)
                    for image in range(
                        -_WANNIER90_KMESH_NSUPCELL,
                        _WANNIER90_KMESH_NSUPCELL + 1,
                    )
                }
            )
        )
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for values in product(*candidate_axes):
        if not any(values):
            continue
        signed = tuple(int(value) for value in values)
        shell_distance = distance(signed)
        if shell_distance <= max_target_distance + tolerance:
            candidates.append((shell_distance, signed))
    candidates.sort()

    catalog: dict[tuple[int, int, int], dict[str, Any]] = {}
    previous_distance = 0.0
    cursor = 0
    shell_index = 0
    while cursor < len(candidates):
        while (
            cursor < len(candidates)
            and candidates[cursor][0] <= previous_distance + tolerance
        ):
            cursor += 1
        if cursor == len(candidates):
            break
        center = candidates[cursor][0]
        stop = cursor
        while stop < len(candidates) and candidates[stop][0] < center + tolerance:
            stop += 1
        members = tuple(values for _distance, values in candidates[cursor:stop])
        absolute_members = {tuple(abs(value) for value in item) for item in members}
        shell_index += 1
        matched_targets = absolute_members & target_absolute_numerators
        for values in matched_targets:
            expected_orbit = set(_sign_orbit(values))
            if absolute_members != {values} or set(members) != expected_orbit:
                raise ValueError(
                    "target transfer is merged or truncated by the fixed shell search: "
                    f"{values} -> {sorted(members)}"
                )
            moment = np.zeros((3, 3), dtype=np.float64)
            for signed in members:
                vector = np.asarray(signed, dtype=np.float64) @ step_vectors
                moment += np.outer(vector, vector)
            catalog[values] = {
                "shell_index": shell_index,
                "distance_angstrom_inv": center,
                "multiplicity": len(members),
                "merged_absolute_numerators": [list(values)],
                "b1_column": [
                    moment[0, 0],
                    moment[1, 1],
                    moment[2, 2],
                    moment[0, 1],
                    moment[1, 2],
                    moment[2, 0],
                ],
            }
        previous_distance = center
        cursor = stop

    records: list[dict[str, Any]] = []
    for values in sorted(target_absolute_numerators):
        if values not in catalog:
            raise RuntimeError(f"failed to map transfer shell {values}")
        record = catalog[values]
        records.append({"absolute_numerator": list(values), **record})
    records.sort(key=lambda record: int(record["shell_index"]))
    return records


def _shell_batch_b1_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    columns = np.asarray([record["b1_column"] for record in records], dtype=np.float64).T
    singular_values = np.linalg.svd(columns, compute_uv=False)
    rank = int(np.linalg.matrix_rank(columns))
    target = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    weights, *_ = np.linalg.lstsq(columns, target, rcond=None)
    residual = columns @ weights - target
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    return {
        "rank": rank,
        "singular_values": singular_values.tolist(),
        "condition_number": condition_number,
        "weights_angstrom_squared": weights.tolist(),
        "max_abs_completeness_residual": float(np.max(np.abs(residual))),
    }


def _shell_batch_rank(records: list[dict[str, Any]]) -> int:
    return int(_shell_batch_b1_metrics(records)["rank"])


def _pack_stock_vasp_shell_batches(
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Pack full sign shells into non-overcomplete, B1-complete W90 lists."""

    by_multiplicity = {
        multiplicity: [
            record for record in records if int(record["multiplicity"]) == multiplicity
        ]
        for multiplicity in (2, 4, 8)
    }
    if sum(len(items) for items in by_multiplicity.values()) != len(records):
        raise ValueError("accepted orthorhombic inventory has an unexpected shell multiplicity")
    axis_shells = list(by_multiplicity[2])
    if not axis_shells:
        raise ValueError("B1-complete batching requires axial support shells")
    uncovered_axis = {
        tuple(record["absolute_numerator"]) for record in axis_shells
    }

    def with_axis_supports(
        base: list[dict[str, Any]],
        count: int,
    ) -> list[dict[str, Any]]:
        best: tuple[int, tuple[int, ...], tuple[dict[str, Any], ...]] | None = None
        base_keys = {tuple(record["absolute_numerator"]) for record in base}
        for support in combinations(axis_shells, count):
            support_keys = {
                tuple(record["absolute_numerator"]) for record in support
            }
            if base_keys & support_keys:
                continue
            trial = [*base, *support]
            if _shell_batch_rank(trial) != 3:
                continue
            score = sum(key in uncovered_axis for key in support_keys)
            indices = tuple(int(record["shell_index"]) for record in support)
            candidate = (score, tuple(-value for value in indices), support)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            raise RuntimeError("failed to construct a B1-complete axial support set")
        trial = [*base, *best[2]]
        for record in trial:
            uncovered_axis.discard(tuple(record["absolute_numerator"]))
        return trial

    batches: list[list[dict[str, Any]]] = []
    for record in by_multiplicity[8]:
        batches.append(with_axis_supports([record], 2))

    four_by_zero_axis: dict[int, list[dict[str, Any]]] = {
        axis: [] for axis in range(3)
    }
    for record in by_multiplicity[4]:
        zero_axis = next(
            index for index, value in enumerate(record["absolute_numerator"]) if not value
        )
        four_by_zero_axis[zero_axis].append(record)

    while all(four_by_zero_axis[axis] for axis in range(3)):
        batch = [four_by_zero_axis[axis].pop(0) for axis in range(3)]
        if _shell_batch_rank(batch) != 3:
            raise RuntimeError("three-axis multiplicity-four batch is not B1 complete")
        batches.append(batch)

    nonempty = [axis for axis in range(3) if four_by_zero_axis[axis]]
    while len(nonempty) >= 2:
        nonempty.sort(key=lambda axis: len(four_by_zero_axis[axis]), reverse=True)
        first, second = nonempty[:2]
        pattern = (first, first, second)
        if len(four_by_zero_axis[first]) < 2:
            pattern = (first, second, second)
        found: list[dict[str, Any]] | None = None
        for first_pair in combinations(four_by_zero_axis[pattern[0]], 2):
            if pattern[0] != pattern[1]:
                break
            for third in four_by_zero_axis[pattern[2]]:
                trial = [*first_pair, third]
                if _shell_batch_rank(trial) == 3:
                    found = trial
                    break
            if found is not None:
                break
        if found is None:
            pool = [
                record
                for axis in nonempty
                for record in four_by_zero_axis[axis]
            ]
            for trial_tuple in combinations(pool, 3):
                if _shell_batch_rank(list(trial_tuple)) == 3:
                    found = list(trial_tuple)
                    break
        if found is None:
            break
        for record in found:
            zero_axis = next(
                index for index, value in enumerate(record["absolute_numerator"]) if not value
            )
            four_by_zero_axis[zero_axis].remove(record)
        batches.append(found)
        nonempty = [axis for axis in range(3) if four_by_zero_axis[axis]]

    remaining_four = [
        record for axis in range(3) for record in four_by_zero_axis[axis]
    ]
    while len(remaining_four) >= 2:
        pair = remaining_four[:2]
        if _shell_batch_rank(pair) == 2:
            batches.append(with_axis_supports(pair, 1))
            del remaining_four[:2]
        else:
            batches.append(with_axis_supports([remaining_four.pop(0)], 2))
    if remaining_four:
        batches.append(with_axis_supports([remaining_four.pop()], 2))

    while uncovered_axis:
        target_key = min(uncovered_axis)
        target = next(
            record
            for record in axis_shells
            if tuple(record["absolute_numerator"]) == target_key
        )
        uncovered_axis.discard(target_key)
        batches.append(with_axis_supports([target], 2))

    for batch in batches:
        multiplicity = sum(int(record["multiplicity"]) for record in batch)
        if len(batch) != 3 or multiplicity > 12:
            raise RuntimeError("stock-VASP shell batch violates fixed-list limits")
        b1 = _shell_batch_b1_metrics(batch)
        if b1["rank"] != 3:
            raise RuntimeError("stock-VASP shell batch is not B1 complete")
        if b1["max_abs_completeness_residual"] > _TRANSFER_KMESH_TOLERANCE:
            raise RuntimeError(
                "stock-VASP shell batch fails the Wannier90 B1 tolerance: "
                f"{b1['max_abs_completeness_residual']}"
            )
    covered = {
        tuple(record["absolute_numerator"])
        for batch in batches
        for record in batch
    }
    expected = {tuple(record["absolute_numerator"]) for record in records}
    if covered != expected:
        raise RuntimeError("stock-VASP shell batches do not cover the transfer inventory")
    return batches


def build_active_transfer_plan(
    *,
    source_root: Path,
    local_field_shell: int,
) -> dict[str, Any]:
    """Build a source-bound, inversion-closed stock-VASP transfer plan."""

    if local_field_shell not in (0, 1):
        raise ValueError("local_field_shell must be 0 or 1")
    source = load_tpt_bulk_source(source_root)
    poscar_sha256 = _sha256(source.root / "POSCAR")
    if poscar_sha256 != TPT_BULK_POSCAR_SHA256:
        raise ValueError(
            "accepted TPT bulk POSCAR hash mismatch during transfer planning: "
            f"{poscar_sha256}"
        )
    mesh = tuple(int(value) for value in TPT_BULK_SOURCE_MESH)
    q_seed = {
        tuple(int(value) for value in values)
        for values in product(*(_half_open_mesh_axis(size) for size in mesh))
    }
    g_seed = {(0, 0, 0)}
    if local_field_shell == 1:
        for axis in range(3):
            for sign in (-1, 1):
                value = [0, 0, 0]
                value[axis] = sign
                g_seed.add(tuple(value))
    physical = {
        tuple(q[axis] + mesh[axis] * g[axis] for axis in range(3))
        for q in q_seed
        for g in g_seed
    }
    physical |= {tuple(-value for value in transfer) for transfer in tuple(physical)}
    absolute_shells = {
        tuple(abs(value) for value in transfer)
        for transfer in physical
        if any(transfer)
    }
    shell_records = _build_wannier90_shell_records(
        reciprocal_rows=source.reciprocal_rows_angstrom_inv,
        mesh=mesh,
        target_absolute_numerators=absolute_shells,
        kmesh_tolerance=_TRANSFER_KMESH_TOLERANCE,
    )
    batches = _pack_stock_vasp_shell_batches(shell_records)
    export_physical = {(0, 0, 0)}
    for values in absolute_shells:
        export_physical.update(_sign_orbit(values))

    # Integer-only ordering keeps labels and chunk boundaries independent of
    # floating-point reciprocal-metric rounding across platforms.
    positive = sorted(
        transfer
        for transfer in physical
        if any(transfer) and _positive_transfer(transfer)
    )
    labels = {(0, 0, 0): 0}
    for label, transfer in enumerate(positive, start=1):
        labels[transfer] = label
        labels[tuple(-value for value in transfer)] = -label
    channels = []
    for transfer in sorted(physical, key=lambda value: (abs(labels[value]), labels[value])):
        q_shift, g_integer = _canonical_q_and_g(transfer, mesh)
        channels.append(
            {
                "label": labels[transfer],
                "physical_q_numerator": list(transfer),
                "q_mesh_shift": list(q_shift),
                "g_integer": list(g_integer),
                "source": (
                    "exact_paw_orthonormality_identity"
                    if transfer == (0, 0, 0)
                    else "stock_vasp_paw_mmn"
                ),
            }
        )

    shell_occurrence_batches: dict[tuple[int, int, int], list[int]] = {}
    for batch_index, batch in enumerate(batches):
        for record in batch:
            key = tuple(record["absolute_numerator"])
            shell_occurrence_batches.setdefault(key, []).append(batch_index)
    shell_owner_batch = {
        key: min(indices) for key, indices in shell_occurrence_batches.items()
    }

    batch_payloads = []
    for batch_index, batch in enumerate(batches):
        batch = sorted(batch, key=lambda record: int(record["shell_index"]))
        exported = sorted(
            {
                signed
                for record in batch
                for signed in _sign_orbit(tuple(record["absolute_numerator"]))
            }
        )
        nntot = sum(int(record["multiplicity"]) for record in batch)
        b1_metrics = _shell_batch_b1_metrics(batch)
        owned_shells = {
            tuple(record["absolute_numerator"])
            for record in batch
            if shell_owner_batch[tuple(record["absolute_numerator"])] == batch_index
        }
        batch_payloads.append(
            {
                "batch_index": batch_index,
                "name": f"qg{local_field_shell}-batch-{batch_index:03d}",
                "search_shells": max(int(record["shell_index"]) for record in batch),
                "shell_list": [int(record["shell_index"]) for record in batch],
                "shell_absolute_numerators": [
                    record["absolute_numerator"] for record in batch
                ],
                "nntot": nntot,
                "expected_mmn_blocks": 192 * nntot,
                "fixed_shell_svd_rank": b1_metrics["rank"],
                "b1_metrics": b1_metrics,
                "b1_complete": True,
                "skip_b1_tests": False,
                "shell_occurrences": [
                    {
                        "absolute_numerator": record["absolute_numerator"],
                        "shell_index": int(record["shell_index"]),
                        "occurrence_count": len(
                            shell_occurrence_batches[
                                tuple(record["absolute_numerator"])
                            ]
                        ),
                        "canonical_owner_batch": shell_owner_batch[
                            tuple(record["absolute_numerator"])
                        ],
                        "is_canonical_owner": shell_owner_batch[
                            tuple(record["absolute_numerator"])
                        ]
                        == batch_index,
                    }
                    for record in batch
                ],
                "retained_physical_q_numerators": [
                    list(value)
                    for value in exported
                    if value in physical
                    and tuple(abs(item) for item in value) in owned_shells
                ],
                "duplicate_validation_q_numerators": [
                    list(value)
                    for value in exported
                    if value in physical
                    and tuple(abs(item) for item in value) not in owned_shells
                ],
                "discarded_extra_physical_q_numerators": [
                    list(value) for value in exported if value not in physical
                ],
            }
        )

    chunk_count = (len(channels) + _TRANSFER_CHUNK_CHANNELS - 1) // _TRANSFER_CHUNK_CHANNELS
    return {
        "schema": "tpt-bulk-active16-paw-transfer-plan-v2",
        "status": "planned_not_exported_not_hf_authority",
        "source": {
            "root": str(source.root),
            "hr_sha256": source.hr_sha256,
            "poscar_sha256": poscar_sha256,
            "mesh": list(mesh),
            "reciprocal_rows_angstrom_inv": source.reciprocal_rows_angstrom_inv.tolist(),
        },
        "inventory": {
            "local_field_shell": local_field_shell,
            "g_seed": [list(value) for value in sorted(g_seed)],
            "canonical_q_domain": "accepted_gamma_grid_componentwise_-N/2+1_to_+N/2",
            "physical_q_numerator_definition": "p=q_mesh_shift+mesh*g_integer",
            "required_channel_count_including_zero": len(physical),
            "stock_vasp_export_channel_count_including_zero": len(export_physical),
            "discarded_stock_shell_extras": len(export_physical - physical),
            "complete_q_residue_count": len(
                {
                    tuple(
                        value % size
                        for value, size in zip(item, mesh, strict=True)
                    )
                    for item in physical
                }
            ),
            "inversion_closed": physical
            == {tuple(-value for value in transfer) for transfer in physical},
            "channels": channels,
        },
        "wannier90": {
            "version": "3.1",
            "kmesh_nsupcell": _WANNIER90_KMESH_NSUPCELL,
            "kmesh_tolerance": _TRANSFER_KMESH_TOLERANCE,
            "shell_count": len(shell_records),
            "max_shell_index": max(int(record["shell_index"]) for record in shell_records),
            "batch_count": len(batch_payloads),
            "neighbour_cap": 12,
            "shell_list_cap": 6,
            "effective_shell_count_cap_due_to_fixed_shell_svd": 3,
            "batching_rule": (
                "every fixed shell list has exactly three independent B1 columns; "
                "skip_B1_tests is forbidden and axial support shells may repeat"
            ),
            "assembly_rule": (
                "retain each shell only from canonical_owner_batch; repeated axial "
                "supports are validation duplicates and must agree before discard"
            ),
            "shell_occurrence_count": sum(
                len(indices) for indices in shell_occurrence_batches.values()
            ),
            "duplicate_shell_occurrence_count": sum(
                len(indices) - 1 for indices in shell_occurrence_batches.values()
            ),
            "shell_owners": [
                {
                    "absolute_numerator": list(key),
                    "canonical_owner_batch": shell_owner_batch[key],
                    "occurrence_batches": shell_occurrence_batches[key],
                }
                for key in sorted(shell_occurrence_batches)
            ],
            "shells": shell_records,
            "batches": batch_payloads,
        },
        "compact_vertex_store": {
            "schema": "tpt-bulk-active16-paw-vertex-chunks-v1",
            "rho_definition": (
                "rho_mn(k,q+G)=<psi_m,k+q|exp(i(q+G).r)|psi_n,k>"
            ),
            "mmn_to_rho_map": "rho(k,q+G)=M_w90(k,q+G)^dagger",
            "mmn_orientation_status": "derived_pending_general_finite_q_asymmetric_canary",
            "dtype": "complex128",
            "array_order": "C",
            "rho_chunk_shape": [f"1..{_TRANSFER_CHUNK_CHANNELS}", 192, 16, 16],
            "chunk_channels": _TRANSFER_CHUNK_CHANNELS,
            "chunk_count": chunk_count,
            "estimated_uncompressed_rho_bytes": len(channels) * 192 * 16 * 16 * 16,
            "manifest_requirements": [
                "exact channel inventory and reverse labels",
                "per-chunk SHA-256 and immutable publication",
                "source seal, submission receipts, active WAVECAR seal, and VASP provenance",
                "MMN exact declared-block exhaustion plus EOF",
                "duplicate-shell replay agreement before deduplication",
                "stable inode, size, and mtime around every streamed input",
            ],
            "microscopic_hf_compatibility": (
                "channels map exactly to q_mesh_shifts/g_integer/rho; the current v2 "
                "loader still requires a materialized NPZ and diagonal scalar weights, "
                "so off-diagonal W_GGprime(q) needs an explicit adapter extension"
            ),
        },
        "authority": {
            "transfer_inventory_authority": False,
            "general_finite_q_orientation_authority": False,
            "screening_authority": False,
            "filling_reference_authority": False,
            "finite_g_convergence_authority": False,
            "production_density_vertex_authority": False,
            "production_hf_authority": False,
        },
    }


def prepare_active_transfer_plan(
    *,
    source_root: Path,
    output_root: Path,
    local_field_shell: int,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace transfer plan: {output_root}")
    plan = build_active_transfer_plan(
        source_root=source_root,
        local_field_shell=local_field_shell,
    )
    plan["planner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    staging = output_root.with_name(
        f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    try:
        path = staging / "TRANSFER_PLAN.json"
        _write_json_new(path, plan)
        path.chmod(0o444)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            raise FileExistsError(f"refusing to replace transfer plan: {output_root}")
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_unique_line(
    lines: list[str],
    *,
    key: str,
    replacement: str,
) -> None:
    indices = [
        index
        for index, line in enumerate(lines)
        if line.split("#", 1)[0].strip().lower().startswith(f"{key.lower()} ")
        or line.split("#", 1)[0].strip().lower().startswith(f"{key.lower()}=")
    ]
    if len(indices) != 1:
        raise ValueError(f"expected exactly one {key} line, found {indices}")
    lines[indices[0]] = replacement


def render_oracle_incar(parent_text: str, case: OracleCase) -> str:
    """Return a restart-only VASP input with an explicit Wannier shell inventory."""

    required_fragments = (
        "ALGO = Nothing",
        "ISTART = 1",
        "ICHARG = 0",
        "ISYM = -1",
        "NBANDS = 600",
        "NUM_WANN = 320",
        "LWANNIER90 = .TRUE.",
        "LSORBIT = .TRUE.",
        'WANNIER90_WIN = "',
    )
    missing = [fragment for fragment in required_fragments if fragment not in parent_text]
    if missing:
        raise ValueError(f"accepted parent INCAR is missing required restart tags: {missing}")

    lines = parent_text.splitlines()
    active_parallel_tags = [
        line
        for line in lines
        if line.split("#", 1)[0].strip().lower().startswith(("kpar", "ncore"))
    ]
    if active_parallel_tags:
        raise ValueError(
            f"accepted parent unexpectedly has active KPAR/NCORE tags: {active_parallel_tags}"
        )
    parallel_index = next(
        index for index, line in enumerate(lines) if line.strip().lower() == "parallel"
    )
    lines[parallel_index + 1 : parallel_index + 1] = [
        f"KPAR = {TPT_BULK_VASP_KPAR}",
        f"NCORE = {TPT_BULK_VASP_NCORE}",
        "NWRITE = 2",
    ]
    _replace_unique_line(lines, key="LWAVE", replacement="LWAVE = .FALSE.")
    _replace_unique_line(
        lines,
        key="kmesh_tol",
        replacement=f"kmesh_tol = {case.kmesh_tolerance}",
    )

    if any(
        line.split("#", 1)[0].strip().lower().startswith("lwrite_mmn_amn")
        for line in lines
    ):
        _replace_unique_line(
            lines,
            key="LWRITE_MMN_AMN",
            replacement="LWRITE_MMN_AMN = .TRUE.",
        )
    else:
        index = next(
            index
            for index, line in enumerate(lines)
            if line.split("#", 1)[0].strip().lower().startswith("lwannier90")
        )
        lines.insert(index + 1, "LWRITE_MMN_AMN = .TRUE.")

    if any(
        line.split("#", 1)[0].strip().lower().startswith("search_shells")
        or line.split("#", 1)[0].strip().lower().startswith("shell_list")
        for line in lines
    ):
        raise ValueError("parent INCAR unexpectedly already declares explicit Wannier shells")
    kmesh_index = next(
        index
        for index, line in enumerate(lines)
        if line.split("#", 1)[0].strip().lower().startswith("kmesh_tol")
    )
    lines[kmesh_index + 1 : kmesh_index + 1] = [
        f"search_shells = {case.search_shells}",
        "shell_list = " + ", ".join(str(value) for value in case.shell_list),
    ]

    rendered = "\n".join(lines) + "\n"
    forbidden = ("LWANNIER90_RUN = .TRUE.", "ALGO = Normal", "ALGO = Fast")
    if any(text in rendered for text in forbidden):
        raise ValueError("rendered oracle INCAR enables an unsupported electronic/Wannier run")
    return rendered


def render_active_oracle_incar(
    parent_text: str,
    case: OracleCase,
    *,
    kpar: int = 4,
) -> str:
    """Render a no-update 16-band transformed-WAVECAR PAW export input."""

    lines = render_oracle_incar(parent_text, case).splitlines()
    _replace_unique_line(lines, key="NBANDS", replacement="NBANDS = 16")
    _replace_unique_line(lines, key="NUM_WANN", replacement="NUM_WANN = 16")
    _replace_unique_line(lines, key="KPAR", replacement=f"KPAR = {int(kpar)}")
    if any(
        line.split("#", 1)[0].strip().lower().startswith("nelect")
        for line in lines
    ):
        _replace_unique_line(lines, key="NELECT", replacement="NELECT = 8")
    else:
        insertion = next(
            index
            for index, line in enumerate(lines)
            if line.split("#", 1)[0].strip().lower().startswith("nbands")
        )
        lines.insert(insertion, "NELECT = 8")
    guiding_lines = [
        index
        for index, line in enumerate(lines)
        if line.split("#", 1)[0].strip().lower().startswith("guiding_centres")
    ]
    if len(guiding_lines) > 1:
        raise ValueError("parent INCAR contains multiple guiding_centres tags")
    if guiding_lines:
        lines[guiding_lines[0]] = "guiding_centres = false"
    if any(
        line.split("#", 1)[0].strip().lower().startswith("lscdm")
        for line in lines
    ):
        _replace_unique_line(lines, key="LSCDM", replacement="LSCDM = .TRUE.")
    else:
        insertion = next(
            index
            for index, line in enumerate(lines)
            if line.split("#", 1)[0].strip().lower().startswith("lwannier90")
        )
        lines.insert(insertion + 1, "LSCDM = .TRUE.")

    projection_begin = [
        index for index, line in enumerate(lines) if line.strip().lower() == "begin projections"
    ]
    projection_end = [
        index for index, line in enumerate(lines) if line.strip().lower() == "end projections"
    ]
    if bool(projection_begin) != bool(projection_end) or len(projection_begin) > 1:
        raise ValueError("malformed Wannier90 projection block in parent INCAR")
    if projection_begin:
        begin = projection_begin[0]
        end = projection_end[0]
        if end <= begin:
            raise ValueError("malformed Wannier90 projection block ordering")
        del lines[begin : end + 1]

    rendered = "\n".join(lines) + "\n"
    required = (
        "ALGO = Nothing",
        "ISTART = 1",
        "ICHARG = 0",
        "NBANDS = 16",
        "NELECT = 8",
        "NUM_WANN = 16",
        "LSCDM = .TRUE.",
        "LWRITE_MMN_AMN = .TRUE.",
        "LWAVE = .FALSE.",
        f"KPAR = {int(kpar)}",
        "NCORE = 1",
    )
    missing = [item for item in required if item not in rendered]
    if missing:
        raise RuntimeError(f"active oracle INCAR lost required tags: {missing}")
    return rendered


def _validate_parent(parent_root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name, expected in _PARENT_HASHES.items():
        path = parent_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing accepted-parent source: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"accepted-parent {name} hash mismatch: expected {expected}, got {actual}"
            )
        stat = path.stat()
        inventory[name] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "size_bytes": stat.st_size,
            "mtime_ns_at_preparation": stat.st_mtime_ns,
        }
    return inventory


def _copy_wavecar(source: Path, destination: Path, *, mode: str) -> None:
    if mode not in {"reflink", "full"}:
        raise ValueError(f"unsupported WAVECAR copy mode: {mode}")
    command = ["cp"]
    if mode == "reflink":
        command.append("--reflink=always")
    # This cluster's older coreutils has no --reflink=never spelling.  Plain
    # cp performs the required byte copy and the inode check below rejects
    # link-based fallbacks.
    command.extend(
        [
            "--preserve=mode,timestamps",
            "--",
            str(source),
            str(destination),
        ]
    )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"{mode} WAVECAR clone failed; refusing hardlink/symlink fallback: "
            f"{result.stderr.strip()}"
        )
    source_stat = source.stat()
    destination_stat = destination.stat()
    if source_stat.st_ino == destination_stat.st_ino:
        raise RuntimeError("WAVECAR clone shares the source inode")
    if destination.is_symlink():
        raise RuntimeError("WAVECAR clone must not be a symlink")


def _write_text_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Strictly serialize and publish one JSON artifact without replacement."""

    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to replace existing JSON artifact: {path}")
    staged = unique_staging_path(path)
    try:
        write_json_artifact(payload, staged)
        publish_staged_file_noreplace(staged, path)
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def prepare_active_wavecar(
    *,
    source_root: Path,
    parent_root: Path,
    output_root: Path,
    active_ranks_1based: tuple[int, int] = (233, 248),
    serialized_occupied_rank: int = 8,
) -> dict[str, Any]:
    """Build a small active-state WAVECAR for PAW-complete VASP export.

    The serialized occupations only make the restart file internally usable;
    they do not authorize the physical filling of a later HF calculation.
    """

    ensure_not_running_compute_on_login_node("prepare TPT bulk active WAVECAR")
    source_root = source_root.expanduser().resolve()
    parent_root = parent_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace active WAVECAR capsule: {output_root}")
    first, last = (int(value) for value in active_ranks_1based)
    active_rank = last - first + 1
    if active_rank <= 0 or not 0 <= serialized_occupied_rank <= active_rank:
        raise ValueError("invalid active rank or serialized occupation rank")

    repository_root = Path(__file__).resolve().parents[3]
    runtime_hashes = {
        relative: _sha256(repository_root / relative)
        for relative in _RUNTIME_SOURCE_RELATIVE_PATHS
    }
    source = load_tpt_bulk_source(source_root)
    eigensystem = build_tpt_bulk_active_eigensystem(
        source,
        mesh=TPT_BULK_SOURCE_MESH,
        active_ranks_1based=(first, last),
    )
    checkpoint_path = parent_root / "wannier90.chk"
    with Wannier90CheckpointReader(
        checkpoint_path,
        expected_sha256=TPT_BULK_WANNIER90_CHK_SHA256,
    ) as checkpoint:
        chk = checkpoint.metadata
        if (chk.num_kpts, chk.num_bands, chk.num_wann) != (192, 600, 320):
            raise ValueError("checkpoint dimensions are not 192 x 600 x 320")
        if chk.mp_grid != TPT_BULK_SOURCE_MESH:
            raise ValueError("checkpoint mesh is not the accepted 12x4x4 mesh")
        if not np.allclose(
            chk.kpoints_fractional,
            eigensystem.k_fractional,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("checkpoint and active eigensystem k ordering disagree")
        dft_to_active = np.empty(
            (chk.num_kpts, chk.num_bands, active_rank),
            dtype=np.complex128,
        )
        for ik in range(chk.num_kpts):
            dft_to_active[ik] = (
                checkpoint.dft_to_wannier_gauge(ik)
                @ eigensystem.active_eigenvectors[:, :, ik]
            )

    occupations = np.zeros((eigensystem.nk, active_rank), dtype=np.float64)
    occupations[:, :serialized_occupied_rank] = 1.0
    output_root.mkdir(parents=True)
    wavecar_path = output_root / "WAVECAR"
    try:
        writer = write_projected_spinor_wavecar(
            parent_root / "WAVECAR",
            wavecar_path,
            dft_to_active_gauge_by_k=dft_to_active,
            active_eigenvalues_ev=eigensystem.active_energies_ev.T,
            active_occupations=occupations,
            expected_source_sha256=TPT_BULK_WAVECAR_SHA256,
            projection_workers=8,
        )
        wavecar_hash = _sha256(wavecar_path)
        seal = {
            "schema": "tpt-bulk-active-spinor-wavecar-v1",
            "status": "prepared_not_yet_paw_oracle_qualified",
            "scope": {
                "material": "Ta8Te20Pd12",
                "cell": "periodic_40_atom_bulk",
                "mesh": list(TPT_BULK_SOURCE_MESH),
                "num_kpoints": eigensystem.nk,
                "source_num_bands": chk.num_bands,
                "active_ranks_1based": [first, last],
                "active_num_bands": active_rank,
            },
            "derivation": {
                "active_state": "|phi_a(k)> = sum_n |psi_n(k)> A_na(k)",
                "active_gauge": "A(k) = U_dis(k) U(k) E_active(k)",
                "paw_overlap_identity": (
                    "A_source^dagger M_PAW(source,target,Q) A_target; VASP "
                    "recomputes the linear PAW projector/augmentation terms from "
                    "the transformed pseudo coefficients"
                ),
                "source_coefficient_precision": str(np.dtype("<c8")),
                "active_coefficient_precision": writer["coefficient_dtype"],
                "precision_rationale": (
                    "Store transformed active coefficients as complex128 so the "
                    "basis rotation does not add a second binary32 quantization and "
                    "can be tested against the 1e-10 microscopic closure contract."
                ),
            },
            "serialized_restart_metadata": {
                "occupied_active_rank": serialized_occupied_rank,
                "physical_filling_authority": False,
                "purpose": "VASP_restart_parseability_and_no_electronic_update_only",
            },
            "inputs": {
                "source_root": str(source_root),
                "hr_sha256": source.hr_sha256,
                "parent_root": str(parent_root),
                "source_wavecar_sha256": TPT_BULK_WAVECAR_SHA256,
                "checkpoint_sha256": TPT_BULK_WANNIER90_CHK_SHA256,
            },
            "output": {
                **writer,
                "sha256": wavecar_hash,
                "mode_octal": oct(wavecar_path.stat().st_mode & 0o777),
            },
            "numerics": {
                "max_hr_antihermitian_before_symmetrization_ev": (
                    eigensystem.max_antihermitian_before_symmetrization_ev
                ),
                "max_active_eigenpair_residual_ev": (
                    eigensystem.max_eigenpair_residual_ev
                ),
            },
            **_selected_provenance_preparation_fields(runtime_hashes),
            "authority": {
                "active_wavecar_basis_transform": True,
                "paw_overlap_exporter": False,
                "arbitrary_finite_q_plus_g": False,
                "screening": False,
                "filling_reference": False,
                "production_density_vertex": False,
                "production_hf": False,
            },
            "next_required_gate": (
                "Replay the native-neighbour and selected pure-G VASP PAW MMN "
                "canaries from this active WAVECAR and compare against direct "
                "A_source^dagger M_600 A_target projections."
            ),
        }
        seal_path = output_root / "ACTIVE_WAVECAR_SEAL.json"
        _write_json_new(seal_path, seal)
        seal_path.chmod(0o444)
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    return seal


def prepare_oracle_capsule(
    *,
    parent_root: Path,
    output_root: Path,
    case: OracleCase,
    wavecar_copy_mode: str = "reflink",
) -> dict[str, Any]:
    parent_root = parent_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace oracle capsule: {output_root}")
    if not TPT_BULK_VASP_NCL_EXECUTABLE.is_file():
        raise FileNotFoundError(f"missing VASP NCL executable: {TPT_BULK_VASP_NCL_EXECUTABLE}")

    repository_root = Path(__file__).resolve().parents[3]
    runtime_hashes = {
        relative: _sha256(repository_root / relative)
        for relative in _RUNTIME_SOURCE_RELATIVE_PATHS
    }
    parent_inventory = _validate_parent(parent_root)
    rendered_incar = render_oracle_incar((parent_root / "INCAR").read_text(), case)

    staging = output_root.with_name(
        f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    if staging.exists():
        raise FileExistsError(f"unexpected staging collision: {staging}")
    staging.mkdir(parents=True)
    try:
        for name in ("KPOINTS", "POSCAR", "POTCAR"):
            shutil.copy2(parent_root / name, staging / name)
        _copy_wavecar(
            parent_root / "WAVECAR",
            staging / "WAVECAR",
            mode=wavecar_copy_mode,
        )
        _write_text_new(staging / "INCAR", rendered_incar)

        capsule_inventory: dict[str, dict[str, Any]] = {}
        for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "WAVECAR"):
            path = staging / name
            capsule_inventory[name] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        for name in ("KPOINTS", "POSCAR", "POTCAR", "WAVECAR"):
            if capsule_inventory[name]["sha256"] != parent_inventory[name]["sha256"]:
                raise RuntimeError(f"copied oracle input differs from accepted parent: {name}")

        seal = {
            "schema": "tpt-bulk-vasp-paw-oracle-capsule-v1",
            "status": "prepared_not_submitted",
            "case": {
                "name": case.name,
                "authority": case.authority,
                "kmesh_tolerance": case.kmesh_tolerance,
                "search_shells": case.search_shells,
                "shell_list": list(case.shell_list),
            },
            "scope": {
                "material": "Ta8Te20Pd12",
                "cell": "periodic_40_atom_bulk",
                "mesh": [12, 4, 4],
                "num_kpoints": 192,
                "num_bands": 600,
                "encut_ev": 300.0,
            },
            "safety": {
                "algo": "Nothing",
                "istart": 1,
                "icharg": 0,
                "lwave": False,
                "lcharge": False,
                "source_wavecar_inode_isolated": True,
                "wavecar_copy_mode": wavecar_copy_mode,
                "kpar": TPT_BULK_VASP_KPAR,
                "ncore": TPT_BULK_VASP_NCORE,
                "layout_rationale": (
                    "Use one 32-rank MPI world spread as 16 ranks on each of two "
                    "exclusive nodes, with four exact 48-k-point groups. PEAD requires "
                    "NCORE=1 and NPAR=8 divides NBANDS=600 exactly. Sixteen ranks per node "
                    "reduce MMN-stage rank replication relative to the failed 24- and "
                    "32-rank-per-node layouts while preserving the exact source band count."
                ),
                "source_parent_must_remain_unchanged": True,
                "production_density_vertex_authority": False,
                "production_hf_authority": False,
            },
            "parent_inventory": parent_inventory,
            "capsule_inventory": capsule_inventory,
            "runtime": {
                "vasp_modules": list(TPT_BULK_VASP_MODULES),
                "vasp_ncl_executable": str(TPT_BULK_VASP_NCL_EXECUTABLE),
                "vasp_ncl_executable_sha256": _sha256(TPT_BULK_VASP_NCL_EXECUTABLE),
                **_selected_provenance_preparation_fields(runtime_hashes),
            },
            "execution": {
                "required_host_class": "Slurm_compute_node",
                "required_account": "hmt03",
                "compatible_partitions": [
                    "regular256",
                    "regular6430",
                    "regular",
                    "long",
                    "long_old",
                ],
                "required_shape": "two_exclusive_at_least_56_cpu_256GB_nodes_32_ranks_16_per_node",
                "required_launcher": "Intel_MPI_mpirun_not_srun",
                "expected_npar_per_kpar_group": TPT_BULK_VASP_NPAR,
                "expected_runtime_nbands": 600,
                "explicit_time_limit": None,
                "working_directory": str(output_root),
                "command": (
                    "source /etc/profile.d/modules.sh; module purge; "
                    f"module load {' '.join(TPT_BULK_VASP_MODULES)}; "
                    "export I_MPI_PIN_DOMAIN=core MV2_ENABLE_AFFINITY=0; "
                    f"cd {output_root}; "
                    f"mpirun -np ${{SLURM_NTASKS}} {TPT_BULK_VASP_NCL_EXECUTABLE}"
                ),
            },
            "verdict": {
                "submission_authorized": False,
                "reason": (
                    "Preparation alone is non-authorizing. Recheck live scheduler state, "
                    "verify the seal, and create a separate submission receipt before sbatch."
                ),
            },
        }
        write_json_artifact(seal, staging / "SOURCE_SEAL.json")
        for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "WAVECAR", "SOURCE_SEAL.json"):
            (staging / name).chmod(0o444)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            raise FileExistsError(f"refusing to replace oracle capsule: {output_root}")
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return seal


def prepare_active_oracle_capsule(
    *,
    parent_root: Path,
    active_wavecar_root: Path,
    output_root: Path,
    case: OracleCase,
    wavecar_copy_mode: str = "full",
    active_layout: str = "regular64",
) -> dict[str, Any]:
    """Prepare a 16-band VASP PAW-overlap replay from the transformed WAVECAR."""

    parent_root = parent_root.expanduser().resolve()
    active_wavecar_root = active_wavecar_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace active oracle capsule: {output_root}")
    if active_layout not in _ACTIVE_ORACLE_LAYOUTS:
        raise ValueError(f"unsupported active oracle layout: {active_layout}")
    layout = _ACTIVE_ORACLE_LAYOUTS[active_layout]
    if 192 % layout["kpar"] or layout["mpi_ranks"] % layout["kpar"]:
        raise ValueError("active oracle layout does not divide the k/rank inventories")
    if layout["npar"] != layout["mpi_ranks"] // layout["kpar"]:
        raise ValueError("active oracle NPAR is inconsistent with MPI ranks/KPAR")
    if 16 % layout["npar"]:
        raise ValueError("active oracle NPAR does not divide NBANDS=16")
    active_seal_path = active_wavecar_root / "ACTIVE_WAVECAR_SEAL.json"
    active_wavecar_path = active_wavecar_root / "WAVECAR"
    active_seal = _read_json_object(
        active_seal_path,
        domain="active WAVECAR seal",
    )
    if active_seal.get("schema") != "tpt-bulk-active-spinor-wavecar-v1":
        raise ValueError("active WAVECAR seal schema mismatch")
    if active_seal.get("status") != "prepared_not_yet_paw_oracle_qualified":
        raise ValueError("active WAVECAR seal status mismatch")
    if active_seal.get("scope", {}).get("active_ranks_1based") != [233, 248]:
        raise ValueError("active WAVECAR ranks are not 233:248")
    if active_seal.get("scope", {}).get("mesh") != [12, 4, 4]:
        raise ValueError("active WAVECAR mesh mismatch")
    _, _, active_source_drift = _selected_provenance_inventory(
        active_seal,
        domain="active WAVECAR seal",
    )
    if active_source_drift:
        raise ValueError(
            "active WAVECAR selected provenance content changed; a new active "
            "WAVECAR seal is required before oracle handoff"
        )
    expected_active_hash = active_seal.get("output", {}).get("sha256")
    if _sha256(active_wavecar_path) != expected_active_hash:
        raise ValueError("active WAVECAR hash mismatch")
    if active_wavecar_path.stat().st_mode & 0o222:
        raise PermissionError("active WAVECAR must be read-only")

    parent_inventory: dict[str, dict[str, Any]] = {}
    for name, expected in _PARENT_HASHES.items():
        if name == "WAVECAR":
            continue
        path = parent_root / name
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"accepted-parent {name} hash mismatch")
        parent_inventory[name] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    rendered_incar = render_active_oracle_incar(
        (parent_root / "INCAR").read_text(),
        case,
        kpar=layout["kpar"],
    )
    repository_root = Path(__file__).resolve().parents[3]
    runtime_hashes = {
        relative: _sha256(repository_root / relative)
        for relative in _RUNTIME_SOURCE_RELATIVE_PATHS
    }

    staging = output_root.with_name(
        f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    try:
        for name in ("KPOINTS", "POSCAR", "POTCAR"):
            shutil.copy2(parent_root / name, staging / name)
        _copy_wavecar(
            active_wavecar_path,
            staging / "WAVECAR",
            mode=wavecar_copy_mode,
        )
        _write_text_new(staging / "INCAR", rendered_incar)
        capsule_inventory = {
            name: {
                "sha256": _sha256(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "WAVECAR")
        }
        for name in ("KPOINTS", "POSCAR", "POTCAR"):
            if capsule_inventory[name]["sha256"] != parent_inventory[name]["sha256"]:
                raise RuntimeError(f"active oracle copy changed {name}")
        if capsule_inventory["WAVECAR"]["sha256"] != expected_active_hash:
            raise RuntimeError("active oracle WAVECAR copy changed")
        seal = {
            "schema": "tpt-bulk-active16-vasp-paw-oracle-capsule-v1",
            "status": "prepared_not_submitted",
            "case": {
                "name": case.name,
                "authority": case.authority,
                "kmesh_tolerance": case.kmesh_tolerance,
                "search_shells": case.search_shells,
                "shell_list": list(case.shell_list),
            },
            "scope": {
                "material": "Ta8Te20Pd12",
                "cell": "periodic_40_atom_bulk",
                "mesh": [12, 4, 4],
                "num_kpoints": 192,
                "num_bands": 16,
                "active_ranks_1based": [233, 248],
                "encut_ev": 300.0,
            },
            "active_wavecar": {
                "seal_path": str(active_seal_path),
                "seal_sha256": _sha256(active_seal_path),
                "wavecar_sha256": expected_active_hash,
                "basis_transform_authority": True,
                "paw_oracle_qualification_pending": True,
            },
            "parent_inventory": parent_inventory,
            "capsule_inventory": capsule_inventory,
            "runtime": {
                "vasp_modules": list(TPT_BULK_VASP_MODULES),
                "vasp_ncl_executable": str(TPT_BULK_VASP_NCL_EXECUTABLE),
                "vasp_ncl_executable_sha256": _sha256(TPT_BULK_VASP_NCL_EXECUTABLE),
                **_selected_provenance_preparation_fields(runtime_hashes),
            },
            "execution": {
                "required_host_class": "Slurm_compute_node",
                "required_account": "hmt03",
                "layout_profile": active_layout,
                "required_partition": layout["partition"],
                "required_shape": (
                    f"one_exclusive_node_{layout['allocated_cpus']}_allocated_cpu_"
                    f"{layout['mpi_ranks']}_mpi_rank_world"
                ),
                "physical_node_cpus": layout.get(
                    "physical_node_cpus", layout["allocated_cpus"]
                ),
                "required_launcher": "Intel_MPI_mpirun_not_srun",
                "submission_layout": {
                    "nodes": 1,
                    "allocated_cpus": layout["allocated_cpus"],
                    "mpi_ranks": layout["mpi_ranks"],
                    "ranks_per_node": layout["ranks_per_node"],
                    "kpar": layout["kpar"],
                    "ncore": layout["ncore"],
                    "npar": layout["npar"],
                    "nbands": 16,
                },
                "kpar": layout["kpar"],
                "ncore": layout["ncore"],
                "expected_npar_per_kpar_group": layout["npar"],
                "expected_runtime_nbands": 16,
                "explicit_time_limit": None,
                "working_directory": str(output_root),
                "command": (
                    "source /etc/profile.d/modules.sh; module purge; "
                    f"module load {' '.join(TPT_BULK_VASP_MODULES)}; "
                    "export I_MPI_PIN_DOMAIN=core MV2_ENABLE_AFFINITY=0; "
                    f"cd {output_root}; "
                    f"mpirun -np {layout['mpi_ranks']} {TPT_BULK_VASP_NCL_EXECUTABLE}"
                ),
            },
            "authority": {
                "submission": False,
                "selected_paw_overlap_canary": False,
                "arbitrary_finite_q_plus_g": False,
                "production_density_vertex": False,
                "production_hf": False,
            },
            "next_required_gate": (
                "Run this capsule and compare its 16x16 PAW MMN blocks against "
                "direct active projection of the immutable 600-band oracle."
            ),
        }
        seal_path = staging / "SOURCE_SEAL.json"
        _write_json_new(seal_path, seal)
        for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "WAVECAR", "SOURCE_SEAL.json"):
            (staging / name).chmod(0o444)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return seal


def compare_wannier90_mmn_files(
    *,
    reference_path: Path,
    candidate_path: Path,
    num_bands: int,
    num_kpoints: int,
    nntot: int,
    reference_sha256: str | None = None,
) -> dict[str, Any]:
    """Stream and compare complete MMN inventories without materializing either file."""

    reference_path = reference_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    candidate_initial = candidate_path.stat()
    if candidate_initial.st_mode & 0o222:
        raise PermissionError("candidate wannier90.mmn must be read-only before validation")

    reference_blocks = iter_wannier90_mmn_blocks(
        reference_path,
        expected_num_bands=num_bands,
        expected_num_kpts=num_kpoints,
        expected_nntot=nntot,
        expected_sha256=reference_sha256,
    )
    candidate_blocks = iter_wannier90_mmn_blocks(
        candidate_path,
        expected_num_bands=num_bands,
        expected_num_kpts=num_kpoints,
        expected_nntot=nntot,
    )
    sentinel = object()
    block_count = 0
    max_abs_difference = 0.0
    max_location: dict[str, Any] | None = None
    shift_counts: dict[str, int] = {}
    num_matrix_elements = 0
    sum_squared_difference = 0.0
    sum_squared_reference = 0.0
    exceedance_counts = {threshold: 0 for threshold in (1.0e-10, 1.0e-9, 1.0e-8)}
    for reference, candidate in zip_longest(
        reference_blocks,
        candidate_blocks,
        fillvalue=sentinel,
    ):
        if reference is sentinel or candidate is sentinel:
            raise ValueError("MMN block inventories have different lengths")
        reference_header = (
            reference.source_k_0based,
            reference.target_k_0based,
            reference.target_cell_shift_integer,
        )
        candidate_header = (
            candidate.source_k_0based,
            candidate.target_k_0based,
            candidate.target_cell_shift_integer,
        )
        if candidate_header != reference_header:
            raise ValueError(
                f"MMN block header mismatch at block {block_count}: "
                f"expected {reference_header}, got {candidate_header}"
            )
        difference = np.abs(
            candidate.overlap_source_bra_target_ket
            - reference.overlap_source_bra_target_ket
        )
        local_flat = int(np.argmax(difference))
        local_max = float(difference.reshape(-1)[local_flat])
        num_matrix_elements += difference.size
        sum_squared_difference += float(np.sum(np.square(difference), dtype=np.float64))
        sum_squared_reference += float(
            np.sum(np.square(np.abs(reference.overlap_source_bra_target_ket)), dtype=np.float64)
        )
        for threshold in exceedance_counts:
            exceedance_counts[threshold] += int(np.count_nonzero(difference > threshold))
        if local_max > max_abs_difference:
            band_bra, band_ket = np.unravel_index(local_flat, difference.shape)
            max_abs_difference = local_max
            max_location = {
                "block_index": block_count,
                "source_k_0based": reference.source_k_0based,
                "target_k_0based": reference.target_k_0based,
                "target_cell_shift_integer": list(reference.target_cell_shift_integer),
                "source_bra_band_0based": int(band_bra),
                "target_ket_band_0based": int(band_ket),
            }
        shift_key = ",".join(str(value) for value in reference.target_cell_shift_integer)
        shift_counts[shift_key] = shift_counts.get(shift_key, 0) + 1
        block_count += 1

    expected_blocks = num_kpoints * nntot
    if block_count != expected_blocks:
        raise ValueError(
            f"MMN block count mismatch: expected {expected_blocks}, got {block_count}"
        )
    candidate_final = candidate_path.stat()
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(candidate_initial, field) != getattr(candidate_final, field)
        for field in stat_fields
    ):
        raise RuntimeError("candidate wannier90.mmn changed during validation")
    candidate_sha256 = _sha256(candidate_path)
    candidate_after_hash = candidate_path.stat()
    if any(
        getattr(candidate_initial, field) != getattr(candidate_after_hash, field)
        for field in stat_fields
    ):
        raise RuntimeError("candidate wannier90.mmn changed during hashing")
    return {
        "num_bands": num_bands,
        "num_kpoints": num_kpoints,
        "nntot": nntot,
        "block_count": block_count,
        "max_abs_difference": max_abs_difference,
        "max_difference_location": max_location,
        "num_matrix_elements": num_matrix_elements,
        "rms_abs_difference": float(
            np.sqrt(sum_squared_difference / num_matrix_elements)
        ),
        "relative_frobenius_difference": float(
            np.sqrt(sum_squared_difference / sum_squared_reference)
        ),
        "exceedance_counts": {
            f"greater_than_{threshold:.0e}": count
            for threshold, count in exceedance_counts.items()
        },
        "candidate_sha256": candidate_sha256,
        "candidate_size_bytes": candidate_initial.st_size,
        "shift_counts": shift_counts,
        "header_inventory_exact_match": True,
    }


def compare_projected_active_mmn_files(
    *,
    source_root: Path,
    parent_root: Path,
    reference_path: Path,
    candidate_path: Path,
    nntot: int = 6,
    reference_sha256: str | None = None,
    candidate_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare an active MMN to A_source^dagger M_full A_target block by block."""

    ensure_not_running_compute_on_login_node("project and compare TPT bulk PAW MMN")
    source = load_tpt_bulk_source(source_root)
    eigensystem = build_tpt_bulk_active_eigensystem(
        source,
        mesh=TPT_BULK_SOURCE_MESH,
        active_ranks_1based=(233, 248),
    )
    with Wannier90CheckpointReader(
        parent_root / "wannier90.chk",
        expected_sha256=TPT_BULK_WANNIER90_CHK_SHA256,
    ) as checkpoint:
        chk = checkpoint.metadata
        if (chk.num_kpts, chk.num_bands, chk.num_wann) != (192, 600, 320):
            raise ValueError("checkpoint dimensions are not 192 x 600 x 320")
        if not np.allclose(
            chk.kpoints_fractional,
            eigensystem.k_fractional,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("checkpoint and active eigensystem k ordering disagree")
        active_gauge = np.empty((192, 600, 16), dtype=np.complex128)
        for ik in range(192):
            active_gauge[ik] = (
                checkpoint.dft_to_wannier_gauge(ik)
                @ eigensystem.active_eigenvectors[:, :, ik]
            )

    reference_path = reference_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    initial_stats: dict[str, os.stat_result] = {}
    for label, path in (("reference", reference_path), ("candidate", candidate_path)):
        initial_stats[label] = path.stat()
        if initial_stats[label].st_mode & 0o222:
            raise PermissionError(f"{label} MMN must be read-only before validation")
    reference_blocks = iter_wannier90_mmn_blocks(
        reference_path,
        expected_num_bands=600,
        expected_num_kpts=192,
        expected_nntot=nntot,
        expected_sha256=reference_sha256,
    )
    candidate_blocks = iter_wannier90_mmn_blocks(
        candidate_path,
        expected_num_bands=16,
        expected_num_kpts=192,
        expected_nntot=nntot,
        expected_sha256=candidate_sha256,
    )

    block_count = 0
    element_count = 0
    max_abs_difference = 0.0
    max_location: dict[str, Any] | None = None
    sum_squared_difference = 0.0
    sum_squared_reference = 0.0
    exceedance_counts = {1.0e-10: 0, 1.0e-9: 0, 1.0e-8: 0}
    shift_counts: dict[str, int] = {}
    for reference, candidate in zip_longest(reference_blocks, candidate_blocks):
        if reference is None or candidate is None:
            raise ValueError("full and active MMN inventories have unequal lengths")
        reference_header = (
            reference.source_k_0based,
            reference.target_k_0based,
            reference.target_cell_shift_integer,
        )
        candidate_header = (
            candidate.source_k_0based,
            candidate.target_k_0based,
            candidate.target_cell_shift_integer,
        )
        if reference_header != candidate_header:
            raise ValueError(
                f"projected MMN header mismatch at block {block_count}: "
                f"{reference_header} != {candidate_header}"
            )
        source_gauge = active_gauge[reference.source_k_0based]
        target_gauge = active_gauge[reference.target_k_0based]
        projected = source_gauge.conj().T @ (
            reference.overlap_source_bra_target_ket @ target_gauge
        )
        difference = np.abs(
            candidate.overlap_source_bra_target_ket - projected
        )
        local_flat = int(np.argmax(difference))
        local_max = float(difference.reshape(-1)[local_flat])
        if local_max > max_abs_difference:
            active_bra, active_ket = np.unravel_index(local_flat, difference.shape)
            max_abs_difference = local_max
            max_location = {
                "block_index": block_count,
                "source_k_0based": reference.source_k_0based,
                "target_k_0based": reference.target_k_0based,
                "target_cell_shift_integer": list(
                    reference.target_cell_shift_integer
                ),
                "active_source_bra_0based": int(active_bra),
                "active_target_ket_0based": int(active_ket),
            }
        element_count += difference.size
        sum_squared_difference += float(
            np.sum(np.square(difference), dtype=np.float64)
        )
        sum_squared_reference += float(
            np.sum(np.square(np.abs(projected)), dtype=np.float64)
        )
        for threshold in exceedance_counts:
            exceedance_counts[threshold] += int(np.count_nonzero(difference > threshold))
        shift_key = ",".join(str(value) for value in reference.target_cell_shift_integer)
        shift_counts[shift_key] = shift_counts.get(shift_key, 0) + 1
        block_count += 1

    expected_blocks = 192 * nntot
    if block_count != expected_blocks:
        raise ValueError(
            f"projected MMN block count mismatch: {block_count}/{expected_blocks}"
        )
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    for label, path in (("reference", reference_path), ("candidate", candidate_path)):
        after_stream = path.stat()
        if any(
            getattr(initial_stats[label], field) != getattr(after_stream, field)
            for field in stat_fields
        ):
            raise RuntimeError(f"{label} MMN changed during projected validation")
    resolved_reference_hash = reference_sha256 or _sha256(reference_path)
    resolved_candidate_hash = candidate_sha256 or _sha256(candidate_path)
    for label, path in (("reference", reference_path), ("candidate", candidate_path)):
        after_hash = path.stat()
        if any(
            getattr(initial_stats[label], field) != getattr(after_hash, field)
            for field in stat_fields
        ):
            raise RuntimeError(f"{label} MMN changed during projected hashing")
    return {
        "reference_num_bands": 600,
        "candidate_num_bands": 16,
        "num_kpoints": 192,
        "nntot": nntot,
        "block_count": block_count,
        "num_matrix_elements": element_count,
        "max_abs_difference": max_abs_difference,
        "max_difference_location": max_location,
        "rms_abs_difference": float(
            np.sqrt(sum_squared_difference / element_count)
        ),
        "relative_frobenius_difference": float(
            np.sqrt(sum_squared_difference / sum_squared_reference)
        ),
        "exceedance_counts": {
            f"greater_than_{threshold:.0e}": count
            for threshold, count in exceedance_counts.items()
        },
        "reference_sha256": resolved_reference_hash,
        "candidate_sha256": resolved_candidate_hash,
        "shift_counts": shift_counts,
        "header_inventory_exact_match": True,
        "tolerance": TPT_BULK_ACTIVE_MMN_PROJECTION_TOLERANCE,
        "tolerance_rationale": (
            "One binary32 epsilon, fixed by the accepted complex64 parent WAVECAR "
            "precision; complex128 active storage avoids a second such quantization. "
            "This does not relax the separate 1e-10 microscopic-HF closure gate."
        ),
        "reference_text_rounding_bound": TPT_BULK_ACTIVE_MMN_TEXT_ROUNDING_BOUND,
        "within_reference_text_rounding_bound": (
            max_abs_difference <= TPT_BULK_ACTIVE_MMN_TEXT_ROUNDING_BOUND
        ),
        "passed": (
            max_abs_difference <= TPT_BULK_ACTIVE_MMN_PROJECTION_TOLERANCE
        ),
    }


def validate_pure_g_mmn_file(
    *,
    candidate_path: Path,
    num_bands: int,
    num_kpoints: int,
    expected_shifts: tuple[tuple[int, int, int], ...],
) -> dict[str, Any]:
    """Stream a pure-G MMN and validate inventory plus rho(-G)=rho(G)^dagger."""

    candidate_path = candidate_path.expanduser().resolve()
    candidate_initial = candidate_path.stat()
    if candidate_initial.st_mode & 0o222:
        raise PermissionError("candidate wannier90.mmn must be read-only before validation")
    expected_shift_set = set(expected_shifts)
    if len(expected_shift_set) != len(expected_shifts):
        raise ValueError("expected pure-G shifts must be unique")
    for shift in expected_shifts:
        if tuple(-value for value in shift) not in expected_shift_set:
            raise ValueError(f"missing reverse shift in expected inventory: {shift}")

    block_count = 0
    shift_counts = {",".join(str(value) for value in shift): 0 for shift in expected_shifts}
    current_source: int | None = None
    current_blocks: dict[tuple[int, int, int], np.ndarray] = {}
    max_reverse_closure = 0.0
    max_reverse_location: dict[str, Any] | None = None
    sum_squared_reverse_closure = 0.0
    num_reverse_elements = 0

    def close_source(source_k: int, blocks: dict[tuple[int, int, int], np.ndarray]) -> None:
        nonlocal max_reverse_closure
        nonlocal max_reverse_location
        nonlocal sum_squared_reverse_closure
        nonlocal num_reverse_elements
        if set(blocks) != expected_shift_set:
            raise ValueError(
                f"pure-G source k={source_k} shift inventory mismatch: {sorted(blocks)}"
            )
        for shift in expected_shifts:
            reverse = tuple(-value for value in shift)
            if shift > reverse:
                continue
            difference = np.abs(blocks[shift] - blocks[reverse].conj().T)
            local_flat = int(np.argmax(difference))
            local_max = float(difference.reshape(-1)[local_flat])
            sum_squared_reverse_closure += float(
                np.sum(np.square(difference), dtype=np.float64)
            )
            num_reverse_elements += difference.size
            if local_max > max_reverse_closure:
                band_bra, band_ket = np.unravel_index(local_flat, difference.shape)
                max_reverse_closure = local_max
                max_reverse_location = {
                    "source_k_0based": source_k,
                    "shift": list(shift),
                    "reverse_shift": list(reverse),
                    "source_bra_band_0based": int(band_bra),
                    "target_ket_band_0based": int(band_ket),
                }

    for block in iter_wannier90_mmn_blocks(
        candidate_path,
        expected_num_bands=num_bands,
        expected_num_kpts=num_kpoints,
        expected_nntot=len(expected_shifts),
    ):
        if block.target_k_0based != block.source_k_0based:
            raise ValueError(
                "pure-G MMN contains a non-same-k row: "
                f"{block.source_k_0based} -> {block.target_k_0based}"
            )
        if current_source is None:
            current_source = block.source_k_0based
        elif block.source_k_0based != current_source:
            close_source(current_source, current_blocks)
            if block.source_k_0based != current_source + 1:
                raise ValueError("pure-G MMN source-k blocks are not contiguous and ordered")
            current_source = block.source_k_0based
            current_blocks = {}
        shift = block.target_cell_shift_integer
        if shift not in expected_shift_set:
            raise ValueError(f"unexpected pure-G integer shift: {shift}")
        if shift in current_blocks:
            raise ValueError(f"duplicate pure-G shift at source k={current_source}: {shift}")
        current_blocks[shift] = block.overlap_source_bra_target_ket
        shift_counts[",".join(str(value) for value in shift)] += 1
        block_count += 1
    if current_source is None:
        raise ValueError("pure-G MMN has no blocks")
    close_source(current_source, current_blocks)

    expected_blocks = num_kpoints * len(expected_shifts)
    if block_count != expected_blocks or current_source != num_kpoints - 1:
        raise ValueError(
            f"pure-G MMN inventory is incomplete: blocks={block_count}, "
            f"last_source={current_source}, expected_blocks={expected_blocks}"
        )
    candidate_final = candidate_path.stat()
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(candidate_initial, field) != getattr(candidate_final, field)
        for field in stat_fields
    ):
        raise RuntimeError("pure-G candidate wannier90.mmn changed during validation")
    candidate_sha256 = _sha256(candidate_path)
    candidate_after_hash = candidate_path.stat()
    if any(
        getattr(candidate_initial, field) != getattr(candidate_after_hash, field)
        for field in stat_fields
    ):
        raise RuntimeError("pure-G candidate wannier90.mmn changed during hashing")
    return {
        "num_bands": num_bands,
        "num_kpoints": num_kpoints,
        "nntot": len(expected_shifts),
        "block_count": block_count,
        "same_k_inventory": True,
        "shift_counts": shift_counts,
        "reverse_closure_max_abs_difference": max_reverse_closure,
        "reverse_closure_max_location": max_reverse_location,
        "reverse_closure_rms_abs_difference": float(
            np.sqrt(sum_squared_reverse_closure / num_reverse_elements)
        ),
        "candidate_sha256": candidate_sha256,
        "candidate_size_bytes": candidate_initial.st_size,
    }


def _read_win_kpoint_numerators(
    path: Path,
    *,
    mesh: tuple[int, int, int] = TPT_BULK_SOURCE_MESH,
) -> tuple[tuple[int, int, int], ...]:
    lines = path.read_text().splitlines()
    begin = next(
        index for index, line in enumerate(lines) if line.strip().lower() == "begin kpoints"
    )
    end = next(
        index
        for index in range(begin + 1, len(lines))
        if lines[index].strip().lower() == "end kpoints"
    )
    points: list[tuple[int, int, int]] = []
    for line in lines[begin + 1 : end]:
        values = tuple(float(value) for value in line.split())
        if len(values) != 3:
            raise ValueError(f"invalid Wannier90 k-point row: {line!r}")
        numerator = tuple(
            int(round(value * size))
            for value, size in zip(values, mesh, strict=True)
        )
        if any(
            abs(value - item / size) > 1.0e-10
            for value, item, size in zip(values, numerator, mesh, strict=True)
        ):
            raise ValueError(f"off-mesh Wannier90 k point: {values}")
        points.append(numerator)
    if len(points) != int(np.prod(mesh)) or len(set(points)) != len(points):
        raise ValueError("Wannier90 k-point inventory is not the exact source mesh")

    def fft_axis(size: int) -> tuple[int, ...]:
        if size == 1:
            return (0,)
        if size <= 0 or size % 2:
            raise ValueError("Wannier90 source mesh must have even nontrivial axes")
        return tuple(range(0, size // 2 + 1)) + tuple(
            range(-(size // 2) + 1, 0)
        )

    expected_order = tuple(
        (x, y, z)
        for z in fft_axis(mesh[2])
        for y in fft_axis(mesh[1])
        for x in fft_axis(mesh[0])
    )
    if tuple(points) != expected_order:
        raise ValueError(
            "Wannier90 k points are not in the accepted Gamma-grid FFT ordering"
        )
    return tuple(points)


def _mixed_qg_expected_transfers() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        sorted(
            {
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
                *_sign_orbit((18, 2, 2)),
            }
        )
    )


def validate_general_qg_mmn_file(
    *,
    candidate_path: Path,
    num_bands: int,
    kpoint_numerators: tuple[tuple[int, int, int], ...],
    expected_transfers: tuple[tuple[int, int, int], ...],
    mesh: tuple[int, int, int] = TPT_BULK_SOURCE_MESH,
    check_matrix_reverse_closure: bool,
) -> dict[str, Any]:
    """Validate finite-q+G MMN headers and optional matrix reverse closure."""

    candidate_path = candidate_path.expanduser().resolve()
    initial = candidate_path.stat()
    if initial.st_mode & 0o222:
        raise PermissionError("candidate wannier90.mmn must be read-only before validation")
    expected = set(expected_transfers)
    if len(expected) != len(expected_transfers):
        raise ValueError("general finite-q+G transfer inventory contains duplicates")
    if expected != {tuple(-value for value in transfer) for transfer in expected}:
        raise ValueError("general finite-q+G transfer inventory is not reverse closed")
    if len(kpoint_numerators) != int(np.prod(mesh)):
        raise ValueError("general finite-q+G k-point inventory has the wrong size")

    header_keys: set[tuple[int, int, tuple[int, int, int]]] = set()
    source_transfers: dict[int, set[tuple[int, int, int]]] = {}
    matrices: dict[
        tuple[int, int, tuple[int, int, int]], np.ndarray
    ] | None = {} if check_matrix_reverse_closure else None
    transfer_counts = {",".join(map(str, transfer)): 0 for transfer in expected_transfers}
    block_count = 0
    for block in iter_wannier90_mmn_blocks(
        candidate_path,
        expected_num_bands=num_bands,
        expected_num_kpts=len(kpoint_numerators),
        expected_nntot=len(expected_transfers),
    ):
        source_k = block.source_k_0based
        target_k = block.target_k_0based
        shift = block.target_cell_shift_integer
        key = (source_k, target_k, shift)
        if key in header_keys:
            raise ValueError(f"duplicate general finite-q+G MMN header: {key}")
        header_keys.add(key)
        transfer = tuple(
            kpoint_numerators[target_k][axis]
            - kpoint_numerators[source_k][axis]
            + mesh[axis] * shift[axis]
            for axis in range(3)
        )
        if transfer not in expected:
            raise ValueError(f"unexpected general finite-q+G transfer: {transfer}")
        per_source = source_transfers.setdefault(source_k, set())
        if transfer in per_source:
            raise ValueError(
                f"duplicate physical transfer at source k={source_k}: {transfer}"
            )
        per_source.add(transfer)
        transfer_counts[",".join(map(str, transfer))] += 1
        if matrices is not None:
            matrices[key] = block.overlap_source_bra_target_ket
        block_count += 1

    num_kpoints = len(kpoint_numerators)
    if set(source_transfers) != set(range(num_kpoints)):
        raise ValueError("general finite-q+G MMN source-k inventory is incomplete")
    for source_k, transfers in source_transfers.items():
        if transfers != expected:
            raise ValueError(
                f"general finite-q+G source k={source_k} inventory mismatch"
            )
    for source_k, target_k, shift in header_keys:
        reverse = (target_k, source_k, tuple(-value for value in shift))
        if reverse not in header_keys:
            raise ValueError(f"missing reverse MMN header for {(source_k, target_k, shift)}")

    max_reverse = 0.0
    max_reverse_location: dict[str, Any] | None = None
    sum_squared_reverse = 0.0
    reverse_elements = 0
    if matrices is not None:
        for key, matrix in matrices.items():
            reverse_key = (key[1], key[0], tuple(-value for value in key[2]))
            if key > reverse_key:
                continue
            difference = np.abs(matrix - matrices[reverse_key].conj().T)
            local_flat = int(np.argmax(difference))
            local_max = float(difference.reshape(-1)[local_flat])
            sum_squared_reverse += float(np.sum(np.square(difference), dtype=np.float64))
            reverse_elements += difference.size
            if local_max > max_reverse:
                band_bra, band_ket = np.unravel_index(local_flat, difference.shape)
                max_reverse = local_max
                max_reverse_location = {
                    "header": [key[0], key[1], *key[2]],
                    "reverse_header": [reverse_key[0], reverse_key[1], *reverse_key[2]],
                    "source_bra_band_0based": int(band_bra),
                    "target_ket_band_0based": int(band_ket),
                }

    final = candidate_path.stat()
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(initial, field) != getattr(final, field) for field in stat_fields):
        raise RuntimeError("general finite-q+G MMN changed during validation")
    candidate_sha256 = _sha256(candidate_path)
    after_hash = candidate_path.stat()
    if any(
        getattr(initial, field) != getattr(after_hash, field) for field in stat_fields
    ):
        raise RuntimeError("general finite-q+G MMN changed during hashing")
    return {
        "num_bands": num_bands,
        "num_kpoints": num_kpoints,
        "nntot": len(expected_transfers),
        "block_count": block_count,
        "header_inventory_exact": True,
        "header_reverse_closed": True,
        "transfer_counts": transfer_counts,
        "matrix_reverse_closure_checked": matrices is not None,
        "reverse_closure_max_abs_difference": max_reverse if matrices is not None else None,
        "reverse_closure_max_location": max_reverse_location,
        "reverse_closure_rms_abs_difference": (
            float(np.sqrt(sum_squared_reverse / reverse_elements))
            if reverse_elements
            else None
        ),
        "candidate_sha256": candidate_sha256,
        "candidate_size_bytes": initial.st_size,
    }


def _load_active_submission_receipt(
    *,
    output_root: Path,
    job_id: int,
    source_seal_sha256: str,
    case_name: str,
    expected_layout: dict[str, int],
) -> tuple[dict[str, Any], str]:
    receipt_path = output_root / f"SUBMISSION-{job_id}.json"
    initial = receipt_path.stat()
    if initial.st_mode & 0o222:
        raise PermissionError("active submission receipt must be read-only")
    receipt = _read_json_object(
        receipt_path,
        domain="active submission receipt",
    )
    expected = {
        "schema": "tpt-bulk-active16-paw-oracle-submission-v1",
        "job_id": job_id,
        "source_seal_sha256": source_seal_sha256,
        "case": case_name,
        "account": "hmt03",
        "launcher": "Intel_MPI_mpirun_not_srun",
        "layout": expected_layout,
        "authority": {
            "vasp_execution_authorized": True,
            "scientific_authority": False,
        },
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"active submission receipt {key} mismatch: "
                f"expected {value!r}, got {receipt.get(key)!r}"
            )
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    final = receipt_path.stat()
    if any(getattr(initial, field) != getattr(final, field) for field in stat_fields):
        raise RuntimeError("active submission receipt changed while it was parsed")
    digest = _sha256(receipt_path)
    after_hash = receipt_path.stat()
    if any(
        getattr(initial, field) != getattr(after_hash, field) for field in stat_fields
    ):
        raise RuntimeError("active submission receipt changed while it was hashed")
    return receipt, digest


def validate_active_oracle_outputs(
    *,
    source_root: Path,
    parent_root: Path,
    reference_root: Path,
    output_root: Path,
    job_id: int,
    revalidation_seal_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a completed active-16 PAW MMN against its full-band oracle."""

    ensure_not_running_compute_on_login_node("validate TPT bulk active PAW oracle")
    source_root = source_root.expanduser().resolve()
    parent_root = parent_root.expanduser().resolve()
    reference_root = reference_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    seal_path = output_root / "SOURCE_SEAL.json"
    seal = _read_json_object(seal_path, domain="active oracle source seal")
    if seal.get("schema") != "tpt-bulk-active16-vasp-paw-oracle-capsule-v1":
        raise ValueError("active oracle source seal schema mismatch")
    if seal.get("status") != "prepared_not_submitted":
        raise ValueError("active oracle source seal status mismatch")
    case_name = seal.get("case", {}).get("name")
    if case_name not in ORACLE_CASES:
        raise ValueError("active oracle case mismatch")
    case = ORACLE_CASES[case_name]
    if seal.get("case", {}).get("shell_list") != list(case.shell_list):
        raise ValueError("active oracle shell inventory mismatch")
    layout_profile = seal.get("execution", {}).get("layout_profile")
    if layout_profile not in _ACTIVE_ORACLE_LAYOUTS:
        raise ValueError("active oracle layout profile mismatch")
    expected_layout = {
        "nodes": 1,
        **{
            key: _ACTIVE_ORACLE_LAYOUTS[layout_profile][key]
            for key in (
                "allocated_cpus",
                "mpi_ranks",
                "ranks_per_node",
                "kpar",
                "ncore",
                "npar",
            )
        },
        "nbands": 16,
    }
    if seal.get("execution", {}).get("submission_layout") != expected_layout:
        raise ValueError("active oracle sealed submission layout mismatch")
    source_seal_sha256 = _sha256(seal_path)
    _submission_receipt, submission_receipt_sha256 = (
        _load_active_submission_receipt(
            output_root=output_root,
            job_id=job_id,
            source_seal_sha256=source_seal_sha256,
            case_name=case_name,
            expected_layout=expected_layout,
        )
    )

    (
        preparation_hashes,
        validation_hashes,
        source_drift,
        revalidation_authorization,
    ) = _validate_postflight_provenance(
        owner=seal.get("runtime"),
        domain="active oracle source seal",
        output_root=output_root,
        job_id=job_id,
        source_seal_sha256=source_seal_sha256,
        submission_receipt_sha256=submission_receipt_sha256,
        revalidation_seal_path=revalidation_seal_path,
    )
    for name, expected in seal["capsule_inventory"].items():
        if _sha256(output_root / name) != expected["sha256"]:
            raise ValueError(f"active oracle sealed input changed: {name}")

    required_outputs = (
        "OUTCAR",
        "wannier90.amn",
        "wannier90.eig",
        "wannier90.mmn",
        "wannier90.win",
        "wannier90.wout",
    )
    missing = [name for name in required_outputs if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing active oracle outputs: {missing}")
    for name in required_outputs:
        path = output_root / name
        if path.stat().st_size <= 0:
            raise ValueError(f"empty active oracle output: {path}")
        path.chmod(path.stat().st_mode & ~0o222)

    outcar_text = (output_root / "OUTCAR").read_text(errors="replace")
    required_patterns = (
        rf"running\s+{expected_layout['mpi_ranks']} mpi-ranks,\s+on\s+1 nodes",
        rf"distrk:\s+each k-point on\s+{expected_layout['npar']} cores,"
        rf"\s+{expected_layout['kpar']} groups",
        rf"distr:\s+one band on NCORE=\s*{expected_layout['ncore']} cores,"
        rf"\s+{expected_layout['npar']} groups",
        r"Computing MMN \(overlap matrix elements\)",
    )
    missing_patterns = [
        pattern for pattern in required_patterns if re.search(pattern, outcar_text) is None
    ]
    if missing_patterns:
        raise ValueError(
            f"active oracle OUTCAR is missing runtime evidence: {missing_patterns}"
        )
    forbidden_outcar = (
        "The number of bands has been changed",
        "random initialization beyond band",
        "I REFUSE TO CONTINUE",
    )
    present_forbidden = [text for text in forbidden_outcar if text in outcar_text]
    if present_forbidden:
        raise ValueError(
            f"active oracle OUTCAR contains forbidden evidence: {present_forbidden}"
        )
    accounting = _query_slurm_accounting(job_id)
    if (
        accounting["nodes"] != str(expected_layout["nodes"])
        or accounting["allocated_cpus"] != str(expected_layout["allocated_cpus"])
    ):
        raise ValueError("active oracle accounting does not match the sealed layout")

    source = load_tpt_bulk_source(source_root)
    eigensystem = build_tpt_bulk_active_eigensystem(
        source,
        mesh=TPT_BULK_SOURCE_MESH,
        active_ranks_1based=(233, 248),
    )
    candidate_eig = read_wannier90_eig(
        output_root / "wannier90.eig",
        num_bands=16,
        num_kpts=192,
    )
    eig_max_abs_difference_ev = float(
        np.max(np.abs(candidate_eig - eigensystem.active_energies_ev.T))
    )
    if eig_max_abs_difference_ev > TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV:
        raise ValueError(
            "active oracle eigenvalue replay exceeds tolerance: "
            f"{eig_max_abs_difference_ev}"
        )

    if case_name == "native":
        reference_mmn_hash = (
            "03b4666d34c16af3ec391e1494c47ff0f54b396b64c26f34c864d270bd73a5ff"
        )
        certificate_name = "ARTIFACT-REVALIDATION-507717.json"
        certificate_hash = (
            "eb567ffb3babd407881fe5b6d47b107cefd035690e23732321875c4f8be8a754"
        )
        nntot = 6
    elif case_name == "pure_g":
        reference_mmn_hash = (
            "c9a4ece460ae466595a16f83319b899ced8dfa04c14124bf716b7f600b32fda8"
        )
        certificate_name = "ARTIFACT-REVALIDATION-508057.json"
        certificate_hash = (
            "40b8e73bfff3425d291d53080a6ced9a06bcab6b6008d8302d0c490088174fdc"
        )
        nntot = 6
    else:
        if (
            TPT_BULK_MIXED_QG_REFERENCE_JOB_ID is None
            or not TPT_BULK_MIXED_QG_REFERENCE_CERTIFICATE_NAME
            or not TPT_BULK_MIXED_QG_REFERENCE_CERTIFICATE_SHA256
        ):
            raise ValueError(
                "mixed-q+G full-band reference has no externally pinned certificate"
            )
        certificate_name = TPT_BULK_MIXED_QG_REFERENCE_CERTIFICATE_NAME
        certificate_hash = TPT_BULK_MIXED_QG_REFERENCE_CERTIFICATE_SHA256
        certificate_path = reference_root / certificate_name
        if certificate_path.stat().st_mode & 0o222:
            raise PermissionError("mixed-q+G full-band certificate must be read-only")
        if _sha256(certificate_path) != certificate_hash:
            raise ValueError("mixed-q+G full-band certificate hash is not the pinned hash")
        certificate = _read_json_object(
            certificate_path,
            domain="mixed-q+G full-band reference certificate",
        )
        expected_verdict = {
            "full_band_mixed_qg_paw_oracle_inventory_validated": True,
            "active_projection_replay_validated": False,
            "arbitrary_finite_q_plus_g_exporter_validated": False,
            "production_density_vertex_authority": False,
            "screening_authority": False,
            "filling_reference_authority": False,
            "production_hf_authority": False,
        }
        if (
            certificate.get("schema")
            != "tpt-bulk-mixed-qg-paw-overlap-validation-v1"
            or certificate.get("status")
            != "passed_full_band_mixed_qg_paw_oracle_inventory"
            or certificate.get("job_id") != TPT_BULK_MIXED_QG_REFERENCE_JOB_ID
            or certificate.get("case", {}).get("name") != "mixed_qg"
            or certificate.get("case", {}).get("shell_list") != [1, 2, 6529]
            or certificate.get("verdict") != expected_verdict
        ):
            raise ValueError("mixed-q+G full-band reference certificate is invalid")
        reference_seal = reference_root / "SOURCE_SEAL.json"
        reference_submission = (
            reference_root
            / f"SUBMISSION-{TPT_BULK_MIXED_QG_REFERENCE_JOB_ID}.json"
        )
        reference_validation_job_id = certificate.get("validation_job_id")
        if not isinstance(reference_validation_job_id, int):
            raise ValueError("mixed-q+G certificate lacks a validation job id")
        reference_validation_submission = (
            reference_root
            / f"VALIDATION-SUBMISSION-{reference_validation_job_id}.json"
        )
        if _sha256(reference_seal) != certificate.get("source_seal_sha256"):
            raise ValueError("mixed-q+G certificate is not bound to SOURCE_SEAL.json")
        if _sha256(reference_submission) != certificate.get(
            "submission_receipt_sha256"
        ):
            raise ValueError("mixed-q+G certificate is not bound to submission receipt")
        if _sha256(reference_validation_submission) != certificate.get(
            "validation_submission_receipt_sha256"
        ):
            raise ValueError(
                "mixed-q+G certificate is not bound to validation submission receipt"
            )
        expected_full_outputs = {
            "OUTCAR",
            "wannier90.amn",
            "wannier90.eig",
            "wannier90.mmn",
            "wannier90.win",
            "wannier90.wout",
        }
        if set(certificate.get("output_hashes", {})) != expected_full_outputs:
            raise ValueError("mixed-q+G certificate output inventory is incomplete")
        for name, digest in certificate["output_hashes"].items():
            if _sha256(reference_root / name) != digest:
                raise ValueError(f"mixed-q+G certified output hash mismatch: {name}")
        reference_mmn_hash = certificate["output_hashes"]["wannier90.mmn"]
        nntot = len(_mixed_qg_expected_transfers())
    certificate_path = reference_root / certificate_name
    if _sha256(certificate_path) != certificate_hash:
        raise ValueError("full-band PAW oracle certificate hash mismatch")
    projected_comparison = compare_projected_active_mmn_files(
        source_root=source_root,
        parent_root=parent_root,
        reference_path=reference_root / "wannier90.mmn",
        candidate_path=output_root / "wannier90.mmn",
        nntot=nntot,
        reference_sha256=reference_mmn_hash,
    )
    if not projected_comparison["passed"]:
        raise ValueError(
            "active PAW projection replay exceeds its representation-derived "
            f"tolerance: {projected_comparison['max_abs_difference']} > "
            f"{TPT_BULK_ACTIVE_MMN_PROJECTION_TOLERANCE}"
        )

    pure_g_validation: dict[str, Any] | None = None
    general_qg_validation: dict[str, Any] | None = None
    if case_name == "pure_g":
        pure_g_validation = validate_pure_g_mmn_file(
            candidate_path=output_root / "wannier90.mmn",
            num_bands=16,
            num_kpoints=192,
            expected_shifts=(
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ),
        )
        if pure_g_validation["reverse_closure_max_abs_difference"] > 1.0e-10:
            raise ValueError("active pure-G reverse closure exceeds 1e-10")
    elif case_name == "mixed_qg":
        parent_kpoints = _read_win_kpoint_numerators(parent_root / "wannier90.win")
        output_kpoints = _read_win_kpoint_numerators(output_root / "wannier90.win")
        if output_kpoints != parent_kpoints:
            raise ValueError("active mixed-q+G k-point ordering differs from parent")
        general_qg_validation = validate_general_qg_mmn_file(
            candidate_path=output_root / "wannier90.mmn",
            num_bands=16,
            kpoint_numerators=output_kpoints,
            expected_transfers=_mixed_qg_expected_transfers(),
            check_matrix_reverse_closure=True,
        )
        if general_qg_validation["reverse_closure_max_abs_difference"] > 1.0e-10:
            raise ValueError("active mixed-q+G reverse closure exceeds 1e-10")

    output_hashes = {name: _sha256(output_root / name) for name in required_outputs}
    if (
        revalidation_authorization is not None
        and output_hashes != revalidation_authorization["expected_output_hashes"]
    ):
        raise ValueError("active artifact output hashes do not match authorization")
    report = {
        "schema": "tpt-bulk-active16-paw-overlap-validation-v1",
        "status": f"passed_active16_{case_name}_paw_projection_replay",
        **_selected_provenance_report_fields(
            preparation_hashes,
            validation_hashes,
            source_drift,
        ),
        "job_id": job_id,
        "case": {
            "name": case_name,
            "shell_list": list(case.shell_list),
            "authority": case.authority,
        },
        "scope": seal["scope"],
        "source_seal_sha256": source_seal_sha256,
        "submission_receipt_sha256": submission_receipt_sha256,
        "active_wavecar_seal_sha256": seal["active_wavecar"]["seal_sha256"],
        "full_band_reference": {
            "root": str(reference_root),
            "certificate": certificate_name,
            "certificate_sha256": certificate_hash,
            "mmn_sha256": reference_mmn_hash,
        },
        "accounting": accounting,
        "eigenvalue_replay": {
            "max_abs_difference_ev": eig_max_abs_difference_ev,
            "tolerance_ev": TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV,
            "passed": True,
        },
        "projected_mmn_replay": projected_comparison,
        "pure_g_validation": pure_g_validation,
        "general_qg_validation": general_qg_validation,
        "output_hashes": output_hashes,
        "external_revalidation_authorization_sha256": (
            revalidation_authorization["authorization_sha256"]
            if revalidation_authorization is not None
            else None
        ),
        "verdict": {
            "active_wavecar_basis_transform_validated": True,
            "selected_paw_overlap_canary_validated": True,
            "general_finite_q_plus_g_orientation_canary_validated": (
                case_name == "mixed_qg"
            ),
            "arbitrary_finite_q_plus_g_exporter_validated": False,
            "screening_authority": False,
            "filling_reference_authority": False,
            "production_density_vertex_authority": False,
            "production_hf_authority": False,
        },
    }
    report_name = (
        f"ACTIVE-ARTIFACT-REVALIDATION-{job_id}.json"
        if revalidation_authorization is not None
        else f"ACTIVE-VALIDATION-{job_id}.json"
    )
    report_path = output_root / report_name
    _write_json_new(report_path, report)
    report_path.chmod(0o444)
    return report


def _load_and_validate_submission_receipt(
    *,
    path: Path,
    job_id: int,
    source_seal_sha256: str,
    expected_case: str,
) -> tuple[dict[str, Any], str]:
    """Parse, hash, and cross-bind the immutable Slurm submission receipt."""

    if not path.is_file():
        raise FileNotFoundError(f"missing immutable submission receipt: {path}")
    initial = path.stat()
    if initial.st_mode & 0o222:
        raise PermissionError("submission receipt must be read-only")
    receipt = _read_json_object(path, domain="oracle submission receipt")
    if receipt.get("schema") != "tpt-bulk-vasp-paw-oracle-submission-v6":
        raise ValueError("unexpected oracle submission-receipt schema")
    expected = {
        "job_id": job_id,
        "account": "hmt03",
        "nodes": 2,
        "ntasks": TPT_BULK_VASP_MPI_RANKS,
        "ntasks_per_node": 16,
        "exclusive": True,
        "source_seal_sha256": source_seal_sha256,
        "launcher": f"mpirun -np {TPT_BULK_VASP_MPI_RANKS}",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"submission receipt {key} mismatch: expected {value!r}, "
                f"got {receipt.get(key)!r}"
            )
    expected_runtime = {
        "mpi_ranks": TPT_BULK_VASP_MPI_RANKS,
        "kpar_groups": TPT_BULK_VASP_KPAR,
        "ncore": TPT_BULK_VASP_NCORE,
        "npar": TPT_BULK_VASP_NPAR,
        "nbands": 600,
    }
    if receipt.get("expected_runtime") != expected_runtime:
        raise ValueError("submission receipt does not bind the exact runtime decomposition")
    if expected_case == "pure_g":
        expected_oracle_case = {
            "name": "pure_g",
            "kmesh_tolerance": "1.0e-7",
            "search_shells": 2200,
            "shell_list": [28, 60, 2195],
            "expected_transfers": ["+Gx", "-Gx", "+Gy", "-Gy", "+Gz", "-Gz"],
        }
        if receipt.get("oracle_case") != expected_oracle_case:
            raise ValueError("submission receipt does not bind the pure-G oracle case")
    elif expected_case == "mixed_qg":
        expected_oracle_case = {
            "name": "mixed_qg",
            "kmesh_tolerance": "1.0e-7",
            "search_shells": 6529,
            "shell_list": [1, 2, 6529],
            "expected_physical_q_numerators": [
                list(value) for value in _mixed_qg_expected_transfers()
            ],
        }
        if receipt.get("oracle_case") != expected_oracle_case:
            raise ValueError("submission receipt does not bind the mixed-q+G oracle case")
    required_denials = {
        "production_density_vertex_authority",
        "production_hf_authority",
    }
    if expected_case == "mixed_qg":
        required_denials.update(
            {
                "arbitrary_finite_q_plus_g_exporter_authority",
                "screening_authority",
                "filling_reference_authority",
            }
        )
    for key in required_denials:
        if receipt.get(key) is not False:
            raise ValueError(f"submission receipt must explicitly deny {key}")
    for key, value in receipt.items():
        if key.endswith("authority") and value is not False:
            raise ValueError(f"submission receipt must deny {key}")
    final = path.stat()
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(initial, field) != getattr(final, field) for field in stat_fields):
        raise RuntimeError("submission receipt changed while it was parsed")
    digest = _sha256(path)
    after_hash = path.stat()
    if any(
        getattr(initial, field) != getattr(after_hash, field) for field in stat_fields
    ):
        raise RuntimeError("submission receipt changed while it was hashed")
    return receipt, digest


def _load_mixed_qg_validation_submission_receipt(
    *,
    output_root: Path,
    validation_job_id: int,
    science_job_id: int,
    source_seal_sha256: str,
    science_submission_sha256: str,
    validator_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Cross-bind the zero-science Slurm validator to its science dependency."""

    path = output_root / f"VALIDATION-SUBMISSION-{validation_job_id}.json"
    initial = path.stat()
    if initial.st_mode & 0o222:
        raise PermissionError("mixed-q+G validation submission receipt must be read-only")
    receipt = _read_json_object(
        path,
        domain="mixed-q+G validation submission receipt",
    )
    expected = {
        "schema": "tpt-bulk-mixed-qg-validation-submission-v2",
        "validation_job_id": validation_job_id,
        "science_job_id": science_job_id,
        "dependency": f"afterok:{science_job_id}",
        "account": "hmt03",
        "partitions": ["regular256", "regular6430"],
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 64,
        "exclusive": True,
        "explicit_time_limit": None,
        "action": "validate-mixed-qg",
        "source_seal_sha256": source_seal_sha256,
        "science_submission_sha256": science_submission_sha256,
        "validator_source_sha256": validator_source_sha256,
        "zero_science_postflight": True,
        "production_density_vertex_authority": False,
        "production_hf_authority": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(
                f"mixed-q+G validation receipt {key} mismatch: "
                f"expected {value!r}, got {receipt.get(key)!r}"
            )
    for key, value in receipt.items():
        if key.endswith("authority") and value is not False:
            raise ValueError(f"mixed-q+G validation receipt must deny {key}")
    stat_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    final = path.stat()
    if any(getattr(initial, field) != getattr(final, field) for field in stat_fields):
        raise RuntimeError("mixed-q+G validation receipt changed while it was parsed")
    digest = _sha256(path)
    after_hash = path.stat()
    if any(
        getattr(initial, field) != getattr(after_hash, field) for field in stat_fields
    ):
        raise RuntimeError("mixed-q+G validation receipt changed while it was hashed")
    return receipt, digest


def _load_external_revalidation_authorization(
    *,
    path: Path,
    output_root: Path,
    job_id: int,
    source_seal_sha256: str,
    submission_receipt_sha256: str,
    source_drift: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Load a read-only authorization outside the mutable work capsule."""

    path = path.expanduser().resolve()
    if path == output_root or output_root in path.parents:
        raise ValueError("artifact revalidation authorization must be outside work capsule")
    if path.stat().st_mode & 0o222:
        raise PermissionError("artifact revalidation authorization must be read-only")
    authorization = _read_json_object(
        path,
        domain="artifact revalidation authorization",
    )
    if authorization.get("schema") != "tpt-bulk-paw-artifact-revalidation-v1":
        raise ValueError("unexpected artifact revalidation authorization schema")
    expected = {
        "science_job_id": job_id,
        "source_seal_sha256": source_seal_sha256,
        "submission_receipt_sha256": submission_receipt_sha256,
        "source_drift": source_drift,
        "zero_science_revalidation": True,
        "production_density_vertex_authority": False,
        "production_hf_authority": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise ValueError(f"artifact revalidation authorization does not bind {key}")
    report_path = path.parent / authorization["existing_validation_report_name"]
    if _sha256(report_path) != authorization.get("existing_validation_report_sha256"):
        raise ValueError("artifact revalidation authorization does not bind prior report")
    prior_report = _read_json_object(
        report_path,
        domain="prior validation report",
    )
    if prior_report.get("job_id") != job_id:
        raise ValueError("prior validation report has the wrong science job id")
    if prior_report.get("source_seal_sha256") != source_seal_sha256:
        raise ValueError("prior validation report has the wrong source seal")
    if prior_report.get("submission_receipt_sha256") != submission_receipt_sha256:
        raise ValueError("prior validation report has the wrong submission receipt")
    if prior_report.get("output_hashes") != authorization.get("expected_output_hashes"):
        raise ValueError("authorization output hashes differ from prior validation report")
    return {**authorization, "authorization_path": str(path), "authorization_sha256": _sha256(path)}


def _query_slurm_accounting(job_id: int) -> dict[str, Any]:
    command = [
        "sacct",
        "-j",
        str(job_id),
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode,Elapsed,NNodes,AllocCPUS,MaxRSS",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    job_rows = [row for row in rows if row[0] == str(job_id)]
    if len(job_rows) != 1:
        raise RuntimeError(f"expected one sacct parent row for job {job_id}, got {job_rows}")
    accounting = dict(
        zip(
            (
                "job_id",
                "state",
                "exit_code",
                "elapsed",
                "nodes",
                "allocated_cpus",
                "max_rss",
            ),
            job_rows[0],
            strict=True,
        )
    )
    if accounting["state"] != "COMPLETED" or accounting["exit_code"] != "0:0":
        raise RuntimeError(f"oracle job did not complete successfully: {accounting}")
    accounting["raw_rows"] = rows
    return accounting


def validate_native_oracle_outputs(
    *,
    parent_root: Path,
    output_root: Path,
    job_id: int,
    revalidation_seal_path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless a native VASP PAW MMN fully replays the accepted parent."""

    ensure_not_running_compute_on_login_node("TPT bulk PAW oracle postflight")
    parent_root = parent_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    seal_path = output_root / "SOURCE_SEAL.json"
    seal = _read_json_object(seal_path, domain="native oracle source seal")
    if seal.get("schema") != "tpt-bulk-vasp-paw-oracle-capsule-v1":
        raise ValueError("unexpected PAW oracle source-seal schema")
    if seal.get("case", {}).get("name") != "native":
        raise ValueError("native replay postflight requires a native oracle capsule")
    if seal.get("scope", {}).get("num_bands") != 600:
        raise ValueError("oracle source seal does not require 600 bands")
    if seal.get("scope", {}).get("num_kpoints") != 192:
        raise ValueError("oracle source seal does not require 192 k points")
    source_seal_sha256 = _sha256(seal_path)
    submission_path = output_root / f"SUBMISSION-{job_id}.json"
    _submission_receipt, submission_receipt_sha256 = (
        _load_and_validate_submission_receipt(
            path=submission_path,
            job_id=job_id,
            source_seal_sha256=source_seal_sha256,
            expected_case="native",
        )
    )
    (
        preparation_hashes,
        validation_hashes,
        source_drift,
        revalidation_authorization,
    ) = _validate_postflight_provenance(
        owner=seal.get("runtime"),
        domain="native oracle source seal",
        output_root=output_root,
        job_id=job_id,
        source_seal_sha256=source_seal_sha256,
        submission_receipt_sha256=submission_receipt_sha256,
        revalidation_seal_path=revalidation_seal_path,
    )

    parent_inventory = _validate_parent(parent_root)
    for name, expected in seal["parent_inventory"].items():
        if parent_inventory[name]["sha256"] != expected["sha256"]:
            raise ValueError(f"accepted-parent source drift after oracle run: {name}")
    for name, expected in seal["capsule_inventory"].items():
        actual = _sha256(output_root / name)
        if actual != expected["sha256"]:
            raise ValueError(f"oracle capsule input drift after run: {name}")

    required_outputs = (
        "OUTCAR",
        "wannier90.amn",
        "wannier90.eig",
        "wannier90.mmn",
        "wannier90.win",
        "wannier90.wout",
    )
    missing = [name for name in required_outputs if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing native oracle output files: {missing}")
    for name in required_outputs:
        path = output_root / name
        if path.stat().st_size <= 0:
            raise ValueError(f"empty native oracle output: {path}")
        path.chmod(path.stat().st_mode & ~0o222)

    outcar_text = (output_root / "OUTCAR").read_text(errors="replace")
    required_outcar = (
        "running   32 mpi-ranks, on    2 nodes",
        "distrk:  each k-point on    8 cores,    4 groups",
        "distr:  one band on NCORE=   1 cores,    8 groups",
        "Computing MMN (overlap matrix elements)",
    )
    missing_outcar = [text for text in required_outcar if text not in outcar_text]
    if missing_outcar:
        raise ValueError(f"OUTCAR is missing required runtime evidence: {missing_outcar}")
    forbidden_outcar = (
        "The number of bands has been changed",
        "random initialization beyond band",
        "I REFUSE TO CONTINUE",
    )
    present_forbidden = [text for text in forbidden_outcar if text in outcar_text]
    if present_forbidden:
        raise ValueError(f"OUTCAR contains forbidden runtime evidence: {present_forbidden}")

    accounting = _query_slurm_accounting(job_id)
    accepted_eig = read_wannier90_eig(
        parent_root / "wannier90.eig",
        num_bands=600,
        num_kpts=192,
        expected_sha256=TPT_BULK_WANNIER90_EIG_SHA256,
    )
    candidate_eig = read_wannier90_eig(
        output_root / "wannier90.eig",
        num_bands=600,
        num_kpts=192,
    )
    eig_max_abs_difference_ev = float(np.max(np.abs(candidate_eig - accepted_eig)))
    if eig_max_abs_difference_ev > TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV:
        raise ValueError(
            "native oracle eig replay exceeds tolerance: "
            f"{eig_max_abs_difference_ev} > {TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV}"
        )

    mmn_comparison = compare_wannier90_mmn_files(
        reference_path=parent_root / "wannier90.mmn",
        candidate_path=output_root / "wannier90.mmn",
        num_bands=600,
        num_kpoints=192,
        nntot=6,
        reference_sha256=TPT_BULK_WANNIER90_MMN_SHA256,
    )
    if mmn_comparison["max_abs_difference"] > TPT_BULK_NATIVE_MMN_MAX_ABS_TOLERANCE:
        raise ValueError(
            "native PAW MMN replay exceeds tolerance: "
            f"{mmn_comparison['max_abs_difference']} > "
            f"{TPT_BULK_NATIVE_MMN_MAX_ABS_TOLERANCE}"
        )

    output_hashes = {
        name: (
            mmn_comparison["candidate_sha256"]
            if name == "wannier90.mmn"
            else _sha256(output_root / name)
        )
        for name in required_outputs
    }
    if (
        revalidation_authorization is not None
        and output_hashes != revalidation_authorization["expected_output_hashes"]
    ):
        raise ValueError("artifact revalidation output hashes do not match authorization")
    report = {
        "schema": "tpt-bulk-native-paw-overlap-validation-v1",
        "status": "passed_native_paw_overlap_replay",
        **_selected_provenance_report_fields(
            preparation_hashes,
            validation_hashes,
            source_drift,
        ),
        "job_id": job_id,
        "source_seal_sha256": source_seal_sha256,
        "submission_receipt_sha256": submission_receipt_sha256,
        "accounting": accounting,
        "scope": seal["scope"],
        "case": seal["case"],
        "external_revalidation_authorization_sha256": (
            revalidation_authorization["authorization_sha256"]
            if revalidation_authorization is not None
            else None
        ),
        "accepted_parent_hashes": {
            **{name: entry["sha256"] for name, entry in parent_inventory.items()},
            "wannier90.eig": TPT_BULK_WANNIER90_EIG_SHA256,
            "wannier90.mmn": TPT_BULK_WANNIER90_MMN_SHA256,
        },
        "output_hashes": output_hashes,
        "eigenvalue_replay": {
            "max_abs_difference_ev": eig_max_abs_difference_ev,
            "tolerance_ev": TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV,
            "passed": True,
        },
        "mmn_replay": {
            **mmn_comparison,
            "tolerance": TPT_BULK_NATIVE_MMN_MAX_ABS_TOLERANCE,
            "passed": True,
        },
        "verdict": {
            "native_vasp_paw_overlap_path_validated": True,
            "arbitrary_finite_g_exporter_validated": False,
            "production_density_vertex_authority": False,
            "screening_authority": False,
            "filling_reference_authority": False,
            "production_hf_authority": False,
        },
    }
    report_name = (
        f"ARTIFACT-REVALIDATION-{job_id}.json"
        if revalidation_authorization is not None
        else f"VALIDATION-{job_id}.json"
    )
    report_path = output_root / report_name
    _write_json_new(report_path, report)
    report_path.chmod(0o444)
    return report


def validate_pure_g_oracle_outputs(
    *,
    parent_root: Path,
    output_root: Path,
    job_id: int,
    revalidation_seal_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the combined +/-Gx,+/-Gy,+/-Gz VASP PAW oracle inventory."""

    ensure_not_running_compute_on_login_node("TPT bulk pure-G PAW oracle postflight")
    parent_root = parent_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    seal_path = output_root / "SOURCE_SEAL.json"
    seal = _read_json_object(seal_path, domain="pure-G oracle source seal")
    if seal.get("schema") != "tpt-bulk-vasp-paw-oracle-capsule-v1":
        raise ValueError("unexpected PAW oracle source-seal schema")
    if seal.get("case", {}).get("name") != "pure_g":
        raise ValueError("pure-G postflight requires a pure_g oracle capsule")
    if seal.get("case", {}).get("shell_list") != [28, 60, 2195]:
        raise ValueError("pure-G source seal has the wrong shell inventory")
    if seal.get("scope", {}).get("num_bands") != 600:
        raise ValueError("pure-G source seal does not require 600 bands")
    if seal.get("scope", {}).get("num_kpoints") != 192:
        raise ValueError("pure-G source seal does not require 192 k points")

    source_seal_sha256 = _sha256(seal_path)
    submission_path = output_root / f"SUBMISSION-{job_id}.json"
    _submission_receipt, submission_receipt_sha256 = (
        _load_and_validate_submission_receipt(
            path=submission_path,
            job_id=job_id,
            source_seal_sha256=source_seal_sha256,
            expected_case="pure_g",
        )
    )
    (
        preparation_hashes,
        validation_hashes,
        source_drift,
        revalidation_authorization,
    ) = _validate_postflight_provenance(
        owner=seal.get("runtime"),
        domain="pure-G oracle source seal",
        output_root=output_root,
        job_id=job_id,
        source_seal_sha256=source_seal_sha256,
        submission_receipt_sha256=submission_receipt_sha256,
        revalidation_seal_path=revalidation_seal_path,
    )

    parent_inventory = _validate_parent(parent_root)
    for name, expected in seal["parent_inventory"].items():
        if parent_inventory[name]["sha256"] != expected["sha256"]:
            raise ValueError(f"accepted-parent source drift after pure-G run: {name}")
    for name, expected in seal["capsule_inventory"].items():
        actual = _sha256(output_root / name)
        if actual != expected["sha256"]:
            raise ValueError(f"pure-G capsule input drift after run: {name}")

    required_outputs = (
        "OUTCAR",
        "wannier90.amn",
        "wannier90.eig",
        "wannier90.mmn",
        "wannier90.win",
        "wannier90.wout",
    )
    missing = [name for name in required_outputs if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing pure-G oracle output files: {missing}")
    for name in required_outputs:
        path = output_root / name
        if path.stat().st_size <= 0:
            raise ValueError(f"empty pure-G oracle output: {path}")
        path.chmod(path.stat().st_mode & ~0o222)

    outcar_text = (output_root / "OUTCAR").read_text(errors="replace")
    required_outcar = (
        "running   32 mpi-ranks, on    2 nodes",
        "distrk:  each k-point on    8 cores,    4 groups",
        "distr:  one band on NCORE=   1 cores,    8 groups",
        "Computing MMN (overlap matrix elements)",
    )
    missing_outcar = [text for text in required_outcar if text not in outcar_text]
    if missing_outcar:
        raise ValueError(f"OUTCAR is missing required runtime evidence: {missing_outcar}")
    forbidden_outcar = (
        "The number of bands has been changed",
        "random initialization beyond band",
        "I REFUSE TO CONTINUE",
    )
    present_forbidden = [text for text in forbidden_outcar if text in outcar_text]
    if present_forbidden:
        raise ValueError(f"OUTCAR contains forbidden runtime evidence: {present_forbidden}")

    accounting = _query_slurm_accounting(job_id)
    accepted_eig = read_wannier90_eig(
        parent_root / "wannier90.eig",
        num_bands=600,
        num_kpts=192,
        expected_sha256=TPT_BULK_WANNIER90_EIG_SHA256,
    )
    candidate_eig = read_wannier90_eig(
        output_root / "wannier90.eig",
        num_bands=600,
        num_kpts=192,
    )
    eig_max_abs_difference_ev = float(np.max(np.abs(candidate_eig - accepted_eig)))
    if eig_max_abs_difference_ev > TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV:
        raise ValueError(
            "pure-G eig replay exceeds tolerance: "
            f"{eig_max_abs_difference_ev} > {TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV}"
        )

    expected_shifts = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    mmn_validation = validate_pure_g_mmn_file(
        candidate_path=output_root / "wannier90.mmn",
        num_bands=600,
        num_kpoints=192,
        expected_shifts=expected_shifts,
    )
    if (
        mmn_validation["reverse_closure_max_abs_difference"]
        > TPT_BULK_PURE_G_REVERSE_CLOSURE_TOLERANCE
    ):
        raise ValueError(
            "pure-G PAW MMN reverse closure exceeds tolerance: "
            f"{mmn_validation['reverse_closure_max_abs_difference']} > "
            f"{TPT_BULK_PURE_G_REVERSE_CLOSURE_TOLERANCE}"
        )

    output_hashes = {
        name: (
            mmn_validation["candidate_sha256"]
            if name == "wannier90.mmn"
            else _sha256(output_root / name)
        )
        for name in required_outputs
    }
    if (
        revalidation_authorization is not None
        and output_hashes != revalidation_authorization["expected_output_hashes"]
    ):
        raise ValueError("artifact revalidation output hashes do not match authorization")
    report = {
        "schema": "tpt-bulk-pure-g-paw-overlap-validation-v1",
        "status": "passed_combined_pure_g_paw_oracle",
        **_selected_provenance_report_fields(
            preparation_hashes,
            validation_hashes,
            source_drift,
        ),
        "job_id": job_id,
        "source_seal_sha256": source_seal_sha256,
        "submission_receipt_sha256": submission_receipt_sha256,
        "accounting": accounting,
        "scope": seal["scope"],
        "case": seal["case"],
        "external_revalidation_authorization_sha256": (
            revalidation_authorization["authorization_sha256"]
            if revalidation_authorization is not None
            else None
        ),
        "accepted_parent_hashes": {
            **{name: entry["sha256"] for name, entry in parent_inventory.items()},
            "wannier90.eig": TPT_BULK_WANNIER90_EIG_SHA256,
        },
        "output_hashes": output_hashes,
        "eigenvalue_replay": {
            "max_abs_difference_ev": eig_max_abs_difference_ev,
            "tolerance_ev": TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV,
            "passed": True,
        },
        "mmn_validation": {
            **mmn_validation,
            "reverse_closure_tolerance": TPT_BULK_PURE_G_REVERSE_CLOSURE_TOLERANCE,
            "passed": True,
        },
        "verdict": {
            "combined_pure_g_vasp_paw_oracle_validated": True,
            "arbitrary_finite_q_plus_g_exporter_validated": False,
            "production_density_vertex_authority": False,
            "screening_authority": False,
            "filling_reference_authority": False,
            "production_hf_authority": False,
        },
    }
    report_name = (
        f"ARTIFACT-REVALIDATION-{job_id}.json"
        if revalidation_authorization is not None
        else f"VALIDATION-{job_id}.json"
    )
    report_path = output_root / report_name
    _write_json_new(report_path, report)
    report_path.chmod(0o444)
    return report


def validate_mixed_qg_oracle_outputs(
    *,
    parent_root: Path,
    output_root: Path,
    job_id: int,
    validation_job_id: int,
    revalidation_seal_path: Path | None = None,
) -> dict[str, Any]:
    """Seal the minimal B1-complete full-band mixed finite-q+G PAW oracle."""

    ensure_not_running_compute_on_login_node("TPT bulk mixed-q+G PAW oracle postflight")
    parent_root = parent_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    seal_path = output_root / "SOURCE_SEAL.json"
    seal = _read_json_object(seal_path, domain="mixed-q+G oracle source seal")
    case = ORACLE_CASES["mixed_qg"]
    if seal.get("schema") != "tpt-bulk-vasp-paw-oracle-capsule-v1":
        raise ValueError("unexpected mixed-q+G PAW oracle source-seal schema")
    if seal.get("case", {}).get("name") != case.name:
        raise ValueError("mixed-q+G postflight requires a mixed_qg capsule")
    if seal.get("case", {}).get("shell_list") != list(case.shell_list):
        raise ValueError("mixed-q+G source seal has the wrong shell inventory")
    if seal.get("scope", {}).get("num_bands") != 600:
        raise ValueError("mixed-q+G source seal does not require 600 bands")
    if seal.get("scope", {}).get("num_kpoints") != 192:
        raise ValueError("mixed-q+G source seal does not require 192 k points")

    source_seal_sha256 = _sha256(seal_path)
    submission_path = output_root / f"SUBMISSION-{job_id}.json"
    _submission_receipt, submission_receipt_sha256 = (
        _load_and_validate_submission_receipt(
            path=submission_path,
            job_id=job_id,
            source_seal_sha256=source_seal_sha256,
            expected_case=case.name,
        )
    )
    (
        preparation_hashes,
        validation_hashes,
        source_drift,
        revalidation_authorization,
    ) = _validate_postflight_provenance(
        owner=seal.get("runtime"),
        domain="mixed-q+G oracle source seal",
        output_root=output_root,
        job_id=job_id,
        source_seal_sha256=source_seal_sha256,
        submission_receipt_sha256=submission_receipt_sha256,
        revalidation_seal_path=revalidation_seal_path,
    )
    _validation_receipt, validation_submission_sha256 = (
        _load_mixed_qg_validation_submission_receipt(
            output_root=output_root,
            validation_job_id=validation_job_id,
            science_job_id=job_id,
            source_seal_sha256=source_seal_sha256,
            science_submission_sha256=submission_receipt_sha256,
            validator_source_sha256=validation_hashes[
                "src/mean_field/devtools/tpt_bulk_paw_oracle.py"
            ],
        )
    )
    validation_runtime = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_num_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }
    expected_validation_runtime = {
        "slurm_job_id": str(validation_job_id),
        "slurm_job_num_nodes": "1",
        "slurm_ntasks": "1",
        "slurm_cpus_per_task": "64",
    }
    if validation_runtime != expected_validation_runtime:
        raise ValueError(
            "mixed-q+G validation runtime does not match its sealed submission: "
            f"{validation_runtime}"
        )

    parent_inventory = _validate_parent(parent_root)
    for name, expected in seal["parent_inventory"].items():
        if parent_inventory[name]["sha256"] != expected["sha256"]:
            raise ValueError(f"accepted-parent source drift after mixed-q+G run: {name}")
    for name, expected in seal["capsule_inventory"].items():
        if _sha256(output_root / name) != expected["sha256"]:
            raise ValueError(f"mixed-q+G capsule input drift after run: {name}")

    required_outputs = (
        "OUTCAR",
        "wannier90.amn",
        "wannier90.eig",
        "wannier90.mmn",
        "wannier90.win",
        "wannier90.wout",
    )
    missing = [name for name in required_outputs if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing mixed-q+G oracle outputs: {missing}")
    for name in required_outputs:
        path = output_root / name
        if path.stat().st_size <= 0:
            raise ValueError(f"empty mixed-q+G oracle output: {path}")
        path.chmod(path.stat().st_mode & ~0o222)

    outcar_text = (output_root / "OUTCAR").read_text(errors="replace")
    required_outcar = (
        "running   32 mpi-ranks, on    2 nodes",
        "distrk:  each k-point on    8 cores,    4 groups",
        "distr:  one band on NCORE=   1 cores,    8 groups",
        "Computing MMN (overlap matrix elements)",
    )
    missing_outcar = [text for text in required_outcar if text not in outcar_text]
    if missing_outcar:
        raise ValueError(
            f"mixed-q+G OUTCAR is missing runtime evidence: {missing_outcar}"
        )
    forbidden_outcar = (
        "The number of bands has been changed",
        "random initialization beyond band",
        "I REFUSE TO CONTINUE",
    )
    present_forbidden = [text for text in forbidden_outcar if text in outcar_text]
    if present_forbidden:
        raise ValueError(
            f"mixed-q+G OUTCAR contains forbidden evidence: {present_forbidden}"
        )

    accounting = _query_slurm_accounting(job_id)
    accepted_eig = read_wannier90_eig(
        parent_root / "wannier90.eig",
        num_bands=600,
        num_kpts=192,
        expected_sha256=TPT_BULK_WANNIER90_EIG_SHA256,
    )
    candidate_eig = read_wannier90_eig(
        output_root / "wannier90.eig",
        num_bands=600,
        num_kpts=192,
    )
    eig_max_abs_difference_ev = float(np.max(np.abs(candidate_eig - accepted_eig)))
    if eig_max_abs_difference_ev > TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV:
        raise ValueError(
            "mixed-q+G eig replay exceeds tolerance: "
            f"{eig_max_abs_difference_ev}"
        )

    parent_kpoints = _read_win_kpoint_numerators(parent_root / "wannier90.win")
    output_kpoints = _read_win_kpoint_numerators(output_root / "wannier90.win")
    if output_kpoints != parent_kpoints:
        raise ValueError("mixed-q+G output k-point ordering differs from accepted parent")

    if TPT_BULK_MIXED_QG_PREFLIGHT_PATH.stat().st_mode & 0o222:
        raise PermissionError("mixed-q+G Wannier90 shell preflight must be read-only")
    if (
        _sha256(TPT_BULK_MIXED_QG_PREFLIGHT_PATH)
        != TPT_BULK_MIXED_QG_PREFLIGHT_SHA256
    ):
        raise ValueError("mixed-q+G Wannier90 shell preflight hash mismatch")
    shell_preflight = _read_json_object(
        TPT_BULK_MIXED_QG_PREFLIGHT_PATH,
        domain="mixed-q+G Wannier90 shell preflight",
    )
    preflight_transfers = {
        tuple(row["physical_q_numerator"])
        for row in shell_preflight.get("source_kpoint_one_rows", [])
    }
    if (
        shell_preflight.get("schema")
        != "tpt-bulk-w90-general-transfer-b1-preflight-v1"
        or shell_preflight.get("status")
        != "passed_b1_complete_shell_1_2_6529"
        or shell_preflight.get("shell_list") != [1, 2, 6529]
        or shell_preflight.get("search_shells") != 6529
        or shell_preflight.get("nntot") != 12
        or shell_preflight.get("b1_complete") is not True
        or shell_preflight.get("skip_b1_tests") is not False
        or preflight_transfers != set(_mixed_qg_expected_transfers())
        or shell_preflight.get("authority", {}).get("shell_index_mapping") is not True
        or shell_preflight.get("authority", {}).get("paw_overlap_authority") is not False
        or shell_preflight.get("authority", {}).get("production_hf_authority")
        is not False
    ):
        raise ValueError("mixed-q+G Wannier90 shell preflight semantics mismatch")
    mmn_validation = validate_general_qg_mmn_file(
        candidate_path=output_root / "wannier90.mmn",
        num_bands=600,
        kpoint_numerators=output_kpoints,
        expected_transfers=_mixed_qg_expected_transfers(),
        check_matrix_reverse_closure=False,
    )
    output_hashes = {
        name: (
            mmn_validation["candidate_sha256"]
            if name == "wannier90.mmn"
            else _sha256(output_root / name)
        )
        for name in required_outputs
    }
    if (
        revalidation_authorization is not None
        and output_hashes != revalidation_authorization["expected_output_hashes"]
    ):
        raise ValueError("mixed-q+G output hashes do not match authorization")

    report = {
        "schema": "tpt-bulk-mixed-qg-paw-overlap-validation-v1",
        "status": "passed_full_band_mixed_qg_paw_oracle_inventory",
        **_selected_provenance_report_fields(
            preparation_hashes,
            validation_hashes,
            source_drift,
        ),
        "job_id": job_id,
        "validation_job_id": validation_job_id,
        "source_seal_sha256": source_seal_sha256,
        "submission_receipt_sha256": submission_receipt_sha256,
        "validation_submission_receipt_sha256": validation_submission_sha256,
        "validation_runtime": validation_runtime,
        "accounting": accounting,
        "scope": seal["scope"],
        "case": seal["case"],
        "external_revalidation_authorization_sha256": (
            revalidation_authorization["authorization_sha256"]
            if revalidation_authorization is not None
            else None
        ),
        "accepted_parent_hashes": {
            **{name: entry["sha256"] for name, entry in parent_inventory.items()},
            "wannier90.eig": TPT_BULK_WANNIER90_EIG_SHA256,
        },
        "output_hashes": output_hashes,
        "eigenvalue_replay": {
            "max_abs_difference_ev": eig_max_abs_difference_ev,
            "tolerance_ev": TPT_BULK_NATIVE_EIG_MAX_ABS_TOLERANCE_EV,
            "passed": True,
        },
        "shell_preflight": {
            "path": str(TPT_BULK_MIXED_QG_PREFLIGHT_PATH),
            "sha256": TPT_BULK_MIXED_QG_PREFLIGHT_SHA256,
            "wannier90_nnkp_sha256": shell_preflight["hashes"]["wannier90.nnkp"],
            "b1_complete": True,
            "nntot": 12,
        },
        "mmn_validation": mmn_validation,
        "verdict": {
            "full_band_mixed_qg_paw_oracle_inventory_validated": True,
            "active_projection_replay_validated": False,
            "arbitrary_finite_q_plus_g_exporter_validated": False,
            "production_density_vertex_authority": False,
            "screening_authority": False,
            "filling_reference_authority": False,
            "production_hf_authority": False,
        },
    }
    report_name = (
        f"ARTIFACT-REVALIDATION-{job_id}.json"
        if revalidation_authorization is not None
        else f"VALIDATION-{job_id}.json"
    )
    report_path = output_root / report_name
    _write_json_new(report_path, report)
    report_path.chmod(0o444)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "prepare",
            "prepare-active-wavecar",
            "prepare-active-oracle",
            "prepare-transfer-plan",
            "validate-active",
            "validate-native",
            "validate-pure-g",
            "validate-mixed-qg",
        ),
        default="prepare",
    )
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=Path("/data/work/tmp/toZZY/moreiter"),
        help="Accepted same-parent DFT/Wannier directory.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/work/ziyuzhu/TPT235-bulk-wannier1_source_v1"),
        help="Accepted 320-Wannier HR/POSCAR source directory.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--active-wavecar-root",
        type=Path,
        default=Path("/data/work/ziyuzhu/tpt_bulk_active16_wavecar_v2_c128"),
        help="Source-sealed transformed active-16 WAVECAR capsule.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Certified full-600-band PAW oracle root for active validation.",
    )
    parser.add_argument("--case", choices=tuple(ORACLE_CASES))
    parser.add_argument(
        "--active-layout",
        choices=tuple(_ACTIVE_ORACLE_LAYOUTS),
        default="regular64",
        help="Exact MPI/KPAR/NPAR profile for a transformed-WAVECAR oracle.",
    )
    parser.add_argument("--job-id", type=int)
    parser.add_argument(
        "--validation-job-id",
        type=int,
        help="Slurm job running the zero-science mixed-q+G validator.",
    )
    parser.add_argument(
        "--local-field-shell",
        type=int,
        choices=(0, 1),
        help="Planning-only local-field seed: 0 for G=0, 1 for 0 plus +/-b_i.",
    )
    parser.add_argument(
        "--revalidation-seal",
        type=Path,
        help="Read-only external authorization for zero-science artifact revalidation.",
    )
    parser.add_argument(
        "--wavecar-copy-mode",
        choices=("reflink", "full"),
        default="full",
        help=(
            "Independent WAVECAR clone mode. Full copy is the Lustre-safe default; "
            "hardlinks and symlinks are forbidden."
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if (
        args.validation_job_id is not None
        and args.action != "validate-mixed-qg"
    ):
        raise SystemExit(
            "--validation-job-id is only valid for --action validate-mixed-qg"
        )
    if args.action == "prepare-transfer-plan":
        if args.local_field_shell is None:
            raise SystemExit(
                "--local-field-shell is required for --action prepare-transfer-plan"
            )
        if args.case is not None or args.job_id is not None:
            raise SystemExit(
                "--case/--job-id are not valid for --action prepare-transfer-plan"
            )
        payload = prepare_active_transfer_plan(
            source_root=args.source_root,
            output_root=args.output_root,
            local_field_shell=args.local_field_shell,
        )
    elif args.action == "prepare":
        if args.case is None:
            raise SystemExit("--case is required for --action prepare")
        payload = prepare_oracle_capsule(
            parent_root=args.parent_root,
            output_root=args.output_root,
            case=ORACLE_CASES[args.case],
            wavecar_copy_mode=args.wavecar_copy_mode,
        )
    elif args.action == "prepare-active-wavecar":
        if args.case is not None or args.job_id is not None:
            raise SystemExit(
                "--case/--job-id are not valid for --action prepare-active-wavecar"
            )
        payload = prepare_active_wavecar(
            source_root=args.source_root,
            parent_root=args.parent_root,
            output_root=args.output_root,
        )
    elif args.action == "prepare-active-oracle":
        if args.case is None:
            raise SystemExit("--case is required for --action prepare-active-oracle")
        if args.job_id is not None:
            raise SystemExit("--job-id is not valid for --action prepare-active-oracle")
        payload = prepare_active_oracle_capsule(
            parent_root=args.parent_root,
            active_wavecar_root=args.active_wavecar_root,
            output_root=args.output_root,
            case=ORACLE_CASES[args.case],
            wavecar_copy_mode=args.wavecar_copy_mode,
            active_layout=args.active_layout,
        )
    elif args.action == "validate-active":
        if args.job_id is None or args.reference_root is None:
            raise SystemExit(
                "--job-id and --reference-root are required for --action validate-active"
            )
        if args.case is not None:
            raise SystemExit("--case is inferred from SOURCE_SEAL.json")
        payload = validate_active_oracle_outputs(
            source_root=args.source_root,
            parent_root=args.parent_root,
            reference_root=args.reference_root,
            output_root=args.output_root,
            job_id=args.job_id,
            revalidation_seal_path=args.revalidation_seal,
        )
    else:
        if args.job_id is None:
            raise SystemExit("--job-id is required for validation actions")
        if args.case is not None:
            raise SystemExit("--case is inferred from SOURCE_SEAL.json during validation")
        validators = {
            "validate-native": validate_native_oracle_outputs,
            "validate-pure-g": validate_pure_g_oracle_outputs,
            "validate-mixed-qg": validate_mixed_qg_oracle_outputs,
        }
        validator = validators[args.action]
        validation_kwargs: dict[str, Any] = {}
        if args.action == "validate-mixed-qg":
            if args.validation_job_id is None:
                raise SystemExit(
                    "--validation-job-id is required for --action validate-mixed-qg"
                )
            validation_kwargs["validation_job_id"] = args.validation_job_id
        payload = validator(
            parent_root=args.parent_root,
            output_root=args.output_root,
            job_id=args.job_id,
            revalidation_seal_path=args.revalidation_seal,
            **validation_kwargs,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ORACLE_CASES",
    "OracleCase",
    "build_active_transfer_plan",
    "compare_projected_active_mmn_files",
    "compare_wannier90_mmn_files",
    "prepare_active_oracle_capsule",
    "prepare_active_transfer_plan",
    "prepare_active_wavecar",
    "prepare_oracle_capsule",
    "render_active_oracle_incar",
    "render_oracle_incar",
    "validate_active_oracle_outputs",
    "validate_general_qg_mmn_file",
    "validate_mixed_qg_oracle_outputs",
    "validate_native_oracle_outputs",
    "validate_pure_g_mmn_file",
    "validate_pure_g_oracle_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
