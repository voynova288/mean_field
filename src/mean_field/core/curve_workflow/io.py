from __future__ import annotations

"""Deterministic full-workflow artifacts for generic exact-grid curve evidence."""

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import write_json_artifact, write_npz_artifact, write_text_artifact
from .contracts import (
    ComputeCertificate,
    CurveDomainReceipt,
    DiscreteBranchNode,
    EnumerationReceipt,
    ExactSavedGridCurve,
    ExactSavedGridCurveBundle,
    ObservableReceipt,
    SavedGridReceipt,
    SourceAuthorityReceipt,
    ValueTransformReceipt,
    canonical_array_sha256,
    certify_enumerated_branch_closure,
    make_source_authority_receipt,
)
from .raster import (
    CurveAgreementCriterion,
    RasterAxesCalibration,
    RasterBundleComparison,
    RasterCurveExtraction,
    RasterEvaluationPlan,
    RasterExtractionPolicy,
    RasterSourceReceipt,
    SelectionLineageReceipt,
    compare_raster_to_all_branches,
    create_raster_evaluation_plan,
    extract_raster_curve,
)

METADATA_FILENAME = "metadata.json"
ARRAYS_FILENAME = "arrays.npz"
CURVES_CSV_FILENAME = "curves.csv"


@dataclass(frozen=True)
class CurveWorkflowArtifactPaths:
    """Fixed generic filenames produced by one artifact export."""

    metadata_json: Path
    arrays_npz: Path
    curves_csv: Path


@dataclass(frozen=True)
class LoadedCurveWorkflow:
    """Fully reconstructed and revalidated workflow evidence."""

    bundle: ExactSavedGridCurveBundle
    raster: RasterCurveExtraction | None
    evaluation_plan: RasterEvaluationPlan | None
    comparison: RasterBundleComparison | None
    paths: CurveWorkflowArtifactPaths


def _source_authority_metadata(
    receipt: SourceAuthorityReceipt,
) -> dict[str, str]:
    receipt.validate_live_state()
    return {
        "authority_id": receipt.authority_id,
        "canonical_payload_json": receipt.canonical_payload_json,
        "payload_sha256": receipt.payload_sha256,
    }


def _grid_metadata(grid: SavedGridReceipt) -> dict[str, object]:
    return {
        "source_id": grid.source_id,
        "x_units": grid.x_units,
        "domain": asdict(grid.domain),
        "point_indices_sha256": grid.point_indices_sha256,
        "x_sha256": grid.x_sha256,
    }


def _bundle_metadata(bundle: ExactSavedGridCurveBundle) -> dict[str, object]:
    curves: list[dict[str, object]] = []
    for index, curve in enumerate(bundle.curves):
        curves.append(
            {
                "array_prefix": f"curve_{index:04d}",
                "terminal_id": curve.terminal_id,
                "terminal_payload_sha256": curve.terminal_payload_sha256,
                "saved_grid": _grid_metadata(curve.saved_grid),
                "observable": asdict(curve.observable),
                "value_transform": asdict(curve.value_transform),
                "raw_y_sha256": curve.raw_y_sha256,
                "output_y_sha256": curve.output_y_sha256,
            }
        )
    closure = bundle.branch_closure
    return {
        "schema": "mean_field.curve_workflow.v3",
        "bundle_fingerprint": bundle.bundle_fingerprint,
        "source_authority": _source_authority_metadata(bundle.source_authority),
        "branch_closure": {
            "authority": closure.authority,
            "supplied_finite_tree_structurally_resolved": closure.supplied_finite_tree_structurally_resolved,
            "root_id": closure.root_id,
            "terminal_ids": list(closure.terminal_ids),
            "computed_terminal_ids": list(closure.computed_terminal_ids),
            "rejected_terminal_ids": list(closure.rejected_terminal_ids),
            "nodes": [asdict(node) for node in closure.nodes],
            "enumeration_receipt": asdict(closure.enumeration_receipt),
        },
        "compute_certificate": asdict(bundle.compute_certificate),
        "curves": curves,
    }


def _raster_metadata(
    raster: RasterCurveExtraction,
    evaluation_plan: RasterEvaluationPlan | None,
) -> dict[str, object]:
    return {
        "source": asdict(raster.source),
        "calibration": asdict(raster.calibration),
        "policy": asdict(raster.policy),
        "component_label": raster.component_label,
        "component_pixel_sha256": raster.component_pixel_sha256,
        "closed_dark_frame_detected": raster.closed_dark_frame_detected,
        "boundary_flag_count": int(np.count_nonzero(raster.boundary_flags)),
        "source_bytes_stored": raster.source_bytes is not None,
        "evaluation_binding": "unbound" if evaluation_plan is None else "plan_bound",
    }


def _plan_metadata(plan: RasterEvaluationPlan) -> dict[str, object]:
    return {
        "compute_bundle_fingerprint": plan.compute_bundle_fingerprint,
        "expected_source": asdict(plan.expected_source),
        "calibration": asdict(plan.calibration),
        "extraction_policy": asdict(plan.extraction_policy),
        "value_kind": plan.value_kind,
        "value_units": plan.value_units,
        "value_semantics": plan.value_semantics,
        "transforms_identical": plan.transforms_identical,
        "criterion": None if plan.criterion is None else asdict(plan.criterion),
        "selection_lineage": asdict(plan.selection_lineage),
        "preregistration_evidence_sha256": plan.preregistration_evidence_sha256,
        "expected_closed_frame_present": plan.expected_closed_frame_present,
        "criterion_eligible": plan.criterion_eligible,
        "plan_fingerprint": plan.plan_fingerprint,
    }


def _comparison_metadata(comparison: RasterBundleComparison) -> dict[str, object]:
    branches: list[dict[str, object]] = []
    for item in comparison.branch_metrics:
        branches.append(
            {
                "terminal_id": item.terminal_id,
                "rmse": item.rmse,
                "mae": item.mae,
                "maximum_absolute_error": item.maximum_absolute_error,
                "mean_error": item.mean_error,
                "crossings": [asdict(value) for value in item.crossings],
                "exact_grid_extrema": [asdict(value) for value in item.exact_grid_extrema],
                "threshold_checks": [[name, passed] for name, passed in item.threshold_checks],
            }
        )
    return {
        "plan_fingerprint": comparison.plan.plan_fingerprint,
        "value_kind": comparison.value_kind,
        "decision": comparison.decision,
        "transforms_identical": comparison.transforms_identical,
        "raster_crossings": [asdict(value) for value in comparison.raster_crossings],
        "branches": branches,
    }


def _array_payload(
    bundle: ExactSavedGridCurveBundle,
    raster: RasterCurveExtraction | None,
    comparison: RasterBundleComparison | None,
) -> dict[str, np.ndarray]:
    authority = bundle.source_authority
    arrays: dict[str, np.ndarray] = {
        "source_authority_authority_id_utf8": np.frombuffer(
            authority.authority_id.encode("utf-8"), dtype=np.uint8
        ),
        "source_authority_canonical_payload_json_utf8": np.frombuffer(
            authority.canonical_payload_json.encode("utf-8"), dtype=np.uint8
        ),
        "source_authority_payload_sha256_ascii": np.frombuffer(
            authority.payload_sha256.encode("ascii"), dtype=np.uint8
        ),
    }
    for index, curve in enumerate(bundle.curves):
        prefix = f"curve_{index:04d}"
        arrays[f"{prefix}_point_indices"] = curve.saved_grid.point_indices
        arrays[f"{prefix}_x"] = curve.saved_grid.x
        arrays[f"{prefix}_raw_y"] = curve.raw_y
        arrays[f"{prefix}_output_y"] = curve.output_y
    if raster is not None:
        for name in (
            "component_pixel_x",
            "component_pixel_y",
            "centerline_pixel_x",
            "centerline_pixel_y",
            "x",
            "y",
            "pixel_uncertainty_y",
            "line_thickness_uncertainty_y",
            "total_uncertainty_y",
            "boundary_flags",
            "source_rgba",
        ):
            arrays[f"raster_{name}"] = getattr(raster, name)
        if raster.source_bytes is not None:
            arrays["raster_source_bytes"] = np.frombuffer(raster.source_bytes, dtype=np.uint8)
    if comparison is not None:
        for index, item in enumerate(comparison.branch_metrics):
            prefix = f"comparison_{index:04d}"
            arrays[f"{prefix}_branch_y_at_raster_x"] = item.branch_y_at_raster_x
            arrays[f"{prefix}_residual"] = item.residual
    return arrays


def _curve_csv(bundle: ExactSavedGridCurveBundle) -> str:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "terminal_id",
            "point_index",
            "x",
            "raw_y",
            "output_y",
            "x_units",
            "input_units",
            "output_units",
            "source_authority_id",
            "source_authority_payload_sha256",
            "source_authority_canonical_payload_json",
        ]
    )
    for curve in bundle.curves:
        for point_index, x, raw_y, output_y in zip(
            curve.saved_grid.point_indices,
            curve.saved_grid.x,
            curve.raw_y,
            curve.output_y,
            strict=True,
        ):
            writer.writerow(
                [
                    curve.terminal_id,
                    int(point_index),
                    format(float(x), ".17g"),
                    format(float(raw_y), ".17g"),
                    format(float(output_y), ".17g"),
                    curve.saved_grid.x_units,
                    curve.value_transform.input_units,
                    curve.value_transform.output_units,
                    bundle.source_authority.authority_id,
                    bundle.source_authority.payload_sha256,
                    bundle.source_authority.canonical_payload_json,
                ]
            )
    return buffer.getvalue()


def _require_equal(actual: object, expected: object, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"artifact {name} mismatch: {actual!r} != {expected!r}")


def _require_array_equal(actual: object, expected: object, *, name: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"artifact {name} array does not match re-derived evidence")


def _validate_comparison_identity(
    bundle: ExactSavedGridCurveBundle,
    raster: RasterCurveExtraction,
    plan: RasterEvaluationPlan,
    comparison: RasterBundleComparison,
) -> None:
    expected = compare_raster_to_all_branches(bundle, plan, raster)
    if _comparison_metadata(comparison) != _comparison_metadata(expected):
        raise ValueError("comparison metadata does not derive from bundle, plan, and raster")
    for supplied, derived in zip(comparison.branch_metrics, expected.branch_metrics, strict=True):
        _require_array_equal(
            supplied.branch_y_at_raster_x,
            derived.branch_y_at_raster_x,
            name=f"comparison {supplied.terminal_id} interpolated values",
        )
        _require_array_equal(
            supplied.residual,
            derived.residual,
            name=f"comparison {supplied.terminal_id} residuals",
        )


def write_curve_workflow_artifacts(
    bundle: ExactSavedGridCurveBundle,
    output_directory: str | Path,
    *,
    raster: RasterCurveExtraction | None = None,
    evaluation_plan: RasterEvaluationPlan | None = None,
    comparison: RasterBundleComparison | None = None,
) -> CurveWorkflowArtifactPaths:
    """Write one fixed-name JSON/NPZ/CSV workflow without authority claims."""

    if comparison is not None:
        if raster is None:
            raster = comparison.raster
        elif raster is not comparison.raster:
            raise ValueError("comparison must reference the supplied raster object")
        if evaluation_plan is None:
            evaluation_plan = comparison.plan
        elif evaluation_plan is not comparison.plan:
            raise ValueError("comparison must reference the supplied evaluation-plan object")
    if evaluation_plan is not None and evaluation_plan.compute_bundle_fingerprint != bundle.bundle_fingerprint:
        raise ValueError("evaluation plan is bound to a different compute bundle")
    if raster is not None and evaluation_plan is not None and (
        raster.source != evaluation_plan.expected_source
        or raster.calibration != evaluation_plan.calibration
        or raster.policy != evaluation_plan.extraction_policy
    ):
        raise ValueError("raster does not match the supplied evaluation-plan bindings")
    if comparison is not None:
        assert raster is not None and evaluation_plan is not None
        _validate_comparison_identity(bundle, raster, evaluation_plan, comparison)

    metadata = _bundle_metadata(bundle)
    if raster is not None:
        metadata["raster"] = _raster_metadata(raster, evaluation_plan)
    if evaluation_plan is not None:
        metadata["evaluation_plan"] = _plan_metadata(evaluation_plan)
    if comparison is not None:
        metadata["comparison"] = _comparison_metadata(comparison)
    arrays = _array_payload(bundle, raster, comparison)
    csv_text = _curve_csv(bundle)
    metadata["array_sha256"] = {
        key: canonical_array_sha256(value) for key, value in sorted(arrays.items())
    }
    metadata["curves_csv_sha256"] = sha256(csv_text.encode("utf-8")).hexdigest()

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    return CurveWorkflowArtifactPaths(
        metadata_json=write_json_artifact(metadata, root / METADATA_FILENAME),
        arrays_npz=write_npz_artifact(arrays, root / ARRAYS_FILENAME),
        curves_csv=write_text_artifact(csv_text, root / CURVES_CSV_FILENAME),
    )


def _load_bundle(metadata: dict[str, object], payload: np.lib.npyio.NpzFile) -> ExactSavedGridCurveBundle:
    authority_record = metadata.get("source_authority")
    if not isinstance(authority_record, dict):
        raise ValueError("artifact must contain source-authority metadata")
    canonical_payload_json = authority_record.get("canonical_payload_json")
    if not isinstance(canonical_payload_json, str):
        raise ValueError("source-authority canonical payload must be a string")
    source_authority = make_source_authority_receipt(
        authority_record.get("authority_id"),
        json.loads(canonical_payload_json),
    )
    _require_equal(
        _source_authority_metadata(source_authority),
        authority_record,
        name="source authority",
    )
    authority_arrays = {
        "source_authority_authority_id_utf8": source_authority.authority_id.encode("utf-8"),
        "source_authority_canonical_payload_json_utf8": source_authority.canonical_payload_json.encode("utf-8"),
        "source_authority_payload_sha256_ascii": source_authority.payload_sha256.encode("ascii"),
    }
    for key, expected in authority_arrays.items():
        if key not in payload.files:
            raise ValueError(f"artifact NPZ omits {key}")
        _require_equal(bytes(np.asarray(payload[key], dtype=np.uint8)), expected, name=key)

    curve_records = metadata.get("curves")
    if not isinstance(curve_records, list) or not curve_records:
        raise ValueError("artifact must contain nonempty curve metadata")
    curves: list[ExactSavedGridCurve] = []
    for record in curve_records:
        if not isinstance(record, dict):
            raise ValueError("curve metadata records must be mappings")
        prefix = str(record["array_prefix"])
        grid_record = record["saved_grid"]
        if not isinstance(grid_record, dict):
            raise ValueError("saved-grid metadata must be a mapping")
        grid = SavedGridReceipt(
            source_id=grid_record["source_id"],
            point_indices=payload[f"{prefix}_point_indices"],
            x=payload[f"{prefix}_x"],
            x_units=grid_record["x_units"],
            domain=CurveDomainReceipt(**grid_record["domain"]),
        )
        _require_equal(grid.point_indices_sha256, grid_record["point_indices_sha256"], name=f"{prefix} point-index hash")
        _require_equal(grid.x_sha256, grid_record["x_sha256"], name=f"{prefix} x hash")
        curve = ExactSavedGridCurve(
            terminal_id=record["terminal_id"],
            terminal_payload_sha256=record["terminal_payload_sha256"],
            saved_grid=grid,
            observable=ObservableReceipt(**record["observable"]),
            value_transform=ValueTransformReceipt(**record["value_transform"]),
            raw_y=payload[f"{prefix}_raw_y"],
            output_y=payload[f"{prefix}_output_y"],
        )
        _require_equal(curve.raw_y_sha256, record["raw_y_sha256"], name=f"{prefix} raw hash")
        _require_equal(curve.output_y_sha256, record["output_y_sha256"], name=f"{prefix} output hash")
        curves.append(curve)

    closure_record = metadata["branch_closure"]
    if not isinstance(closure_record, dict):
        raise ValueError("branch closure metadata must be a mapping")
    nodes = tuple(DiscreteBranchNode(**record) for record in closure_record["nodes"])
    enumeration = EnumerationReceipt(**closure_record["enumeration_receipt"])
    closure = certify_enumerated_branch_closure(nodes, enumeration)
    _require_equal(closure.root_id, closure_record["root_id"], name="closure root")
    _require_equal(closure.terminal_ids, tuple(closure_record["terminal_ids"]), name="closure terminals")
    _require_equal(closure.computed_terminal_ids, tuple(closure_record["computed_terminal_ids"]), name="closure computed terminals")
    _require_equal(closure.rejected_terminal_ids, tuple(closure_record["rejected_terminal_ids"]), name="closure rejected terminals")
    _require_equal(closure.authority, closure_record["authority"], name="closure authority")
    _require_equal(
        closure.supplied_finite_tree_structurally_resolved,
        closure_record["supplied_finite_tree_structurally_resolved"],
        name="closure structural status",
    )
    compute_record = dict(metadata["compute_certificate"])
    compute_record["computed_terminal_ids"] = tuple(compute_record["computed_terminal_ids"])
    bundle = ExactSavedGridCurveBundle(
        curves=tuple(curves),
        branch_closure=closure,
        compute_certificate=ComputeCertificate(**compute_record),
        source_authority=source_authority,
    )
    _require_equal(bundle.bundle_fingerprint, metadata["bundle_fingerprint"], name="bundle fingerprint")
    return bundle


def _validate_csv(csv_bytes: bytes, bundle: ExactSavedGridCurveBundle) -> None:
    if b"\r" in csv_bytes or not csv_bytes.endswith(b"\n"):
        raise ValueError("curves CSV must use LF endings and end with LF")
    text = csv_bytes.decode("utf-8")
    reader = csv.reader(StringIO(text, newline=""))
    rows = list(reader)
    expected_header = [
        "terminal_id",
        "point_index",
        "x",
        "raw_y",
        "output_y",
        "x_units",
        "input_units",
        "output_units",
        "source_authority_id",
        "source_authority_payload_sha256",
        "source_authority_canonical_payload_json",
    ]
    if not rows or rows[0] != expected_header:
        raise ValueError("curves CSV header does not match the workflow schema")
    expected_rows: list[tuple[object, ...]] = []
    for curve in bundle.curves:
        for point_index, x, raw_y, output_y in zip(
            curve.saved_grid.point_indices,
            curve.saved_grid.x,
            curve.raw_y,
            curve.output_y,
            strict=True,
        ):
            expected_rows.append(
                (
                    curve.terminal_id,
                    int(point_index),
                    float(x),
                    float(raw_y),
                    float(output_y),
                    curve.saved_grid.x_units,
                    curve.value_transform.input_units,
                    curve.value_transform.output_units,
                    bundle.source_authority.authority_id,
                    bundle.source_authority.payload_sha256,
                    bundle.source_authority.canonical_payload_json,
                )
            )
    if len(rows) - 1 != len(expected_rows):
        raise ValueError("curves CSV row count does not match NPZ curves")
    for row_index, (row, expected) in enumerate(zip(rows[1:], expected_rows, strict=True), start=2):
        if len(row) != len(expected_header):
            raise ValueError(f"curves CSV row {row_index} has the wrong field count")
        parsed = (
            row[0],
            int(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
        )
        if parsed != expected:
            raise ValueError(f"curves CSV row {row_index} does not match NPZ curve evidence")


def _load_raster(
    record: dict[str, object],
    payload: np.lib.npyio.NpzFile,
    *,
    plan_present: bool,
) -> RasterCurveExtraction:
    expected_binding = "plan_bound" if plan_present else "unbound"
    _require_equal(record.get("evaluation_binding"), expected_binding, name="raster evaluation binding")
    source_receipt = RasterSourceReceipt(**record["source"])
    if bool(record["source_bytes_stored"]):
        if "raster_source_bytes" not in payload.files:
            raise ValueError("raster metadata declares source bytes but NPZ payload omits them")
        source: object = bytes(np.asarray(payload["raster_source_bytes"], dtype=np.uint8))
    else:
        source = Image.fromarray(np.asarray(payload["raster_source_rgba"], dtype=np.uint8), mode="RGBA")
    derived = extract_raster_curve(
        source,
        RasterAxesCalibration(**record["calibration"]),
        RasterExtractionPolicy(**record["policy"]),
        expected_source=source_receipt,
    )
    for name in (
        "component_pixel_x",
        "component_pixel_y",
        "centerline_pixel_x",
        "centerline_pixel_y",
        "x",
        "y",
        "pixel_uncertainty_y",
        "line_thickness_uncertainty_y",
        "total_uncertainty_y",
        "boundary_flags",
        "source_rgba",
    ):
        _require_array_equal(getattr(derived, name), payload[f"raster_{name}"], name=f"raster {name}")
    _require_equal(derived.component_label, record["component_label"], name="raster component label")
    _require_equal(derived.component_pixel_sha256, record["component_pixel_sha256"], name="raster component hash")
    _require_equal(derived.closed_dark_frame_detected, record["closed_dark_frame_detected"], name="raster frame detection")
    _require_equal(int(np.count_nonzero(derived.boundary_flags)), record["boundary_flag_count"], name="raster boundary count")
    return derived


def _load_plan(record: dict[str, object], bundle: ExactSavedGridCurveBundle) -> RasterEvaluationPlan:
    criterion_record = record["criterion"]
    criterion = None if criterion_record is None else CurveAgreementCriterion(**criterion_record)
    plan = create_raster_evaluation_plan(
        bundle,
        expected_source=RasterSourceReceipt(**record["expected_source"]),
        calibration=RasterAxesCalibration(**record["calibration"]),
        extraction_policy=RasterExtractionPolicy(**record["extraction_policy"]),
        value_kind=record["value_kind"],
        criterion=criterion,
        selection_lineage=SelectionLineageReceipt(**record["selection_lineage"]),
        preregistration_evidence_sha256=record["preregistration_evidence_sha256"],
        expected_closed_frame_present=record["expected_closed_frame_present"],
    )
    _require_equal(_plan_metadata(plan), record, name="re-derived evaluation plan")
    return plan


def _validate_loaded_comparison(
    record: dict[str, object],
    payload: np.lib.npyio.NpzFile,
    bundle: ExactSavedGridCurveBundle,
    raster: RasterCurveExtraction,
    plan: RasterEvaluationPlan,
) -> RasterBundleComparison:
    comparison = compare_raster_to_all_branches(bundle, plan, raster)
    _require_equal(_comparison_metadata(comparison), record, name="re-derived comparison")
    for index, item in enumerate(comparison.branch_metrics):
        prefix = f"comparison_{index:04d}"
        _require_array_equal(
            item.branch_y_at_raster_x,
            payload[f"{prefix}_branch_y_at_raster_x"],
            name=f"{prefix} interpolated values",
        )
        _require_array_equal(item.residual, payload[f"{prefix}_residual"], name=f"{prefix} residual")
    return comparison


def load_curve_workflow_artifacts(output_directory: str | Path) -> LoadedCurveWorkflow:
    """Reconstruct and independently revalidate every workflow layer present."""

    root = Path(output_directory)
    paths = CurveWorkflowArtifactPaths(
        metadata_json=root / METADATA_FILENAME,
        arrays_npz=root / ARRAYS_FILENAME,
        curves_csv=root / CURVES_CSV_FILENAME,
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("schema") != "mean_field.curve_workflow.v3":
        raise ValueError("unsupported curve-workflow artifact schema")
    csv_bytes = paths.curves_csv.read_bytes()
    _require_equal(sha256(csv_bytes).hexdigest(), metadata.get("curves_csv_sha256"), name="curves CSV hash")
    expected_array_hashes = metadata.get("array_sha256")
    if not isinstance(expected_array_hashes, dict) or not expected_array_hashes:
        raise ValueError("artifact must contain array hashes")

    with np.load(paths.arrays_npz, allow_pickle=False) as payload:
        _require_equal(tuple(sorted(payload.files)), tuple(sorted(expected_array_hashes)), name="NPZ array-key inventory")
        for key, expected_hash in expected_array_hashes.items():
            _require_equal(canonical_array_sha256(payload[key]), expected_hash, name=f"{key} array hash")
        bundle = _load_bundle(metadata, payload)
        _validate_csv(csv_bytes, bundle)

        raster_record = metadata.get("raster")
        plan_record = metadata.get("evaluation_plan")
        raster = None if raster_record is None else _load_raster(
            raster_record,
            payload,
            plan_present=plan_record is not None,
        )
        plan = None if plan_record is None else _load_plan(plan_record, bundle)
        if raster is not None and plan is not None and (
            raster.source != plan.expected_source
            or raster.calibration != plan.calibration
            or raster.policy != plan.extraction_policy
        ):
            raise ValueError("loaded raster does not match evaluation-plan bindings")
        comparison_record = metadata.get("comparison")
        if comparison_record is not None:
            if raster is None or plan is None:
                raise ValueError("comparison artifacts require both raster and evaluation plan")
            comparison = _validate_loaded_comparison(
                comparison_record,
                payload,
                bundle,
                raster,
                plan,
            )
        else:
            comparison = None
    return LoadedCurveWorkflow(
        bundle=bundle,
        raster=raster,
        evaluation_plan=plan,
        comparison=comparison,
        paths=paths,
    )


def load_exact_grid_curve_bundle(output_directory: str | Path) -> ExactSavedGridCurveBundle:
    """Convenience bundle-only view backed by the full workflow revalidator."""

    return load_curve_workflow_artifacts(output_directory).bundle


__all__ = [
    "ARRAYS_FILENAME",
    "CURVES_CSV_FILENAME",
    "METADATA_FILENAME",
    "CurveWorkflowArtifactPaths",
    "LoadedCurveWorkflow",
    "load_curve_workflow_artifacts",
    "load_exact_grid_curve_bundle",
    "write_curve_workflow_artifacts",
]
