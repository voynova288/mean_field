from __future__ import annotations

"""Factory-derived raster evidence, immutable evaluation plans, and all-curve comparison."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy import ndimage

from .analysis import (
    CurveValueKind,
    ExactGridExtremum,
    ZeroCrossing,
    compatible_value_convention,
    exact_grid_local_extrema,
    exact_zero_piecewise_linear_crossings,
    explicit_piecewise_linear_interpolation,
)
from .contracts import (
    ExactSavedGridCurveBundle,
    canonical_array_sha256,
    canonical_json_sha256,
    immutable_finite_array,
)

RasterConnectivity = Literal[4, 8]
AgreementDecision = Literal["evidence_only", "criterion_satisfied", "criterion_not_satisfied"]

_EXTRACTION_TOKEN = object()
_PLAN_TOKEN = object()
_COMPARISON_TOKEN = object()


def _text(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be nonempty")
    return text


def _sha256_text(value: str, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _freeze_uint8(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != ndim or any(size <= 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-D uint8 array")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.uint8).reshape(contiguous.shape)


@dataclass(frozen=True)
class RasterSourceReceipt:
    """Digest and decoded dimensions of the exact raster source."""

    sha256: str
    width: int
    height: int
    mode: str
    hash_basis: Literal["source_bytes", "decoded_pixels"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", _sha256_text(self.sha256, name="RasterSourceReceipt.sha256"))
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("raster dimensions must be positive")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "mode", _text(self.mode, name="mode"))
        if self.hash_basis not in {"source_bytes", "decoded_pixels"}:
            raise ValueError(f"unsupported raster hash basis {self.hash_basis!r}")


@dataclass(frozen=True)
class RasterAxesCalibration:
    """Fixed affine mapping from pixel coordinates to data coordinates."""

    pixel_x_left: int
    pixel_x_right: int
    x_left: float
    x_right: float
    pixel_y_top: int
    pixel_y_bottom: int
    y_top: float
    y_bottom: float
    x_units: str
    y_units: str

    def __post_init__(self) -> None:
        for name in ("pixel_x_left", "pixel_x_right", "pixel_y_top", "pixel_y_bottom"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact int")
        if self.pixel_x_left < 0 or self.pixel_y_top < 0:
            raise ValueError("calibration pixel bounds must be nonnegative")
        if self.pixel_x_left >= self.pixel_x_right:
            raise ValueError("pixel_x_left must be less than pixel_x_right")
        if self.pixel_y_top >= self.pixel_y_bottom:
            raise ValueError("pixel_y_top must be less than pixel_y_bottom")
        for name in ("x_left", "x_right", "y_top", "y_bottom"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.x_left >= self.x_right:
            raise ValueError("open-interval raster extraction requires x_left < x_right")
        if self.y_top == self.y_bottom:
            raise ValueError("calibration y range must be nonzero")
        object.__setattr__(self, "x_units", _text(self.x_units, name="x_units"))
        object.__setattr__(self, "y_units", _text(self.y_units, name="y_units"))

    def pixel_x_to_data(self, pixel_x: object) -> np.ndarray:
        values = np.asarray(pixel_x, dtype=np.float64)
        fraction = (values - self.pixel_x_left) / (self.pixel_x_right - self.pixel_x_left)
        return self.x_left + fraction * (self.x_right - self.x_left)

    def pixel_y_to_data(self, pixel_y: object) -> np.ndarray:
        values = np.asarray(pixel_y, dtype=np.float64)
        fraction = (values - self.pixel_y_top) / (self.pixel_y_bottom - self.pixel_y_top)
        return self.y_top + fraction * (self.y_bottom - self.y_top)

    @property
    def y_units_per_pixel(self) -> float:
        return abs((self.y_bottom - self.y_top) / (self.pixel_y_bottom - self.pixel_y_top))


@dataclass(frozen=True)
class RasterExtractionPolicy:
    """Predeclared threshold, connectivity, frame, and component rules."""

    dark_threshold: int
    connectivity: RasterConnectivity = 8
    minimum_pixels: int = 3
    minimum_columns: int = 3
    minimum_column_fraction: float = 0.5
    auto_exclude_closed_dark_frame: bool = False
    closed_frame_edge_fraction: float = 0.8
    frame_interior_margin: int = 1

    def __post_init__(self) -> None:
        if type(self.dark_threshold) is not int or not 0 <= self.dark_threshold <= 255:
            raise ValueError("dark_threshold must be an exact int in [0, 255]")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        for name in ("minimum_pixels", "minimum_columns"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact int")
        if type(self.frame_interior_margin) is not int or self.frame_interior_margin < 0:
            raise ValueError("frame_interior_margin must be a nonnegative exact int")
        for name in ("minimum_column_fraction", "closed_frame_edge_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if type(self.auto_exclude_closed_dark_frame) is not bool:
            raise TypeError("auto_exclude_closed_dark_frame must be bool")


@dataclass(frozen=True, init=False)
class RasterCurveExtraction:
    """Factory-derived component pixels, centerline, calibration, and uncertainty."""

    source: RasterSourceReceipt
    calibration: RasterAxesCalibration
    policy: RasterExtractionPolicy
    component_label: int
    component_pixel_x: np.ndarray
    component_pixel_y: np.ndarray
    centerline_pixel_x: np.ndarray
    centerline_pixel_y: np.ndarray
    x: np.ndarray
    y: np.ndarray
    pixel_uncertainty_y: np.ndarray
    line_thickness_uncertainty_y: np.ndarray
    total_uncertainty_y: np.ndarray
    boundary_flags: np.ndarray
    closed_dark_frame_detected: bool
    source_rgba: np.ndarray
    source_bytes: bytes | None
    component_pixel_sha256: str = field(init=False)

    def __init__(
        self,
        *,
        _token: object,
        source: RasterSourceReceipt,
        calibration: RasterAxesCalibration,
        policy: RasterExtractionPolicy,
        component_label: int,
        component_pixel_x: object,
        component_pixel_y: object,
        centerline_pixel_x: object,
        centerline_pixel_y: object,
        x: object,
        y: object,
        pixel_uncertainty_y: object,
        line_thickness_uncertainty_y: object,
        total_uncertainty_y: object,
        boundary_flags: object,
        closed_dark_frame_detected: bool,
        source_rgba: object,
        source_bytes: bytes | None,
    ) -> None:
        if _token is not _EXTRACTION_TOKEN:
            raise TypeError("RasterCurveExtraction is factory-only; use extract_raster_curve")
        if not isinstance(source, RasterSourceReceipt):
            raise TypeError("source must be a RasterSourceReceipt")
        if not isinstance(calibration, RasterAxesCalibration) or not isinstance(policy, RasterExtractionPolicy):
            raise TypeError("calibration and policy must use raster contract types")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "policy", policy)
        if type(component_label) is not int or component_label <= 0:
            raise ValueError("component_label must be a positive exact int")
        object.__setattr__(self, "component_label", component_label)
        specs = {
            "component_pixel_x": (component_pixel_x, np.dtype("<i8")),
            "component_pixel_y": (component_pixel_y, np.dtype("<i8")),
            "centerline_pixel_x": (centerline_pixel_x, np.dtype("<i8")),
            "centerline_pixel_y": (centerline_pixel_y, np.dtype("<f8")),
            "x": (x, np.dtype("<f8")),
            "y": (y, np.dtype("<f8")),
            "pixel_uncertainty_y": (pixel_uncertainty_y, np.dtype("<f8")),
            "line_thickness_uncertainty_y": (line_thickness_uncertainty_y, np.dtype("<f8")),
            "total_uncertainty_y": (total_uncertainty_y, np.dtype("<f8")),
            "boundary_flags": (boundary_flags, np.dtype("|b1")),
        }
        for name, (value, dtype) in specs.items():
            object.__setattr__(self, name, immutable_finite_array(value, dtype=dtype, name=name))
        if self.component_pixel_x.shape != self.component_pixel_y.shape:
            raise ValueError("component pixel x/y arrays must match")
        center_shape = self.centerline_pixel_x.shape
        for name in (
            "centerline_pixel_y",
            "x",
            "y",
            "pixel_uncertainty_y",
            "line_thickness_uncertainty_y",
            "total_uncertainty_y",
            "boundary_flags",
        ):
            if getattr(self, name).shape != center_shape:
                raise ValueError(f"{name} must match the centerline shape")
        if self.x.size > 1 and np.any(np.diff(self.x) <= 0.0):
            raise ValueError("raster centerline x must be strictly increasing")
        if not np.array_equal(self.total_uncertainty_y, self.pixel_uncertainty_y + self.line_thickness_uncertainty_y):
            raise ValueError("total uncertainty must equal pixel plus line-thickness uncertainty")
        if type(closed_dark_frame_detected) is not bool:
            raise TypeError("closed_dark_frame_detected must be bool")
        object.__setattr__(self, "closed_dark_frame_detected", closed_dark_frame_detected)
        rgba = _freeze_uint8(source_rgba, name="source_rgba", ndim=3)
        if rgba.shape != (source.height, source.width, 4):
            raise ValueError("source_rgba shape does not match source receipt")
        object.__setattr__(self, "source_rgba", rgba)
        payload = None if source_bytes is None else bytes(source_bytes)
        if source.hash_basis == "source_bytes":
            if not payload or sha256(payload).hexdigest() != source.sha256:
                raise ValueError("stored source bytes do not match raster source receipt")
            with Image.open(BytesIO(payload)) as image:
                decoded = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            if not np.array_equal(decoded, rgba):
                raise ValueError("stored decoded pixels do not derive from stored source bytes")
        else:
            if payload is not None:
                raise ValueError("decoded-pixel receipts must not carry source bytes")
            if canonical_array_sha256(rgba) != source.sha256:
                raise ValueError("stored decoded pixels do not match raster source receipt")
        object.__setattr__(self, "source_bytes", payload)
        pixels = np.column_stack((self.component_pixel_x, self.component_pixel_y))
        object.__setattr__(self, "component_pixel_sha256", canonical_array_sha256(pixels))


@dataclass(frozen=True)
class CurveAgreementCriterion:
    """Optional preregistered numerical thresholds, checked for every curve."""

    maximum_rmse: float | None = None
    maximum_mae: float | None = None
    maximum_absolute_error: float | None = None
    maximum_absolute_mean_error: float | None = None

    def __post_init__(self) -> None:
        names = ("maximum_rmse", "maximum_mae", "maximum_absolute_error", "maximum_absolute_mean_error")
        if all(getattr(self, name) is None for name in names):
            raise ValueError("an agreement criterion must declare at least one threshold")
        for name in names:
            raw = getattr(self, name)
            if raw is None:
                continue
            value = float(raw)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SelectionLineageReceipt:
    """Plan-local selection and preregistration facts, never a compute receipt."""

    solver_target_isolated: bool
    raster_evaluation_postfreeze: bool
    contract_preregistered_before_final_run: bool
    contract_selected_blind_to_prior_target_comparison: bool | None

    def __post_init__(self) -> None:
        for name in ("solver_target_isolated", "raster_evaluation_postfreeze", "contract_preregistered_before_final_run"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        blind = self.contract_selected_blind_to_prior_target_comparison
        if blind is not None and type(blind) is not bool:
            raise TypeError("contract_selected_blind_to_prior_target_comparison must be bool or None")

    @property
    def criterion_lineage_satisfied(self) -> bool:
        return (
            self.solver_target_isolated
            and self.raster_evaluation_postfreeze
            and self.contract_preregistered_before_final_run
            and self.contract_selected_blind_to_prior_target_comparison is True
        )


def _evaluation_plan_payload(plan: object) -> dict[str, object]:
    return {
        "compute_bundle_fingerprint": getattr(plan, "compute_bundle_fingerprint"),
        "expected_source": asdict(getattr(plan, "expected_source")),
        "calibration": asdict(getattr(plan, "calibration")),
        "extraction_policy": asdict(getattr(plan, "extraction_policy")),
        "value_kind": getattr(plan, "value_kind"),
        "value_units": getattr(plan, "value_units"),
        "value_semantics": getattr(plan, "value_semantics"),
        "transforms_identical": getattr(plan, "transforms_identical"),
        "criterion": (
            None
            if getattr(plan, "criterion") is None
            else asdict(getattr(plan, "criterion"))
        ),
        "selection_lineage": asdict(getattr(plan, "selection_lineage")),
        "preregistration_evidence_sha256": getattr(plan, "preregistration_evidence_sha256"),
        "expected_closed_frame_present": getattr(plan, "expected_closed_frame_present"),
    }


def _evaluation_plan_fingerprint(plan: object) -> str:
    return canonical_json_sha256(
        _evaluation_plan_payload(plan),
        namespace="mean_field.raster_evaluation_plan.v1",
    )


@dataclass(frozen=True, init=False)
class RasterEvaluationPlan:
    """Factory-created immutable binding of compute, target, policy, and criterion."""

    compute_bundle_fingerprint: str
    expected_source: RasterSourceReceipt
    calibration: RasterAxesCalibration
    extraction_policy: RasterExtractionPolicy
    value_kind: CurveValueKind
    value_units: str
    value_semantics: str
    transforms_identical: bool
    criterion: CurveAgreementCriterion | None
    selection_lineage: SelectionLineageReceipt
    preregistration_evidence_sha256: str | None
    expected_closed_frame_present: bool | None
    plan_fingerprint: str = field(init=False)

    def __init__(
        self,
        *,
        _token: object,
        compute_bundle_fingerprint: str,
        expected_source: RasterSourceReceipt,
        calibration: RasterAxesCalibration,
        extraction_policy: RasterExtractionPolicy,
        value_kind: CurveValueKind,
        value_units: str,
        value_semantics: str,
        transforms_identical: bool,
        criterion: CurveAgreementCriterion | None,
        selection_lineage: SelectionLineageReceipt,
        preregistration_evidence_sha256: str | None,
        expected_closed_frame_present: bool | None,
    ) -> None:
        if _token is not _PLAN_TOKEN:
            raise TypeError("RasterEvaluationPlan is factory-only; use create_raster_evaluation_plan")
        object.__setattr__(self, "compute_bundle_fingerprint", _sha256_text(compute_bundle_fingerprint, name="compute_bundle_fingerprint"))
        if not isinstance(expected_source, RasterSourceReceipt):
            raise TypeError("expected_source must be a RasterSourceReceipt")
        if not isinstance(calibration, RasterAxesCalibration) or not isinstance(extraction_policy, RasterExtractionPolicy):
            raise TypeError("calibration and extraction_policy must use raster contract types")
        if value_kind not in {"raw", "output"}:
            raise ValueError(f"unsupported value_kind {value_kind!r}")
        if type(transforms_identical) is not bool:
            raise TypeError("transforms_identical must be bool")
        if criterion is not None and not isinstance(criterion, CurveAgreementCriterion):
            raise TypeError("criterion must be CurveAgreementCriterion or None")
        if not isinstance(selection_lineage, SelectionLineageReceipt):
            raise TypeError("selection_lineage must be a SelectionLineageReceipt")
        prereg = None if preregistration_evidence_sha256 is None else _sha256_text(
            preregistration_evidence_sha256,
            name="preregistration_evidence_sha256",
        )
        if expected_closed_frame_present is not None and type(expected_closed_frame_present) is not bool:
            raise TypeError("expected_closed_frame_present must be bool or None")
        object.__setattr__(self, "expected_source", expected_source)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "extraction_policy", extraction_policy)
        object.__setattr__(self, "value_kind", value_kind)
        object.__setattr__(self, "value_units", _text(value_units, name="value_units"))
        object.__setattr__(self, "value_semantics", _text(value_semantics, name="value_semantics"))
        object.__setattr__(self, "transforms_identical", transforms_identical)
        object.__setattr__(self, "criterion", criterion)
        object.__setattr__(self, "selection_lineage", selection_lineage)
        object.__setattr__(self, "preregistration_evidence_sha256", prereg)
        object.__setattr__(self, "expected_closed_frame_present", expected_closed_frame_present)
        object.__setattr__(self, "plan_fingerprint", _evaluation_plan_fingerprint(self))

    @property
    def criterion_eligible(self) -> bool:
        return (
            self.criterion is not None
            and self.preregistration_evidence_sha256 is not None
            and self.selection_lineage.criterion_lineage_satisfied
        )


def create_raster_evaluation_plan(
    bundle: ExactSavedGridCurveBundle,
    *,
    expected_source: RasterSourceReceipt,
    calibration: RasterAxesCalibration,
    extraction_policy: RasterExtractionPolicy,
    value_kind: CurveValueKind = "output",
    criterion: CurveAgreementCriterion | None = None,
    selection_lineage: SelectionLineageReceipt,
    preregistration_evidence_sha256: str | None = None,
    expected_closed_frame_present: bool | None = None,
) -> RasterEvaluationPlan:
    """Bind a plan-bound raster evaluation; held-out status comes from lineage."""

    bundle.domain.require_open_interval(operation="raster evaluation planning")
    value_units, value_semantics, identical = compatible_value_convention(bundle, value_kind)
    if calibration.x_units != bundle.curves[0].saved_grid.x_units:
        raise ValueError("plan calibration and exact-grid x units do not match")
    if calibration.y_units != value_units:
        raise ValueError("plan calibration y units do not match the selected value convention")
    return RasterEvaluationPlan(
        _token=_PLAN_TOKEN,
        compute_bundle_fingerprint=bundle.bundle_fingerprint,
        expected_source=expected_source,
        calibration=calibration,
        extraction_policy=extraction_policy,
        value_kind=value_kind,
        value_units=value_units,
        value_semantics=value_semantics,
        transforms_identical=identical,
        criterion=criterion,
        selection_lineage=selection_lineage,
        preregistration_evidence_sha256=preregistration_evidence_sha256,
        expected_closed_frame_present=expected_closed_frame_present,
    )


@dataclass(frozen=True)
class BranchRasterMetrics:
    """Raster residual metrics and exact-grid feature evidence for one curve."""

    terminal_id: str
    rmse: float
    mae: float
    maximum_absolute_error: float
    mean_error: float
    branch_y_at_raster_x: np.ndarray
    residual: np.ndarray
    crossings: tuple[ZeroCrossing, ...]
    exact_grid_extrema: tuple[ExactGridExtremum, ...]
    threshold_checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal_id", _text(self.terminal_id, name="terminal_id"))
        for name in ("rmse", "mae", "maximum_absolute_error", "mean_error"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        branch = immutable_finite_array(self.branch_y_at_raster_x, dtype=np.dtype("<f8"), name="branch_y_at_raster_x")
        residual = immutable_finite_array(self.residual, dtype=np.dtype("<f8"), name="residual")
        if branch.shape != residual.shape:
            raise ValueError("interpolated curve and residual shapes must match")
        object.__setattr__(self, "branch_y_at_raster_x", branch)
        object.__setattr__(self, "residual", residual)
        expected_metrics = {
            "rmse": float(np.sqrt(np.mean(residual * residual))),
            "mae": float(np.mean(np.abs(residual))),
            "maximum_absolute_error": float(np.max(np.abs(residual))),
            "mean_error": float(np.mean(residual)),
        }
        tolerance = 32.0 * np.finfo(np.float64).eps
        for name, expected in expected_metrics.items():
            if not np.isclose(getattr(self, name), expected, atol=tolerance, rtol=tolerance):
                raise ValueError(f"{name} does not match the residual array")
        checks: list[tuple[str, bool]] = []
        for name, passed in self.threshold_checks:
            if type(passed) is not bool:
                raise TypeError("threshold check outcomes must be exact bool values")
            checks.append((_text(name, name="threshold check name"), passed))
        if len({name for name, _passed in checks}) != len(checks):
            raise ValueError("threshold check names must be unique")
        object.__setattr__(self, "threshold_checks", tuple(checks))


@dataclass(frozen=True, init=False)
class RasterBundleComparison:
    """Factory-only plan/bundle/raster comparison with no branch ranking."""

    plan: RasterEvaluationPlan
    raster: RasterCurveExtraction
    branch_metrics: tuple[BranchRasterMetrics, ...]
    raster_crossings: tuple[ZeroCrossing, ...]
    decision: AgreementDecision
    transforms_identical: bool

    def __init__(
        self,
        *,
        _token: object,
        bundle: ExactSavedGridCurveBundle,
        plan: RasterEvaluationPlan,
        raster: RasterCurveExtraction,
        branch_metrics: tuple[BranchRasterMetrics, ...],
        raster_crossings: tuple[ZeroCrossing, ...],
        decision: AgreementDecision,
        transforms_identical: bool,
    ) -> None:
        if _token is not _COMPARISON_TOKEN:
            raise TypeError(
                "RasterBundleComparison is factory-only; use compare_raster_to_all_branches"
            )
        if not isinstance(bundle, ExactSavedGridCurveBundle):
            raise TypeError("bundle must be an ExactSavedGridCurveBundle")
        if not isinstance(plan, RasterEvaluationPlan):
            raise TypeError("plan must be a RasterEvaluationPlan")
        if plan.plan_fingerprint != _evaluation_plan_fingerprint(plan):
            raise ValueError("evaluation-plan fingerprint does not match its bound fields")
        if not isinstance(raster, RasterCurveExtraction):
            raise TypeError("raster must be a RasterCurveExtraction")
        if plan.compute_bundle_fingerprint != bundle.bundle_fingerprint:
            raise ValueError("comparison plan is bound to a different compute bundle")
        if raster.source != plan.expected_source:
            raise ValueError("raster source does not equal the plan expected source")
        if raster.calibration != plan.calibration:
            raise ValueError("raster calibration does not equal the plan calibration")
        if raster.policy != plan.extraction_policy:
            raise ValueError("raster policy does not equal the plan extraction policy")
        if (
            plan.expected_closed_frame_present is not None
            and raster.closed_dark_frame_detected
            != plan.expected_closed_frame_present
        ):
            raise ValueError("raster does not satisfy the plan expected frame condition")

        metrics = tuple(branch_metrics)
        if not metrics or any(not isinstance(item, BranchRasterMetrics) for item in metrics):
            raise TypeError("branch_metrics must contain BranchRasterMetrics records")
        expected_ids = bundle.branch_closure.computed_terminal_ids
        if tuple(item.terminal_id for item in metrics) != expected_ids:
            raise ValueError("comparison terminal IDs must equal every computed terminal ID")
        expected_metrics = _derive_branch_metrics(bundle, plan, raster)
        if not _branch_metric_sequences_equal(metrics, expected_metrics):
            raise ValueError("comparison branch metrics are not factory-derived")
        expected_crossings = exact_zero_piecewise_linear_crossings(raster.x, raster.y)
        if tuple(raster_crossings) != expected_crossings:
            raise ValueError("raster crossings are not factory-derived")
        expected_decision = _comparison_decision(plan, raster, expected_metrics)
        if decision != expected_decision:
            raise ValueError("comparison decision is not factory-derived")
        if type(transforms_identical) is not bool or transforms_identical != plan.transforms_identical:
            raise ValueError("comparison transform identity is not factory-derived")

        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "raster", raster)
        object.__setattr__(self, "branch_metrics", metrics)
        object.__setattr__(self, "raster_crossings", expected_crossings)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "transforms_identical", transforms_identical)

    @property
    def value_kind(self) -> CurveValueKind:
        return self.plan.value_kind

    @property
    def criterion(self) -> CurveAgreementCriterion | None:
        return self.plan.criterion


@dataclass(frozen=True)
class _LoadedRasterSource:
    image: Image.Image
    receipt: RasterSourceReceipt
    source_rgba: np.ndarray
    source_bytes: bytes | None


def _load_raster_source(
    source: str | Path | bytes | bytearray | Image.Image | np.ndarray,
) -> _LoadedRasterSource:
    payload: bytes | None = None
    if isinstance(source, (str, Path)):
        payload = Path(source).read_bytes()
        image = Image.open(BytesIO(payload))
        image.load()
    elif isinstance(source, (bytes, bytearray)):
        payload = bytes(source)
        image = Image.open(BytesIO(payload))
        image.load()
    elif isinstance(source, Image.Image):
        image = source.copy()
    else:
        pixels = np.asarray(source)
        if pixels.dtype != np.uint8:
            raise TypeError("raster NumPy arrays must have uint8 dtype")
        image = Image.fromarray(pixels)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    receipt = RasterSourceReceipt(
        sha256=sha256(payload).hexdigest() if payload is not None else canonical_array_sha256(rgba),
        width=image.width,
        height=image.height,
        mode=image.mode if payload is not None else "RGBA",
        hash_basis="source_bytes" if payload is not None else "decoded_pixels",
    )
    return _LoadedRasterSource(image=image, receipt=receipt, source_rgba=rgba, source_bytes=payload)


def raster_source_receipt(
    source: str | Path | bytes | bytearray | Image.Image | np.ndarray,
) -> RasterSourceReceipt:
    """Return a digest receipt without retaining a mutable image object."""

    return _load_raster_source(source).receipt


def _grayscale_on_white(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float64)
    alpha = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.asarray(np.rint(luminance), dtype=np.uint8)


def _closed_frame_on_full_mask(
    mask: np.ndarray,
    calibration: RasterAxesCalibration,
    edge_fraction: float,
) -> bool:
    top = mask[calibration.pixel_y_top, calibration.pixel_x_left : calibration.pixel_x_right + 1]
    bottom = mask[calibration.pixel_y_bottom, calibration.pixel_x_left : calibration.pixel_x_right + 1]
    left = mask[calibration.pixel_y_top : calibration.pixel_y_bottom + 1, calibration.pixel_x_left]
    right = mask[calibration.pixel_y_top : calibration.pixel_y_bottom + 1, calibration.pixel_x_right]
    return all(float(np.mean(edge)) >= edge_fraction for edge in (top, bottom, left, right))


def detect_raster_frame_calibration(
    source: str | Path | bytes | bytearray | Image.Image | np.ndarray,
    *,
    x_left: float,
    x_right: float,
    y_top: float,
    y_bottom: float,
    x_units: str,
    y_units: str,
    dark_threshold: int,
    edge_fraction: float = 0.8,
) -> RasterAxesCalibration:
    """Auto-detect one closed dark frame and attach caller-supplied fixed axis bounds."""

    if type(dark_threshold) is not int or not 0 <= dark_threshold <= 255:
        raise ValueError("dark_threshold must be an exact int in [0, 255]")
    if not 0.0 <= float(edge_fraction) <= 1.0:
        raise ValueError("edge_fraction must lie in [0, 1]")
    loaded = _load_raster_source(source)
    mask = _grayscale_on_white(loaded.image) <= dark_threshold
    row_counts = np.count_nonzero(mask, axis=1)
    column_counts = np.count_nonzero(mask, axis=0)
    if int(np.max(row_counts)) == 0 or int(np.max(column_counts)) == 0:
        raise ValueError("could not auto-detect a unique closed dark frame")
    horizontal = np.flatnonzero(row_counts >= float(edge_fraction) * int(np.max(row_counts)))
    vertical = np.flatnonzero(column_counts >= float(edge_fraction) * int(np.max(column_counts)))
    if horizontal.size < 2 or vertical.size < 2:
        raise ValueError("could not auto-detect a unique closed dark frame")
    calibration = RasterAxesCalibration(
        pixel_x_left=int(vertical[0]),
        pixel_x_right=int(vertical[-1]),
        x_left=x_left,
        x_right=x_right,
        pixel_y_top=int(horizontal[0]),
        pixel_y_bottom=int(horizontal[-1]),
        y_top=y_top,
        y_bottom=y_bottom,
        x_units=x_units,
        y_units=y_units,
    )
    if not _closed_frame_on_full_mask(mask, calibration, float(edge_fraction)):
        raise ValueError("auto-detected bounds do not form a closed dark frame")
    return calibration


def extract_raster_curve(
    source: str | Path | bytes | bytearray | Image.Image | np.ndarray,
    calibration: RasterAxesCalibration,
    policy: RasterExtractionPolicy,
    *,
    expected_source: RasterSourceReceipt | None = None,
) -> RasterCurveExtraction:
    """Factory-derive the unique qualifying dark component and its centerline."""

    loaded = _load_raster_source(source)
    if expected_source is not None and loaded.receipt != expected_source:
        raise ValueError("raster source does not match the expected hash-bound receipt")
    if calibration.pixel_x_right >= loaded.image.width or calibration.pixel_y_bottom >= loaded.image.height:
        raise ValueError("raster calibration lies outside the decoded image")
    grayscale = _grayscale_on_white(loaded.image)
    mask = grayscale <= policy.dark_threshold
    closed_frame_detected = _closed_frame_on_full_mask(mask, calibration, policy.closed_frame_edge_fraction)

    label_left = calibration.pixel_x_left
    label_right = calibration.pixel_x_right
    label_top = calibration.pixel_y_top
    label_bottom = calibration.pixel_y_bottom
    if closed_frame_detected and policy.auto_exclude_closed_dark_frame:
        exclusion = policy.frame_interior_margin + 1
        label_left += exclusion
        label_right -= exclusion
        label_top += exclusion
        label_bottom -= exclusion
        if label_left > label_right or label_top > label_bottom:
            raise ValueError("frame exclusion and interior margin leave no labelable raster interior")

    bounded = np.zeros_like(mask, dtype=bool)
    bounded[label_top : label_bottom + 1, label_left : label_right + 1] = mask[
        label_top : label_bottom + 1,
        label_left : label_right + 1,
    ]
    structure = ndimage.generate_binary_structure(2, 1 if policy.connectivity == 4 else 2)
    labels, count = ndimage.label(bounded, structure=structure)
    qualifying: list[tuple[int, np.ndarray, np.ndarray]] = []
    available_columns = label_right - label_left + 1
    for label in range(1, count + 1):
        pixel_y, pixel_x = np.nonzero(labels == label)
        if pixel_x.size < policy.minimum_pixels:
            continue
        column_count = np.unique(pixel_x).size
        if column_count < policy.minimum_columns:
            continue
        if column_count / available_columns < policy.minimum_column_fraction:
            continue
        qualifying.append((label, pixel_x, pixel_y))
    if len(qualifying) != 1:
        raise ValueError(
            "raster extraction requires one unique qualifying component; "
            f"found {len(qualifying)}"
        )
    component_label, pixel_x, pixel_y = qualifying[0]
    center_x = np.unique(pixel_x)
    center_y = np.empty(center_x.shape, dtype=np.float64)
    half_span_pixels = np.empty(center_x.shape, dtype=np.float64)
    boundary_flags = np.zeros(center_x.shape, dtype=bool)
    for index, column in enumerate(center_x):
        rows = pixel_y[pixel_x == column]
        median = float(np.median(rows))
        center_y[index] = median
        half_span_pixels[index] = max(median - float(np.min(rows)), float(np.max(rows)) - median)
        boundary_flags[index] = bool(
            column in {label_left, label_right}
            or np.any(rows == label_top)
            or np.any(rows == label_bottom)
        )
    data_x = calibration.pixel_x_to_data(center_x)
    data_y = calibration.pixel_y_to_data(center_y)
    pixel_uncertainty = np.full(center_x.shape, 0.5 * calibration.y_units_per_pixel)
    line_uncertainty = half_span_pixels * calibration.y_units_per_pixel
    return RasterCurveExtraction(
        _token=_EXTRACTION_TOKEN,
        source=loaded.receipt,
        calibration=calibration,
        policy=policy,
        component_label=component_label,
        component_pixel_x=pixel_x,
        component_pixel_y=pixel_y,
        centerline_pixel_x=center_x,
        centerline_pixel_y=center_y,
        x=data_x,
        y=data_y,
        pixel_uncertainty_y=pixel_uncertainty,
        line_thickness_uncertainty_y=line_uncertainty,
        total_uncertainty_y=pixel_uncertainty + line_uncertainty,
        boundary_flags=boundary_flags,
        closed_dark_frame_detected=closed_frame_detected,
        source_rgba=loaded.source_rgba,
        source_bytes=loaded.source_bytes,
    )


def _criterion_checks(
    metrics: dict[str, float],
    criterion: CurveAgreementCriterion | None,
) -> tuple[tuple[str, bool], ...]:
    if criterion is None:
        return ()
    mapping = (
        ("maximum_rmse", "rmse"),
        ("maximum_mae", "mae"),
        ("maximum_absolute_error", "maximum_absolute_error"),
        ("maximum_absolute_mean_error", "absolute_mean_error"),
    )
    return tuple(
        (threshold_name, metrics[metric_name] <= float(getattr(criterion, threshold_name)))
        for threshold_name, metric_name in mapping
        if getattr(criterion, threshold_name) is not None
    )


def _comparison_decision(
    plan: RasterEvaluationPlan,
    raster: RasterCurveExtraction,
    metrics: tuple[BranchRasterMetrics, ...],
) -> AgreementDecision:
    if plan.criterion is None:
        return "evidence_only"
    frame_matches = (
        plan.expected_closed_frame_present is None
        or raster.closed_dark_frame_detected == plan.expected_closed_frame_present
    )
    checks_pass = all(all(passed for _name, passed in item.threshold_checks) for item in metrics)
    if (
        plan.criterion_eligible
        and frame_matches
        and not np.any(raster.boundary_flags)
        and checks_pass
    ):
        return "criterion_satisfied"
    return "criterion_not_satisfied"


def _derive_branch_metrics(
    bundle: ExactSavedGridCurveBundle,
    plan: RasterEvaluationPlan,
    raster: RasterCurveExtraction,
) -> tuple[BranchRasterMetrics, ...]:
    results: list[BranchRasterMetrics] = []
    for curve in bundle.curves:
        exact_y = curve.raw_y if plan.value_kind == "raw" else curve.output_y
        interpolated = explicit_piecewise_linear_interpolation(
            curve.saved_grid.x,
            exact_y,
            raster.x,
            domain=curve.saved_grid.domain,
        )
        residual = interpolated - raster.y
        values = {
            "rmse": float(np.sqrt(np.mean(residual * residual))),
            "mae": float(np.mean(np.abs(residual))),
            "maximum_absolute_error": float(np.max(np.abs(residual))),
            "mean_error": float(np.mean(residual)),
        }
        values["absolute_mean_error"] = abs(values["mean_error"])
        results.append(
            BranchRasterMetrics(
                terminal_id=curve.terminal_id,
                rmse=values["rmse"],
                mae=values["mae"],
                maximum_absolute_error=values["maximum_absolute_error"],
                mean_error=values["mean_error"],
                branch_y_at_raster_x=interpolated,
                residual=residual,
                crossings=exact_zero_piecewise_linear_crossings(
                    curve.saved_grid.x, exact_y
                ),
                exact_grid_extrema=exact_grid_local_extrema(
                    curve.saved_grid.point_indices,
                    curve.saved_grid.x,
                    exact_y,
                ),
                threshold_checks=_criterion_checks(values, plan.criterion),
            )
        )
    return tuple(results)


def _branch_metric_sequences_equal(
    actual: tuple[BranchRasterMetrics, ...],
    expected: tuple[BranchRasterMetrics, ...],
) -> bool:
    if len(actual) != len(expected):
        return False
    scalar_fields = (
        "terminal_id",
        "rmse",
        "mae",
        "maximum_absolute_error",
        "mean_error",
        "crossings",
        "exact_grid_extrema",
        "threshold_checks",
    )
    for supplied, derived in zip(actual, expected, strict=True):
        if any(getattr(supplied, name) != getattr(derived, name) for name in scalar_fields):
            return False
        if not np.array_equal(
            supplied.branch_y_at_raster_x, derived.branch_y_at_raster_x
        ) or not np.array_equal(supplied.residual, derived.residual):
            return False
    return True


def compare_raster_to_all_branches(
    bundle: ExactSavedGridCurveBundle,
    plan: RasterEvaluationPlan,
    raster: RasterCurveExtraction,
) -> RasterBundleComparison:
    """Verify every plan binding and compare the raster with every computed curve."""

    bundle.domain.require_open_interval(operation="raster comparison")
    if not isinstance(plan, RasterEvaluationPlan):
        raise TypeError("comparison requires a factory-created RasterEvaluationPlan")
    if plan.plan_fingerprint != _evaluation_plan_fingerprint(plan):
        raise ValueError("evaluation-plan fingerprint does not match its bound fields")
    if not isinstance(raster, RasterCurveExtraction):
        raise TypeError("comparison requires a factory-created RasterCurveExtraction")
    if plan.compute_bundle_fingerprint != bundle.bundle_fingerprint:
        raise ValueError("evaluation plan is bound to a different compute-bundle fingerprint")
    if raster.source != plan.expected_source:
        raise ValueError("raster extraction does not match the evaluation plan's expected source")
    if raster.calibration != plan.calibration:
        raise ValueError("raster extraction calibration does not match the evaluation plan")
    if raster.policy != plan.extraction_policy:
        raise ValueError("raster extraction policy does not match the evaluation plan")
    if (
        plan.expected_closed_frame_present is not None
        and raster.closed_dark_frame_detected != plan.expected_closed_frame_present
    ):
        raise ValueError("raster does not satisfy the plan expected frame condition")
    units, semantics, identical = compatible_value_convention(bundle, plan.value_kind)
    if (units, semantics, identical) != (plan.value_units, plan.value_semantics, plan.transforms_identical):
        raise ValueError("evaluation-plan value convention does not match the compute bundle")
    if plan.calibration.y_units != units:
        raise ValueError("raster calibration y units do not match the evaluation-plan value convention")
    metrics = _derive_branch_metrics(bundle, plan, raster)
    return RasterBundleComparison(
        _token=_COMPARISON_TOKEN,
        bundle=bundle,
        plan=plan,
        raster=raster,
        branch_metrics=metrics,
        raster_crossings=exact_zero_piecewise_linear_crossings(raster.x, raster.y),
        decision=_comparison_decision(plan, raster, metrics),
        transforms_identical=identical,
    )


__all__ = [
    "AgreementDecision",
    "BranchRasterMetrics",
    "CurveAgreementCriterion",
    "RasterAxesCalibration",
    "RasterBundleComparison",
    "RasterConnectivity",
    "RasterCurveExtraction",
    "RasterEvaluationPlan",
    "RasterExtractionPolicy",
    "RasterSourceReceipt",
    "SelectionLineageReceipt",
    "compare_raster_to_all_branches",
    "create_raster_evaluation_plan",
    "detect_raster_frame_calibration",
    "extract_raster_curve",
    "raster_source_receipt",
]
