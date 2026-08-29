from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json

import numpy as np
import pytest

from mean_field.core.curve_workflow import (
    CurveAgreementCriterion,
    CurveDomainReceipt,
    DiscreteBranchNode,
    EnumerationReceipt,
    ExactGridObservableEvaluation,
    ObservableReceipt,
    RasterAxesCalibration,
    RasterBundleComparison,
    RasterCurveExtraction,
    RasterEvaluationPlan,
    RasterExtractionPolicy,
    RasterSourceReceipt,
    SavedGridReceipt,
    SelectionLineageReceipt,
    SourceAuthorityReceipt,
    ValueTransformReceipt,
    all_branch_pointwise_spread,
    build_exact_grid_curve_bundle,
    canonical_array_sha256,
    center_at_exact_requested_x,
    certify_enumerated_branch_closure,
    compare_raster_to_all_branches,
    create_raster_evaluation_plan,
    detect_raster_frame_calibration,
    exact_grid_local_extrema,
    exact_zero_piecewise_linear_crossings,
    explicit_piecewise_linear_interpolation,
    extract_raster_curve,
    load_curve_workflow_artifacts,
    make_source_authority_receipt,
    write_curve_workflow_artifacts,
)
from mean_field.core.plotting import plot_exact_grid_curve_bundle


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _lineage(*, eligible: bool = True) -> SelectionLineageReceipt:
    return SelectionLineageReceipt(
        solver_target_isolated=eligible,
        raster_evaluation_postfreeze=eligible,
        contract_preregistered_before_final_run=eligible,
        contract_selected_blind_to_prior_target_comparison=eligible,
    )


def _tree(*terminal_ids: str, rejected: str | None = None) -> tuple[DiscreteBranchNode, ...]:
    children = terminal_ids + (() if rejected is None else (rejected,))
    nodes = [DiscreteBranchNode("root", None, children, "expanded")]
    nodes.extend(DiscreteBranchNode(value, "root") for value in terminal_ids)
    if rejected is not None:
        nodes.append(DiscreteBranchNode(rejected, "root", status="rejected", rejection_reason="invalid"))
    return tuple(nodes)


def _enumeration(*terminal_ids: str, frontier: int = 0) -> EnumerationReceipt:
    return EnumerationReceipt(
        algorithm_id="test-enumerator",
        algorithm_version="1",
        source_input_sha256=_digest("source-input"),
        choice_inventory_sha256=_digest("choice-inventory"),
        unconsumed_frontier_count=frontier,
        terminal_payload_hashes={terminal_id: _digest(f"payload:{terminal_id}") for terminal_id in terminal_ids},
        system_claims_exhaustive_enumeration=False,
    )


def _transform(
    *,
    input_units: str = "meV",
    output_units: str = "meV",
    scale: float = 1.0,
    offset: float = 0.0,
    semantics: str = "identity observable convention",
    common: bool = True,
) -> ValueTransformReceipt:
    return ValueTransformReceipt(
        input_units=input_units,
        output_units=output_units,
        scale=scale,
        offset=offset,
        semantics=semantics,
        common_across_branches=common,
    )


def _evaluation(
    terminal_id: str,
    raw_y: np.ndarray,
    *,
    x: np.ndarray | None = None,
    transform: ValueTransformReceipt | None = None,
    domain: CurveDomainReceipt | None = None,
    payload_digest: str | None = None,
) -> ExactGridObservableEvaluation:
    values = np.asarray(raw_y, dtype=float)
    if x is None:
        x = np.arange(values.size, dtype=float)
    transform = _transform() if transform is None else transform
    domain = CurveDomainReceipt("open_interval") if domain is None else domain
    return ExactGridObservableEvaluation(
        branch_source_id=terminal_id,
        terminal_payload_sha256=payload_digest or _digest(f"payload:{terminal_id}"),
        saved_grid=SavedGridReceipt(
            source_id="saved-grid",
            point_indices=np.arange(values.size),
            x=x,
            x_units="V/nm",
            domain=domain,
        ),
        observable=ObservableReceipt(
            "observable",
            "fixed basis",
            transform.input_units,
            "valid exact-grid evaluation",
        ),
        value_transform=transform,
        raw_y=values,
        output_y=transform.scale * values + transform.offset,
    )


@dataclass
class _Adapter:
    branch_tree: tuple[DiscreteBranchNode, ...]
    enumeration_receipt: EnumerationReceipt
    evaluations: dict[str, ExactGridObservableEvaluation]
    source_authority_id: str = "test_candidate_only_source.v1"

    @property
    def source_authority(self):
        return make_source_authority_receipt(
            self.source_authority_id,
            {"candidate_only": True, "production": False},
        )

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    def evaluate_terminal(self, terminal_id: str) -> ExactGridObservableEvaluation:
        self.calls.append(terminal_id)
        return self.evaluations[terminal_id]


def _bundle(
    values: dict[str, np.ndarray],
    *,
    transforms: dict[str, ValueTransformReceipt] | None = None,
    domain: CurveDomainReceipt | None = None,
):
    terminal_ids = tuple(values)
    transforms = transforms or {}
    adapter = _Adapter(
        branch_tree=_tree(*reversed(terminal_ids)),
        enumeration_receipt=_enumeration(*reversed(terminal_ids)),
        evaluations={
            terminal_id: _evaluation(
                terminal_id,
                value,
                transform=transforms.get(terminal_id),
                domain=domain,
            )
            for terminal_id, value in values.items()
        },
    )
    return build_exact_grid_curve_bundle(adapter), adapter


def _raster(*, touches_boundary: bool = False):
    pixels = np.full((7, 5), 255, dtype=np.uint8)
    pixels[3, :] = 0 if touches_boundary else 255
    if not touches_boundary:
        pixels[3, 1:4] = 0
    calibration = RasterAxesCalibration(0, 4, 0.0, 4.0, 0, 6, 3.0, -3.0, "V/nm", "meV")
    policy = RasterExtractionPolicy(
        dark_threshold=5,
        connectivity=4,
        minimum_pixels=3,
        minimum_columns=3,
        minimum_column_fraction=0.5,
    )
    return extract_raster_curve(pixels, calibration, policy), pixels, calibration, policy


def _plan(bundle, raster, *, criterion=None, lineage=None, preregistration=True, expected_frame=None):
    return create_raster_evaluation_plan(
        bundle,
        expected_source=raster.source,
        calibration=raster.calibration,
        extraction_policy=raster.policy,
        value_kind="output",
        criterion=criterion,
        selection_lineage=_lineage() if lineage is None else lineage,
        preregistration_evidence_sha256=_digest("preregistration") if preregistration else None,
        expected_closed_frame_present=expected_frame,
    )


def test_crossings_extrema_center_and_spread_report_transform_identity() -> None:
    x = np.arange(7, dtype=float)
    y = np.asarray([-1.0, 0.0, 0.0, 1.0, -1.0, 2.0, 3.0])
    crossings = exact_zero_piecewise_linear_crossings(x, y)
    assert [(item.kind, item.x_left, item.x_right) for item in crossings] == [
        ("zero_plateau", 1.0, 2.0),
        ("piecewise_linear", 3.0, 4.0),
        ("piecewise_linear", 4.0, 5.0),
    ]
    extrema = exact_grid_local_extrema(np.arange(7), x, y)
    assert [(item.kind, item.array_index) for item in extrema] == [("maximum", 3), ("minimum", 4)]
    centered = center_at_exact_requested_x(x, y, requested_x=3.0)
    np.testing.assert_array_equal(centered.y, y - 1.0)
    with pytest.raises(ValueError, match="exactly one"):
        center_at_exact_requested_x(x, y, requested_x=3.5)

    bundle, _adapter = _bundle({"a": y, "b": y + 2.0})
    spread = all_branch_pointwise_spread(bundle)
    np.testing.assert_array_equal(spread.spread, np.full(y.shape, 2.0))
    assert spread.transforms_identical
    assert spread.value_units == "meV"


def test_value_transform_identity_and_ev_to_mev_with_offset_are_exact() -> None:
    identity = _evaluation("a", np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(identity.raw_y, identity.output_y)
    assert identity.value_transform.is_identity

    converted = _evaluation(
        "a",
        np.asarray([0.001, 0.002]),
        transform=_transform(
            input_units="eV",
            output_units="meV",
            scale=1000.0,
            offset=5.0,
            semantics="eV to meV plus declared baseline",
        ),
    )
    np.testing.assert_array_equal(converted.output_y, np.asarray([6.0, 7.0]))
    with pytest.raises(ValueError, match=r"scale \* raw_y \+ offset"):
        ExactGridObservableEvaluation(
            branch_source_id=converted.branch_source_id,
            terminal_payload_sha256=converted.terminal_payload_sha256,
            saved_grid=converted.saved_grid,
            observable=converted.observable,
            value_transform=converted.value_transform,
            raw_y=converted.raw_y,
            output_y=np.asarray([1.0, 2.0]),
        )

    overflowing = _transform(
        scale=np.finfo(np.float64).max,
        semantics="deliberately overflowing affine transform",
    )
    finite_base = _evaluation("a", np.asarray([2.0, 3.0]))
    with pytest.raises(ValueError, match="non-finite affine intermediate"):
        ExactGridObservableEvaluation(
            branch_source_id=finite_base.branch_source_id,
            terminal_payload_sha256=finite_base.terminal_payload_sha256,
            saved_grid=finite_base.saved_grid,
            observable=finite_base.observable,
            value_transform=overflowing,
            raw_y=finite_base.raw_y,
            output_y=np.zeros(2),
        )


def test_saved_grid_accepts_unique_nonmonotone_indices_and_rejects_overflow() -> None:
    receipt = SavedGridReceipt(
        source_id="nonmonotone-source-index",
        point_indices=np.asarray([7, 2, 9], dtype=np.int64),
        x=np.asarray([0.0, 1.0, 2.0]),
        x_units="arb",
        domain=CurveDomainReceipt("open_interval"),
    )
    assert receipt.point_indices.tolist() == [7, 2, 9]
    with pytest.raises(OverflowError, match="signed int64"):
        SavedGridReceipt(
            source_id="overflow",
            point_indices=np.asarray([np.iinfo(np.int64).max + 1], dtype=np.uint64),
            x=np.asarray([0.0]),
            x_units="arb",
            domain=CurveDomainReceipt("open_interval"),
        )


def test_source_authority_is_factory_only_strict_hash_bound_and_canonical() -> None:
    receipt = make_source_authority_receipt(
        "candidate-source.v1",
        {"z": [True, None], "a": {"finite": 1.25}},
    )
    assert receipt.canonical_payload_json == '{"a":{"finite":1.25},"z":[true,null]}'
    assert receipt.payload_sha256 == sha256(
        receipt.canonical_payload_json.encode("utf-8")
    ).hexdigest()
    with pytest.raises(TypeError, match="factory-only"):
        SourceAuthorityReceipt(  # type: ignore[call-arg]
            _token=object(),
            authority_id=receipt.authority_id,
            canonical_payload_json=receipt.canonical_payload_json,
            payload_sha256=receipt.payload_sha256,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        make_source_authority_receipt("bad.v1", {"bad": float("nan")})


def test_source_authority_changes_the_bundle_fingerprint() -> None:
    values = {"a": np.asarray([0.0, 1.0])}
    first, _ = _bundle(values)
    second_adapter = _Adapter(
        branch_tree=_tree("a"),
        enumeration_receipt=_enumeration("a"),
        evaluations={"a": _evaluation("a", values["a"])},
        source_authority_id="different_candidate_source.v1",
    )
    second = build_exact_grid_curve_bundle(second_adapter)
    assert first.source_authority != second.source_authority
    assert first.bundle_fingerprint != second.bundle_fingerprint


def test_structural_closure_enumeration_frontier_and_no_rejection_override() -> None:
    receipt = _enumeration("b", "a")
    certificate = certify_enumerated_branch_closure(_tree("b", "a"), receipt)
    assert certificate.authority == "supplied_finite_tree_structurally_resolved"
    assert certificate.supplied_finite_tree_structurally_resolved
    assert not certificate.enumeration_receipt.system_claims_exhaustive_enumeration

    with pytest.raises(ValueError, match="frontier"):
        certify_enumerated_branch_closure(_tree("a"), _enumeration("a", frontier=1))

    adapter = _Adapter(
        branch_tree=_tree("a", rejected="bad"),
        enumeration_receipt=_enumeration("a", "bad"),
        evaluations={"a": _evaluation("a", np.asarray([0.0, 1.0]))},
    )
    with pytest.raises(ValueError, match="every supplied leaf computed"):
        build_exact_grid_curve_bundle(adapter)
    assert adapter.calls == []
    with pytest.raises(TypeError):
        build_exact_grid_curve_bundle(adapter, rejected_terminals_acceptable=True)  # type: ignore[call-arg]


def test_builder_checks_exact_terminal_payload_digest_before_bundle() -> None:
    adapter = _Adapter(
        branch_tree=_tree("a"),
        enumeration_receipt=_enumeration("a"),
        evaluations={
            "a": _evaluation(
                "a",
                np.asarray([0.0, 1.0]),
                payload_digest=_digest("wrong-payload"),
            )
        },
    )
    with pytest.raises(ValueError, match="terminal payload digest mismatch"):
        build_exact_grid_curve_bundle(adapter)


def test_branch_specific_incompatible_transforms_block_spread_and_plan() -> None:
    compatible_output, _ = _bundle(
        {"a": np.zeros(5), "b": np.zeros(5)},
        transforms={
            "a": _transform(offset=0.0, common=False),
            "b": _transform(offset=1.0, common=False),
        },
    )
    compatible_spread = all_branch_pointwise_spread(compatible_output)
    assert not compatible_spread.transforms_identical
    np.testing.assert_array_equal(compatible_spread.spread, np.ones(5))

    incompatible_output, _ = _bundle(
        {"a": np.zeros(5), "b": np.zeros(5)},
        transforms={
            "a": _transform(semantics="baseline A", common=False),
            "b": _transform(semantics="baseline B", common=False),
        },
    )
    with pytest.raises(ValueError, match="compatible transform semantics"):
        all_branch_pointwise_spread(incompatible_output)
    raster, _pixels, calibration, policy = _raster()
    with pytest.raises(ValueError, match="compatible transform semantics"):
        create_raster_evaluation_plan(
            incompatible_output,
            expected_source=raster.source,
            calibration=calibration,
            extraction_policy=policy,
            value_kind="output",
            selection_lineage=_lineage(),
        )

    incompatible_raw, _ = _bundle(
        {"a": np.zeros(5), "b": np.zeros(5)},
        transforms={
            "a": _transform(input_units="eV", output_units="meV", scale=1000.0, common=False),
            "b": _transform(input_units="meV", output_units="meV", common=False),
        },
    )
    with pytest.raises(ValueError, match="common input units"):
        all_branch_pointwise_spread(incompatible_raw, value_kind="raw")


def test_open_interval_supported_and_periodic_comparison_fails_until_implemented() -> None:
    open_domain = CurveDomainReceipt("open_interval")
    result = explicit_piecewise_linear_interpolation(
        [0.0, 1.0],
        [0.0, 2.0],
        [0.5],
        domain=open_domain,
    )
    np.testing.assert_array_equal(result, np.asarray([1.0]))
    periodic = CurveDomainReceipt("periodic", period=2.0, seam=-1.0)
    with pytest.raises(NotImplementedError, match="periodic seam"):
        explicit_piecewise_linear_interpolation([0.0, 1.0], [0.0, 2.0], [0.5], domain=periodic)
    periodic_bundle, _ = _bundle({"a": np.zeros(5)}, domain=periodic)
    with pytest.raises(NotImplementedError, match="open_interval"):
        all_branch_pointwise_spread(periodic_bundle)


def test_extraction_and_plan_are_factory_only() -> None:
    with pytest.raises(TypeError):
        RasterCurveExtraction()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        RasterEvaluationPlan()  # type: ignore[call-arg]


def test_comparison_is_factory_only_and_rejects_provenance_bypass() -> None:
    raster, _pixels, _calibration, _policy = _raster()
    bundle, _ = _bundle({"a": np.zeros(5)})
    plan = _plan(bundle, raster)
    derived = compare_raster_to_all_branches(bundle, plan, raster)
    with pytest.raises(TypeError, match="factory-only"):
        RasterBundleComparison(  # type: ignore[call-arg]
            _token=object(),
            bundle=bundle,
            plan=plan,
            raster=raster,
            branch_metrics=derived.branch_metrics,
            raster_crossings=derived.raster_crossings,
            decision=derived.decision,
            transforms_identical=derived.transforms_identical,
        )


def test_full_mask_frame_detection_preserves_frame_connected_curve_with_margin() -> None:
    pixels = np.full((20, 30), 255, dtype=np.uint8)
    pixels[1, 1:29] = 0
    pixels[18, 1:29] = 0
    pixels[1:19, 1] = 0
    pixels[1:19, 28] = 0
    pixels[10, 1:29] = 0
    calibration = RasterAxesCalibration(1, 28, 0.0, 27.0, 1, 18, 1.0, -1.0, "V/nm", "meV")
    policy = RasterExtractionPolicy(
        dark_threshold=5,
        minimum_pixels=10,
        minimum_columns=10,
        minimum_column_fraction=0.3,
        auto_exclude_closed_dark_frame=True,
        frame_interior_margin=1,
    )
    extraction = extract_raster_curve(pixels, calibration, policy)
    assert extraction.closed_dark_frame_detected
    assert extraction.centerline_pixel_x.tolist() == list(range(3, 27))
    assert np.all(extraction.centerline_pixel_y == 10.0)

    detected = detect_raster_frame_calibration(
        pixels,
        x_left=0.0,
        x_right=27.0,
        y_top=1.0,
        y_bottom=-1.0,
        x_units="V/nm",
        y_units="meV",
        dark_threshold=5,
    )
    assert detected == calibration


def test_line_uncertainty_bounds_median_to_both_edges_plus_half_pixel() -> None:
    pixels = np.full((12, 7), 255, dtype=np.uint8)
    pixels[2:9, 2:5] = 0
    pixels[3:5, 1] = 0
    pixels[3:5, 5] = 0
    extraction = extract_raster_curve(
        pixels,
        RasterAxesCalibration(0, 6, 0.0, 6.0, 0, 11, 11.0, 0.0, "V/nm", "meV"),
        RasterExtractionPolicy(
            dark_threshold=5,
            minimum_pixels=10,
            minimum_columns=3,
            minimum_column_fraction=0.4,
        ),
    )
    for column, median, total in zip(
        extraction.centerline_pixel_x,
        extraction.centerline_pixel_y,
        extraction.total_uncertainty_y,
        strict=True,
    ):
        rows = extraction.component_pixel_y[extraction.component_pixel_x == column]
        required = max(median - rows.min(), rows.max() - median) + 0.5
        assert total >= required * extraction.calibration.y_units_per_pixel


def test_plan_binding_posthoc_criterion_ineligible_and_decisions_renamed() -> None:
    raster, _pixels, _calibration, _policy = _raster()
    bundle, _ = _bundle({"a": np.zeros(5), "b": np.zeros(5)})

    evidence_plan = _plan(bundle, raster, criterion=None, preregistration=False)
    assert compare_raster_to_all_branches(bundle, evidence_plan, raster).decision == "evidence_only"

    criterion = CurveAgreementCriterion(maximum_rmse=0.1, maximum_absolute_error=0.1)
    posthoc_plan = _plan(bundle, raster, criterion=criterion, preregistration=False)
    assert not posthoc_plan.criterion_eligible
    assert compare_raster_to_all_branches(bundle, posthoc_plan, raster).decision == "criterion_not_satisfied"
    with pytest.raises(TypeError):
        compare_raster_to_all_branches(  # type: ignore[call-arg]
            bundle,
            evidence_plan,
            raster,
            criterion=criterion,
        )

    eligible_plan = _plan(bundle, raster, criterion=criterion)
    comparison = compare_raster_to_all_branches(bundle, eligible_plan, raster)
    assert comparison.decision == "criterion_satisfied"
    assert [item.terminal_id for item in comparison.branch_metrics] == ["a", "b"]

    wrong_source = RasterSourceReceipt(
        sha256=_digest("wrong-source"),
        width=raster.source.width,
        height=raster.source.height,
        mode=raster.source.mode,
        hash_basis=raster.source.hash_basis,
    )
    wrong_plan = create_raster_evaluation_plan(
        bundle,
        expected_source=wrong_source,
        calibration=raster.calibration,
        extraction_policy=raster.policy,
        value_kind="output",
        selection_lineage=_lineage(),
    )
    with pytest.raises(ValueError, match="expected source"):
        compare_raster_to_all_branches(bundle, wrong_plan, raster)


def test_boundary_samples_and_expected_frame_block_criterion_satisfaction() -> None:
    raster, _pixels, _calibration, _policy = _raster(touches_boundary=True)
    bundle, _ = _bundle({"a": np.zeros(5)})
    criterion = CurveAgreementCriterion(maximum_rmse=0.1)
    boundary_plan = _plan(bundle, raster, criterion=criterion)
    assert np.any(raster.boundary_flags)
    assert compare_raster_to_all_branches(bundle, boundary_plan, raster).decision == "criterion_not_satisfied"

    no_boundary, _pixels, _calibration, _policy = _raster()
    expected_frame_plan = _plan(bundle, no_boundary, criterion=criterion, expected_frame=True)
    with pytest.raises(ValueError, match="expected frame condition"):
        compare_raster_to_all_branches(bundle, expected_frame_plan, no_boundary)


def test_full_bundle_raster_plan_comparison_roundtrip_and_csv_rows(tmp_path) -> None:
    raster, _pixels, _calibration, _policy = _raster()
    bundle, _ = _bundle({"a": np.zeros(5), "b": np.zeros(5)})
    plan = _plan(bundle, raster, criterion=CurveAgreementCriterion(maximum_rmse=0.1))
    comparison = compare_raster_to_all_branches(bundle, plan, raster)
    paths = write_curve_workflow_artifacts(
        bundle,
        tmp_path / "workflow",
        raster=raster,
        evaluation_plan=plan,
        comparison=comparison,
    )
    loaded = load_curve_workflow_artifacts(tmp_path / "workflow")
    assert loaded.bundle.bundle_fingerprint == bundle.bundle_fingerprint
    assert loaded.bundle.source_authority == bundle.source_authority
    assert loaded.raster is not None
    assert loaded.evaluation_plan is not None
    assert loaded.evaluation_plan.plan_fingerprint == plan.plan_fingerprint
    assert loaded.comparison is not None
    assert loaded.comparison.decision == "criterion_satisfied"
    assert paths.metadata_json.name == "metadata.json"
    assert paths.arrays_npz.name == "arrays.npz"
    assert paths.curves_csv.name == "curves.csv"


def test_tampered_metadata_arrays_and_even_rehashed_csv_are_rejected(tmp_path) -> None:
    def write_case(name: str):
        raster, _pixels, _calibration, _policy = _raster()
        bundle, _ = _bundle({"a": np.zeros(5)})
        plan = _plan(bundle, raster, criterion=CurveAgreementCriterion(maximum_rmse=0.1))
        comparison = compare_raster_to_all_branches(bundle, plan, raster)
        root = tmp_path / name
        write_curve_workflow_artifacts(bundle, root, raster=raster, evaluation_plan=plan, comparison=comparison)
        return root

    authority_root = write_case("authority")
    authority_path = authority_root / "metadata.json"
    authority_metadata = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_metadata["source_authority"]["canonical_payload_json"] = (
        '{"candidate_only":false,"production":false}'
    )
    authority_path.write_text(
        json.dumps(authority_metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source authority"):
        load_curve_workflow_artifacts(authority_root)

    authority_npz_root = write_case("authority-npz")
    authority_npz_path = authority_npz_root / "arrays.npz"
    with np.load(authority_npz_path, allow_pickle=False) as loaded:
        authority_arrays = {key: loaded[key].copy() for key in loaded.files}
    authority_key = "source_authority_authority_id_utf8"
    authority_arrays[authority_key][0] ^= np.uint8(1)
    np.savez(authority_npz_path, **authority_arrays)
    authority_npz_metadata_path = authority_npz_root / "metadata.json"
    authority_npz_metadata = json.loads(
        authority_npz_metadata_path.read_text(encoding="utf-8")
    )
    authority_npz_metadata["array_sha256"][authority_key] = canonical_array_sha256(
        authority_arrays[authority_key]
    )
    authority_npz_metadata_path.write_text(
        json.dumps(authority_npz_metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=authority_key):
        load_curve_workflow_artifacts(authority_npz_root)

    metadata_root = write_case("metadata")
    metadata_path = metadata_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["comparison"]["decision"] = "evidence_only"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="re-derived comparison"):
        load_curve_workflow_artifacts(metadata_root)

    arrays_root = write_case("arrays")
    arrays_path = arrays_root / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    arrays["curve_0000_output_y"][0] = 2.0
    np.savez(arrays_path, **arrays)
    with pytest.raises(ValueError, match="array hash"):
        load_curve_workflow_artifacts(arrays_root)

    csv_root = write_case("csv")
    csv_path = csv_root / "curves.csv"
    rows = list(csv.reader(StringIO(csv_path.read_text(encoding="utf-8"))))
    rows[1][8] = "forged-authority"
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    forged_csv = output.getvalue()
    csv_path.write_text(forged_csv, encoding="utf-8")
    metadata_path = csv_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["curves_csv_sha256"] = sha256(forged_csv.encode("utf-8")).hexdigest()
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match NPZ"):
        load_curve_workflow_artifacts(csv_root)


def test_plot_accepts_only_bound_comparison_and_labels_value_convention() -> None:
    raster, _pixels, _calibration, _policy = _raster()
    bundle, _ = _bundle({"a": np.zeros(5)})
    plan = _plan(bundle, raster)
    comparison = compare_raster_to_all_branches(bundle, plan, raster)

    class _Axis:
        def __init__(self):
            self.labels: list[str] = []
            self.title = ""

        def plot(self, *_args, **kwargs):
            self.labels.append(kwargs["label"])

        def errorbar(self, *_args, **kwargs):
            self.labels.append(kwargs["label"])

        def set_xlabel(self, _value):
            pass

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def legend(self):
            pass

    axis = _Axis()
    assert plot_exact_grid_curve_bundle(bundle, comparison=comparison, ax=axis) is axis
    assert "held-out raster evaluation" in axis.labels
    assert "output" in axis.ylabel
    assert comparison.plan.value_semantics in axis.title
    with pytest.raises(TypeError, match="bound RasterBundleComparison"):
        plot_exact_grid_curve_bundle(bundle, comparison=raster, ax=_Axis())  # type: ignore[arg-type]
