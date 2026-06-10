from __future__ import annotations

import pytest

from mean_field.core.validation import (
    ValidationCheck,
    ValidationReport,
    format_validation_value,
    status_from_bool,
    validate_valley,
)
from mean_field.systems.RnG_hBN.validation import ValidationCheck as RLGValidationCheck
from mean_field.systems.RnG_hBN.validation import ValidationReport as RLGValidationReport
from mean_field.systems.atmg.validation import ValidationCheck as ATMGValidationCheck
from mean_field.systems.atmg.validation import ValidationReport as ATMGValidationReport
from mean_field.systems.tdbg.validation import ValidationCheck as TDBGValidationCheck
from mean_field.systems.tdbg.validation import ValidationReport as TDBGValidationReport
from mean_field.systems.tmbg.validation import ValidationCheck as TMBGValidationCheck
from mean_field.systems.tmbg.validation import ValidationReport as TMBGValidationReport


def test_validate_valley_normalizes_and_rejects_invalid_values() -> None:
    assert validate_valley(1) == 1
    assert validate_valley(-1) == -1
    with pytest.raises(ValueError, match="Expected valley"):
        validate_valley(0)


def test_validation_report_markdown_and_counts() -> None:
    report = ValidationReport(
        title="demo",
        checks=(
            ValidationCheck("a", status_from_bool(True), "passed", 1.25),
            ValidationCheck("b", status_from_bool(False), "failed", "bad"),
            ValidationCheck("c", "skipped", "not run", None),
        ),
    )

    assert report.failure_count == 1
    assert report.skipped_count == 1
    assert report.has_failures
    assert report.has_skips
    markdown = report.to_markdown()
    assert "# demo" in markdown
    assert "[pass] a (1.250000e+00): passed" in markdown
    assert "[fail] b (bad): failed" in markdown
    assert "- skipped: 1" in markdown
    assert format_validation_value(None) == ""


def test_validation_report_combine_and_system_validation_modules_reexport_core_type() -> None:
    report_a = ValidationReport("a", (ValidationCheck("a", "pass", "ok"),))
    report_b = ValidationReport("b", (ValidationCheck("b", "skipped", "later"),))
    merged = ValidationReport.combine("merged", report_a, report_b)

    assert [check.name for check in merged.checks] == ["a", "b"]
    assert merged.skipped_count == 1
    assert RLGValidationCheck is ValidationCheck
    assert RLGValidationReport is ValidationReport
    assert ATMGValidationCheck is ValidationCheck
    assert ATMGValidationReport is ValidationReport
    assert TDBGValidationCheck is ValidationCheck
    assert TDBGValidationReport is ValidationReport
    assert TMBGValidationCheck is ValidationCheck
    assert TMBGValidationReport is ValidationReport


def test_system_validation_markdown_uses_core_report_format() -> None:
    atmg_report = ATMGValidationReport(
        "atmg",
        (ATMGValidationCheck("skip", "skipped", "not applicable", None),),
    )
    tdbg_report = TDBGValidationReport(
        "tdbg",
        (TDBGValidationCheck("residual", "pass", "ok", 1.0e-12),),
    )

    assert "- skipped: 1" in atmg_report.to_markdown()
    assert "[pass] residual (1.000000e-12): ok" in tdbg_report.to_markdown()
