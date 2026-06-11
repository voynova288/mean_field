from __future__ import annotations

import pytest

from mean_field.core.validation import ValidationCheck, ValidationReport, status_from_bool


def test_validation_check_accepts_value_and_val_aliases() -> None:
    with_value = ValidationCheck(name="a", status="pass", detail="ok", value=1.25)
    with_val = ValidationCheck(name="b", status="fail", detail="bad", val=2.5)

    assert with_value.passed
    assert with_value.value == 1.25
    assert with_value.val == 1.25
    assert with_value.to_dict()["passed"] is True
    assert not with_val.passed
    assert with_val.value == 2.5
    assert with_val.val == 2.5


def test_validation_check_rejects_double_payload_alias() -> None:
    with pytest.raises(ValueError, match="either value= or val="):
        ValidationCheck(name="bad", status="pass", detail="bad", value=1.0, val=2.0)


def test_validation_report_counts_skips_and_combines() -> None:
    left = ValidationReport(
        title="left",
        checks=(ValidationCheck(name="ok", status="pass", detail="ok"),),
    )
    right = ValidationReport(
        title="right",
        checks=(
            ValidationCheck(name="bad", status="fail", detail="bad", value=3.0),
            ValidationCheck(name="skip", status="skipped", detail="skip"),
        ),
    )

    combined = ValidationReport.combine("combined", left, right)

    assert combined.failure_count == 1
    assert combined.skipped_count == 1
    assert combined.has_failures
    assert combined.has_skips
    payload = combined.to_dict()
    assert payload["failure_count"] == 1
    assert payload["checks"][1]["name"] == "bad"
    assert payload["checks"][1]["value"] == 3.0
    markdown = combined.to_markdown()
    assert "bad" in markdown
    assert "3.000000e+00" in markdown


def test_status_from_bool() -> None:
    assert status_from_bool(True) == "pass"
    assert status_from_bool(False) == "fail"
