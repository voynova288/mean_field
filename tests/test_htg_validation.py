from __future__ import annotations

from mean_field.systems.htg import build_htg_lattice, htg_validation_report
from mean_field.systems.htg.validation import ValidationCheck, validate_lattice


def test_htg_validation_check_converts_to_core_report() -> None:
    check = ValidationCheck("demo", False, 0.2, 0.1)

    core_check = check.to_core_check()
    report = htg_validation_report("HTG demo", (check,))

    assert check.status == "fail"
    assert core_check.status == "fail"
    assert core_check.value == 0.2
    assert "tolerance=0.1" in core_check.detail
    assert report.failure_count == 1
    assert report.to_dict()["checks"][0]["name"] == "demo"


def test_htg_lattice_validation_can_be_reported_with_core_container() -> None:
    checks = validate_lattice(build_htg_lattice(1.5, n_shells=0))

    report = htg_validation_report("HTG lattice", checks)

    assert report.title == "HTG lattice"
    assert report.failure_count == 0
    assert {check.name for check in report.checks} >= {"q_norm_equal", "g_contains_zero"}
