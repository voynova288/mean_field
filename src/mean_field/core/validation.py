from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ValidationStatus = Literal["pass", "fail", "skipped"]
ValidationValue = float | int | str | None


def validate_valley(valley: int) -> int:
    value = int(valley)
    if value not in {-1, 1}:
        raise ValueError(f"valley must be ±1, got {valley}")
    return value


def status_from_bool(condition: bool) -> ValidationStatus:
    return "pass" if bool(condition) else "fail"


def format_validation_value(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


@dataclass(frozen=True, init=False)
class ValidationCheck:
    """Status/detail validation record shared by system validation modules.

    The historical system modules used both ``value=`` and ``val=`` for the
    numerical payload.  This class accepts both spellings and exposes both
    ``.value`` and the compatibility alias ``.val`` so wrappers can migrate
    without changing every call site at once.
    """

    name: str
    status: ValidationStatus
    detail: str
    value: ValidationValue = None

    def __init__(
        self,
        name: str,
        status: ValidationStatus,
        detail: str,
        value: ValidationValue = None,
        *,
        val: ValidationValue = None,
    ) -> None:
        if value is not None and val is not None:
            raise ValueError("Pass either value= or val=, not both.")
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "detail", str(detail))
        object.__setattr__(self, "value", val if value is None else value)

    @property
    def val(self) -> ValidationValue:
        return self.value

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "value": self.value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ValidationReport:
    title: str
    checks: tuple[ValidationCheck, ...]

    @property
    def failure_count(self) -> int:
        return sum(check.status == "fail" for check in self.checks)

    @property
    def skipped_count(self) -> int:
        return sum(check.status == "skipped" for check in self.checks)

    @property
    def has_failures(self) -> bool:
        return self.failure_count > 0

    @property
    def has_skips(self) -> bool:
        return self.skipped_count > 0

    @classmethod
    def combine(cls, title: str, *reports: "ValidationReport") -> "ValidationReport":
        merged: list[ValidationCheck] = []
        for report in reports:
            merged.extend(report.checks)
        return cls(title=title, checks=tuple(merged))

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
            "has_failures": self.has_failures,
            "has_skips": self.has_skips,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for check in self.checks:
            value_text = format_validation_value(check.value)
            suffix = f" ({value_text})" if value_text else ""
            lines.append(f"- [{check.status}] {check.name}{suffix}: {check.detail}")
        lines.append("")
        lines.append(f"- failures: {self.failure_count}")
        if self.skipped_count:
            lines.append(f"- skipped: {self.skipped_count}")
        return "\n".join(lines)


__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "ValidationStatus",
    "ValidationValue",
    "format_validation_value",
    "validate_valley",
    "status_from_bool",
]
