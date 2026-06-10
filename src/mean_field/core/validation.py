from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DEFAULT_VALID_VALLEYS: tuple[int, ...] = (-1, 1)
ValidationStatus = Literal["pass", "fail", "skipped"]


def validate_valley(valley: int, valid_valleys: tuple[int, ...] = DEFAULT_VALID_VALLEYS) -> int:
    """Normalize and validate a graphene valley label."""

    normalized = int(valley)
    if normalized not in valid_valleys:
        raise ValueError(f"Expected valley in {valid_valleys}, got {normalized}")
    return normalized


def status_from_bool(condition: bool) -> ValidationStatus:
    """Return the standard status token for a boolean validation condition."""

    return "pass" if bool(condition) else "fail"


def format_validation_value(value: object | None) -> str:
    """Format a validation value consistently for markdown reports."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


@dataclass(frozen=True)
class ValidationCheck:
    """A single named validation assertion.

    System modules should keep physics-specific diagnostics in ``detail`` and
    ``value`` while reusing this result container for reporting.
    """

    name: str
    status: ValidationStatus
    detail: str
    value: float | int | str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass(frozen=True)
class ValidationReport:
    """Collection of validation checks with common markdown reporting."""

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
    "DEFAULT_VALID_VALLEYS",
    "ValidationCheck",
    "ValidationReport",
    "ValidationStatus",
    "format_validation_value",
    "status_from_bool",
    "validate_valley",
]
