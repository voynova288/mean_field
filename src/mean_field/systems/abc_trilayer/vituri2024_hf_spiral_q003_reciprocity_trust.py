"""Externally reviewed trust pins for the q003 reciprocity certifier."""

from typing import Final

PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256: Final[str] = (
    "6fcb86709ac5393ed78e2b89dadff5812d662d26f5ce4d9278fd02de293f8da9"
)
PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256: Final[str] = (
    "ecebfd0dcea1a9b63cb66a651b7a851d45c22d9bca15c5e911354874f2f10b36"
)

__all__ = [
    "PINNED_Q003_RECIPROCITY_CERTIFIER_REVIEW_SHA256",
    "PINNED_Q003_RECIPROCITY_CERTIFIER_SHA256",
]
