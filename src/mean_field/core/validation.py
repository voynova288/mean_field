from __future__ import annotations

DEFAULT_VALID_VALLEYS: tuple[int, ...] = (-1, 1)


def validate_valley(valley: int, valid_valleys: tuple[int, ...] = DEFAULT_VALID_VALLEYS) -> int:
    """Normalize and validate a graphene valley label."""
    normalized = int(valley)
    if normalized not in valid_valleys:
        raise ValueError(f"Expected valley in {valid_valleys}, got {normalized}")
    return normalized
