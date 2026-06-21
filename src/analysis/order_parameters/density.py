from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .schema import StateLabel


def average_projector_occupations(projector_kab: np.ndarray) -> np.ndarray:
    """Return k-averaged diagonal occupations from ``(state,state,k)`` projectors."""

    projector = np.asarray(projector_kab, dtype=np.complex128)
    if projector.ndim != 3 or projector.shape[0] != projector.shape[1]:
        raise ValueError(f"Expected projector shape (state,state,k), got {projector.shape}")
    occ = np.zeros(projector.shape[0], dtype=float)
    for ik in range(projector.shape[2]):
        occ += np.real(np.diag(projector[:, :, ik]))
    return occ / float(projector.shape[2])


def occupation_table(projector_kab: np.ndarray, labels: Sequence[StateLabel]) -> list[dict[str, object]]:
    """Return occupation rows keyed by generic ``StateLabel`` records."""

    occ = average_projector_occupations(projector_kab)
    rows: list[dict[str, object]] = []
    for label in labels:
        item = dict(label.metadata)
        item.update(
            {
                "index": int(label.index),
                "spin": label.spin,
                "valley": label.valley,
                "band": label.band,
                "sector": label.sector,
                "fold": label.fold,
                "layer": label.layer,
                "sublattice": label.sublattice,
                "active": bool(label.active),
                "occupation": float(occ[int(label.index)]),
            }
        )
        rows.append(item)
    return rows


def signed_polarization(
    projector_kab: np.ndarray,
    labels: Sequence[StateLabel],
    sign: Callable[[StateLabel], float],
    *,
    active_only: bool = False,
) -> float:
    """Return ``sum_i sign(label_i) n_i`` over all or active labels."""

    occ = average_projector_occupations(projector_kab)
    total = 0.0
    for label in labels:
        if active_only and not bool(label.active):
            continue
        total += float(sign(label)) * float(occ[int(label.index)])
    return float(total)


__all__ = ["average_projector_occupations", "occupation_table", "signed_polarization"]
