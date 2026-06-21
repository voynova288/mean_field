from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .schema import StateLabel


def coherence_norm(
    projector_kab: np.ndarray,
    labels: Sequence[StateLabel],
    left_filter: Callable[[StateLabel], bool],
    right_filter: Callable[[StateLabel], bool],
) -> float:
    """Return the k-averaged Frobenius norm between two labeled subspaces."""

    projector = np.asarray(projector_kab, dtype=np.complex128)
    if projector.ndim != 3 or projector.shape[0] != projector.shape[1]:
        raise ValueError(f"Expected projector shape (state,state,k), got {projector.shape}")
    left = [int(label.index) for label in labels if left_filter(label)]
    right = [int(label.index) for label in labels if right_filter(label)]
    if not left or not right:
        return 0.0
    total = 0.0
    for ik in range(projector.shape[2]):
        total += float(np.linalg.norm(projector[np.ix_(left, right, [ik])][:, :, 0]))
    return float(total / float(projector.shape[2]))


def ivc_amplitude(projector_kab: np.ndarray, labels: Sequence[StateLabel], *, active_only: bool = True) -> float:
    """Return a valley-coherence amplitude summed over spin sectors."""

    spins = sorted({label.spin for label in labels if label.spin is not None}, key=str)
    total = 0.0
    for spin in spins:
        total += coherence_norm(
            projector_kab,
            labels,
            lambda label, spin=spin: label.spin == spin
            and label.valley == 1
            and (bool(label.active) or not active_only),
            lambda label, spin=spin: label.spin == spin
            and label.valley == -1
            and (bool(label.active) or not active_only),
        )
    return float(total)


__all__ = ["coherence_norm", "ivc_amplitude"]
