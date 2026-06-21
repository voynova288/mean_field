from __future__ import annotations

import numpy as np


def folded_translation_order_parameters(
    density_blocks: np.ndarray,
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    spin_index: int = 0,
    valley_index: int = 0,
    folds_per_primitive: int = 2,
) -> dict[str, np.ndarray | float]:
    """Fold-off-diagonal translation-breaking order diagnostic.

    The default is the doubled-cell convention used by the Polshyn tMBG
    adapter: each primitive band contributes fold 0 and fold 1, and the target
    order is ``|rho_{fold0,fold1}|`` for the target primitive band.
    """

    if int(folds_per_primitive) != 2:
        raise NotImplementedError("Only two-fold doubled-cell translation order is currently implemented")
    density = np.asarray(density_blocks, dtype=np.complex128)
    projected_indices = tuple(int(index) for index in projected_indices)
    target_pos = projected_indices.index(int(target_band_index))
    fold0 = np.asarray([2 * iprim for iprim in range(len(projected_indices))], dtype=int)
    fold1 = fold0 + 1
    sector = density[int(spin_index), int(valley_index)]
    target_raw = np.abs(sector[2 * target_pos, 2 * target_pos + 1, :])
    all_raw = np.sqrt(np.sum(np.abs(sector[np.ix_(fold0, fold1, np.arange(sector.shape[-1]))]) ** 2, axis=(0, 1)))
    return {
        "target_raw": np.asarray(target_raw, dtype=float),
        "all_raw": np.asarray(all_raw, dtype=float),
        "target_x2": np.asarray(2.0 * target_raw, dtype=float),
        "all_x2": np.asarray(2.0 * all_raw, dtype=float),
        "target_x2_min": float(np.min(2.0 * target_raw)),
        "target_x2_mean": float(np.mean(2.0 * target_raw)),
        "target_x2_max": float(np.max(2.0 * target_raw)),
        "all_x2_min": float(np.min(2.0 * all_raw)),
        "all_x2_mean": float(np.mean(2.0 * all_raw)),
        "all_x2_max": float(np.max(2.0 * all_raw)),
    }


__all__ = ["folded_translation_order_parameters"]
