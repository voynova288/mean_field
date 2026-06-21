from __future__ import annotations

from ._polshyn_shared import *  # noqa: F401,F403
from ._polshyn_types import *  # noqa: F401,F403

def reference_diagonal_for_projected_indices(projected_indices: tuple[int, ...], target_band_index: int) -> np.ndarray:
    """Reference density for Polshyn's conduction-band filling convention.

    The experimental/paper filling ``nu=7/2`` counts electrons added into the
    target conduction C=2 band.  Therefore the target band reference is empty,
    not half-filled as in charge-neutral two-flat-band TBG conventions.  Remote
    bands below the target are part of the subtraction-method sea and are filled
    in the reference; remote bands above the target are empty.
    """

    return folded_reference_diagonal_by_primitive_index(
        tuple(int(index) for index in projected_indices),
        target_band_index=int(target_band_index),
        folds_per_primitive=2,
        lower_reference=1.0,
        target_reference=0.0,
        upper_reference=0.0,
    )


def occupation_counts_nu_7over2(projected_indices: tuple[int, ...], target_band_index: int) -> np.ndarray:
    indices = tuple(int(index) for index in projected_indices)
    target_fold_indices = folded_indices_for_primitive_band(
        indices,
        target_band_index=int(target_band_index),
        folds_per_primitive=2,
    )
    if len(target_fold_indices) != 2:
        raise ValueError("Expected exactly one primitive target band, folded into two supercell bands")
    lower_count = sum(1 for index in indices if int(index) < int(target_band_index))
    full = 2 * int(lower_count) + len(target_fold_indices)
    partial = 2 * int(lower_count) + 1
    return fixed_sector_occupation_counts(
        n_spin=2,
        n_eta=2,
        default_count=full,
        overrides={(0, 0): partial},
        n_band=2 * len(indices),
    )


def primitive_nu_from_counts(occupation_counts: np.ndarray, reference_diagonal: np.ndarray, *, area_ratio: int) -> float:
    return primitive_filling_from_occupation_counts(
        occupation_counts,
        reference_diagonal=reference_diagonal,
        area_ratio=int(area_ratio),
        n_band=int(np.asarray(reference_diagonal, dtype=float).size),
    )



def polshyn_nu_7over2_filling_summary(
    projected_indices: tuple[int, ...],
    *,
    target_band_index: int,
    area_ratio: int = 2,
) -> PolshynFillingSummary:
    indices = tuple(int(index) for index in projected_indices)
    target = int(target_band_index)
    if target not in indices:
        raise ValueError(f"target_band_index={target} is not present in projected_indices={indices}")
    target_position = indices.index(target)
    target_fold_indices = folded_indices_for_primitive_band(
        indices,
        target_band_index=target,
        folds_per_primitive=2,
    )
    reference = reference_diagonal_for_projected_indices(indices, target)
    counts = occupation_counts_nu_7over2(indices, target)
    primitive_nu = primitive_nu_from_counts(counts, reference, area_ratio=int(area_ratio))
    return PolshynFillingSummary(
        projected_indices=indices,
        target_band_index=target,
        target_primitive_position=int(target_position),
        target_fold_indices=target_fold_indices,
        nb=2 * len(indices),
        area_ratio=int(area_ratio),
        reference_diagonal=reference,
        occupation_counts=counts,
        primitive_nu=float(primitive_nu),
        matches_expected_filling=bool(np.isclose(primitive_nu, 3.5, atol=1.0e-12)),
    )













def cdw_density_blocks(
    *,
    projected_indices: tuple[int, ...],
    target_band_index: int,
    n_spin: int,
    n_eta: int,
    nb: int,
    nk: int,
    reference_diagonal: np.ndarray,
) -> np.ndarray:
    """Maximal translation-breaking initializer for the K+ spin-up target band."""

    projected_indices = tuple(int(index) for index in projected_indices)
    target_primitive_pos = projected_indices.index(int(target_band_index))
    target_fold_indices = (2 * target_primitive_pos, 2 * target_primitive_pos + 1)
    reference = np.diag(np.asarray(reference_diagonal, dtype=float)).astype(np.complex128)
    density = np.zeros((int(n_spin), int(n_eta), int(nb), int(nb), int(nk)), dtype=np.complex128)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            projector = np.zeros((int(nb), int(nb)), dtype=np.complex128)
            # Lower remote bands are filled in the reference and stay filled in the initializer.
            for iprim, band_index in enumerate(projected_indices):
                if int(band_index) < int(target_band_index):
                    projector[2 * iprim, 2 * iprim] = 1.0
                    projector[2 * iprim + 1, 2 * iprim + 1] = 1.0
            if ispin == 0 and ieta == 0:
                i0, i1 = target_fold_indices
                projector[i0, i0] = 0.5
                projector[i1, i1] = 0.5
                projector[i0, i1] = 0.5
                projector[i1, i0] = 0.5
            else:
                i0, i1 = target_fold_indices
                projector[i0, i0] = 1.0
                projector[i1, i1] = 1.0
            for ik in range(int(nk)):
                density[ispin, ieta, :, :, ik] = projector - reference
    return density

__all__ = [name for name in globals() if not name.startswith('__')]
