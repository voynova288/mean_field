from __future__ import annotations

import numpy as np

from .band_classifier import BandClassification
from .bm import AllBandBMSolution
from .form_factor import PRODUCTION_FORM_FACTOR_MODE, compute_lambda_stack
from .grid import CRPAKGrid


def occupation_step(
    energies_mev: np.ndarray,
    *,
    fermi_level_mev: float = 0.0,
    fermi_tol_mev: float = 1.0e-10,
    fermi_occupancy: float = 0.5,
) -> np.ndarray:
    energies = np.asarray(energies_mev, dtype=float)
    occ = np.zeros_like(energies, dtype=float)
    occ[energies < fermi_level_mev - fermi_tol_mev] = 1.0
    at_fermi = np.abs(energies - fermi_level_mev) <= fermi_tol_mev
    occ[at_fermi] = float(fermi_occupancy)
    return occ


def cnp_index_occupation(classification: BandClassification, ieta: int, ik: int) -> np.ndarray:
    """Return Zhang CNP reference occupations by band index.

    The two CNP flat bands are classified explicitly: remote valence and the
    lower flat band are occupied; the upper flat band and remote conduction
    bands are empty.  This avoids assigning f=1/2 at numerical Dirac zeroes.
    """

    if classification.flat_indices.shape[2] != 2:
        raise ValueError(
            "cnp_index occupation expects exactly two flat bands per valley/k; "
            f"got {classification.flat_indices.shape[2]}."
        )
    occ = np.zeros(classification.n_band, dtype=float)
    occ[np.asarray(classification.remote_below_mask[int(ieta), int(ik)], dtype=bool)] = 1.0
    lower_flat = int(np.min(classification.flat_indices[int(ieta), int(ik)]))
    occ[lower_flat] = 1.0
    return occ


def _occupation_for_mode(
    energies_mev: np.ndarray,
    classification: BandClassification,
    ieta: int,
    ik: int,
    *,
    occupation_mode: str,
    fermi_level_mev: float,
    fermi_tol_mev: float,
) -> np.ndarray:
    mode = str(occupation_mode).strip().lower().replace("-", "_")
    if mode in {"cnp_index", "index", "zhang_cnp"}:
        return cnp_index_occupation(classification, ieta, ik)
    if mode in {"energy_step", "step", "fermi_step"}:
        return occupation_step(
            energies_mev,
            fermi_level_mev=fermi_level_mev,
            fermi_tol_mev=fermi_tol_mev,
        )
    raise ValueError(f"Unsupported cRPA occupation_mode: {occupation_mode!r}")


def lindhard_matrix(
    left_energies_mev: np.ndarray,
    right_energies_mev: np.ndarray,
    *,
    eta_mev: float = 1.0,
    fermi_level_mev: float = 0.0,
    fermi_tol_mev: float = 1.0e-10,
    left_occupations: np.ndarray | None = None,
    right_occupations: np.ndarray | None = None,
) -> np.ndarray:
    """Return chi0_mn(q) with rows at k+q and columns at k."""

    left_e = np.asarray(left_energies_mev, dtype=float)
    right_e = np.asarray(right_energies_mev, dtype=float)
    if left_occupations is None:
        f_left = occupation_step(left_e, fermi_level_mev=fermi_level_mev, fermi_tol_mev=fermi_tol_mev)
    else:
        f_left = np.asarray(left_occupations, dtype=float)
    if right_occupations is None:
        f_right = occupation_step(right_e, fermi_level_mev=fermi_level_mev, fermi_tol_mev=fermi_tol_mev)
    else:
        f_right = np.asarray(right_occupations, dtype=float)
    if f_left.shape != left_e.shape:
        raise ValueError(f"left occupation shape {f_left.shape} does not match left energies {left_e.shape}")
    if f_right.shape != right_e.shape:
        raise ValueError(f"right occupation shape {f_right.shape} does not match right energies {right_e.shape}")
    numerator = f_left[:, None] - f_right[None, :]
    denominator = right_e[None, :] - left_e[:, None]
    if eta_mev <= 0.0:
        out = np.zeros_like(denominator, dtype=float)
        mask = np.abs(denominator) > 0.0
        out[mask] = numerator[mask] / denominator[mask]
        return out
    eta = float(eta_mev)
    return numerator * denominator / (denominator * denominator + eta * eta)


def _compute_chi0(
    solution: AllBandBMSolution,
    classification: BandClassification,
    grid: CRPAKGrid,
    q_index: int | tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    *,
    pair_mode: str,
    eta_mev: float = 1.0,
    spin_degeneracy: float = 2.0,
    fermi_level_mev: float = 0.0,
    fermi_tol_mev: float = 1.0e-10,
    symmetrize: bool = True,
    form_factor_mode: str = PRODUCTION_FORM_FACTOR_MODE,
    occupation_mode: str = "cnp_index",
) -> np.ndarray:
    """Compute static polarizability for one q_tilde."""

    if solution.nk != grid.nk:
        raise ValueError(f"solution/grid nk mismatch: {solution.nk} != {grid.nk}")
    pair_mode = str(pair_mode).lower()
    if pair_mode not in {"full", "flat_flat", "constrained"}:
        raise ValueError(f"Unsupported cRPA pair_mode: {pair_mode}")
    n_q = len(q_shifts)
    chi0 = np.zeros((n_q, n_q), dtype=np.complex128)
    prefactor = float(spin_degeneracy) / float(grid.nk)

    for ieta in range(solution.n_eta):
        for ik in range(solution.nk):
            ikq, wrap = grid.shifted_index_and_wrap(ik, q_index)
            left_vecs = solution.uk[:, :, ieta, ikq]
            right_vecs = solution.uk[:, :, ieta, ik]
            wrapped_q_shifts = tuple((int(qm) + wrap[0], int(qn) + wrap[1]) for qm, qn in q_shifts)
            lambda_stack = compute_lambda_stack(
                left_vecs,
                right_vecs,
                grid_shape=solution.grid_shape,
                q_shifts=wrapped_q_shifts,
                local_basis_size=solution.nlocal,
                form_factor_mode=form_factor_mode,
            )
            lambdas = np.moveaxis(lambda_stack, 0, -1)
            left_energies = solution.spectrum[:, ieta, ikq]
            right_energies = solution.spectrum[:, ieta, ik]
            lindhard = lindhard_matrix(
                left_energies,
                right_energies,
                eta_mev=eta_mev,
                fermi_level_mev=fermi_level_mev,
                fermi_tol_mev=fermi_tol_mev,
                left_occupations=_occupation_for_mode(
                    left_energies,
                    classification,
                    ieta,
                    ikq,
                    occupation_mode=occupation_mode,
                    fermi_level_mev=fermi_level_mev,
                    fermi_tol_mev=fermi_tol_mev,
                ),
                right_occupations=_occupation_for_mode(
                    right_energies,
                    classification,
                    ieta,
                    ik,
                    occupation_mode=occupation_mode,
                    fermi_level_mev=fermi_level_mev,
                    fermi_tol_mev=fermi_tol_mev,
                ),
            )
            left_flat = classification.flat_mask[ieta, ikq]
            right_flat = classification.flat_mask[ieta, ik]
            flat_flat = np.logical_and(left_flat[:, None], right_flat[None, :])
            if pair_mode == "constrained":
                lindhard[flat_flat] = 0.0
            elif pair_mode == "flat_flat":
                lindhard[~flat_flat] = 0.0
            chi0 += prefactor * np.einsum(
                "mnQ,mn,mnP->QP",
                lambdas.conjugate(),
                lindhard,
                lambdas,
                optimize=True,
            )

    if symmetrize:
        chi0 = 0.5 * (chi0 + chi0.conjugate().T)
    return chi0


def compute_full_chi0(
    solution: AllBandBMSolution,
    classification: BandClassification,
    grid: CRPAKGrid,
    q_index: int | tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    **kwargs: object,
) -> np.ndarray:
    return _compute_chi0(
        solution,
        classification,
        grid,
        q_index,
        q_shifts,
        pair_mode="full",
        **kwargs,
    )


def compute_flat_flat_chi0(
    solution: AllBandBMSolution,
    classification: BandClassification,
    grid: CRPAKGrid,
    q_index: int | tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    **kwargs: object,
) -> np.ndarray:
    return _compute_chi0(
        solution,
        classification,
        grid,
        q_index,
        q_shifts,
        pair_mode="flat_flat",
        **kwargs,
    )


def compute_constrained_chi0(
    solution: AllBandBMSolution,
    classification: BandClassification,
    grid: CRPAKGrid,
    q_index: int | tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    **kwargs: object,
) -> np.ndarray:
    """Compute the constrained static polarizability for one q_tilde."""

    return _compute_chi0(
        solution,
        classification,
        grid,
        q_index,
        q_shifts,
        pair_mode="constrained",
        **kwargs,
    )


def compute_constrained_chi0_by_subtraction(
    solution: AllBandBMSolution,
    classification: BandClassification,
    grid: CRPAKGrid,
    q_index: int | tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    **kwargs: object,
) -> np.ndarray:
    full = compute_full_chi0(solution, classification, grid, q_index, q_shifts, **kwargs)
    flat_flat = compute_flat_flat_chi0(solution, classification, grid, q_index, q_shifts, **kwargs)
    return full - flat_flat


def constrained_sum_identity_error(
    solution: AllBandBMSolution,
    classification: BandClassification,
    grid: CRPAKGrid,
    q_index: int | tuple[int, int],
    q_shifts: tuple[tuple[int, int], ...],
    **kwargs: object,
) -> float:
    direct = compute_constrained_chi0(solution, classification, grid, q_index, q_shifts, **kwargs)
    subtraction = compute_constrained_chi0_by_subtraction(
        solution,
        classification,
        grid,
        q_index,
        q_shifts,
        **kwargs,
    )
    return float(np.max(np.abs(direct - subtraction)))
