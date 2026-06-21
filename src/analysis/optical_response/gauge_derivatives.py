from __future__ import annotations

import numpy as np

from .gauge_data import GeneralizedDerivativeData, PairGeneralizedDerivativeData

def covariant_derivative_matrix(matrix: np.ndarray, comma_derivative: np.ndarray, dcov: np.ndarray) -> np.ndarray:
    """Full-band covariant derivative ``A_{;d} = ∂_d A - D^d A + A D^d``.

    This is the full-matrix version of the WannierBerri
    ``Matrix_GenDer_ln`` pattern.  Use
    :func:`wannierberri_matrix_gen_derivative_ln` when an exact comparison to
    WannierBerri's ``ln``/``nn`` block API is needed.

    Parameters
    ----------
    matrix:
        Shape ``(nb, nb, *extra)``.
    comma_derivative:
        Shape ``(nb, nb, *extra, ndim)``.
    dcov:
        Shape ``(ndim, nb, nb)``.
    """

    A = np.asarray(matrix, dtype=np.complex128)
    dA = np.asarray(comma_derivative, dtype=np.complex128)
    D = np.asarray(dcov, dtype=np.complex128)
    if A.ndim < 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must have shape (nb,nb,*extra), got {A.shape}")
    nb = A.shape[0]
    if D.ndim != 3 or D.shape[1:] != (nb, nb):
        raise ValueError(f"dcov must have shape (ndim,{nb},{nb}), got {D.shape}")
    ndim = D.shape[0]
    expected_dA = A.shape + (ndim,)
    if dA.shape != expected_dA:
        raise ValueError(f"comma_derivative has shape {dA.shape}, expected {expected_dA}")

    extra_shape = A.shape[2:]
    nextra = int(np.prod(extra_shape, dtype=int)) if extra_shape else 1
    A_flat = A.reshape(nb, nb, nextra)
    dA_flat = dA.reshape(nb, nb, nextra, ndim)
    out = np.empty((nb, nb, nextra, ndim), dtype=np.complex128)
    for axis in range(ndim):
        for iextra in range(nextra):
            block = A_flat[:, :, iextra]
            out[:, :, iextra, axis] = dA_flat[:, :, iextra, axis] - D[axis] @ block + block @ D[axis]
    return out.reshape(A.shape + (ndim,))

def wannierberri_matrix_gen_derivative_ln(
    matrix: np.ndarray,
    comma_derivative: np.ndarray,
    dcov: np.ndarray,
    inn: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Exact ``Matrix_GenDer_ln.ln`` block formula from WannierBerri.

    Mirrors ``reference/upstream/wannier-berri/wannierberri/formula/formula.py``:

    ```text
    summ = dA.ln
    summ -= einsum("mld,lnb...->mnb...d", D.ln, A.nn)
    summ += einsum("mlb...,lnd->mnb...d", A.ll, D.ln)
    ```

    Here ``inn`` and ``out`` are band-index arrays, ``matrix`` has shape
    ``(nb,nb,*extra)``, ``comma_derivative`` has shape
    ``(nb,nb,*extra,ndim)``, and ``dcov`` has shape ``(ndim,nb,nb)``.
    """

    A = np.asarray(matrix, dtype=np.complex128)
    dA = np.asarray(comma_derivative, dtype=np.complex128)
    D = np.asarray(dcov, dtype=np.complex128)
    inn = np.asarray(inn, dtype=int)
    out = np.asarray(out, dtype=int)
    if A.ndim < 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must have shape (nb,nb,*extra), got {A.shape}")
    nb = A.shape[0]
    if D.ndim != 3 or D.shape[1:] != (nb, nb):
        raise ValueError(f"dcov must have shape (ndim,{nb},{nb}), got {D.shape}")
    expected_dA = A.shape + (D.shape[0],)
    if dA.shape != expected_dA:
        raise ValueError(f"comma_derivative has shape {dA.shape}, expected {expected_dA}")
    summ = dA[np.ix_(out, inn)].copy()
    D_ln = np.moveaxis(D[:, out][:, :, inn], 0, -1)  # (out, inn, ndim)
    A_nn = A[np.ix_(inn, inn)]
    A_ll = A[np.ix_(out, out)]
    summ -= np.einsum("mld,ln...->mn...d", D_ln, A_nn, optimize=True)
    summ += np.einsum("ml...,lnd->mn...d", A_ll, D_ln, optimize=True)
    return summ

def wannierberri_matrix_gen_derivative_nn(
    matrix: np.ndarray,
    comma_derivative: np.ndarray,
    dcov: np.ndarray,
    inn: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Exact ``Matrix_GenDer_ln.nn`` block formula from WannierBerri."""

    A = np.asarray(matrix, dtype=np.complex128)
    dA = np.asarray(comma_derivative, dtype=np.complex128)
    D = np.asarray(dcov, dtype=np.complex128)
    inn = np.asarray(inn, dtype=int)
    out = np.asarray(out, dtype=int)
    if A.ndim < 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must have shape (nb,nb,*extra), got {A.shape}")
    nb = A.shape[0]
    if D.ndim != 3 or D.shape[1:] != (nb, nb):
        raise ValueError(f"dcov must have shape (ndim,{nb},{nb}), got {D.shape}")
    expected_dA = A.shape + (D.shape[0],)
    if dA.shape != expected_dA:
        raise ValueError(f"comma_derivative has shape {dA.shape}, expected {expected_dA}")
    summ = dA[np.ix_(inn, inn)].copy()
    D_nl = np.moveaxis(D[:, inn][:, :, out], 0, -1)  # (inn, out, ndim)
    D_ln = np.moveaxis(D[:, out][:, :, inn], 0, -1)  # (out, inn, ndim)
    A_ln = A[np.ix_(out, inn)]
    A_nl = A[np.ix_(inn, out)]
    summ -= np.einsum("mld,ln...->mn...d", D_nl, A_ln, optimize=True)
    summ += np.einsum("ml...,lnd->mn...d", A_nl, D_ln, optimize=True)
    return summ

def berry_connection_generalized_derivative(
    velocity_h: np.ndarray,
    energies: np.ndarray,
    *,
    denominator_cutoff: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta: float | None = None,
) -> GeneralizedDerivativeData:
    """Gauge-free generalized derivative of the interband Berry connection.

    This is the Hamiltonian-derivative sum rule used for shift current.  It is
    compatible with the WannierBerri convention ``A_H=iD_H`` and includes the
    extra ``-i W^{ab}_{nm}/(E_n-E_m)`` term when ``second_velocity_h`` is
    supplied.  The W term is required for nonlinear tight-binding Hamiltonians;
    it vanishes for linear continuum Dirac blocks.

    If ``principal_value_eta`` is supplied and positive, intermediate-state
    denominators use the WannierBerri/Wannier90 shift-current principal-value
    regularizer ``x/(x^2+eta^2)``.  The direct optical denominator
    ``E_n-E_m`` remains exact, matching ``ShiftCurrentFormula``'s use of
    ``D_H_Pval`` only inside generalized-derivative commutator terms.
    """

    V = np.asarray(velocity_h, dtype=np.complex128)
    e = np.asarray(energies, dtype=float)
    if V.ndim != 3 or V.shape[1:] != (e.size, e.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {V.shape} for nb={e.size}")
    W = None if second_velocity_h is None else np.asarray(second_velocity_h, dtype=np.complex128)
    if W is not None and W.shape != (V.shape[0], V.shape[0], e.size, e.size):
        raise ValueError(f"second_velocity_h has shape {W.shape}, expected {(V.shape[0], V.shape[0], e.size, e.size)}")

    ndim, nb, _ = V.shape
    values = np.zeros((ndim, ndim, nb, nb), dtype=np.complex128)
    skipped = 0
    cutoff = float(denominator_cutoff)
    pv_eta = None if principal_value_eta is None else float(principal_value_eta)
    if pv_eta is not None and pv_eta < 0.0:
        raise ValueError(f"principal_value_eta must be non-negative, got {principal_value_eta}")
    for n in range(nb):
        for m in range(nb):
            if n == m:
                continue
            e_nm = float(e[n] - e[m])
            if abs(e_nm) <= cutoff:
                skipped += 1
                continue
            delta_v = V[:, n, n] - V[:, m, m]
            for deriv_axis in range(ndim):
                for conn_axis in range(ndim):
                    total = 0.0j
                    for ell in range(nb):
                        if ell == n or ell == m:
                            continue
                        e_lm = float(e[ell] - e[m])
                        e_nl = float(e[n] - e[ell])
                        if abs(e_lm) <= cutoff or abs(e_nl) <= cutoff:
                            skipped += 1
                            continue
                        inv_lm = (1.0 / e_lm) if not pv_eta else (e_lm / (e_lm * e_lm + pv_eta * pv_eta))
                        inv_nl = (1.0 / e_nl) if not pv_eta else (e_nl / (e_nl * e_nl + pv_eta * pv_eta))
                        total += (
                            V[conn_axis, n, ell] * V[deriv_axis, ell, m] * inv_lm
                            - V[deriv_axis, n, ell] * V[conn_axis, ell, m] * inv_nl
                        )
                    total *= 1.0j / e_nm
                    total += (
                        1.0j
                        / (e_nm * e_nm)
                        * (V[deriv_axis, n, m] * delta_v[conn_axis] + V[conn_axis, n, m] * delta_v[deriv_axis])
                    )
                    if W is not None:
                        total += -1.0j * W[deriv_axis, conn_axis, n, m] / e_nm
                    values[deriv_axis, conn_axis, n, m] = total
    return GeneralizedDerivativeData(values=values, skipped_small_denominators=skipped)

def berry_connection_pair(velocity_h: np.ndarray, energies: np.ndarray, n: int, m: int, *, denominator_cutoff: float = 1.0e-10) -> np.ndarray:
    """Return ``A^a_{nm} = -i <u_n|∂_a H|u_m>/(E_n-E_m)`` for one pair."""

    V = np.asarray(velocity_h, dtype=np.complex128)
    e = np.asarray(energies, dtype=float)
    n = int(n)
    m = int(m)
    if V.ndim != 3 or V.shape[1:] != (e.size, e.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {V.shape} for nb={e.size}")
    if n == m or n < 0 or m < 0 or n >= e.size or m >= e.size:
        raise ValueError(f"Invalid band pair ({n},{m}) for nb={e.size}")
    denom = float(e[n] - e[m])
    if abs(denom) <= float(denominator_cutoff):
        return np.zeros(V.shape[0], dtype=np.complex128)
    return -1.0j * V[:, n, m] / denom

def berry_connection_generalized_derivative_pair(
    velocity_h: np.ndarray,
    energies: np.ndarray,
    n: int,
    m: int,
    *,
    denominator_cutoff: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta: float | None = None,
) -> PairGeneralizedDerivativeData:
    """Selected-pair generalized derivative in ``O(ndim^2 * nb)`` time.

    This is the production path for selected transitions.  It returns the same
    ``values[:, :, n, m]`` entries as
    :func:`berry_connection_generalized_derivative` without constructing the
    full ``(ndim,ndim,nb,nb)`` tensor.  ``principal_value_eta`` has the same
    WannierBerri/Wannier90 meaning as in the full-tensor function.
    """

    V = np.asarray(velocity_h, dtype=np.complex128)
    e = np.asarray(energies, dtype=float)
    n = int(n)
    m = int(m)
    if V.ndim != 3 or V.shape[1:] != (e.size, e.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {V.shape} for nb={e.size}")
    W = None if second_velocity_h is None else np.asarray(second_velocity_h, dtype=np.complex128)
    if W is not None and W.shape != (V.shape[0], V.shape[0], e.size, e.size):
        raise ValueError(f"second_velocity_h has shape {W.shape}, expected {(V.shape[0], V.shape[0], e.size, e.size)}")
    if n == m or n < 0 or m < 0 or n >= e.size or m >= e.size:
        raise ValueError(f"Invalid band pair ({n},{m}) for nb={e.size}")

    ndim, nb, _ = V.shape
    cutoff = float(denominator_cutoff)
    pv_eta = None if principal_value_eta is None else float(principal_value_eta)
    if pv_eta is not None and pv_eta < 0.0:
        raise ValueError(f"principal_value_eta must be non-negative, got {principal_value_eta}")
    e_nm = float(e[n] - e[m])
    if abs(e_nm) <= cutoff:
        return PairGeneralizedDerivativeData(values=np.zeros((ndim, ndim), dtype=np.complex128), skipped_small_denominators=1)

    ell = np.arange(nb)
    mask = (ell != n) & (ell != m)
    e_lm = e - e[m]
    e_nl = e[n] - e
    valid = mask & (np.abs(e_lm) > cutoff) & (np.abs(e_nl) > cutoff)
    skipped = int(np.count_nonzero(mask) - np.count_nonzero(valid))

    values = np.zeros((ndim, ndim), dtype=np.complex128)
    delta_v = V[:, n, n] - V[:, m, m]
    for deriv_axis in range(ndim):
        for conn_axis in range(ndim):
            if np.any(valid):
                inv_lm = 1.0 / e_lm[valid] if not pv_eta else e_lm[valid] / (e_lm[valid] * e_lm[valid] + pv_eta * pv_eta)
                inv_nl = 1.0 / e_nl[valid] if not pv_eta else e_nl[valid] / (e_nl[valid] * e_nl[valid] + pv_eta * pv_eta)
                term = (
                    V[conn_axis, n, valid] * V[deriv_axis, valid, m] * inv_lm
                    - V[deriv_axis, n, valid] * V[conn_axis, valid, m] * inv_nl
                )
                total = np.sum(term, dtype=np.complex128)
            else:
                total = 0.0j
            total *= 1.0j / e_nm
            total += (
                1.0j
                / (e_nm * e_nm)
                * (V[deriv_axis, n, m] * delta_v[conn_axis] + V[conn_axis, n, m] * delta_v[deriv_axis])
            )
            if W is not None:
                total += -1.0j * W[deriv_axis, conn_axis, n, m] / e_nm
            values[deriv_axis, conn_axis] = total
    return PairGeneralizedDerivativeData(values=values, skipped_small_denominators=skipped)

def wannierberri_shift_current_internal_imn(
    velocity_h: np.ndarray,
    energies: np.ndarray,
    *,
    sc_eta: float,
    second_velocity_h: np.ndarray | None = None,
    denominator_cutoff: float = 1.0e-10,
) -> np.ndarray:
    """Port of WannierBerri ``ShiftCurrentFormula`` internal-term integrand.

    Returns ``Imn[n,m,a,b,c]`` for one k point, matching
    ``reference/upstream/wannier-berri/wannierberri/calculators/dynamic.py``
    with ``external_terms=False``.  The returned real tensor is the quantity
    that WannierBerri multiplies by occupation differences and spectral delta
    functions.  Optical indices ``b,c`` are symmetrized as in the reference.

    This is intentionally separate from the phase-derivative helpers above:
    it is a line-by-line dynamic shift-current formula port including the
    Wannier90/WannierBerri ``sc_eta`` principal-value regularizer.
    """

    V_axis = np.asarray(velocity_h, dtype=np.complex128)
    e = np.asarray(energies, dtype=float)
    if V_axis.ndim != 3 or V_axis.shape[1:] != (e.size, e.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {V_axis.shape} for nb={e.size}")
    ndim, nb, _ = V_axis.shape
    W_axis = None if second_velocity_h is None else np.asarray(second_velocity_h, dtype=np.complex128)
    if W_axis is not None and W_axis.shape != (ndim, ndim, nb, nb):
        raise ValueError(f"second_velocity_h has shape {W_axis.shape}, expected {(ndim, ndim, nb, nb)}")
    eta = float(sc_eta)
    if eta < 0.0:
        raise ValueError(f"sc_eta must be non-negative, got {sc_eta}")

    # WannierBerri index convention at one k: V_H[n,m,a].
    V = np.moveaxis(V_axis, 0, -1)
    del2e = np.zeros((nb, nb, ndim, ndim), dtype=np.complex128)
    if W_axis is not None:
        del2e = np.transpose(W_axis, (2, 3, 0, 1))  # (n,m,axis0,axis1)

    d_eig = e[:, None] - e[None, :]
    inv = np.zeros_like(d_eig, dtype=float)
    mask = np.abs(d_eig) > float(denominator_cutoff)
    inv[mask] = 1.0 / d_eig[mask]

    D_H = -V * inv[:, :, None]
    if eta == 0.0:
        pval = inv.copy()
    else:
        pval = np.zeros_like(d_eig, dtype=float)
        pval[mask] = d_eig[mask] / (d_eig[mask] * d_eig[mask] + eta * eta)
    D_H_Pval = -V * pval[:, :, None]

    sum_HD = (
        np.einsum("nlc,lma->nmca", V, D_H_Pval, optimize=True)
        - np.einsum("nnc,nma->nmca", V, D_H_Pval, optimize=True)
        - np.einsum("nla,lmc->nmca", D_H_Pval, V, optimize=True)
        + np.einsum("nma,mmc->nmca", D_H_Pval, V, optimize=True)
    )
    DV_bit = (
        np.einsum("nmc,nna->nmca", D_H, V, optimize=True)
        - np.einsum("nmc,mma->nmca", D_H, V, optimize=True)
        + np.einsum("nma,nnc->nmca", D_H, V, optimize=True)
        - np.einsum("nma,mmc->nmca", D_H, V, optimize=True)
    )
    # WannierBerri uses ``data_K.dEig_inv.swapaxes(2, 1)`` for this final
    # optical-pair denominator, while ``D_H`` and ``D_H_Pval`` above use the
    # unswapped denominator.  Keep that asymmetry for line-by-line parity with
    # ``calculators/dynamic.py::ShiftCurrentFormula``.
    A_gen_der = 1.0j * (del2e + sum_HD + DV_bit) * inv.T[:, :, None, None]
    A_H = 1.0j * D_H
    imn = -np.einsum("nmca,mnb->nmabc", A_gen_der, A_H, optimize=True).imag
    return imn + np.swapaxes(imn, 3, 4)

def wannierberri_shift_current_group_trace(
    imn: np.ndarray,
    initial_group: np.ndarray | list[int] | tuple[int, ...],
    final_group: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    """Trace/sum WannierBerri shift-current ``Imn`` over two band groups."""

    tensor = np.asarray(imn, dtype=float)
    inn = np.asarray(initial_group, dtype=int)
    out = np.asarray(final_group, dtype=int)
    if tensor.ndim != 5 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f"imn must have shape (nb,nb,ndim,ndim,ndim), got {tensor.shape}")
    return tensor[np.ix_(inn, out)].sum(axis=(0, 1))

__all__ = [
    "covariant_derivative_matrix",
    "wannierberri_matrix_gen_derivative_ln",
    "wannierberri_matrix_gen_derivative_nn",
    "berry_connection_generalized_derivative",
    "berry_connection_pair",
    "berry_connection_generalized_derivative_pair",
    "wannierberri_shift_current_internal_imn",
    "wannierberri_shift_current_group_trace",
]
