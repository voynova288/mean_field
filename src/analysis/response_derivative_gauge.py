from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HamiltonianGaugeData:
    """Hamiltonian-gauge derivative ingredients.

    This mirrors the WannierBerri convention used in
    ``wannierberri/data_K/data_K.py``:

    ``D_H = -Xbar('Ham', 1) * dEig_inv[..., None]`` and
    ``A_H = 1j * D_H`` when no external position/Berry-connection terms are
    present.  Eigenvectors are assumed to be stored as columns.
    """

    energies: np.ndarray  # (nb,)
    eigenvectors: np.ndarray  # (basis, nb)
    velocity_h: np.ndarray  # (ndim, nb, nb), <u_n|d_a H|u_m>
    energy_difference_inverse: np.ndarray  # (nb, nb), 1/(E_n-E_m), diag/small gaps -> 0
    dcov: np.ndarray  # (ndim, nb, nb), WannierBerri D_H^a
    berry_connection: np.ndarray  # (ndim, nb, nb), A_H^a = i D_H^a + external terms
    second_velocity_h: np.ndarray | None = None  # (ndim, ndim, nb, nb), <u_n|d_a d_b H|u_m>


@dataclass(frozen=True)
class GeneralizedDerivativeData:
    """Generalized derivative of the Berry connection.

    ``values[deriv_axis, connection_axis, n, m]`` stores
    ``(A^{connection_axis}_{n m})_{; deriv_axis}`` in the same index/order
    convention as the existing shift-current code.
    """

    values: np.ndarray
    skipped_small_denominators: int


@dataclass(frozen=True)
class PairGeneralizedDerivativeData:
    """Selected-pair generalized derivative.

    ``values[deriv_axis, connection_axis]`` stores
    ``(A^{connection_axis}_{n m})_{; deriv_axis}`` for one pair ``(n,m)``.
    """

    values: np.ndarray
    skipped_small_denominators: int


def energy_difference_inverse(energies: np.ndarray, *, cutoff: float = 1.0e-10) -> np.ndarray:
    """Return ``1/(E_n-E_m)`` with diagonal/small denominators set to zero.

    This follows WannierBerri ``data_K.py::dEig_inv`` in spirit: near-degenerate
    denominators are excluded rather than assigned arbitrary phases.
    """

    e = np.asarray(energies, dtype=float)
    if e.ndim != 1:
        raise ValueError(f"energies must be 1D, got shape {e.shape}")
    diff = e[:, None] - e[None, :]
    out = np.zeros_like(diff, dtype=float)
    mask = np.abs(diff) > float(cutoff)
    out[mask] = 1.0 / diff[mask]
    return out


def degenerate_band_groups(energies: np.ndarray, *, threshold: float = 1.0e-4) -> list[tuple[int, int]]:
    """Return contiguous WannierBerri-style degenerate band groups.

    Each group is a half-open interval ``(start, stop)``.  Adjacent bands are
    placed in the same group when their energy separation is at most
    ``threshold``.  This mirrors the band-group logic used before
    WannierBerri dynamic calculators call ``trace_ln`` over subspaces.
    """

    e = np.asarray(energies, dtype=float)
    if e.ndim != 1:
        raise ValueError(f"energies must be 1D, got shape {e.shape}")
    if e.size == 0:
        return []
    borders = [0]
    gaps = np.diff(e)
    borders.extend((np.where(gaps > float(threshold))[0] + 1).tolist())
    borders.append(int(e.size))
    return [(int(a), int(b)) for a, b in zip(borders, borders[1:])]


def group_indices(group: tuple[int, int]) -> np.ndarray:
    """Convert a half-open band group to an integer index array."""

    start, stop = int(group[0]), int(group[1])
    if stop < start:
        raise ValueError(f"Invalid group {group}")
    return np.arange(start, stop, dtype=int)


def random_block_unitary(groups: list[tuple[int, int]], nb: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Return a block-diagonal random unitary for gauge-covariance tests.

    WannierBerri's ``random_gauge`` applies random unitary rotations inside
    degenerate subspaces.  This helper provides the same validation pattern for
    local tests.  Singleton groups receive a random U(1) phase.
    """

    generator = np.random.default_rng(rng)
    unitary = np.eye(int(nb), dtype=np.complex128)
    for start, stop in groups:
        start = int(start)
        stop = int(stop)
        size = stop - start
        if size <= 0:
            raise ValueError(f"Invalid group {(start, stop)}")
        z = generator.normal(size=(size, size)) + 1.0j * generator.normal(size=(size, size))
        q, r = np.linalg.qr(z)
        phases = np.diag(r)
        phases = phases / np.where(np.abs(phases) > 0.0, np.abs(phases), 1.0)
        unitary[start:stop, start:stop] = q * phases.conjugate()[None, :]
    return unitary


def apply_band_gauge_to_matrix(matrix: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    """Apply ``X -> G† X G`` to a band-space covariant matrix.

    ``matrix`` has shape ``(nb,nb,*extra)`` and ``gauge`` has shape
    ``(nb,nb)``.  This is the finite-dimensional version of the Hamiltonian
    gauge covariance used by WannierBerri.
    """

    X = np.asarray(matrix, dtype=np.complex128)
    G = np.asarray(gauge, dtype=np.complex128)
    if X.ndim < 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"matrix must have shape (nb,nb,*extra), got {X.shape}")
    nb = X.shape[0]
    if G.shape != (nb, nb):
        raise ValueError(f"gauge has shape {G.shape}, expected {(nb, nb)}")
    return np.einsum("am,ab...,bn->mn...", G.conjugate(), X, G, optimize=True)


def apply_band_gauge_to_axis_matrix(matrix: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    """Apply ``X^a -> G† X^a G`` to ``(ndim,nb,nb,*extra)`` arrays."""

    X = np.asarray(matrix, dtype=np.complex128)
    G = np.asarray(gauge, dtype=np.complex128)
    if X.ndim < 3 or X.shape[1] != X.shape[2]:
        raise ValueError(f"matrix must have shape (ndim,nb,nb,*extra), got {X.shape}")
    nb = X.shape[1]
    if G.shape != (nb, nb):
        raise ValueError(f"gauge has shape {G.shape}, expected {(nb, nb)}")
    return np.einsum("am,xab...,bn->xmn...", G.conjugate(), X, G, optimize=True)


def trace_subspace(matrix: np.ndarray, group: tuple[int, int]) -> np.ndarray:
    """Trace a covariant matrix over a band subspace.

    This is the scalar/subspace invariant that survives arbitrary unitary
    rotations inside the group, analogous to WannierBerri's ``trace_ln`` use in
    dynamic calculators.
    """

    X = np.asarray(matrix, dtype=np.complex128)
    idx = group_indices(group)
    if X.ndim < 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"matrix must have shape (nb,nb,*extra), got {X.shape}")
    return np.trace(X[np.ix_(idx, idx)], axis1=0, axis2=1)


def matrix_in_eigenbasis(eigenvectors: np.ndarray, operators: np.ndarray) -> np.ndarray:
    """Transform one or more basis-space operators into Hamiltonian gauge.

    Parameters
    ----------
    eigenvectors:
        Matrix with eigenvectors as columns, shape ``(basis, nb)``.
    operators:
        Either ``(basis,basis)`` or ``(..., basis, basis)``.

    Returns
    -------
    np.ndarray
        Shape ``(..., nb, nb)`` with entries ``<u_n|O|u_m>``.
    """

    u = np.asarray(eigenvectors, dtype=np.complex128)
    if u.ndim != 2:
        raise ValueError(f"eigenvectors must be 2D with eigenvectors as columns, got {u.shape}")
    ops = np.asarray(operators, dtype=np.complex128)
    if ops.shape[-2:] != (u.shape[0], u.shape[0]):
        raise ValueError(f"operator trailing shape {ops.shape[-2:]} incompatible with eigenvectors {u.shape}")
    return np.einsum("bn,...bc,cm->...nm", u.conjugate(), ops, u, optimize=True)


def hamiltonian_gauge_data(
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    dhdk: np.ndarray,
    *,
    denominator_cutoff: float = 1.0e-10,
    d2hdk: np.ndarray | None = None,
    external_connection: np.ndarray | None = None,
) -> HamiltonianGaugeData:
    """Build WannierBerri-style Hamiltonian-gauge derivative ingredients.

    ``dhdk`` must have shape ``(ndim,basis,basis)``.  ``d2hdk`` is optional and
    must have shape ``(ndim,ndim,basis,basis)`` when supplied.  Momentum units
    are whatever units the derivatives use; the caller must document them.
    """

    e = np.asarray(energies, dtype=float)
    u = np.asarray(eigenvectors, dtype=np.complex128)
    raw_dh = np.asarray(dhdk, dtype=np.complex128)
    if e.ndim != 1:
        raise ValueError(f"energies must be 1D, got shape {e.shape}")
    if u.shape != (raw_dh.shape[-1], e.size):
        raise ValueError(f"eigenvectors shape {u.shape} incompatible with dhdk {raw_dh.shape} and energies {e.shape}")
    if raw_dh.ndim != 3 or raw_dh.shape[-2:] != (u.shape[0], u.shape[0]):
        raise ValueError(f"dhdk must have shape (ndim,basis,basis), got {raw_dh.shape}")

    velocity_h = matrix_in_eigenbasis(u, raw_dh)
    inv = energy_difference_inverse(e, cutoff=float(denominator_cutoff))
    dcov = -velocity_h * inv[None, :, :]
    berry = 1.0j * dcov
    if external_connection is not None:
        abar = np.asarray(external_connection, dtype=np.complex128)
        if abar.shape != berry.shape:
            raise ValueError(f"external_connection has shape {abar.shape}, expected {berry.shape}")
        berry = berry + abar

    second_velocity_h = None
    if d2hdk is not None:
        raw_d2 = np.asarray(d2hdk, dtype=np.complex128)
        expected = (raw_dh.shape[0], raw_dh.shape[0], u.shape[0], u.shape[0])
        if raw_d2.shape != expected:
            raise ValueError(f"d2hdk has shape {raw_d2.shape}, expected {expected}")
        second_velocity_h = matrix_in_eigenbasis(u, raw_d2)

    return HamiltonianGaugeData(
        energies=e,
        eigenvectors=u,
        velocity_h=velocity_h,
        energy_difference_inverse=inv,
        dcov=dcov,
        berry_connection=berry,
        second_velocity_h=second_velocity_h,
    )


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


def shift_integrand_from_pair_generalized_derivative(
    berry_connection: np.ndarray,
    pair_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
) -> float:
    """Return ``Im[A_mn^b (A_nm^b)_{;a}]`` from a selected-pair derivative.

    ``pair_gen_der`` has shape ``(ndim,ndim)`` and stores the generalized
    derivative for ``(initial_band, final_band)`` only.
    """

    A = np.asarray(berry_connection, dtype=np.complex128)
    G = np.asarray(pair_gen_der, dtype=np.complex128)
    b = int(optical_axis)
    a = int(deriv_axis)
    n = int(initial_band)
    m = int(final_band)
    if G.ndim != 2:
        raise ValueError(f"pair_gen_der must have shape (ndim,ndim), got {G.shape}")
    return float(np.imag(A[b, m, n] * G[a, b]))


def shift_vector_from_pair_generalized_derivative(
    berry_connection: np.ndarray,
    pair_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
    metric_cutoff: float = 1.0e-24,
) -> float:
    """Selected-pair version of :func:`shift_vector_from_generalized_derivative`."""

    A = np.asarray(berry_connection, dtype=np.complex128)
    b = int(optical_axis)
    n = int(initial_band)
    m = int(final_band)
    metric = float(abs(A[b, m, n]) ** 2)
    if metric <= float(metric_cutoff):
        return float("nan")
    return shift_integrand_from_pair_generalized_derivative(
        A,
        pair_gen_der,
        initial_band=n,
        final_band=m,
        deriv_axis=deriv_axis,
        optical_axis=b,
    ) / metric


def shift_integrand_from_generalized_derivative(
    berry_connection: np.ndarray,
    berry_connection_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
) -> float:
    """Return ``Im[A_mn^b (A_nm^b)_{;a}]`` for a direct transition ``n -> m``.

    ``initial_band`` is the occupied/initial band ``n`` and ``final_band`` is
    the empty/final band ``m``.  The returned quantity is the gauge-invariant
    ``|A|^2 S`` factor used in shift-current integrands.
    """

    A = np.asarray(berry_connection, dtype=np.complex128)
    G = np.asarray(berry_connection_gen_der, dtype=np.complex128)
    b = int(optical_axis)
    a = int(deriv_axis)
    n = int(initial_band)
    m = int(final_band)
    return float(np.imag(A[b, m, n] * G[a, b, n, m]))


def shift_vector_from_generalized_derivative(
    berry_connection: np.ndarray,
    berry_connection_gen_der: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    deriv_axis: int,
    optical_axis: int,
    metric_cutoff: float = 1.0e-24,
) -> float:
    """Return ``S = Im[A_mn (A_nm)_;] / |A_mn|^2`` away from optical zeros."""

    A = np.asarray(berry_connection, dtype=np.complex128)
    b = int(optical_axis)
    n = int(initial_band)
    m = int(final_band)
    metric = float(abs(A[b, m, n]) ** 2)
    if metric <= float(metric_cutoff):
        return float("nan")
    return shift_integrand_from_generalized_derivative(
        A,
        berry_connection_gen_der,
        initial_band=n,
        final_band=m,
        deriv_axis=deriv_axis,
        optical_axis=b,
    ) / metric


def normalized_u1_link(evecs0: np.ndarray, evecs1: np.ndarray, band: int) -> complex:
    """Return the U(1) link ``<u_band(k)|u_band(k+dk)>/abs(...)``."""

    u0 = np.asarray(evecs0, dtype=np.complex128)
    u1 = np.asarray(evecs1, dtype=np.complex128)
    link = complex(np.vdot(u0[:, int(band)], u1[:, int(band)]))
    if abs(link) <= 1.0e-14:
        raise ValueError(f"Vanishing U(1) link for band {band}")
    return link / abs(link)


def link_shift_vector(
    evecs0: np.ndarray,
    evecs1: np.ndarray,
    berry_connection0: np.ndarray,
    berry_connection1: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    optical_axis: int,
    step: float,
) -> float:
    """Gauge-invariant Wilson-link finite-difference shift vector.

    This is the safe phase-derivative route.  It avoids differentiating a raw
    eigenvector or matrix-element phase by parallel-transporting the optical
    matrix element at ``k+step`` back to the gauge at ``k``.
    """

    n = int(initial_band)
    m = int(final_band)
    b = int(optical_axis)
    A0 = np.asarray(berry_connection0, dtype=np.complex128)
    A1 = np.asarray(berry_connection1, dtype=np.complex128)
    if A0.shape != A1.shape or A0.ndim != 3:
        raise ValueError(f"berry connections must both have shape (ndim,nb,nb), got {A0.shape} and {A1.shape}")
    link_m = normalized_u1_link(evecs0, evecs1, m)
    link_n = normalized_u1_link(evecs0, evecs1, n)
    phase_product = A1[b, m, n] * A0[b, n, m] * link_m * np.conj(link_n)
    if abs(phase_product) <= 1.0e-30:
        return float("nan")
    return float(-np.angle(phase_product) / float(step))


__all__ = [
    "GeneralizedDerivativeData",
    "HamiltonianGaugeData",
    "PairGeneralizedDerivativeData",
    "berry_connection_generalized_derivative",
    "berry_connection_generalized_derivative_pair",
    "berry_connection_pair",
    "covariant_derivative_matrix",
    "degenerate_band_groups",
    "energy_difference_inverse",
    "group_indices",
    "hamiltonian_gauge_data",
    "link_shift_vector",
    "matrix_in_eigenbasis",
    "random_block_unitary",
    "normalized_u1_link",
    "shift_integrand_from_generalized_derivative",
    "shift_integrand_from_pair_generalized_derivative",
    "shift_vector_from_generalized_derivative",
    "shift_vector_from_pair_generalized_derivative",
    "trace_subspace",
    "apply_band_gauge_to_axis_matrix",
    "apply_band_gauge_to_matrix",
    "wannierberri_matrix_gen_derivative_ln",
    "wannierberri_matrix_gen_derivative_nn",
    "wannierberri_shift_current_group_trace",
    "wannierberri_shift_current_internal_imn",
]
