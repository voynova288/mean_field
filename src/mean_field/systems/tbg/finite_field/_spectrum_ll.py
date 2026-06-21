from __future__ import annotations

from ._spectrum_shared import *  # noqa: F401,F403

def in_gamma(index: int) -> tuple[int, int]:
    """Return ``(n, gamma)`` for the original 1D LL/sublattice index."""

    i = int(index)
    if i < 0:
        raise ValueError(f"index must be nonnegative, got {index}")
    if i == 0:
        return 0, -3  # matches the original helper; gamma is unused for n=0.
    n = (i - 1) // 2 + 1
    i_gamma = (i - 1) % 2 + 1
    return int(n), int(2 * i_gamma - 3)


@lru_cache(maxsize=16)
def _in_gamma_table(n_h: int) -> tuple[Array, Array, Array]:
    """Cached ``in_gamma`` values and nonzero masks for a LL basis size."""

    n_h_int = int(n_h)
    n_values = np.empty(n_h_int, dtype=np.intp)
    gamma_values = np.empty(n_h_int, dtype=np.intp)
    for index in range(n_h_int):
        n_values[index], gamma_values[index] = in_gamma(index)
    nonzero = n_values != 0
    for arr in (n_values, gamma_values, nonzero):
        arr.setflags(write=False)
    return n_values, gamma_values, nonzero


def projector_para(vec1: complex, vec2: complex) -> float:
    return float((np.conj(vec1) * vec2).real / abs(vec2))


def projector_norm(vec1: complex, vec2: complex) -> float:
    return float((np.conj(vec1) * vec2).imag / abs(vec2))


def associated_laguerre_element(n: int, m: int, cplus: complex, cminus: complex) -> complex:
    """Matrix element used by the LL translation operator."""

    n = int(n)
    m = int(m)
    x = -float((cplus * cminus).real)
    if n >= m:
        prefactor = np.exp(-x / 2.0 + 0.5 * (gammaln(m + 1) - gammaln(n + 1)))
        val = prefactor * cplus ** (n - m) * eval_genlaguerre(m, n - m, x)
    else:
        prefactor = np.exp(-x / 2.0 + 0.5 * (gammaln(n + 1) - gammaln(m + 1)))
        val = prefactor * cminus ** (m - n) * eval_genlaguerre(n, m - n, x)
    val = complex(val)
    return 0.0 + 0.0j if abs(val) < 1e-16 else val


def associated_laguerre_matrix(n_landau: int, qvec: complex, l_b: float) -> Array:
    cplus = -1j * float(l_b) / np.sqrt(2.0) * (qvec.real - 1j * qvec.imag)
    cminus = -1j * float(l_b) / np.sqrt(2.0) * (qvec.real + 1j * qvec.imag)
    n_ll = int(n_landau)
    out = np.zeros((n_ll, n_ll), dtype=np.complex128)
    x = -float((cplus * cminus).real)
    for delta in range(n_ll):
        idx = np.arange(n_ll - delta)
        prefactor = np.exp(
            -x / 2.0 + 0.5 * (gammaln(idx + 1.0) - gammaln(idx + delta + 1.0))
        )
        laguerre = eval_genlaguerre(idx, delta, x)
        lower = prefactor * cplus**delta * laguerre
        lower = np.asarray(lower, dtype=np.complex128)
        lower[np.abs(lower) < 1e-16] = 0.0
        out[idx + delta, idx] = lower
        if delta == 0:
            continue
        upper = prefactor * cminus**delta * laguerre
        upper = np.asarray(upper, dtype=np.complex128)
        upper[np.abs(upper) < 1e-16] = 0.0
        out[idx, idx + delta] = upper
    return out


def tll_matrix(
    tunnel: Array,
    qvec: complex,
    *,
    n_landau: int,
    n_h: int,
    l_b: float,
    theta0: float,
    theta1: float,
    theta2: float,
    sigma_rotation: bool = False,
    valley: Valley = "K",
) -> Array:
    """Return the LL matrix element of ``T exp(-i q.r)``.

    This is the Python form of ``_tLL_v1`` and ``_tLL_v1_valleyKprime``.
    """

    t = np.asarray(tunnel, dtype=np.complex128)
    al = associated_laguerre_matrix(n_landau, qvec, l_b)
    out = np.zeros((n_h, n_h), dtype=np.complex128)
    if n_h == 0:
        return out
    n_values, gamma_values, _ = _in_gamma_table(n_h)
    sqrt2 = np.sqrt(2.0)
    t00 = t[0, 0]
    t01 = t[0, 1]
    t10 = t[1, 0]
    t11 = t[1, 1]
    n_pos = n_values[1:]
    gamma_pos = gamma_values[1:]
    n_row = n_pos[:, None]
    n_col = n_pos[None, :]
    gamma_row = gamma_pos[:, None]
    gamma_col = gamma_pos[None, :]
    is_kprime = valley == "Kprime"

    if not is_kprime:
        if sigma_rotation:
            phase_00 = np.exp(1j * (theta2 - theta1))
            phase_01 = np.exp(-1j * (theta1 - theta0))
            phase_10 = np.exp(1j * (theta2 - theta0))
        else:
            phase_00 = 1.0
            phase_01 = np.exp(1j * theta0)
            phase_10 = np.exp(-1j * theta0)

        out[0, 0] = t11 * al[0, 0]
        if n_pos.size:
            out[1:, 1:] = (
                t00 * gamma_row * gamma_col * phase_00 * al[n_row - 1, n_col - 1]
                + t11 * al[n_row, n_col]
                + t01 * (1j * gamma_row * phase_01) * al[n_row - 1, n_col]
                + t10 * (-1j * gamma_col * phase_10) * al[n_row, n_col - 1]
            ) / 2.0
            out[0, 1:] = (t11 * al[0, n_pos] + t10 * (-1j * gamma_pos * phase_10) * al[0, n_pos - 1]) / sqrt2
            out[1:, 0] = (t11 * al[n_pos, 0] + t01 * (1j * gamma_pos * phase_01) * al[n_pos - 1, 0]) / sqrt2
    else:
        if sigma_rotation:
            phase_00 = np.exp(1j * (theta2 - theta1))
            phase_01 = np.exp(1j * (theta2 - theta0))
            phase_10 = np.exp(-1j * (theta1 - theta0))
        else:
            phase_00 = 1.0
            phase_01 = np.exp(-1j * theta0)
            phase_10 = np.exp(1j * theta0)

        out[0, 0] = t00 * al[0, 0]
        if n_pos.size:
            out[1:, 1:] = (
                t00 * al[n_row, n_col]
                + t11 * gamma_row * gamma_col * phase_00 * al[n_row - 1, n_col - 1]
                + t01 * (1j * gamma_col * phase_01) * al[n_row, n_col - 1]
                + t10 * (-1j * gamma_row * phase_10) * al[n_row - 1, n_col]
            ) / 2.0
            out[1:, 0] = (t00 * al[n_pos, 0] + t10 * (-1j * gamma_pos * phase_10) * al[n_pos - 1, 0]) / sqrt2
            out[0, 1:] = (t00 * al[0, n_pos] + t01 * (1j * gamma_pos * phase_01) * al[0, n_pos - 1]) / sqrt2
    return out

__all__ = [name for name in globals() if not name.startswith('__')]
