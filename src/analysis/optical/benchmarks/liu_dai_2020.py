from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.optical.linear import E2_OVER_H_S
from analysis.optical.magneto import (
    MagnetoOpticalAmplitudes,
    SmallConductivityAngles,
    faraday_kerr_from_sheet_conductivity,
    faraday_kerr_small_conductivity,
)


@dataclass(frozen=True)
class C3TensorDecomposition:
    """C3 in-plane second-order tensor decomposition.

    ``sigma_xxx`` and ``sigma_yxx`` are the two independent components of the
    Liu--Dai 2020 Eq. (3) C3-allowed tensor.  ``residual_norm`` measures the
    Frobenius norm of the part that is not represented by those two components.
    """

    sigma_xxx: complex
    sigma_yxx: complex
    reconstructed: np.ndarray
    residual: np.ndarray
    residual_norm: float


def qah_sheet_conductivity_tensor(
    chern_number: int | float,
    *,
    longitudinal_siemens: complex | float = 0.0,
    conductance_quantum_siemens: float = E2_OVER_H_S,
) -> np.ndarray:
    """Return a simple QAH sheet-conductivity tensor in Siemens.

    The tensor obeys the physical contract ``j_a=sum_b sigma_ab E_b`` and is
    aligned with the repository's default FHS/Kubo orientation. A positive
    Chern number gives

    ```text
    sigma_xy = + C e^2/h,    sigma_yx = - C e^2/h.
    ```

    This is a benchmark/limit helper, not a substitute for a computed optical
    conductivity tensor.
    """

    hall = float(chern_number) * float(conductance_quantum_siemens)
    longitudinal = complex(longitudinal_siemens)
    sigma = np.zeros((2, 2), dtype=np.complex128)
    sigma[0, 0] = longitudinal
    sigma[1, 1] = longitudinal
    sigma[0, 1] = hall
    sigma[1, 0] = -hall
    return sigma


def qah_faraday_kerr_benchmark(
    chern_number: int | float,
    *,
    longitudinal_siemens: complex | float = 0.0,
) -> MagnetoOpticalAmplitudes:
    """Exact Kerr/Faraday amplitudes for the QAH sheet-conductivity limit."""

    return faraday_kerr_from_sheet_conductivity(
        qah_sheet_conductivity_tensor(chern_number, longitudinal_siemens=longitudinal_siemens)
    )


def qah_faraday_kerr_small_benchmark(
    chern_number: int | float,
    *,
    longitudinal_siemens: complex | float = 0.0,
) -> SmallConductivityAngles:
    """Physical small-sheet QAH benchmark.

    Liu-Dai 2020 Eq. (23) prints the opposite transverse sign; this helper is
    consistent with the generic ``j=sigma E`` Maxwell solver instead.
    """

    return faraday_kerr_small_conductivity(
        qah_sheet_conductivity_tensor(chern_number, longitudinal_siemens=longitudinal_siemens)
    )


def c3_inplane_second_order_tensor(
    sigma_xxx: complex | float,
    sigma_yxx: complex | float,
    *,
    ndim: int = 2,
) -> np.ndarray:
    """Construct the C3-allowed in-plane second-order tensor of Eq. (3).

    Tensor indices are ordered as ``sigma[c, a, b]``: current axis first, then
    the two optical axes.  For ``x=0`` and ``y=1`` the nonzero relations are

    ```text
    sigma_xxx = -sigma_xyy = -sigma_yxy = -sigma_yyx
    sigma_yxx =  sigma_xxy =  sigma_xyx = -sigma_yyy
    ```

    The same helper can represent the magnetization-independent sector and the
    part linear in orbital magnetization ``M_z``; combine them at the workflow
    layer as ``sigma_0 + M_z * sigma_z``.
    """

    if ndim < 2:
        raise ValueError(f"ndim must be at least 2, got {ndim}")
    out = np.zeros((ndim, ndim, ndim), dtype=np.complex128)
    a = complex(sigma_xxx)
    b = complex(sigma_yxx)
    x = 0
    y = 1
    out[x, x, x] = a
    out[x, y, y] = -a
    out[y, x, y] = -a
    out[y, y, x] = -a
    out[y, x, x] = b
    out[x, x, y] = b
    out[x, y, x] = b
    out[y, y, y] = -b
    return out


def decompose_c3_inplane_second_order_tensor(tensor: np.ndarray) -> C3TensorDecomposition:
    """Project an in-plane rank-3 tensor onto the Liu--Dai Eq. (3) C3 form."""

    arr = np.asarray(tensor, dtype=np.complex128)
    if arr.shape[0] < 2 or arr.shape[1] < 2 or arr.shape[2] < 2:
        raise ValueError(f"tensor must have shape at least (2,2,2), got {arr.shape}")
    x = 0
    y = 1
    sigma_xxx = np.mean([arr[x, x, x], -arr[x, y, y], -arr[y, x, y], -arr[y, y, x]])
    sigma_yxx = np.mean([arr[y, x, x], arr[x, x, y], arr[x, y, x], -arr[y, y, y]])
    reconstructed = np.zeros_like(arr, dtype=np.complex128)
    reconstructed[:2, :2, :2] = c3_inplane_second_order_tensor(sigma_xxx, sigma_yxx, ndim=2)
    residual = arr - reconstructed
    return C3TensorDecomposition(
        sigma_xxx=complex(sigma_xxx),
        sigma_yxx=complex(sigma_yxx),
        reconstructed=reconstructed,
        residual=residual,
        residual_norm=float(np.linalg.norm(residual)),
    )


__all__ = [
    "C3TensorDecomposition",
    "c3_inplane_second_order_tensor",
    "decompose_c3_inplane_second_order_tensor",
    "qah_faraday_kerr_benchmark",
    "qah_faraday_kerr_small_benchmark",
    "qah_sheet_conductivity_tensor",
]
