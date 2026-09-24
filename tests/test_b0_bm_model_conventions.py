from __future__ import annotations

import numpy as np

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field.model import _construct_diagonal_block, _generate_gvec, _generate_t12, dirac


def _reference_generate_t12(params: TBGParameters, lg: int, zeta: int) -> np.ndarray:
    dim = 4 * lg * lg
    t12 = np.zeros((dim, dim), dtype=np.complex128)
    tmp = t12.reshape(4, lg * lg, 4, lg * lg, order="F")

    idx = np.arange(1, lg * lg + 1, dtype=int).reshape((lg, lg), order="F")
    idx_nn1 = np.roll(idx, shift=(-zeta, zeta), axis=(0, 1))
    idx_nn2 = np.roll(idx, shift=(0, zeta), axis=(0, 1))
    idx_nn12 = np.roll(idx, shift=(-zeta, 0), axis=(0, 1))

    if zeta == 1:
        t0, t1, t2 = params.t0, params.t1, params.t2
    elif zeta == -1:
        t0, t1, t2 = params.t0, params.t2, params.t1
    else:
        raise ValueError(f"Unexpected valley label: {zeta}")

    idx_flat = np.ravel(idx, order="F")
    idx_nn1_flat = np.ravel(idx_nn1, order="F")
    idx_nn2_flat = np.ravel(idx_nn2, order="F")
    idx_nn12_flat = np.ravel(idx_nn12, order="F")

    for ig in range(lg * lg):
        here = int(idx_flat[ig] - 1)
        nn1 = int(idx_nn1_flat[ig] - 1)
        nn2 = int(idx_nn2_flat[ig] - 1)
        nn12 = int(idx_nn12_flat[ig] - 1)
        tmp[2:4, here, 0:2, nn1] = t2
        tmp[0:2, nn1, 2:4, here] = t2
        tmp[2:4, here, 0:2, nn2] = t1
        tmp[0:2, nn2, 2:4, here] = t1
        tmp[2:4, here, 0:2, nn12] = t0
        tmp[0:2, nn12, 2:4, here] = t0

    return t12


def _reference_construct_diagonal_block(params: TBGParameters, gvec: np.ndarray, lg: int, k: complex, zeta: int) -> np.ndarray:
    dim = 4 * lg * lg
    h = np.zeros((dim, dim), dtype=np.complex128)
    sigma0 = np.eye(2, dtype=np.complex128)
    rotation = -params.dtheta_rad / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    div_u = float((params.strain_matrix[0, 0] + params.strain_matrix[1, 1]) / 2.0)

    for ig in range(lg * lg):
        qc = gvec[ig]
        if zeta == 1:
            kb = k - params.kb_point + qc
            kt = k - params.kt + qc
        elif zeta == -1:
            kb = k - params.kt + qc
            kt = k - params.kb_point + qc
        else:
            raise ValueError(f"Unexpected valley label: {zeta}")
        k1 = (np.eye(2) + rotation - params.strain_matrix * params.alpha) @ np.asarray([kb.real, kb.imag], dtype=float)
        k2 = (np.eye(2) - rotation + params.strain_matrix * (1.0 - params.alpha)) @ np.asarray([kt.real, kt.imag], dtype=float)
        left = 4 * ig
        h[left : left + 2, left : left + 2] = params.vf * dirac(complex(k1[0], k1[1]), zeta, 0.0) - (params.deformation_potential * div_u) * sigma0
        h[left + 2 : left + 4, left + 2 : left + 4] = params.vf * dirac(complex(k2[0], k2[1]), zeta, 0.0) + (params.deformation_potential * div_u) * sigma0

    return h


def test_generate_t12_matches_b0_julia_neighbor_convention() -> None:
    params = TBGParameters.from_degrees(1.2, strain=0.0)
    for zeta in (1, -1):
        got = _generate_t12(params, lg=3, zeta=zeta)
        want = _reference_generate_t12(params, lg=3, zeta=zeta)
        assert np.allclose(got, want)


def test_construct_diagonal_block_matches_b0_valley_swap_with_strain() -> None:
    params = TBGParameters.from_degrees(1.2, strain=0.002, alpha=0.5, deformation_potential=-4100.0)
    gvec = _generate_gvec(params, lg=3)
    k = 0.123 + 0.045j

    for zeta in (1, -1):
        got = _construct_diagonal_block(params, gvec, lg=3, k=k, zeta=zeta, sigma_rotation=True)
        want = _reference_construct_diagonal_block(params, gvec, lg=3, k=k, zeta=zeta)
        assert np.allclose(got, want)
