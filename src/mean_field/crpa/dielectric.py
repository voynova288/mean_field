from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DielectricResult:
    epsilon: np.ndarray
    epsilon_inv: np.ndarray
    screened_v: np.ndarray
    effective_epsilon: np.ndarray


def compute_dielectric(chi0_q: np.ndarray, v_q_mev: np.ndarray) -> DielectricResult:
    chi0 = np.asarray(chi0_q, dtype=np.complex128)
    v = np.asarray(v_q_mev, dtype=float)
    if chi0.ndim != 2 or chi0.shape[0] != chi0.shape[1]:
        raise ValueError(f"chi0_q must be square, got {chi0.shape}")
    if v.shape != (chi0.shape[0],):
        raise ValueError(f"Expected v_q shape {(chi0.shape[0],)}, got {v.shape}")

    v_diag = np.diag(v.astype(np.complex128))
    epsilon = np.eye(chi0.shape[0], dtype=np.complex128) + chi0 @ v_diag
    epsilon_inv = np.linalg.inv(epsilon)
    screened_v = v_diag @ epsilon_inv
    effective = np.real(np.diag(epsilon))
    return DielectricResult(
        epsilon=epsilon,
        epsilon_inv=epsilon_inv,
        screened_v=screened_v,
        effective_epsilon=effective,
    )
