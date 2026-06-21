from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .coherence import ivc_amplitude
from .density import occupation_table, signed_polarization
from .schema import OrderParameterResult, StateLabel


def spin_sign(label: StateLabel) -> float:
    if label.spin in {"up", "u", "+", 1, "+1"}:
        return 1.0
    if label.spin in {"down", "d", "-", -1, "-1"}:
        return -1.0
    return 0.0


def valley_sign(label: StateLabel) -> float:
    if label.valley is None:
        return 0.0
    return 1.0 if int(label.valley) == 1 else -1.0


def flavor_order_parameters(projector_kab: np.ndarray, labels: Sequence[StateLabel]) -> OrderParameterResult:
    """Return common spin/valley/coherence diagnostics for labeled projectors."""

    numeric = {
        "spin_polarization": signed_polarization(projector_kab, labels, spin_sign),
        "valley_polarization": signed_polarization(projector_kab, labels, valley_sign),
        "active_spin_polarization": signed_polarization(projector_kab, labels, spin_sign, active_only=True),
        "active_valley_polarization": signed_polarization(projector_kab, labels, valley_sign, active_only=True),
        "ivc_amplitude": ivc_amplitude(projector_kab, labels, active_only=True),
    }
    return OrderParameterResult(
        scalars={key: float(value) for key, value in numeric.items()},
        tables={"occupations": occupation_table(projector_kab, labels)},
        metadata={"label_count": len(tuple(labels))},
    )


def finite_field_valley_spin_order_parameters(
    hamiltonian: np.ndarray,
    energies: np.ndarray,
    mu: float,
    *,
    q: int,
    n_eta: int = 2,
    n_spin: int = 2,
    n_band: int = 2,
) -> dict[str, float]:
    """Return ``s_i eta_j`` order parameters in the historical finite-B convention."""

    pauli = [
        np.array([[1, 0], [0, 1]], dtype=np.complex128),
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.array([[1, 0], [0, -1]], dtype=np.complex128),
    ]
    sub = int(n_band) * int(q)
    identity_band = np.eye(sub, dtype=np.complex128)
    h = np.asarray(hamiltonian, dtype=np.complex128)
    eps = np.asarray(energies, dtype=float)
    out: dict[str, float] = {}
    nk = h.shape[2]
    for ispin, spin_mat in enumerate(pauli):
        for ieta, eta_mat in enumerate(pauli):
            op = np.kron(spin_mat, np.kron(eta_mat, identity_band))
            values = np.zeros_like(eps)
            for ik in range(nk):
                _vals, vecs = np.linalg.eigh(h[:, :, ik])
                values[:, ik] = np.diag(vecs.conj().T @ op @ vecs).real
            out[f"s{ispin}_eta{ieta}"] = float(values[eps <= float(mu)].sum() / eps.size * 8.0)
    return out


__all__ = [
    "finite_field_valley_spin_order_parameters",
    "flavor_order_parameters",
    "spin_sign",
    "valley_sign",
]
