from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .workflow import CRPAResult


@dataclass(frozen=True)
class CRPAScreenedCoulomb:
    """Lookup wrapper for a computed cRPA screening table."""

    result: CRPAResult

    def get_hartree_screened_v(self) -> np.ndarray:
        zero = self.result.q_indices[:, 0] == 0
        zero &= self.result.q_indices[:, 1] == 0
        matches = np.flatnonzero(zero)
        if matches.size == 0:
            raise KeyError("The cRPA result does not contain q_tilde=(0, 0)")
        return np.asarray(self.result.screened_v[matches[0]], dtype=np.complex128)

    def get_fock_epsilon_by_index(self, q_table_index: int, q_shift_index: int) -> float:
        return float(np.real(self.result.effective_epsilon[int(q_table_index), int(q_shift_index)]))

    def nearest_fock_epsilon(self, q_vec: complex) -> float:
        q = complex(q_vec)
        distances = np.abs(self.result.physical_q_vectors.reshape(-1) - q)
        flat = int(np.argmin(distances))
        iq, i_shift = np.unravel_index(flat, self.result.physical_q_vectors.shape)
        return self.get_fock_epsilon_by_index(int(iq), int(i_shift))
