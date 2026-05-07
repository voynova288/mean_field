from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

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

    @cached_property
    def _epsilon_lookup(self) -> tuple[np.ndarray, np.ndarray, cKDTree, LinearNDInterpolator | None]:
        q_values = np.asarray(self.result.physical_q_vectors, dtype=np.complex128).reshape(-1)
        eps_values = np.asarray(self.result.effective_epsilon, dtype=float).reshape(-1)
        finite = np.isfinite(q_values.real) & np.isfinite(q_values.imag) & np.isfinite(eps_values)
        if not np.any(finite):
            raise ValueError("The cRPA result contains no finite Fock epsilon lookup points.")

        coords = np.column_stack((q_values.real[finite], q_values.imag[finite]))
        eps = eps_values[finite]

        # Merge exact duplicate q entries from different (q_tilde, Q)
        # representations.  A rounded key avoids Qhull failures from duplicate
        # points while preserving the table values at physical precision.
        rounded = np.round(coords, decimals=12)
        unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
        if unique.shape[0] != coords.shape[0]:
            sums = np.zeros(unique.shape[0], dtype=float)
            counts = np.zeros(unique.shape[0], dtype=float)
            for idx, value in zip(inverse, eps, strict=True):
                sums[int(idx)] += float(value)
                counts[int(idx)] += 1.0
            coords = unique.astype(float)
            eps = sums / counts

        tree = cKDTree(coords)
        interpolator: LinearNDInterpolator | None
        try:
            interpolator = LinearNDInterpolator(coords, eps, fill_value=np.nan)
        except Exception:
            interpolator = None
        return coords, eps, tree, interpolator

    def fock_epsilon_array(
        self,
        q_vec: complex | np.ndarray,
        *,
        method: str = "linear",
        exact_tol: float = 1.0e-10,
    ) -> float | np.ndarray:
        """Return Fock epsilon values for arbitrary physical momenta.

        ``method="linear"`` uses a piecewise-linear interpolation over the
        stored ``q_tilde + Q`` table and falls back to nearest-neighbour outside
        the convex hull.  Exact table hits are always kept exact.  The old
        nearest-neighbour behaviour remains available as ``method="nearest"``.
        """

        q = np.asarray(q_vec, dtype=np.complex128)
        scalar = q.ndim == 0
        flat_q = q.reshape(-1)
        coords, eps, tree, interpolator = self._epsilon_lookup
        query = np.column_stack((flat_q.real, flat_q.imag))
        distances, nearest = tree.query(query, k=1)
        nearest = np.asarray(nearest, dtype=int)
        values = eps[nearest].astype(float, copy=True)

        resolved_method = str(method).strip().lower()
        if resolved_method not in {"linear", "nearest"}:
            raise ValueError(f"Unsupported Fock epsilon interpolation method: {method!r}")
        if resolved_method == "linear" and interpolator is not None and coords.shape[0] >= 3:
            interpolated = np.asarray(interpolator(query), dtype=float).reshape(-1)
            valid = np.isfinite(interpolated) & (interpolated > 0.0)
            values[valid] = interpolated[valid]

        exact = np.asarray(distances, dtype=float) <= float(exact_tol)
        values[exact] = eps[nearest[exact]]
        values[~np.isfinite(values) | (values <= 0.0)] = eps[nearest[~np.isfinite(values) | (values <= 0.0)]]
        out = values.reshape(q.shape)
        if scalar:
            return float(out.reshape(()))
        return out

    def fock_epsilon(self, q_vec: complex, *, method: str = "linear", exact_tol: float = 1.0e-10) -> float:
        return float(self.fock_epsilon_array(q_vec, method=method, exact_tol=exact_tol))

    def nearest_fock_epsilon(self, q_vec: complex) -> float:
        return self.fock_epsilon(q_vec, method="nearest")
