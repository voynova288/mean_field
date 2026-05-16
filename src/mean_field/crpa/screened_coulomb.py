from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from .workflow import CRPAResult


@dataclass(frozen=True)
class CRPAFockLookupDiagnostics:
    method: str
    q_lookup_failures: int
    q_lookup_fallbacks: int
    q_count: int
    eps_crpa_min: float
    eps_crpa_mean: float
    eps_crpa_max: float
    max_q_reconstruction_residual: float
    max_q_reconstruction_residual_nm_inv: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "method": self.method,
            "q_lookup_failures": self.q_lookup_failures,
            "q_lookup_fallbacks": self.q_lookup_fallbacks,
            "q_count": self.q_count,
            "eps_crpa_min": self.eps_crpa_min,
            "eps_crpa_mean": self.eps_crpa_mean,
            "eps_crpa_max": self.eps_crpa_max,
            "max_q_reconstruction_residual": self.max_q_reconstruction_residual,
            "max_q_reconstruction_residual_nm_inv": self.max_q_reconstruction_residual_nm_inv,
        }


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
    def _reciprocal_basis(self) -> tuple[complex, complex]:
        shifts = np.asarray(self.result.q_shifts, dtype=int)
        q_vectors = np.asarray(self.result.q_vectors, dtype=np.complex128)

        def from_shift(target: tuple[int, int]) -> complex | None:
            matches = np.flatnonzero(np.all(shifts == np.asarray(target, dtype=int), axis=1))
            if matches.size:
                return complex(q_vectors[int(matches[0])])
            return None

        g1 = from_shift((1, 0))
        g2 = from_shift((0, 1))

        q_indices = np.asarray(self.result.q_indices, dtype=int)
        if g1 is None:
            matches = np.flatnonzero(np.all(q_indices == np.asarray((1, 0), dtype=int), axis=1))
            if matches.size:
                g1 = complex(self.result.q_tilde[int(matches[0])] * float(self.result.lk))
        if g2 is None:
            matches = np.flatnonzero(np.all(q_indices == np.asarray((0, 1), dtype=int), axis=1))
            if matches.size:
                g2 = complex(self.result.q_tilde[int(matches[0])] * float(self.result.lk))

        if g1 is None or g2 is None:
            raise ValueError("Cannot reconstruct cRPA reciprocal basis from q_shifts/q_indices.")
        return g1, g2

    @cached_property
    def _q_index_lookup(self) -> dict[tuple[int, int], int]:
        return {tuple(int(v) for v in row): idx for idx, row in enumerate(np.asarray(self.result.q_indices, dtype=int))}

    @cached_property
    def _q_shift_lookup(self) -> dict[tuple[int, int], int]:
        return {tuple(int(v) for v in row): idx for idx, row in enumerate(np.asarray(self.result.q_shifts, dtype=int))}

    @cached_property
    def _q_index_table(self) -> np.ndarray:
        lk = int(self.result.lk)
        table = np.full((lk, lk), -1, dtype=int)
        for idx, row in enumerate(np.asarray(self.result.q_indices, dtype=int)):
            table[int(row[0]) % lk, int(row[1]) % lk] = int(idx)
        return table

    @cached_property
    def _q_shift_table(self) -> tuple[np.ndarray, int, int]:
        shifts = np.asarray(self.result.q_shifts, dtype=int)
        min0 = int(np.min(shifts[:, 0]))
        min1 = int(np.min(shifts[:, 1]))
        max0 = int(np.max(shifts[:, 0]))
        max1 = int(np.max(shifts[:, 1]))
        table = np.full((max0 - min0 + 1, max1 - min1 + 1), -1, dtype=int)
        for idx, row in enumerate(shifts):
            table[int(row[0]) - min0, int(row[1]) - min1] = int(idx)
        return table, min0, min1

    def _fractional_reciprocal_coordinates(self, q_vec: np.ndarray) -> np.ndarray:
        g1, g2 = self._reciprocal_basis
        basis = np.asarray([[g1.real, g2.real], [g1.imag, g2.imag]], dtype=float)
        det = float(np.linalg.det(basis))
        if abs(det) < 1.0e-14:
            raise ValueError("Degenerate cRPA reciprocal basis; cannot decompose physical q vectors.")
        query = np.column_stack((np.asarray(q_vec.real, dtype=float), np.asarray(q_vec.imag, dtype=float)))
        return np.linalg.solve(basis, query.T).T

    def _resolve_matrix_diagonal_epsilon_array(
        self,
        q_vec: np.ndarray,
        *,
        decomposition_tol: float = 1.0e-8,
    ) -> tuple[np.ndarray, int, float, list[str]]:
        q_shape = np.asarray(q_vec).shape
        flat_q = np.asarray(q_vec, dtype=np.complex128).reshape(-1)
        coeffs = self._fractional_reciprocal_coordinates(flat_q)
        lk = int(self.result.lk)
        g1, g2 = self._reciprocal_basis
        scaled = np.rint(coeffs * float(lk)).astype(int)
        mod = np.mod(scaled, lk)
        centered = np.where(mod <= lk // 2, mod, mod - lk).astype(int)
        shifts = ((scaled - centered) // lk).astype(int)
        reconstructed = (shifts[:, 0] + centered[:, 0] / float(lk)) * g1 + (
            shifts[:, 1] + centered[:, 1] / float(lk)
        ) * g2
        residuals = np.abs(flat_q - reconstructed)
        max_residual = float(np.max(residuals)) if residuals.size else 0.0

        q_table = self._q_index_table
        q_table_index = q_table[np.mod(centered[:, 0], lk), np.mod(centered[:, 1], lk)]

        shift_table, shift_min0, shift_min1 = self._q_shift_table
        s0 = shifts[:, 0] - shift_min0
        s1 = shifts[:, 1] - shift_min1
        shift_in_range = (s0 >= 0) & (s0 < shift_table.shape[0]) & (s1 >= 0) & (s1 < shift_table.shape[1])
        q_shift_index = np.full(flat_q.shape, -1, dtype=int)
        q_shift_index[shift_in_range] = shift_table[s0[shift_in_range], s1[shift_in_range]]

        valid = (residuals <= float(decomposition_tol)) & (q_table_index >= 0) & (q_shift_index >= 0)
        values = np.full(flat_q.shape, np.nan, dtype=float)
        values[valid] = np.real(self.result.effective_epsilon[q_table_index[valid], q_shift_index[valid]])

        failure_indices = np.flatnonzero(~valid)
        failures: list[str] = []
        for idx in failure_indices[:5]:
            failures.append(
                f"q={complex(flat_q[int(idx)])!r} coeff=({coeffs[int(idx), 0]:.12g},{coeffs[int(idx), 1]:.12g}) "
                f"q_key=({int(centered[int(idx), 0] % lk)},{int(centered[int(idx), 1] % lk)}) "
                f"shift_key=({int(shifts[int(idx), 0])},{int(shifts[int(idx), 1])}) "
                f"residual={float(residuals[int(idx)]):.3e}"
            )
        return values.reshape(q_shape), int(failure_indices.size), max_residual, failures

    def _matrix_diagonal_epsilon_array(
        self,
        q_vec: np.ndarray,
        *,
        decomposition_tol: float = 1.0e-8,
    ) -> np.ndarray:
        values, failure_count, max_residual, failures = self._resolve_matrix_diagonal_epsilon_array(
            q_vec,
            decomposition_tol=decomposition_tol,
        )
        if failure_count:
            detail = "; ".join(failures)
            raise ValueError(
                "Could not resolve all physical q vectors by exact cRPA matrix-diagonal lookup "
                f"(max_residual={max_residual:.3e}, tol={float(decomposition_tol):.3e}). "
                "Use a full cRPA q table with sufficient Q cutoff for SCF, or use linear/nearest only for labeled off-grid path diagnostics. "
                f"Examples: {detail}"
            )
        return values

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
        method: str = "matrix_diagonal",
        exact_tol: float = 1.0e-10,
        decomposition_tol: float = 1.0e-8,
    ) -> float | np.ndarray:
        """Return Fock epsilon values for arbitrary physical momenta.

        ``method="matrix_diagonal"`` decomposes each physical transfer vector as
        ``q = q_tilde + Q`` and returns the stored matrix diagonal
        ``epsilon(q_tilde)[Q, Q]``.  This is the production HF path.  The
        interpolation methods are retained for off-grid path-band diagnostics.
        """

        q = np.asarray(q_vec, dtype=np.complex128)
        scalar = q.ndim == 0
        resolved_method = str(method).strip().lower()
        if resolved_method not in {"matrix_diagonal", "linear", "nearest"}:
            raise ValueError(f"Unsupported Fock epsilon interpolation method: {method!r}")
        if resolved_method == "matrix_diagonal":
            out = self._matrix_diagonal_epsilon_array(q, decomposition_tol=decomposition_tol)
            if scalar:
                return float(out.reshape(()))
            return out

        flat_q = q.reshape(-1)
        coords, eps, tree, interpolator = self._epsilon_lookup
        query = np.column_stack((flat_q.real, flat_q.imag))
        distances, nearest = tree.query(query, k=1)
        nearest = np.asarray(nearest, dtype=int)
        values = eps[nearest].astype(float, copy=True)

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

    def fock_epsilon(self, q_vec: complex, *, method: str = "matrix_diagonal", exact_tol: float = 1.0e-10) -> float:
        return float(self.fock_epsilon_array(q_vec, method=method, exact_tol=exact_tol))

    def nearest_fock_epsilon(self, q_vec: complex) -> float:
        return self.fock_epsilon(q_vec, method="nearest")

    def fock_lookup_diagnostics(
        self,
        q_vec: complex | np.ndarray,
        *,
        method: str = "matrix_diagonal",
        exact_tol: float = 1.0e-10,
        decomposition_tol: float = 1.0e-8,
    ) -> CRPAFockLookupDiagnostics:
        q = np.asarray(q_vec, dtype=np.complex128)
        flat_q = q.reshape(-1)
        resolved_method = str(method).strip().lower()
        if resolved_method not in {"matrix_diagonal", "linear", "nearest"}:
            raise ValueError(f"Unsupported Fock epsilon interpolation method: {method!r}")

        lattice_a_nm = float(self.result.coulomb_params.graphene_lattice_angstrom) / 10.0
        if resolved_method != "matrix_diagonal":
            values = np.asarray(self.fock_epsilon_array(q, method=resolved_method, exact_tol=exact_tol), dtype=float)
            return CRPAFockLookupDiagnostics(
                method=resolved_method,
                q_lookup_failures=0,
                q_lookup_fallbacks=0,
                q_count=int(flat_q.size),
                eps_crpa_min=float(np.nanmin(values)),
                eps_crpa_mean=float(np.nanmean(values)),
                eps_crpa_max=float(np.nanmax(values)),
                max_q_reconstruction_residual=0.0,
                max_q_reconstruction_residual_nm_inv=0.0,
            )

        values, failures, max_residual, _failure_examples = self._resolve_matrix_diagonal_epsilon_array(
            q,
            decomposition_tol=decomposition_tol,
        )
        values = np.asarray(values, dtype=float).reshape(-1)

        if failures:
            finite_values = values[np.isfinite(values)]
        else:
            finite_values = values
        if finite_values.size == 0:
            eps_min = eps_mean = eps_max = float("nan")
        else:
            eps_min = float(np.nanmin(finite_values))
            eps_mean = float(np.nanmean(finite_values))
            eps_max = float(np.nanmax(finite_values))

        return CRPAFockLookupDiagnostics(
            method=resolved_method,
            q_lookup_failures=int(failures),
            q_lookup_fallbacks=0,
            q_count=int(flat_q.size),
            eps_crpa_min=eps_min,
            eps_crpa_mean=eps_mean,
            eps_crpa_max=eps_max,
            max_q_reconstruction_residual=float(max_residual),
            max_q_reconstruction_residual_nm_inv=float(max_residual / lattice_a_nm),
        )
