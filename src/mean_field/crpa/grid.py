from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..systems.tbg.params import TBGParameters


@dataclass(frozen=True)
class CRPAKGrid:
    """Periodic uniform grid in the primitive moire Brillouin zone."""

    lk: int
    fractional: np.ndarray
    kvec: np.ndarray

    @property
    def nk(self) -> int:
        return int(self.kvec.size)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.lk), int(self.lk))

    def flat_index(self, i: int, j: int) -> int:
        return int((int(i) % self.lk) + self.lk * (int(j) % self.lk))

    def unravel_index(self, index: int) -> tuple[int, int]:
        idx = int(index)
        return (idx % self.lk, idx // self.lk)

    def shifted_index(self, k_index: int, q_index: int | tuple[int, int]) -> int:
        shifted, _ = self.shifted_index_and_wrap(k_index, q_index)
        return shifted

    def centered_q_index(self, q_index: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(q_index, tuple):
            qi, qj = int(q_index[0]) % self.lk, int(q_index[1]) % self.lk
        else:
            qi, qj = self.unravel_index(int(q_index))

        def centered(coord: int) -> int:
            return int(coord) - self.lk if int(coord) > self.lk // 2 else int(coord)

        return (centered(qi), centered(qj))

    def centered_q_vector(self, q_index: int | tuple[int, int]) -> complex:
        qi, qj = self.centered_q_index(q_index)
        return complex((float(qi) / float(self.lk)) * self._g1 + (float(qj) / float(self.lk)) * self._g2)

    @property
    def _g1(self) -> complex:
        if self.lk == 1:
            return 0.0 + 0.0j
        return complex(self.kvec[self.flat_index(1, 0)] * float(self.lk))

    @property
    def _g2(self) -> complex:
        if self.lk == 1:
            return 0.0 + 0.0j
        return complex(self.kvec[self.flat_index(0, 1)] * float(self.lk))

    def shifted_index_and_wrap(self, k_index: int, q_index: int | tuple[int, int]) -> tuple[int, tuple[int, int]]:
        """Return folded ``k + q`` index and reciprocal wrap vector.

        The stored BM eigenvectors live on the first periodic grid tile.  When
        ``k + q`` leaves that tile, the folded state at ``k + q - W`` must be
        used in the periodic gauge with form-factor shifts ``Q + W``.
        """

        ki, kj = self.unravel_index(k_index)
        qi, qj = self.centered_q_index(q_index)
        raw_i = int(ki) + int(qi)
        raw_j = int(kj) + int(qj)
        folded_i = raw_i % self.lk
        folded_j = raw_j % self.lk
        wrap_i = (raw_i - folded_i) // self.lk
        wrap_j = (raw_j - folded_j) // self.lk
        return self.flat_index(folded_i, folded_j), (int(wrap_i), int(wrap_j))


def build_uniform_crpa_grid(params: TBGParameters, lk: int) -> CRPAKGrid:
    """Build an ``lk x lk`` periodic grid without duplicated endpoints."""

    lk = int(lk)
    if lk <= 0:
        raise ValueError(f"lk must be positive, got {lk}")
    frac = np.arange(lk, dtype=float) / float(lk)
    f1, f2 = np.meshgrid(frac, frac, indexing="ij")
    fractional = np.stack([np.ravel(f1, order="F"), np.ravel(f2, order="F")], axis=1)
    kvec = np.ravel(f1 * params.g1 + f2 * params.g2, order="F").astype(np.complex128)
    return CRPAKGrid(lk=lk, fractional=fractional, kvec=kvec)


def build_q_shift_table(q_lg: int) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Return reciprocal-vector shifts and their integer coordinates.

    ``q_lg`` follows the existing HF convention: it must be odd, and
    ``q_lg=3`` gives shifts ``-1, 0, 1`` in each primitive reciprocal
    direction.
    """

    q_lg = int(q_lg)
    if q_lg <= 0 or q_lg % 2 == 0:
        raise ValueError(f"q_lg must be a positive odd integer, got {q_lg}")
    half_width = (q_lg - 1) // 2
    labels = tuple(range(-half_width, half_width + 1))
    shifts = tuple((m, n) for n in labels for m in labels)
    coords = np.asarray(shifts, dtype=int)
    return shifts, coords


def q_shift_vectors(params: TBGParameters, q_shifts: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.asarray([m * params.g1 + n * params.g2 for m, n in q_shifts], dtype=np.complex128)
