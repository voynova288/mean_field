from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LatticeGrid:
    k1: np.ndarray
    k2: np.ndarray
    kvec: np.ndarray
    nk: int
    lk: int
    flag_inv: bool = True


@dataclass(frozen=True)
class KPathNode:
    label: str
    index: int
    k_dist: float
    kvec: complex

    @property
    def kx(self) -> float:
        return float(self.kvec.real)

    @property
    def ky(self) -> float:
        return float(self.kvec.imag)


@dataclass(frozen=True)
class KPath:
    kvec: np.ndarray
    kdist: np.ndarray
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]

    @property
    def nodes(self) -> tuple[KPathNode, ...]:
        return tuple(
            KPathNode(
                label=label,
                index=index,
                k_dist=float(self.kdist[index - 1]),
                kvec=complex(self.kvec[index - 1]),
            )
            for label, index in zip(self.labels, self.node_indices, strict=True)
        )


def build_uniform_lattice(g1: complex, g2: complex, lk: int) -> LatticeGrid:
    frac = np.arange(lk + 1, dtype=float) / float(lk)
    kvec = np.ravel(frac[:, None] * g1 + frac[None, :] * g2, order="F")
    return LatticeGrid(
        k1=frac.copy(),
        k2=frac.copy(),
        kvec=np.asarray(kvec, dtype=np.complex128),
        nk=int(kvec.size),
        lk=int(lk),
        flag_inv=True,
    )


def cumulative_distance(kvec: Iterable[complex]) -> np.ndarray:
    values = np.asarray(list(kvec), dtype=np.complex128)
    if values.size == 0:
        return np.asarray([], dtype=float)
    diffs = np.abs(np.diff(values))
    return np.concatenate([np.asarray([0.0]), np.cumsum(diffs)])



def build_kpath_from_nodes(
    nodes: Iterable[complex],
    labels: Iterable[str],
    segment_point_counts: Iterable[int] | int,
    *,
    duplicate_nodes: bool = False,
) -> KPath:
    node_tuple = tuple(complex(node) for node in nodes)
    label_tuple = tuple(str(label) for label in labels)
    if isinstance(segment_point_counts, int):
        counts_tuple = tuple(int(segment_point_counts) for _ in range(max(0, len(node_tuple) - 1)))
    else:
        counts_tuple = tuple(int(value) for value in segment_point_counts)
    if len(node_tuple) < 2:
        raise ValueError("At least two path nodes are required.")
    if len(node_tuple) != len(label_tuple):
        raise ValueError(f"Expected {len(node_tuple)} labels, got {len(label_tuple)}")
    if len(counts_tuple) != len(node_tuple) - 1:
        raise ValueError(f"Expected {len(node_tuple) - 1} segment counts, got {len(counts_tuple)}")
    if min(counts_tuple) <= 0:
        raise ValueError(f"Segment point counts must be positive, got {counts_tuple}")

    kvec: list[complex] = []
    node_indices: list[int] = [1]
    if duplicate_nodes:
        for segment_index, (start_k, end_k, count) in enumerate(
            zip(node_tuple[:-1], node_tuple[1:], counts_tuple, strict=True)
        ):
            segment = np.linspace(0.0, 1.0, int(count), dtype=float)
            for weight in segment:
                kvec.append(complex(start_k + weight * (end_k - start_k)))
            if segment_index + 1 < len(node_tuple) - 1:
                node_indices.append(len(kvec))
        node_indices.append(len(kvec))
    else:
        kvec.append(complex(node_tuple[0]))
        for start_k, end_k, count in zip(node_tuple[:-1], node_tuple[1:], counts_tuple, strict=True):
            step = (end_k - start_k) / float(count)
            for idx in range(1, int(count) + 1):
                kvec.append(complex(start_k + idx * step))
            node_indices.append(len(kvec))

    kvec_array = np.asarray(kvec, dtype=np.complex128)
    return KPath(
        kvec=kvec_array,
        kdist=cumulative_distance(kvec_array),
        labels=label_tuple,
        node_indices=tuple(node_indices),
    )

def almost_equal_complex(a: complex, b: complex, *, atol: float = 1e-12) -> bool:
    return math.isclose(a.real, b.real, abs_tol=atol) and math.isclose(a.imag, b.imag, abs_tol=atol)
