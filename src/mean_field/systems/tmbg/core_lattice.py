from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class KPathNode:
    label: str
    index: int
    kvec: complex
    k_dist: float

@dataclass(frozen=True)
class KPath:
    kvec: np.ndarray
    kdist: np.ndarray
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]

    @property
    def nodes(self) -> tuple[KPathNode, ...]:
        nodes = []
        for label, idx in zip(self.labels, self.node_indices, strict=True):
            i = int(idx) - 1
            nodes.append(KPathNode(str(label), int(idx), complex(self.kvec[i]), float(self.kdist[i])))
        return tuple(nodes)

def cumulative_distance(kvec: np.ndarray) -> np.ndarray:
    kvec = np.asarray(kvec, dtype=np.complex128)
    if kvec.ndim != 1:
        raise ValueError(f"Expected a 1D path, got shape {kvec.shape}")
    if kvec.size == 0:
        return np.zeros((0,), dtype=float)
    diffs = np.abs(np.diff(kvec))
    return np.concatenate(([0.0], np.cumsum(diffs))).astype(float)
