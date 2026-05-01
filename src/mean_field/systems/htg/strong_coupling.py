from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HTGStrongCouplingClassification:
    family: str
    class_label: str
    flavor_occupations: np.ndarray
    band_occupations: np.ndarray
    flavor_occupation_pattern: tuple[int, ...]
    n_doubly_occupied: int
    n_a_polarized: int
    n_b_polarized: int
    n_empty: int
    nu_z: float

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "class_label": self.class_label,
            "flavor_occupations": [float(value) for value in self.flavor_occupations],
            "band_occupations": self.band_occupations.tolist(),
            "flavor_occupation_pattern": [int(value) for value in self.flavor_occupation_pattern],
            "n_doubly_occupied": int(self.n_doubly_occupied),
            "n_a_polarized": int(self.n_a_polarized),
            "n_b_polarized": int(self.n_b_polarized),
            "n_empty": int(self.n_empty),
            "nu_z": float(self.nu_z),
        }


def flavor_band_occupations(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    """Return per-flavor Chern-sublattice occupations from centered HF density.

    The returned shape is ``(n_spin, n_eta, n_band)`` and each band occupation
    lies near 0 or 1 for an idempotent integer-filling projector.
    """

    density = np.asarray(density, dtype=np.complex128)
    nt, nt_rhs, nk = density.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square density blocks, got {density.shape}")
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"Density dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")

    projector_diagonal = np.zeros((nt, nk), dtype=float)
    for ik in range(nk):
        projector_diagonal[:, ik] = np.real(np.diag(density[:, :, ik])) + 0.5
    return np.mean(projector_diagonal, axis=1).reshape((n_spin, n_eta, n_band), order="F")


def flavor_occupations(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
) -> np.ndarray:
    band_occ = flavor_band_occupations(density, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
    return np.sum(band_occ, axis=2).reshape(-1, order="C")


def classify_htg_family(flavor_occ: np.ndarray, *, atol: float = 0.25) -> tuple[str, tuple[int, ...]]:
    rounded = tuple(sorted((int(round(float(value))) for value in np.asarray(flavor_occ).reshape(-1)), reverse=True))
    if any(value < 0 or value > 2 for value in rounded):
        return "mixed", rounded
    if max(rounded) - min(rounded) <= 1:
        return "FB", rounded
    if max(rounded) == 2 and min(rounded) == 0:
        return "FI", rounded
    residual = max(abs(float(value) - round(float(value))) for value in np.asarray(flavor_occ).reshape(-1))
    return ("mixed" if residual > atol else "unclassified"), rounded


def classify_htg_strong_coupling_state(
    density: np.ndarray,
    *,
    n_spin: int = 2,
    n_eta: int = 2,
    n_band: int = 2,
    occupation_threshold: float = 0.5,
) -> HTGStrongCouplingClassification:
    band_occ = flavor_band_occupations(density, n_spin=n_spin, n_eta=n_eta, n_band=n_band)
    flavor_occ = np.sum(band_occ, axis=2).reshape(-1, order="C")
    family, pattern = classify_htg_family(flavor_occ)

    n_d = 0
    n_a = 0
    n_b = 0
    n_empty = 0
    for occ_a, occ_b in band_occ.reshape((n_spin * n_eta, n_band), order="C"):
        a_filled = float(occ_a) > occupation_threshold
        b_filled = float(occ_b) > occupation_threshold
        if a_filled and b_filled:
            n_d += 1
        elif a_filled:
            n_a += 1
        elif b_filled:
            n_b += 1
        else:
            n_empty += 1

    pieces: list[str] = []
    if n_d:
        pieces.append(f"D{n_d}" if n_d > 1 else "D")
    if n_a:
        pieces.append(f"A{n_a}" if n_a > 1 else "A")
    if n_b:
        pieces.append(f"B{n_b}" if n_b > 1 else "B")
    if not pieces:
        pieces.append("empty")
    class_label = "[" + " ".join(pieces) + "]"
    nu_z = float(np.sum(band_occ[:, :, 0] - band_occ[:, :, 1]))

    return HTGStrongCouplingClassification(
        family=family,
        class_label=class_label,
        flavor_occupations=np.asarray(flavor_occ, dtype=float),
        band_occupations=np.asarray(band_occ, dtype=float),
        flavor_occupation_pattern=pattern,
        n_doubly_occupied=int(n_d),
        n_a_polarized=int(n_a),
        n_b_polarized=int(n_b),
        n_empty=int(n_empty),
        nu_z=nu_z,
    )
