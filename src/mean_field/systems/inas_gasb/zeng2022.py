"""Zeng-Xue-MacDonald 2022 folded-BHZ model definitions.

This module implements the source-attested one-body and basis conventions from
Y. Zeng, F. Xue, and A. H. MacDonald, Phys. Rev. B 105, 125102 (2022),
Eqs. (1)-(3) and (6), arXiv:2112.07523. Energies and lengths are expressed in
``Ry*`` and ``a_B*`` unless a function says otherwise.

The finite momentum/slab regulator and Hartree-Fock contractions are separate:
the paper does not specify their numerical values, so this module must not
silently choose them as historical facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
Spin = Literal["up", "down"]
Band = Literal["c", "v"]


@dataclass(frozen=True)
class Zeng2022Parameters:
    """Dimensionless parameters for Eqs. (2) and (4).

    ``mass_*_over_reduced`` denotes ``m_{e,h}/m`` where
    ``m=m_e*m_h/(m_e+m_h)``. The paper's numerical phase diagrams use the
    particle-hole-symmetric choice ``m_e=m_h=2m``.
    """

    eg_ry: float
    hybridization_ab_ry: float
    q_ab_inv: float
    d_over_ab: float = 0.3
    mass_e_over_reduced: float = 2.0
    mass_h_over_reduced: float = 2.0

    def validate(self) -> None:
        values = np.asarray(
            [
                self.eg_ry,
                self.hybridization_ab_ry,
                self.q_ab_inv,
                self.d_over_ab,
                self.mass_e_over_reduced,
                self.mass_h_over_reduced,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Zeng2022 parameters must be finite")
        if self.hybridization_ab_ry < 0.0 or self.q_ab_inv < 0.0:
            raise ValueError("hybridization and field momentum must be nonnegative")
        if self.d_over_ab < 0.0:
            raise ValueError("interlayer distance must be nonnegative")
        if self.mass_e_over_reduced <= 0.0 or self.mass_h_over_reduced <= 0.0:
            raise ValueError("effective-mass ratios must be positive")


@dataclass(frozen=True)
class ZengSlabBasis:
    """Folded basis ``(band, spin, n)`` at each quasi-momentum.

    The storage order is slab-major, then spin-major, then band-major:
    ``(n, up-c, up-v, down-c, down-v)``. The ordering is explicit because
    Eqs. (5), (6), and (8) route density-matrix indices through slab transfer.
    """

    slab_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.slab_indices:
            raise ValueError("at least one momentum slab is required")
        if len(set(self.slab_indices)) != len(self.slab_indices):
            raise ValueError("slab indices must be unique")
        if tuple(sorted(self.slab_indices)) != self.slab_indices:
            raise ValueError("slab indices must be strictly increasing")

    @property
    def dimension(self) -> int:
        return 4 * len(self.slab_indices)

    def index(self, band: Band, spin: Spin, slab: int) -> int:
        try:
            slab_position = self.slab_indices.index(int(slab))
        except ValueError as exc:
            raise KeyError(f"slab {slab} is not in this basis") from exc
        if band not in {"c", "v"} or spin not in {"up", "down"}:
            raise KeyError((band, spin, slab))
        local = 2 * (0 if spin == "up" else 1) + (0 if band == "c" else 1)
        return 4 * slab_position + local

    def label(self, index: int) -> tuple[Band, Spin, int]:
        if not 0 <= int(index) < self.dimension:
            raise IndexError(index)
        slab_position, local = divmod(int(index), 4)
        spin: Spin = "up" if local < 2 else "down"
        band: Band = "c" if local % 2 == 0 else "v"
        return band, spin, self.slab_indices[slab_position]


def zeng2022_spin_block(
    momentum_ab_inv: FloatArray,
    params: Zeng2022Parameters,
    *,
    spin: Spin,
) -> ComplexArray:
    """Return one 2x2 BHZ spin block from Eq. (2), in ``Ry*``."""

    params.validate()
    p = np.asarray(momentum_ab_inv, dtype=np.float64)
    if p.shape != (2,) or not np.all(np.isfinite(p)):
        raise ValueError("momentum must be a finite length-two vector")
    px, py = map(float, p)
    q = float(params.q_ab_inv)
    ec = ((px - 0.5 * q) ** 2 + py**2) / params.mass_e_over_reduced
    ec += 0.5 * params.eg_ry
    ev = -((px + 0.5 * q) ** 2 + py**2) / params.mass_h_over_reduced
    ev -= 0.5 * params.eg_ry
    if spin == "up":
        cv = params.hybridization_ab_ry * (px + 1j * py)
    elif spin == "down":
        cv = -params.hybridization_ab_ry * (px - 1j * py)
    else:
        raise ValueError(f"unsupported spin {spin!r}")
    return np.asarray([[ec, cv], [cv.conjugate(), ev]], dtype=np.complex128)


def build_zeng2022_folded_h0(
    quasi_momenta_ab_inv: FloatArray,
    basis: ZengSlabBasis,
    params: Zeng2022Parameters,
) -> ComplexArray:
    """Build Eq. (1) in the finite folded slab basis.

    ``quasi_momenta_ab_inv`` has shape ``(nk,2)`` and represents
    ``kappa`` with ``-Q/2 <= kappa_x < Q/2``. The physical momentum in slab
    ``n`` is ``p=kappa+n*Q*xhat``. No regulator choice is made here.
    """

    kappa = np.asarray(quasi_momenta_ab_inv, dtype=np.float64)
    if kappa.ndim != 2 or kappa.shape[1] != 2 or not np.all(np.isfinite(kappa)):
        raise ValueError("quasi_momenta_ab_inv must have shape (nk,2)")
    params.validate()
    out = np.zeros((basis.dimension, basis.dimension, kappa.shape[0]), dtype=np.complex128)
    for ik, point in enumerate(kappa):
        for slab in basis.slab_indices:
            physical = point + np.asarray([slab * params.q_ab_inv, 0.0])
            for spin in ("up", "down"):
                indices = [basis.index("c", spin, slab), basis.index("v", spin, slab)]
                out[np.ix_(indices, indices, [ik])] = zeng2022_spin_block(
                    physical, params, spin=spin
                )[:, :, None]
    error = float(np.max(np.abs(out - np.swapaxes(out.conj(), 0, 1))))
    if error > 1.0e-12:
        raise RuntimeError(f"folded BHZ Hamiltonian is not Hermitian: {error:.3e}")
    return out


def zeng2022_reference_density(basis: ZengSlabBasis, nk: int) -> ComplexArray:
    """Return the Eq. (6) ordinary-electron reference projector.

    Every valence state is filled and every conduction state is empty. Arrays
    use ``P[a,b]=<c_b^dagger c_a>`` orientation, matching the current
    InAs/GaSb ordinary-electron convention.
    """

    if int(nk) != nk or nk < 1:
        raise ValueError("nk must be a positive integer")
    reference = np.zeros((basis.dimension, basis.dimension, int(nk)), dtype=np.complex128)
    for slab in basis.slab_indices:
        for spin in ("up", "down"):
            index = basis.index("v", spin, slab)
            reference[index, index, :] = 1.0
    return reference


def zeng2022_coulomb_components(q_ab_inv: FloatArray, d_over_ab: float) -> tuple[FloatArray, FloatArray]:
    """Return dimensionless ``(V_cc,V_cv)`` from Eq. (4) for strictly positive q.

    In ``Ry*`` and ``a_B*`` units, ``V_cc(q)=4*pi/q`` and
    ``V_cv(q)=V_cc(q)*exp(-q*d)``. The Hartree and Fock q=0 limits require
    separate neutral-mode/self-cell handling and are deliberately rejected.
    """

    q = np.asarray(q_ab_inv, dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("q must be finite and strictly positive; use explicit q=0 handling")
    if not np.isfinite(d_over_ab) or d_over_ab < 0.0:
        raise ValueError("d_over_ab must be finite and nonnegative")
    intra = 4.0 * np.pi / q
    inter = intra * np.exp(-q * float(d_over_ab))
    return intra, inter


def zeng2022_uniform_hartree_limit_ry(
    exciton_density_ab2: float,
    *,
    d_over_ab: float,
) -> tuple[float, float]:
    """Return neutral uniform-layer Hartree potentials ``(Sigma_c,Sigma_v)``.

    In the zero-average gauge, Eq. (7) implies
    ``Sigma_c-Sigma_v=8*pi*d*n_x Ry*`` in dimensionless units. The returned
    potentials are ``(+4*pi*d*n_x,-4*pi*d*n_x)``.
    """

    if not np.isfinite(exciton_density_ab2) or not np.isfinite(d_over_ab):
        raise ValueError("density and distance must be finite")
    if d_over_ab < 0.0:
        raise ValueError("d_over_ab must be nonnegative")
    value = 4.0 * np.pi * float(d_over_ab) * float(exciton_density_ab2)
    return value, -value


__all__ = [
    "Zeng2022Parameters",
    "ZengSlabBasis",
    "build_zeng2022_folded_h0",
    "zeng2022_coulomb_components",
    "zeng2022_reference_density",
    "zeng2022_spin_block",
    "zeng2022_uniform_hartree_limit_ry",
]
