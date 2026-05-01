from __future__ import annotations

from ..tmbg.lattice import TMBGLattice
from ..tmbg.lattice import build_kpath_from_nodes as _build_kpath_from_nodes
from ..tmbg.lattice import build_moire_k_grid as _build_moire_k_grid
from ..tmbg.lattice import build_standard_kpath as _build_standard_kpath
from ..tmbg.lattice import build_tmbg_lattice as _build_tmbg_lattice


ATMGLattice = TMBGLattice


def build_atmg_lattice(
    theta_deg: float,
    *,
    n_shells: int = 5,
    graphene_lattice_constant_nm: float = 0.246,
) -> ATMGLattice:
    return _build_tmbg_lattice(
        theta_deg,
        n_shells=n_shells,
        graphene_lattice_constant_nm=graphene_lattice_constant_nm,
    )


def build_kpath_from_nodes(*args, **kwargs):
    return _build_kpath_from_nodes(*args, **kwargs)


def build_standard_kpath(*args, **kwargs):
    return _build_standard_kpath(*args, **kwargs)


def build_moire_k_grid(*args, **kwargs):
    return _build_moire_k_grid(*args, **kwargs)


__all__ = [
    "ATMGLattice",
    "build_atmg_lattice",
    "build_kpath_from_nodes",
    "build_moire_k_grid",
    "build_standard_kpath",
]
