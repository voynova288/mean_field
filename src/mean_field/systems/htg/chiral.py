from __future__ import annotations

from .lattice import HTGLattice, build_htg_lattice
from .params import HTGParams, theta_deg_from_alpha


MAGIC_ALPHA_ZETA0 = (0.377, 1.197, 1.755, 2.414, 2.991, 3.628, 4.213, 4.840, 5.430)


def build_chiral_lattice_from_alpha(
    alpha: float,
    *,
    n_shells: int = 5,
    params: HTGParams | None = None,
) -> HTGLattice:
    resolved = params if params is not None else HTGParams.chiral()
    theta_deg = theta_deg_from_alpha(alpha, params=resolved)
    return build_htg_lattice(
        theta_deg,
        n_shells=n_shells,
        graphene_lattice_constant_nm=resolved.graphene_lattice_constant_nm,
    )
