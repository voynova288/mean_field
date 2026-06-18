"""Helical twisted quadrilayer graphene (HTQG) continuum model.

Implements the non-interacting single-moiré-domain model of Fujimoto,
Nakatsuji, Vishwanath, and Ledwith (2025), with the convention-locking
lattice/Hamiltonian layer separated from heavier paper reproduction workflows.
"""

from .commensurate import (
    HTQGCommensurateGeometry,
    build_commensurate_geometry,
    commensurate_twist_angles_deg,
    commensurate_twist_angles_rad,
    fujimoto_2025_fig2_checkpoint,
)
from .domains import HTQGDomain, all_domains, canonical_domain_key, domain_displacements, representative_domains
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice import HTQGLattice, build_htqg_lattice, build_standard_kpath
from .model import HTQGModel
from .params import HTQGParams

__all__ = [
    "HTQGCommensurateGeometry",
    "HTQGDomain",
    "HTQGLattice",
    "HTQGParams",
    "HTQGModel",
    "all_domains",
    "build_commensurate_geometry",
    "build_hamiltonian",
    "build_htqg_lattice",
    "build_standard_kpath",
    "canonical_domain_key",
    "commensurate_twist_angles_deg",
    "commensurate_twist_angles_rad",
    "diagonalize_hamiltonian",
    "domain_displacements",
    "fujimoto_2025_fig2_checkpoint",
    "representative_domains",
]
