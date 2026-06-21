"""Finite-magnetic-field BM/Hofstadter spectrum for TBG.

This module ports the non-interacting magnetic spectrum part of the author
code in ``TBG_HartreeFock（作者原始代码）/libs/bmLL*.jl``.  It constructs the
Landau-level basis Hamiltonian at rational flux ``p/q``, diagonalizes the
central ``2q`` Hofstadter subbands on the magnetic Brillouin-zone mesh, and can
build projected density-overlap matrices for the finite-B HF module.

The implementation intentionally keeps file I/O out of the core.  Production
workflows may save the returned arrays in whatever format is convenient; the HF
adapter consumes the same arrays through :class:`MagneticOverlapData`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from fractions import Fraction
from typing import Literal

import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, gammaln

from ....core.magnetic_field import MagneticFlux, choose_magnetic_nq, magnetic_r_orbit_positions, magnetic_reciprocal_vector
from .hf import MagneticOverlapData

Array = np.ndarray
Valley = Literal["K", "Kprime"]

__all__ = [name for name in globals() if not name.startswith('__')]
