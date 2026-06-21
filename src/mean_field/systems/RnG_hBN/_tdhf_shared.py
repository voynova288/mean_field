"""RLG/hBN adapter helpers for the generic TDHF/RPA core.

This module is intentionally a thin system layer: it extracts HF orbitals from a
converged RLG/hBN HF state, builds fixed-q particle-hole labels, and provides an
on-demand HF-basis two-body matrix element backed by the existing layer form
factors and full-Q Coulomb kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from ...core.contracts import HFRunResult as ContractHFRunResult, HFState as ContractHFState
from ...core.hf import (
    ParticleHolePair,
    SpinValleyFlavor,
    TDHFMatrices,
    TDHFOccupationPolicy,
    TDHFStructureResiduals,
    assemble_tdhf_liouvillian,
    build_tdhf_matrices,
    canonical_tdhf_orbitals_from_hf_run_result,
    canonical_tdhf_orbitals_from_hf_state,
    occupied_state_mask,
    validate_tdhf_structures,
)
from .cache import load_layer_overlap_blocks_cache, load_projected_basis_cache
from .hf import (
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNLayerOverlapBlockSet,
    RLGhBNProjectedBasisData,
    rlg_hbn_occupied_state_count,
)

MomentumPolicy = Literal["strict", "mod_integer"]
FiniteQShortcutChannel = Literal["intervalley", "interspin", "inter_spin_valley"]
FiniteQChannel = Literal["intraflavor", "intervalley", "interspin", "inter_spin_valley"]
FINITE_Q_SHORTCUT_CHANNELS: tuple[str, ...] = ("intervalley", "interspin", "inter_spin_valley")
FINITE_Q_FULL_CHANNELS: tuple[str, ...] = ("intraflavor", *FINITE_Q_SHORTCUT_CHANNELS)
FINITE_Q_KNOWN_CHANNELS: tuple[str, ...] = ("all", *FINITE_Q_FULL_CHANNELS)

__all__ = [name for name in globals() if not name.startswith('__')]
