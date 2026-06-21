from __future__ import annotations

"""Dry-run-first helper for historical canonical HF sidecar backfills.

This module deliberately does not mutate historical result directories by
default.  It uses :func:`mean_field.api.load_result` for metadata-only result
inspection and only opens recognized RLG/hBN ``hf_ground_state.npz`` archives
far enough to inspect their key list and small scalar cache metadata.  The
opt-in write path is staging-only: it requires ``--write``, a caller-specified
``--target-root``, and an explicit target allowlist; it never writes into scanned
historical roots.  It never reruns SCF, diagonalizes grids, computes cRPA, or
writes into ``results/`` unless the caller explicitly stages somewhere under an
allowlisted target outside the historical tree.
"""

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.api import load_result
from mean_field.core.io import write_json_artifact, write_text_artifact

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = REPO_ROOT / "results"
_CANONICAL_HF_SIDECAR_KEY = "canonical_hf_run_result"
_CANONICAL_HF_SIDECAR_FILE = "canonical_hf_run_result.json"
_WRITE_MANIFEST_FILE = "backfill_write_manifest.json"
_BACKFILL_AUDIT_FILE = "canonical_hf_backfill_audit.json"
_MANIFEST_PATCH_FILE = "canonical_hf_manifest_patch.json"

_TDBG_ADAPTER = "mean_field.systems.tdbg.projected_hf_contracts.tdbg_projected_hf_result_to_hf_run_result"
_HTG_PRIMITIVE_ADAPTER = "mean_field.systems.htg.mean_field_adapter.htg_hf_run_to_hf_run_result"
_HTG_ADAPTER = "mean_field.systems.htg.supercell_contracts.htg_supercell_hf_run_to_hf_run_result"
_RLG_HBN_ARCHIVE_LOADER = "mean_field.systems.RnG_hBN.tdhf.load_rlg_hbn_tdhf_run_from_archive"
_RLG_HBN_ADAPTER = "mean_field.systems.RnG_hBN.hf_contracts.rlg_hbn_hf_run_to_hf_run_result"

_RLG_HBN_REQUIRED_ARCHIVE_KEYS = frozenset(
    {
        "density",
        "hamiltonian",
        "h0",
        "energies_mev",
        "reference_density",
        "cache_key_basis",
        "cache_key_overlap",
        "cache_dir",
    }
)
# Mirrors the compatibility checks enforced by
# mean_field.systems.RnG_hBN.cache.load_projected_basis_cache and
# load_layer_overlap_blocks_cache.  Keeping the scanner aligned with the loader
# prevents stale cache manifests from being marked write-eligible only to fail
# after the explicit staging command starts loading cache payloads.
_RLG_HBN_EXPECTED_BASIS_PERIODIC_GAUGE = "centered_cell_reciprocal_relabel_pad1_v2"
_RLG_HBN_EXPECTED_FORM_FACTOR_CONVENTION = "physical_q_plus_g_valley_signed_raw_shift_v2"

_TDBG_HF_STATE_CONTRACT_KEYS = frozenset(
    {
        "density",
        "hamiltonian",
        "h0",
        "energies",
        "k_grid_frac",
        "kvec_nm_inv",
        "band_indices",
        "reference_density",
        "mu",
        "iter_energy",
        "iter_err",
        "iter_oda",
        "n_occupied_per_k",
        "lower_band_count",
    }
)
_TDBG_PROJECTED_BASIS_CONTRACT_KEYS = frozenset(
    {
        "wavefunctions",
        "moire_area_nm2",
        "shifts",
        "shift_gvecs",
        "shift_srcmaps",
        "valley_params",
    }
)
_HTG_PRIMITIVE_STATE_CONTRACT_KEYS = frozenset(
    {
        "density",
        "hamiltonian",
        "h0",
        "energies_ev",
        "kvec_nm_inv",
        "k_grid_frac",
        "iter_energy_ev",
        "iter_err",
        "iter_oda",
        "mu",
        "nu",
        "precision",
        "v0",
        "sigma_z",
        "converged",
        "exit_reason",
        "init_mode",
        "seed",
    }
)
_HTG_PRIMITIVE_BASIS_CONTRACT_KEYS = frozenset(
    {
        "wavefunctions",
        "projected_band_indices",
        "central_band_indices",
        "band_sigma_z",
        "reciprocal_grid_shape",
        "reciprocal_grid_origin",
        "moire_cell_area_nm2",
        "model_params",
        "interaction_params",
    }
)
_HTG_SUPERCELL_STATE_CONTRACT_KEYS = frozenset(
    {
        "density",
        "hamiltonian",
        "h0",
        "energies",
        "kvec",
        "k_grid_frac",
        "iter_energy",
        "iter_err",
        "iter_oda",
        "reference_diagonal",
        "fold_representatives",
        "supercell_matrix",
        "primitive_nu",
        "mu",
        "precision",
        "converged",
        "exit_reason",
        "init_mode",
        "seed",
    }
)
_HTG_SUPERCELL_BASIS_CONTRACT_KEYS = frozenset(
    {
        "wavefunctions",
        "primitive_projected_indices",
        "primitive_band_count",
        "super_g1",
        "super_g2",
        "reciprocal_grid_shape",
        "reciprocal_grid_origin",
        "moire_supercell_area_nm2",
        "model_params",
        "interaction_params",
    }
)


@dataclass(frozen=True)
class BackfillCandidate:
    """One scanned historical candidate.

    ``can_backfill_now`` means the existing repository has enough *loader and
    adapter surface* to reconstruct a canonical contract object without heavy
    compute.  It does not mean this dry-run helper will write anything.
    """

    kind: str
    root: str
    system_name: str
    workflow: str
    decision: str
    can_backfill_now: bool
    would_write: bool
    reason: str
    manifest_path: str | None = None
    archive_path: str | None = None
    target_root: str | None = None
    evidence: tuple[str, ...] = ()
    adapters: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "root": self.root,
            "system_name": self.system_name,
            "workflow": self.workflow,
            "decision": self.decision,
            "can_backfill_now": bool(self.can_backfill_now),
            "would_write": bool(self.would_write),
            "reason": self.reason,
            "manifest_path": self.manifest_path,
            "archive_path": self.archive_path,
            "target_root": self.target_root,
            "evidence": list(self.evidence),
            "adapters": list(self.adapters),
            "blockers": list(self.blockers),
            "uncertainty": list(self.uncertainty),
            "metadata": dict(self.metadata),
        }

# Export private constants/helpers too; split modules intentionally import this via star.
__all__ = [name for name in globals() if not name.startswith('__')]
