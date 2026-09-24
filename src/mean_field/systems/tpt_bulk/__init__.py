"""Source-bound adapters for the periodic 40-atom TPT bulk lineage.

The accepted HR remains noninteracting authority.  Interacting evaluation is
available only through the fail-closed ``microscopic_hf`` loader, which requires
an immutable same-parent PAW-complete/all-electron ``rho(k,q+G)`` bundle plus
explicit Hartree, Fock, screening, filling, background, and double-counting
conventions. Raw pseudo ``WAVECAR``/``UNK*`` contractions cannot authorize the
bundle, and no vertex is inferred from HR coefficients, centres, point charges,
or ``.mmn`` alone.
"""

from .microscopic_hf import (
    TPT_BULK_DENSITY_VERTEX_DEFINITION,
    TPT_BULK_MICROSCOPIC_DATA,
    TPT_BULK_MICROSCOPIC_FORMAT,
    TPT_BULK_MICROSCOPIC_MANIFEST,
    TPTBulkHFState,
    TPTBulkInteractionEnergyEvaluation,
    TPTBulkMicroscopicHFInputs,
    build_tpt_bulk_hf_problem,
    evaluate_tpt_bulk_microscopic_hf,
    load_tpt_bulk_microscopic_hf_inputs,
)
from .source import (
    TPT_BULK_HR_SHA256,
    TPT_BULK_NUM_RPTS,
    TPT_BULK_NUM_WANN,
    TPT_BULK_SOURCE_MESH,
    TPTBulkActiveEigensystem,
    TPTBulkSource,
    build_tpt_bulk_active_eigensystem,
    load_tpt_bulk_source,
)

__all__ = [
    "TPT_BULK_DENSITY_VERTEX_DEFINITION",
    "TPT_BULK_HR_SHA256",
    "TPT_BULK_MICROSCOPIC_DATA",
    "TPT_BULK_MICROSCOPIC_FORMAT",
    "TPT_BULK_MICROSCOPIC_MANIFEST",
    "TPT_BULK_NUM_RPTS",
    "TPT_BULK_NUM_WANN",
    "TPT_BULK_SOURCE_MESH",
    "TPTBulkActiveEigensystem",
    "TPTBulkHFState",
    "TPTBulkInteractionEnergyEvaluation",
    "TPTBulkMicroscopicHFInputs",
    "TPTBulkSource",
    "build_tpt_bulk_active_eigensystem",
    "build_tpt_bulk_hf_problem",
    "evaluate_tpt_bulk_microscopic_hf",
    "load_tpt_bulk_microscopic_hf_inputs",
    "load_tpt_bulk_source",
]
