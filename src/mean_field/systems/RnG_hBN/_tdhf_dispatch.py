from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403
from ._tdhf_support import *  # noqa: F401,F403
from ._tdhf_types import *  # noqa: F401,F403
from ._tdhf_orbitals import *  # noqa: F401,F403
from ._tdhf_pairs import *  # noqa: F401,F403
from ._tdhf_q0 import *  # noqa: F401,F403
from ._tdhf_finite_q import *  # noqa: F401,F403

def build_rlg_hbn_tdhf_q_matrices(
    run: RLGhBNHartreeFockRun,
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    channel: FiniteQChannel,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    shortcut_exchange_only: bool = True,
) -> TDHFMatrices:
    """Build a dense finite-q TDHF matrix for one supported RLG/hBN channel."""

    channel_text = str(channel)
    support = _require_rlg_hbn_tdhf_finite_q_mode_supported(
        channel_text,
        shortcut_exchange_only=(False if channel_text == "intraflavor" else shortcut_exchange_only),
        canonical_boundary=False,
    )
    channel_key = support.channel
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift)
    pairs = _filter_rlg_hbn_tdhf_finite_q_pairs(all_pairs, channel_key)
    if len(pairs) > int(max_pairs):
        raise ValueError(
            f"finite-q TDHF sector has {len(pairs)} ph pairs, exceeding max_pairs={max_pairs}; "
            "use channel filtering or raise the explicit Slurm-side limit."
        )
    if channel_key == "intraflavor":
        return build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
            run,
            orbitals,
            pairs,
            q_shift,
            beta=beta,
            structure_tolerance=structure_tolerance,
        )
    return build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        q_shift,
        beta=beta,
        structure_tolerance=structure_tolerance,
    )

def build_rlg_hbn_tdhf_q_matrices_from_canonical_hf(
    run: RLGhBNHartreeFockRun,
    canonical_hf: ContractHFState | ContractHFRunResult,
    q_shift: tuple[int, int] | RLGhBNTDHFMomentumShift,
    *,
    channel: FiniteQChannel,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    shortcut_exchange_only: bool = True,
    validate_legacy_parity: bool = True,
    parity_tolerance: float = 1.0e-8,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
    require_complete_umklapp: bool = True,
    physical_shifts: Sequence[tuple[int, int]] | None = None,
) -> TDHFMatrices:
    """Finite-q TDHF matrices using canonical HFState/HFRunResult orbitals.

    This opt-in bridge reuses the system-specific RLG/hBN finite-q wrapping,
    pair filtering, and layer-overlap assembly.  In the intraflavor channel it
    builds the full Eq. D19 A/B block; in flavor-flip channels it builds the
    guarded conduction-only exchange shortcut.
    """

    channel_text = str(channel)
    support = _require_rlg_hbn_tdhf_finite_q_mode_supported(
        channel_text,
        shortcut_exchange_only=(False if channel_text == "intraflavor" else shortcut_exchange_only),
        canonical_boundary=True,
    )
    channel_key = support.channel
    orbitals = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
        canonical_hf,
        n_spin=run.state.n_spin,
        n_eta=run.state.n_eta,
        n_band=run.state.n_band,
        occupation_policy=occupation_policy,
        projector_tolerance=projector_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    if validate_legacy_parity:
        legacy = build_rlg_hbn_tdhf_orbitals(run.state)
        _validate_rlg_hbn_tdhf_orbital_parity(legacy, orbitals, tolerance=parity_tolerance)
    all_pairs = build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift)
    pairs = _filter_rlg_hbn_tdhf_finite_q_pairs(all_pairs, channel_key)
    if len(pairs) > int(max_pairs):
        raise ValueError(
            f"finite-q TDHF sector has {len(pairs)} ph pairs, exceeding max_pairs={max_pairs}; "
            "use channel filtering or raise the explicit Slurm-side limit."
        )
    if channel_key == "intraflavor":
        return build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs(
            run,
            orbitals,
            pairs,
            q_shift,
            beta=beta,
            structure_tolerance=structure_tolerance,
            require_complete_umklapp=require_complete_umklapp,
            physical_shifts=physical_shifts,
        )
    return build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        q_shift,
        beta=beta,
        structure_tolerance=structure_tolerance,
        require_complete_umklapp=require_complete_umklapp,
        physical_shifts=physical_shifts,
    )


def build_rlg_hbn_tdhf_q0_matrices(
    run: RLGhBNHartreeFockRun,
    *,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    assembly: Literal["vectorized", "generic"] = "vectorized",
) -> TDHFMatrices:
    """Dense q=0 TDHF matrices for smoke tests and small checkpoints.

    Large production runs should use channel filtering and eventually a matvec
    eigensolver.  The default dense assembly is vectorized over k blocks and
    layer form factors rather than calling ``V_hf`` for every matrix element.
    """

    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    if len(pairs) > int(max_pairs):
        raise ValueError(
            f"q=0 TDHF sector has {len(pairs)} ph pairs, exceeding max_pairs={max_pairs}; "
            "use channel filtering, a higher explicit max_pairs on a compute node, or a matvec workflow."
        )
    return build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        beta=beta,
        structure_tolerance=structure_tolerance,
        assembly=assembly,
    )



def build_rlg_hbn_tdhf_q0_matrices_from_canonical_hf(
    run: RLGhBNHartreeFockRun,
    canonical_hf: ContractHFState | ContractHFRunResult,
    *,
    beta: float = 1.0,
    max_pairs: int = 4096,
    structure_tolerance: float = 1.0e-6,
    assembly: Literal["vectorized", "generic"] = "vectorized",
    validate_legacy_parity: bool = True,
    parity_tolerance: float = 1.0e-8,
    occupation_policy: TDHFOccupationPolicy = "projector",
    projector_tolerance: float = 1.0e-7,
    degeneracy_tolerance: float = 1.0e-10,
    flavor_resolution_tolerance: float = 1.0e-8,
) -> TDHFMatrices:
    """Dense q=0 TDHF matrices using canonical HFState/HFRunResult orbitals.

    This is an opt-in bridge from the canonical core TDHF boundary to the
    existing RLG/hBN q=0 matrix assembly.  It reuses the system-specific
    layer-form-factor ``V_hf`` path and, by default, validates that the
    canonical orbitals are parity-equivalent to the legacy RLG/hBN orbital
    builder before assembling matrices.
    """

    orbitals = build_rlg_hbn_tdhf_orbitals_from_canonical_hf(
        canonical_hf,
        n_spin=run.state.n_spin,
        n_eta=run.state.n_eta,
        n_band=run.state.n_band,
        occupation_policy=occupation_policy,
        projector_tolerance=projector_tolerance,
        degeneracy_tolerance=degeneracy_tolerance,
        flavor_resolution_tolerance=flavor_resolution_tolerance,
    )
    if validate_legacy_parity:
        legacy = build_rlg_hbn_tdhf_orbitals(run.state)
        _validate_rlg_hbn_tdhf_orbital_parity(legacy, orbitals, tolerance=parity_tolerance)
    pairs = build_rlg_hbn_tdhf_q0_pairs(orbitals)
    if len(pairs) > int(max_pairs):
        raise ValueError(
            f"q=0 TDHF sector has {len(pairs)} ph pairs, exceeding max_pairs={max_pairs}; "
            "use channel filtering, a higher explicit max_pairs on a compute node, or a matvec workflow."
        )
    return build_rlg_hbn_tdhf_q0_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        beta=beta,
        structure_tolerance=structure_tolerance,
        assembly=assembly,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
