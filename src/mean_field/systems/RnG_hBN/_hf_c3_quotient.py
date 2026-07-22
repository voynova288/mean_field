from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.hf import compute_density_overlap_trace_from_diagonal
from ._hf_basis import (
    _build_c3_fixed_remote_representative_source,
    _c3_transform_raw_components,
)
from ._hf_interaction_path import (
    _contract_layer_fock_term,
    _fock_overlap_shift_for_physical_transfer,
    _maybe_zero_literal_q0_fock_kernel,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_layer_overlap_blocks_between,
    fock_transfer_wrap_masks_between,
)
from ._hf_types import (
    RLGhBNInteractionComponents,
    RLGhBNLayerOverlapBlockSet,
    RLGhBNProjectedBasisData,
)


RLG_HBN_HF_INTERACTION_CONVENTION_VERSION = (
    "actual_node_ws_fixed_variational_copy_v2"
)


@dataclass(frozen=True)
class _RLGhBNHFFixedSourceContext:
    pair: tuple[int, int]
    source_index: int
    source_basis: RLGhBNProjectedBasisData
    copy_transforms: tuple[np.ndarray, np.ndarray, np.ndarray]
    hartree_blocks: RLGhBNLayerOverlapBlockSet
    fock_overlaps: dict[tuple[int, int], np.ndarray]
    fock_kernels: dict[tuple[int, int], np.ndarray]
    fock_weights: dict[tuple[int, int], np.ndarray]


@dataclass(frozen=True)
class _RLGhBNHFFockContractionContext:
    overlaps: dict[tuple[int, int], np.ndarray]
    kernels: dict[tuple[int, int], np.ndarray]
    weights: dict[tuple[int, int], np.ndarray]


@dataclass(frozen=True)
class _RLGhBNHFFixedTargetContext:
    pair: tuple[int, int]
    target_index: int
    target_basis: RLGhBNProjectedBasisData
    copy_transforms: tuple[np.ndarray, np.ndarray, np.ndarray]
    hartree_blocks: RLGhBNLayerOverlapBlockSet
    ordinary_fock: _RLGhBNHFFockContractionContext
    fixed_fock: tuple[_RLGhBNHFFockContractionContext, ...]


@dataclass(frozen=True)
class RLGhBNHFC3QuotientInteractionContext:
    """Precomputed lift-contract-descend data for one variational HF quotient."""

    basis_data: RLGhBNProjectedBasisData
    base_blocks: RLGhBNLayerOverlapBlockSet
    physical_shifts: tuple[tuple[int, int], ...]
    fixed_indices: tuple[int, ...]
    ordinary_fock_overlaps: dict[tuple[int, int], np.ndarray]
    ordinary_fock_kernels: dict[tuple[int, int], np.ndarray]
    ordinary_fock_weights: dict[tuple[int, int], np.ndarray]
    fixed_sources: tuple[_RLGhBNHFFixedSourceContext, ...]
    fixed_targets: tuple[_RLGhBNHFFixedTargetContext, ...]



def _merge_fock_data(
    base: RLGhBNLayerOverlapBlockSet,
    extra: RLGhBNLayerOverlapBlockSet,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    overlaps = dict(base.layer_overlaps)
    overlaps.update(extra.layer_overlaps)
    kernels = dict(base.fock_layer_coulomb)
    kernels.update(extra.fock_layer_coulomb)
    return overlaps, kernels



def _fock_weights_by_overlap_key(
    physical_shifts: tuple[tuple[int, int], ...],
    wrap_masks: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], np.ndarray]:
    result: dict[tuple[int, int], np.ndarray] = {}
    for physical_shift in physical_shifts:
        for wrap, mask in wrap_masks.items():
            key = _fock_overlap_shift_for_physical_transfer(physical_shift, wrap)
            if key not in result:
                result[key] = np.zeros_like(mask, dtype=float)
            result[key] += np.asarray(mask, dtype=float)
    return result



def _build_fock_contraction_context(
    target_basis: RLGhBNProjectedBasisData,
    source_basis: RLGhBNProjectedBasisData,
    physical_shifts: tuple[tuple[int, int], ...],
) -> _RLGhBNHFFockContractionContext:
    wrap_masks = fock_transfer_wrap_masks_between(target_basis, source_basis)
    weights = _fock_weights_by_overlap_key(physical_shifts, wrap_masks)
    keys = tuple(sorted(weights))
    blocks = build_rlg_hbn_layer_overlap_blocks_between(
        target_basis,
        source_basis,
        shifts=keys,
    )
    return _RLGhBNHFFockContractionContext(
        overlaps=dict(blocks.layer_overlaps),
        kernels=dict(blocks.fock_layer_coulomb),
        weights=weights,
    )


def _fixed_copy_sewing(
    source_basis: RLGhBNProjectedBasisData,
    source_copy: int,
    target_copy: int,
) -> np.ndarray:
    if source_basis.periodic_reciprocal_shifts is None:
        raise RuntimeError("fixed representative source has no periodic reciprocal shifts")
    n_spin = int(source_basis.basis.n_spin)
    n_flavor = int(source_basis.basis.n_flavor)
    n_band = int(source_basis.n_band)
    nt = int(source_basis.nt)
    indices = np.arange(nt, dtype=int).reshape((n_spin, n_flavor, n_band), order="F")
    source_shift = source_basis.periodic_reciprocal_shifts[int(source_copy)]
    target_shift = source_basis.periodic_reciprocal_shifts[int(target_copy)]
    result = np.zeros((nt, nt), dtype=np.complex128)
    for flavor in range(n_flavor):
        block = np.zeros((n_band, n_band), dtype=np.complex128)
        for source_band in range(n_band):
            transformed = _c3_transform_raw_components(
                source_basis.basis.wavefunctions[:, source_band, flavor, source_copy],
                source_basis,
                valley=int(source_basis.valleys[flavor]),
                source_total_shift=source_shift,
                target_total_shift=target_shift,
            )
            for target_band in range(n_band):
                block[target_band, source_band] = np.vdot(
                    source_basis.basis.wavefunctions[:, target_band, flavor, target_copy],
                    transformed,
                )
        for spin in range(n_spin):
            block_indices = indices[spin, flavor, :]
            result[np.ix_(block_indices, block_indices)] = block
    return result



def _fixed_copy_transforms(
    source_basis: RLGhBNProjectedBasisData,
    *,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_basis.nk != 3:
        raise ValueError(f"fixed representative source must have nk=3, got {source_basis.nk}")
    s01 = _fixed_copy_sewing(source_basis, 0, 1)
    s12 = _fixed_copy_sewing(source_basis, 1, 2)
    s20 = _fixed_copy_sewing(source_basis, 2, 0)
    identity = np.eye(source_basis.nt, dtype=np.complex128)
    residuals = {
        "s01": float(np.max(np.abs(s01.conj().T @ s01 - identity))),
        "s12": float(np.max(np.abs(s12.conj().T @ s12 - identity))),
        "s20": float(np.max(np.abs(s20.conj().T @ s20 - identity))),
        "cycle": float(np.max(np.abs(s20 @ s12 @ s01 - identity))),
    }
    if max(residuals.values()) > float(tolerance):
        raise RuntimeError(
            "fixed representative copy sewing is not unitary/closed: "
            f"residuals={residuals}, tolerance={float(tolerance):.6e}"
        )
    return identity, s01, s12 @ s01



def _build_fixed_source_context(
    basis_data: RLGhBNProjectedBasisData,
    base_blocks: RLGhBNLayerOverlapBlockSet,
    physical_shifts: tuple[tuple[int, int], ...],
    pair: tuple[int, int],
) -> _RLGhBNHFFixedSourceContext:
    source_basis = _build_c3_fixed_remote_representative_source(
        basis_data,
        basis_data,
        pair,
    )
    copy_transforms = _fixed_copy_transforms(source_basis)
    hartree_blocks = build_rlg_hbn_layer_overlap_blocks(
        source_basis,
        shifts=physical_shifts,
    )
    fock = _build_fock_contraction_context(
        basis_data,
        source_basis,
        physical_shifts,
    )
    mesh = int(basis_data.mesh_size)
    return _RLGhBNHFFixedSourceContext(
        pair=(int(pair[0]), int(pair[1])),
        source_index=int(pair[0]) * mesh + int(pair[1]),
        source_basis=source_basis,
        copy_transforms=copy_transforms,
        hartree_blocks=hartree_blocks,
        fock_overlaps=fock.overlaps,
        fock_kernels=fock.kernels,
        fock_weights=fock.weights,
    )



def _build_fixed_target_context(
    basis_data: RLGhBNProjectedBasisData,
    physical_shifts: tuple[tuple[int, int], ...],
    target_source: _RLGhBNHFFixedSourceContext,
    fixed_sources: tuple[_RLGhBNHFFixedSourceContext, ...],
) -> _RLGhBNHFFixedTargetContext:
    target_basis = target_source.source_basis
    ordinary_fock = _build_fock_contraction_context(
        target_basis,
        basis_data,
        physical_shifts,
    )
    fixed_fock = tuple(
        _build_fock_contraction_context(
            target_basis,
            source.source_basis,
            physical_shifts,
        )
        for source in fixed_sources
    )
    return _RLGhBNHFFixedTargetContext(
        pair=target_source.pair,
        target_index=target_source.source_index,
        target_basis=target_basis,
        copy_transforms=target_source.copy_transforms,
        hartree_blocks=target_source.hartree_blocks,
        ordinary_fock=ordinary_fock,
        fixed_fock=fixed_fock,
    )


def build_rlg_hbn_hf_c3_quotient_interaction_context(
    basis_data: RLGhBNProjectedBasisData,
    base_blocks: RLGhBNLayerOverlapBlockSet,
    *,
    physical_shifts: tuple[tuple[int, int], ...] | None = None,
) -> RLGhBNHFC3QuotientInteractionContext:
    """Precompute the C3-covariant HF contraction provider.

    Ordinary nodes use actual-node Wigner-Seitz transfer folding with exact
    boundary-tie averaging. Nonzero C3-fixed source densities are injected into
    their three periodic-gauge representatives with weight ``1/3``. Responses
    at fixed target copies are then descended with the pairing-adjoint map, so
    the active interaction is the variational quotient ``L^sharp V L``.
    """

    if basis_data.mesh_size <= 0:
        raise ValueError("C3 quotient HF interaction requires a regular mesh")
    if basis_data.periodic_reciprocal_shifts is None:
        raise ValueError("C3 quotient HF interaction requires periodic reciprocal shifts")
    resolved_physical = (
        tuple((int(x), int(y)) for x, y in base_blocks.shifts)
        if physical_shifts is None
        else tuple((int(x), int(y)) for x, y in physical_shifts)
    )
    missing_physical = [shift for shift in resolved_physical if shift not in base_blocks.layer_overlaps]
    if missing_physical:
        raise ValueError(f"base overlap blocks are missing physical shifts: {missing_physical}")

    wrap_masks = fock_transfer_wrap_masks_between(basis_data, basis_data)
    ordinary_weights = _fock_weights_by_overlap_key(resolved_physical, wrap_masks)
    missing_keys = tuple(sorted(key for key in ordinary_weights if key not in base_blocks.layer_overlaps))
    extra_blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=missing_keys)
    ordinary_overlaps, ordinary_kernels = _merge_fock_data(base_blocks, extra_blocks)
    fixed_pairs = tuple(
        (int(pair[0]), int(pair[1]))
        for pair in basis_data.c3_fixed_representative_pairs
    )
    fixed_sources = tuple(
        _build_fixed_source_context(
            basis_data,
            base_blocks,
            resolved_physical,
            pair,
        )
        for pair in fixed_pairs
    )
    fixed_targets = tuple(
        _build_fixed_target_context(
            basis_data,
            resolved_physical,
            target_source,
            fixed_sources,
        )
        for target_source in fixed_sources
    )
    mesh = int(basis_data.mesh_size)
    fixed_indices = tuple(sorted(int(pair[0]) * mesh + int(pair[1]) for pair in fixed_pairs))
    return RLGhBNHFC3QuotientInteractionContext(
        basis_data=basis_data,
        base_blocks=base_blocks,
        physical_shifts=resolved_physical,
        fixed_indices=fixed_indices,
        ordinary_fock_overlaps=ordinary_overlaps,
        ordinary_fock_kernels=ordinary_kernels,
        ordinary_fock_weights=ordinary_weights,
        fixed_sources=fixed_sources,
        fixed_targets=fixed_targets,
    )



def _contract_ws_fock(
    density_delta: np.ndarray,
    overlaps: dict[tuple[int, int], np.ndarray],
    kernels: dict[tuple[int, int], np.ndarray],
    weights: dict[tuple[int, int], np.ndarray],
    *,
    scale: float,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    density = np.asarray(density_delta, dtype=np.complex128)
    result = np.zeros(target_shape, dtype=np.complex128)
    for key, node_weights in weights.items():
        overlap = overlaps[key]
        kernel = _maybe_zero_literal_q0_fock_kernel(key, kernels[key])
        for target_layer in range(overlap.shape[0]):
            for source_layer in range(overlap.shape[0]):
                coeff = float(scale) * np.asarray(node_weights, dtype=float) * kernel[
                    :, :, target_layer, source_layer
                ]
                if np.any(coeff != 0.0):
                    result -= _contract_layer_fock_term(
                        overlap[target_layer],
                        density,
                        coeff,
                        overlap[source_layer],
                    )
    return result



def _contract_hartree_between(
    density_delta: np.ndarray,
    target_blocks: RLGhBNLayerOverlapBlockSet,
    source_blocks: RLGhBNLayerOverlapBlockSet,
    physical_shifts: tuple[tuple[int, int], ...],
    *,
    scale: float,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    density = np.asarray(density_delta, dtype=np.complex128)
    result = np.zeros(target_shape, dtype=np.complex128)
    for shift in physical_shifts:
        target_diagonal = target_blocks.layer_diagonal_overlaps[shift]
        source_diagonal = source_blocks.layer_diagonal_overlaps[shift]
        kernel = source_blocks.hartree_layer_coulomb[shift]
        traces = np.asarray(
            [
                compute_density_overlap_trace_from_diagonal(
                    density,
                    source_diagonal[layer],
                )
                for layer in range(source_diagonal.shape[0])
            ],
            dtype=np.complex128,
        )
        for target_layer in range(target_diagonal.shape[0]):
            prefactor = float(scale) * complex(np.dot(kernel[target_layer, :], traces))
            if prefactor != 0.0:
                result += prefactor * target_diagonal[target_layer]
    return result



def _expanded_fixed_density(
    density_at_fixed_k: np.ndarray,
    context: _RLGhBNHFFixedSourceContext,
) -> np.ndarray:
    source = np.asarray(density_at_fixed_k, dtype=np.complex128)
    expected = (context.source_basis.nt, context.source_basis.nt)
    if source.shape != expected:
        raise ValueError(f"fixed density shape {source.shape} != {expected}")
    # HF stores D[a,b]=<c_a^† c_b>-R[a,b] = P^T[a,b]-R[a,b]. The copy
    # transforms are ket sewings, u_copy = S_copy u_source, so the stored
    # density transforms contragrediently as S* D S^T, not S D S^†.
    expanded = np.empty((*expected, 3), dtype=np.complex128)
    for copy, transform in enumerate(context.copy_transforms):
        expanded[:, :, copy] = (
            transform.conj() @ source @ transform.T / 3.0
        )
    return expanded



def _descend_fixed_hamiltonian(
    physical_hamiltonian: np.ndarray,
    context: _RLGhBNHFFixedTargetContext,
) -> np.ndarray:
    values = np.asarray(physical_hamiltonian, dtype=np.complex128)
    expected = (context.target_basis.nt, context.target_basis.nt, 3)
    if values.shape != expected:
        raise ValueError(
            f"fixed target Hamiltonian shape {values.shape} != {expected}"
        )
    descended = np.zeros(expected[:2], dtype=np.complex128)
    for copy, transform in enumerate(context.copy_transforms):
        descended += (
            transform.conj().T @ values[:, :, copy] @ transform / 3.0
        )
    return descended


def _contract_fixed_target_physical_components(
    ordinary_density: np.ndarray,
    expanded_fixed_densities: tuple[np.ndarray, ...],
    quotient: RLGhBNHFC3QuotientInteractionContext,
    target: _RLGhBNHFFixedTargetContext,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_shape = (target.target_basis.nt, target.target_basis.nt, 3)
    hartree = _contract_hartree_between(
        ordinary_density,
        target.hartree_blocks,
        quotient.base_blocks,
        quotient.physical_shifts,
        scale=scale,
        target_shape=target_shape,
    )
    fock = _contract_ws_fock(
        ordinary_density,
        target.ordinary_fock.overlaps,
        target.ordinary_fock.kernels,
        target.ordinary_fock.weights,
        scale=scale,
        target_shape=target_shape,
    )
    for source, expanded_density, fock_context in zip(
        quotient.fixed_sources,
        expanded_fixed_densities,
        target.fixed_fock,
        strict=True,
    ):
        hartree += _contract_hartree_between(
            expanded_density,
            target.hartree_blocks,
            source.hartree_blocks,
            quotient.physical_shifts,
            scale=scale,
            target_shape=target_shape,
        )
        fock += _contract_ws_fock(
            expanded_density,
            fock_context.overlaps,
            fock_context.kernels,
            fock_context.weights,
            scale=scale,
            target_shape=target_shape,
        )
    return hartree, fock


def _contract_fixed_target_components(
    ordinary_density: np.ndarray,
    expanded_fixed_densities: tuple[np.ndarray, ...],
    quotient: RLGhBNHFC3QuotientInteractionContext,
    target: _RLGhBNHFFixedTargetContext,
    *,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    hartree, fock = _contract_fixed_target_physical_components(
        ordinary_density,
        expanded_fixed_densities,
        quotient,
        target,
        scale=scale,
    )
    return (
        _descend_fixed_hamiltonian(hartree, target),
        _descend_fixed_hamiltonian(fock, target),
    )


def build_rlg_hbn_hf_c3_quotient_interaction_components(
    density_delta: np.ndarray,
    context: RLGhBNHFC3QuotientInteractionContext,
    *,
    v0: float,
    beta: float = 1.0,
) -> RLGhBNInteractionComponents:
    """Apply the variational fixed-copy quotient ``L^sharp V L``."""

    density = np.asarray(density_delta, dtype=np.complex128)
    basis_data = context.basis_data
    expected = (basis_data.nt, basis_data.nt, basis_data.nk)
    if density.shape != expected:
        raise ValueError(f"Expected density_delta shape {expected}, got {density.shape}")
    scale = float(beta) * float(v0) / float(basis_data.nk)
    ordinary_density = density.copy()
    if context.fixed_indices:
        ordinary_density[:, :, np.asarray(context.fixed_indices, dtype=int)] = 0.0

    hartree = _contract_hartree_between(
        ordinary_density,
        context.base_blocks,
        context.base_blocks,
        context.physical_shifts,
        scale=scale,
        target_shape=expected,
    )
    fock = _contract_ws_fock(
        ordinary_density,
        context.ordinary_fock_overlaps,
        context.ordinary_fock_kernels,
        context.ordinary_fock_weights,
        scale=scale,
        target_shape=expected,
    )
    expanded_fixed_densities = tuple(
        _expanded_fixed_density(density[:, :, fixed.source_index], fixed)
        for fixed in context.fixed_sources
    )
    for fixed, expanded_density in zip(
        context.fixed_sources,
        expanded_fixed_densities,
        strict=True,
    ):
        hartree += _contract_hartree_between(
            expanded_density,
            context.base_blocks,
            fixed.hartree_blocks,
            context.physical_shifts,
            scale=scale,
            target_shape=expected,
        )
        fock += _contract_ws_fock(
            expanded_density,
            fixed.fock_overlaps,
            fixed.fock_kernels,
            fixed.fock_weights,
            scale=scale,
            target_shape=expected,
        )

    # Ordinary target blocks already represent one quotient coordinate. Fixed
    # target coordinates represent three physical copies, so replace their
    # base-target values by the pairing-adjoint descent of all physical target
    # copy responses. This chain-rule step makes the active kernel L^sharp V L
    # rather than the non-variational source-only map V L.
    for target in context.fixed_targets:
        target_hartree, target_fock = _contract_fixed_target_components(
            ordinary_density,
            expanded_fixed_densities,
            context,
            target,
            scale=scale,
        )
        hartree[:, :, target.target_index] = target_hartree
        fock[:, :, target.target_index] = target_fock

    return RLGhBNInteractionComponents(
        hartree=hartree,
        fock=fock,
        total=hartree + fock,
    )


__all__ = [
    "RLG_HBN_HF_INTERACTION_CONVENTION_VERSION",
    "RLGhBNHFC3QuotientInteractionContext",
    "build_rlg_hbn_hf_c3_quotient_interaction_components",
    "build_rlg_hbn_hf_c3_quotient_interaction_context",
]
