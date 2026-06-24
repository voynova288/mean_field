from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from analysis.topology import (
    LinkMethod,
    SewingTransform,
    TopologyResult,
    compute_system_topology_from_eigenvectors,
    compute_system_topology_from_grid_result,
    normalize_state_indices,
    reshape_flat_mesh_to_grid,
)

from ._polshyn_reconstruction import reconstruct_polshyn_wang_hf_micro_wavefunctions
from ._polshyn_types import PolshynProjectedBasis, PolshynWangHFState
from .bands import compute_bands_on_grid


def _reciprocal_translation(lattice, *, block_size: int, dn1: int, dn2: int) -> SewingTransform:
    by_g = {tuple(int(v) for v in pair): idx for idx, pair in enumerate(lattice.g_indices)}
    block = int(block_size)

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.complex128)
        if array.shape[0] != block * int(lattice.n_g):
            raise ValueError(f"Expected first axis {block * int(lattice.n_g)}, got {array.shape[0]}")
        out = np.zeros_like(array)
        for target, (n1, n2) in enumerate(lattice.g_indices):
            source = by_g.get((int(n1) + int(dn1), int(n2) + int(dn2)))
            if source is not None:
                out[block * target : block * (target + 1)] = array[block * source : block * (source + 1)]
        return out

    return apply


def boundary_sewing_transforms(lattice) -> tuple[SewingTransform, SewingTransform]:
    return (
        _reciprocal_translation(lattice, block_size=6, dn1=1, dn2=0),
        _reciprocal_translation(lattice, block_size=6, dn1=0, dn2=1),
    )


def _polshyn_basis_layout(basis: PolshynProjectedBasis) -> tuple[int, int, int, int, int, int, int]:
    local = int(basis.local_basis_size)
    if local <= 0:
        raise ValueError(f"Polshyn local_basis_size must be positive, got {local}")
    embedding_shape = tuple(int(value) for value in basis.embedding_shape)
    if len(embedding_shape) != 2 or embedding_shape[0] <= 0 or embedding_shape[1] <= 0:
        raise ValueError(f"Polshyn embedding_shape must contain two positive entries, got {basis.embedding_shape}")
    mesh_x, mesh_y = embedding_shape
    basis_dim = int(basis.basis_dimension)
    expected_basis_dim = local * mesh_x * mesh_y
    if basis_dim != expected_basis_dim:
        raise ValueError(
            "Polshyn basis_dimension does not match basis_F(local=6,embed_x,embed_y): "
            f"{basis_dim} != {local} * {mesh_x} * {mesh_y} = {expected_basis_dim}"
        )
    n_spin = int(basis.n_spin)
    n_valley = int(basis.n_eta)
    if n_spin <= 0 or n_valley <= 0:
        raise ValueError(f"Polshyn n_spin/n_eta must be positive, got {(n_spin, n_valley)}")
    micro_dim = n_spin * n_valley * basis_dim
    return local, mesh_x, mesh_y, basis_dim, n_spin, n_valley, micro_dim


def _polshyn_row_translation(basis: PolshynProjectedBasis, *, d_b1: int, d_b2: int) -> SewingTransform:
    """Return a zero-filled reciprocal-row translation in doubled-cell coordinates.

    The selected vectors passed by the common FHS core have first axis ordered as
    ``spin_major, valley_inner, basis_F(local=6, embed_x, embed_y)``.  A torus
    wrap by ``+B_a`` compares the target row at ``(embed_x, embed_y)`` with the
    source row shifted by ``+1`` along the corresponding doubled-cell reciprocal
    embedding axis, matching the primitive TMBG boundary-sewing convention.
    """

    local, embed_x, embed_y, basis_dim, n_spin, n_valley, micro_dim = _polshyn_basis_layout(basis)
    source_dx, source_dy = int(d_b1), int(d_b2)

    def apply(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.complex128)
        if array.shape[0] != micro_dim:
            raise ValueError(f"Expected first axis {micro_dim} for Polshyn projected-HF sewing, got {array.shape[0]}")
        frames = int(np.prod(array.shape[1:], dtype=np.int64)) if array.ndim > 1 else 1
        matrix = array.reshape((micro_dim, frames), order="C")
        out = np.zeros_like(matrix)
        for ispin in range(n_spin):
            for ivalley in range(n_valley):
                row_start = (ispin * n_valley + ivalley) * basis_dim
                block = matrix[row_start : row_start + basis_dim, :].reshape(
                    (local, embed_x, embed_y, frames),
                    order="F",
                )
                shifted = np.zeros_like(block)
                for target_x in range(embed_x):
                    source_x = target_x + source_dx
                    if source_x < 0 or source_x >= embed_x:
                        continue
                    for target_y in range(embed_y):
                        source_y = target_y + source_dy
                        if source_y < 0 or source_y >= embed_y:
                            continue
                        shifted[:, target_x, target_y, :] = block[:, source_x, source_y, :]
                out[row_start : row_start + basis_dim, :] = shifted.reshape((basis_dim, frames), order="F")
        return out.reshape(array.shape, order="C")

    return apply


def polshyn_projected_hf_boundary_sewing_transforms(
    basis: PolshynProjectedBasis,
) -> tuple[SewingTransform, SewingTransform]:
    """Boundary sewing for Polshyn projected-HF rows over doubled-cell ``B1/B2``.

    This is separate from the public flat diagnostic reconstruction API: it is
    intended only for the topology adapter below, after the flat ``(nk,basis,state)``
    wavefunctions are reshaped to a torus grid.
    """

    return (
        _polshyn_row_translation(basis, d_b1=1, d_b2=0),
        _polshyn_row_translation(basis, d_b1=0, d_b2=1),
    )


def _coerce_polshyn_mesh_shape(raw_shape: object, *, n_k: int, source: str) -> tuple[int, int]:
    try:
        mesh_shape = tuple(int(value) for value in raw_shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"Could not read Polshyn projected-HF mesh shape from {source}: {raw_shape!r}") from exc
    if len(mesh_shape) != 2:
        raise ValueError(f"Polshyn projected-HF mesh shape from {source} must have length two, got {mesh_shape}")
    mesh_b1, mesh_b2 = mesh_shape
    if mesh_b1 <= 0 or mesh_b2 <= 0 or mesh_b1 * mesh_b2 != int(n_k):
        raise ValueError(
            f"Polshyn projected-HF mesh shape {mesh_shape} from {source} is incompatible with n_k={int(n_k)}"
        )
    return int(mesh_b1), int(mesh_b2)


def _polshyn_expected_flat_k_grid(k_grid_frac: np.ndarray, mesh_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(k_grid_frac, dtype=float)
    unique_b1 = np.unique(np.round(arr[:, 0], decimals=14))
    unique_b2 = np.unique(np.round(arr[:, 1], decimals=14))
    if unique_b1.size != int(mesh_shape[0]) or unique_b2.size != int(mesh_shape[1]):
        raise ValueError(
            "Polshyn k_grid_frac unique coordinate counts do not match mesh_shape: "
            f"got {(int(unique_b1.size), int(unique_b2.size))}, expected {mesh_shape}"
        )
    return np.asarray([(float(x), float(y)) for y in unique_b2 for x in unique_b1], dtype=float)


def _validate_polshyn_flat_k_order(k_grid_frac: np.ndarray, mesh_shape: tuple[int, int], *, source: str) -> None:
    arr = np.asarray(k_grid_frac, dtype=float)
    if arr.shape != (int(mesh_shape[0]) * int(mesh_shape[1]), 2):
        raise ValueError(
            f"Polshyn flat k_grid_frac from {source} must have shape ({int(mesh_shape[0]) * int(mesh_shape[1])}, 2), got {arr.shape}"
        )
    expected = _polshyn_expected_flat_k_grid(arr, mesh_shape)
    if not np.allclose(arr, expected, atol=1.0e-12, rtol=0.0):
        raise ValueError(
            "Polshyn flat k_grid_frac order must be iy/f2 outer and ix/f1 inner before order='F' topology reshape; "
            f"source={source!r} does not match that ordering. Pass an explicitly gridded k_grid_frac or rebuild the basis."
        )


def _infer_polshyn_mesh_shape_from_k_grid(k_grid_frac: np.ndarray, *, n_k: int) -> tuple[int, int] | None:
    arr = np.asarray(k_grid_frac, dtype=float)
    if arr.shape != (int(n_k), 2):
        return None
    unique_b1 = np.unique(np.round(arr[:, 0], decimals=14))
    unique_b2 = np.unique(np.round(arr[:, 1], decimals=14))
    if unique_b1.size <= 0 or unique_b2.size <= 0 or unique_b1.size * unique_b2.size != int(n_k):
        return None
    mesh_shape = (int(unique_b1.size), int(unique_b2.size))
    _validate_polshyn_flat_k_order(arr, mesh_shape, source="basis.k_grid_frac")
    return mesh_shape


def _polshyn_projected_hf_mesh_shape(
    basis: PolshynProjectedBasis,
    *,
    n_k: int,
    mesh_shape: tuple[int, int] | None,
) -> tuple[tuple[int, int], str]:
    if mesh_shape is not None:
        return _coerce_polshyn_mesh_shape(mesh_shape, n_k=n_k, source="mesh_shape argument"), "mesh_shape argument"
    raw_k_grid = basis.k_grid_frac
    if raw_k_grid is not None:
        inferred = _infer_polshyn_mesh_shape_from_k_grid(np.asarray(raw_k_grid, dtype=float), n_k=n_k)
        if inferred is not None:
            return inferred, "basis.k_grid_frac unique fractional coordinates"
    raise ValueError(
        "Polshyn projected-HF topology requires explicit mesh_shape or a validated basis.k_grid_frac rectangular grid "
        "with iy/f2 outer, ix/f1 inner flat order. Refusing to infer a topology torus from sqrt(n_k) alone."
    )


def _polshyn_projected_hf_k_grid_frac(
    basis: PolshynProjectedBasis,
    mesh_shape: tuple[int, int],
    k_grid_frac,
) -> np.ndarray | None:
    if k_grid_frac is None:
        raw = basis.k_grid_frac
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=float)
    else:
        arr = np.asarray(k_grid_frac, dtype=float)
    if arr.shape == mesh_shape + (2,):
        return arr
    if arr.shape == (mesh_shape[0] * mesh_shape[1], 2):
        _validate_polshyn_flat_k_order(arr, mesh_shape, source="k_grid_frac argument" if k_grid_frac is not None else "basis.k_grid_frac")
        return reshape_flat_mesh_to_grid(arr, mesh_shape, k_axis=0, order="F")
    raise ValueError(
        "Polshyn projected-HF k_grid_frac must have shape "
        f"({mesh_shape[0] * mesh_shape[1]}, 2) or {mesh_shape + (2,)}, got {arr.shape}"
    )


def _resolve_polshyn_state_indices(
    state_indices: int | Iterable[int] | None,
    band_indices: int | Iterable[int] | None,
) -> int | Iterable[int] | None:
    if state_indices is None:
        return band_indices
    if band_indices is None:
        return state_indices
    if normalize_state_indices(state_indices) != normalize_state_indices(band_indices):
        raise ValueError("state_indices and band_indices are aliases for Polshyn HF states and must match if both are supplied")
    return state_indices


def compute_polshyn_projected_hf_topology(
    basis: PolshynProjectedBasis,
    state: PolshynWangHFState | None = None,
    active_eigenvectors: np.ndarray | None = None,
    state_indices: int | Iterable[int] | None = None,
    *,
    band_indices: int | Iterable[int] | None = None,
    valley: int = 0,
    mesh_shape: tuple[int, int] | None = None,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    boundary_sewing: bool = True,
    diagnostic_no_sewing: bool = False,
    max_dense_elements: int | None = 5_000_000,
    off_sector_atol: float = 1.0e-8,
    hermiticity_atol: float = 1.0e-8,
    stored_energy_atol: float | None = 1.0e-7,
    unitarity_atol: float | None = 1.0e-8,
    link_method: LinkMethod = "polar",
    orientation_sign: float = 1.0,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
    metadata: Mapping[str, object] | None = None,
) -> TopologyResult:
    """Compute FHS topology for selected TMBG/Polshyn projected-HF states.

    The public Polshyn reconstruction facade remains a flat diagnostic API and
    returns topology-ineligible bundles.  This reviewed adapter is the separate
    topology path: reconstruct flat ``(nk,basis,state)`` rows, reshape the flat
    k axis as ``iy/f2`` outer and ``ix/f1`` inner using ``order='F'`` to obtain
    ``(mesh_B1, mesh_B2, basis, state)``, attach doubled-cell ``B1/B2`` boundary
    sewing, and then delegate the FHS link/plaquette calculation to the common
    topology core.
    """

    selected_arg = _resolve_polshyn_state_indices(state_indices, band_indices)
    flat_bundle = reconstruct_polshyn_wang_hf_micro_wavefunctions(
        basis,
        state=state,
        active_eigenvectors=active_eigenvectors,
        include_sewing=False,
        state_indices=selected_arg,
        max_dense_elements=max_dense_elements,
        off_sector_atol=off_sector_atol,
        hermiticity_atol=hermiticity_atol,
        stored_energy_atol=stored_energy_atol,
        unitarity_atol=unitarity_atol,
    )
    basis_metadata = dict(flat_bundle.basis_metadata)
    psi_flat = np.asarray(flat_bundle.psi_micro, dtype=np.complex128)
    if psi_flat.ndim != 3:
        raise ValueError(f"Expected reconstructed Polshyn psi_micro shape (nk,basis,state), got {psi_flat.shape}")
    resolved_mesh_shape, mesh_shape_source = _polshyn_projected_hf_mesh_shape(
        basis,
        n_k=int(psi_flat.shape[0]),
        mesh_shape=mesh_shape,
    )
    psi_grid = reshape_flat_mesh_to_grid(psi_flat, resolved_mesh_shape, k_axis=0, order="F")
    k_grid = _polshyn_projected_hf_k_grid_frac(basis, resolved_mesh_shape, k_grid_frac)
    if sewing_transforms is None:
        active_sewing_transforms = polshyn_projected_hf_boundary_sewing_transforms(basis) if bool(boundary_sewing) else None
    else:
        active_sewing_transforms = tuple(sewing_transforms)
    has_effective_sewing = (
        active_sewing_transforms is not None
        and len(active_sewing_transforms) == 2
        and all(transform is not None for transform in active_sewing_transforms)
    )
    if not has_effective_sewing and not bool(diagnostic_no_sewing):
        raise ValueError(
            "Polshyn projected-HF topology requires doubled-cell B1/B2 boundary sewing. "
            "No-sewing calls are diagnostic-only; pass diagnostic_no_sewing=True to record a non-physical diagnostic."
        )

    selected = tuple(int(index) for index in basis_metadata.get("selected_hf_state_indices", range(psi_grid.shape[-1])))
    if len(selected) != int(psi_grid.shape[-1]):
        raise ValueError(
            "Polshyn projected-HF reconstruction metadata selected_hf_state_indices length "
            f"{len(selected)} does not match reconstructed state count {psi_grid.shape[-1]}"
        )
    column_indices = tuple(range(int(psi_grid.shape[-1])))
    labels = tuple(f"hf_state={index}" for index in selected)

    flat_ineligible_reason = basis_metadata.pop("topology_ineligible_reason", None)
    flat_topology_policy = basis_metadata.pop("topology_policy", None)
    flat_sewing_blocker = basis_metadata.pop("sewing_blocker", None)
    flat_grid_shape_attached = basis_metadata.get("grid_shape_attached")
    flat_sewing_available = basis_metadata.get("sewing_available")
    payload: dict[str, object] = dict(basis_metadata)
    payload.update(
        {
            "topology_adapter": "mean_field.systems.tmbg.topology.compute_polshyn_projected_hf_topology",
            "topology_status": "topology-adapter-with-reviewed-polshyn-doubled-cell-sewing" if has_effective_sewing else "diagnostic-no-sewing-not-physical",
            "topology_eligible": bool(has_effective_sewing),
            "flat_diagnostic_bundle_topology_eligible": False,
            "flat_diagnostic_topology_ineligible_reason": flat_ineligible_reason,
            "flat_diagnostic_topology_policy": flat_topology_policy,
            "flat_diagnostic_sewing_blocker": flat_sewing_blocker,
            "flat_diagnostic_grid_shape_attached": flat_grid_shape_attached,
            "flat_diagnostic_sewing_available": flat_sewing_available,
            "topology_policy": "allowed only through this reviewed Polshyn doubled-cell topology adapter with B1/B2 sewing; no-sewing is diagnostic-only",
            "grid_shape_attached": True,
            "sewing_available": bool(has_effective_sewing),
            "sewing_transforms_attached": bool(has_effective_sewing),
            "diagnostic_no_sewing": bool(diagnostic_no_sewing),
            "physical_validation_status": "software_api_only_pending_slurm_paper_validation" if has_effective_sewing else "not_physical_no_sewing_diagnostic",
            "topology_input_axis_order": "mesh_B1,mesh_B2,microscopic_basis,hf_state",
            "topology_flat_input_axis_order": "nk,microscopic_basis,hf_state",
            "topology_flat_shape_semantics": "flat psi_micro shape is (nk,basis,state) before order='F' reshape",
            "topology_flat_grid_order": "iy/f2 outer, ix/f1 inner; reshaped with order='F' to (mesh_B1,mesh_B2,basis,state)",
            "topology_grid_axes": ["B1", "B2"],
            "topology_grid_shape": [int(resolved_mesh_shape[0]), int(resolved_mesh_shape[1])],
            "topology_grid_shape_source": mesh_shape_source,
            "topology_k_grid_frac_shape": None if k_grid is None else [int(value) for value in k_grid.shape],
            "topology_sewing_axes": ["B1", "B2"],
            "topology_sewing_transforms_count": 0 if active_sewing_transforms is None else int(sum(transform is not None for transform in active_sewing_transforms)),
            "topology_sewing_row_order": "spin_major,valley_inner,basis_F(local=6,embed_x,embed_y)",
            "band_indices_semantics": "Polshyn final-HF state indices; common FHS receives local reconstructed column indices",
            "column_indices": [int(index) for index in column_indices],
            "absolute_band_indices": [int(index) for index in selected],
            "valley_argument_semantics": "diagnostic label; reconstructed projected-HF rows contain spin and both valleys as direct-sum blocks",
            "evidence_paths": [
                "src/mean_field/systems/tmbg/topology.py",
                "src/mean_field/systems/tmbg/_polshyn_reconstruction.py",
                "src/analysis/topology/system.py",
                "src/analysis/topology/core.py",
            ],
            "uncertainty": "Software/API sewing and reshape are lightweight-tested; physical Chern/paper validation remains a separate Slurm target.",
        }
    )
    if metadata is not None:
        payload["caller_metadata"] = dict(metadata)

    return compute_system_topology_from_eigenvectors(
        psi_grid,
        column_indices,
        system="tmbg_polshyn_doubled",
        valley=int(valley),
        k_grid_frac=k_grid,
        sewing_transforms=active_sewing_transforms,
        index_metadata=payload,
        result_band_indices=selected,
        role="hf_state",
        labels=labels,
        link_method=link_method,
        orientation_sign=float(orientation_sign),
        atol=float(atol),
        regularization=float(regularization),
    )


def compute_topology_from_eigenvectors(
    eigenvectors,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    k_grid_frac=None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    return compute_system_topology_from_eigenvectors(
        eigenvectors,
        band_indices,
        system="tmbg",
        valley=valley,
        k_grid_frac=k_grid_frac,
        sewing_transforms=sewing_transforms,
        index_metadata={"boundary_sewing": sewing_transforms is not None},
        orientation_sign=orientation_sign,
    )


def compute_topology_from_grid_result(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    return compute_system_topology_from_grid_result(
        grid_result,
        band_indices,
        system="tmbg",
        valley=valley,
        sewing_transforms=sewing_transforms,
        index_metadata={"boundary_sewing": sewing_transforms is not None},
        orientation_sign=orientation_sign,
    )


def compute_topology_on_grid(
    mesh_size: int,
    lattice,
    params,
    band_indices: int | Iterable[int],
    *,
    valley: int = 1,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    n_bands: int | None = None,
    sewing_transforms: Sequence[SewingTransform | None] | None = None,
    boundary_sewing: bool = True,
    orientation_sign: float = 1.0,
) -> TopologyResult:
    requested = normalize_state_indices(band_indices)
    if n_bands is not None and int(n_bands) <= max(requested):
        raise ValueError(f"n_bands={int(n_bands)} does not include requested band index {max(requested)}")
    if endpoint:
        raise ValueError("Topology FHS meshes must use endpoint=False")
    grid = compute_bands_on_grid(
        int(mesh_size),
        lattice,
        params,
        valley=int(valley),
        n_bands=None if n_bands is None else int(n_bands),
        return_eigenvectors=True,
        endpoint=False,
        frac_shift=(float(frac_shift[0]), float(frac_shift[1])),
    )
    if sewing_transforms is None and bool(boundary_sewing):
        sewing_transforms = boundary_sewing_transforms(lattice)
    return compute_topology_from_grid_result(
        grid,
        requested,
        valley=int(valley),
        sewing_transforms=sewing_transforms,
        orientation_sign=float(orientation_sign),
    )


__all__ = [
    "SewingTransform",
    "TopologyResult",
    "boundary_sewing_transforms",
    "compute_polshyn_projected_hf_topology",
    "compute_topology_from_eigenvectors",
    "compute_topology_from_grid_result",
    "compute_topology_on_grid",
    "polshyn_projected_hf_boundary_sewing_transforms",
]
