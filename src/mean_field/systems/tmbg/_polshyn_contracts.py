from __future__ import annotations

from ._polshyn_shared import *  # noqa: F401,F403
from ._polshyn_types import *  # noqa: F401,F403

def _unavailable_polshyn_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "Polshyn-Wang canonical contract records an already-built projected basis; "
        "use mean_field.systems.tmbg.polshyn_supercell builders for fresh Hamiltonians."
    )

def _unavailable_polshyn_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "Polshyn-Wang canonical contract records post-run arrays; "
        "fresh diagonalization is not performed by the adapter."
    )

def _tmbg_params_summary(params: TMBGParameters) -> dict[str, object]:
    keys = (
        "graphene_lattice_constant_nm",
        "t0",
        "t1",
        "t3",
        "t4",
        "delta",
        "omega",
        "omega_prime",
        "interlayer_potential",
        "staggered_potential",
        "blg_stacking",
        "bernal_convention",
        "model_name",
        "vf",
        "v3",
        "v4",
    )
    out: dict[str, object] = {}
    for key in keys:
        value = getattr(params, key)
        if isinstance(value, str):
            out[key] = value
        else:
            out[key] = float(value)
    return out

def _polshyn_single_particle_model(basis: PolshynProjectedBasis) -> ContractSingleParticleModel:
    return ContractSingleParticleModel(
        system="tmbg_polshyn_doubled",
        lattice=basis.model.lattice_summary(),
        params=_tmbg_params_summary(basis.model.params),
        hamiltonian_builder=_unavailable_polshyn_hamiltonian_builder,
        diagonalizer=_unavailable_polshyn_diagonalizer,
        metadata={
            "source": "mean_field.systems.tmbg.polshyn_supercell",
            "theta_deg": float(basis.model.theta_deg),
            "n_shells": int(basis.model.n_shells),
            "supercell": basis.supercell.as_dict(),
            "projected_indices": [int(value) for value in basis.projected_indices],
            "target_band_index": int(basis.target_band_index),
        },
    )

def _basis_energies_from_flat_h0(h0: np.ndarray) -> np.ndarray:
    h0_array = np.asarray(h0, dtype=np.complex128)
    out = np.zeros((h0_array.shape[0], h0_array.shape[2]), dtype=float)
    for ik in range(h0_array.shape[2]):
        out[:, ik] = np.linalg.eigvalsh(h0_array[:, :, ik])
    return out

def _polshyn_flat_state_index(basis: PolshynProjectedBasis) -> np.ndarray:
    return np.arange(int(basis.n_spin) * int(basis.n_eta) * int(basis.nb), dtype=int).reshape(
        (int(basis.n_spin), int(basis.n_eta), int(basis.nb)),
        order="F",
    )

def _polshyn_folded_band_labels(basis: PolshynProjectedBasis) -> tuple[dict[str, object], ...]:
    labels: list[dict[str, object]] = []
    for primitive_position, primitive_band_index in enumerate(basis.projected_indices):
        for fold_index in range(2):
            labels.append(
                {
                    "folded_band_index": int(2 * primitive_position + fold_index),
                    "primitive_position": int(primitive_position),
                    "primitive_band_index": int(primitive_band_index),
                    "fold_index": int(fold_index),
                    "is_target_band": bool(int(primitive_band_index) == int(basis.target_band_index)),
                }
            )
    if len(labels) != int(basis.nb):
        raise ValueError(f"Polshyn folded band labels length {len(labels)} does not match nb={basis.nb}")
    return tuple(labels)

def _polshyn_active_band_indices(basis: PolshynProjectedBasis) -> tuple[int, ...]:
    labels = np.zeros((int(basis.n_spin) * int(basis.n_eta) * int(basis.nb),), dtype=int)
    state_index = _polshyn_flat_state_index(basis)
    for ispin in range(int(basis.n_spin)):
        for ieta in range(int(basis.n_eta)):
            for iband in range(int(basis.nb)):
                primitive = int(basis.projected_indices[iband // 2])
                labels[int(state_index[ispin, ieta, iband])] = primitive
    return tuple(int(value) for value in labels)

def _polshyn_flavor_labels(basis: PolshynProjectedBasis) -> tuple[str, ...]:
    valley_labels = ("K", "Kprime")
    labels = [""] * (int(basis.n_spin) * int(basis.n_eta) * int(basis.nb))
    state_index = _polshyn_flat_state_index(basis)
    for ispin in range(int(basis.n_spin)):
        for ieta in range(int(basis.n_eta)):
            valley_label = valley_labels[ieta] if ieta < len(valley_labels) else f"eta{ieta}"
            for iband in range(int(basis.nb)):
                labels[int(state_index[ispin, ieta, iband])] = f"spin{ispin}_{valley_label}_folded_band{iband}"
    return tuple(labels)

def _polshyn_reference_density_flat(basis: PolshynProjectedBasis) -> np.ndarray:
    reference_diagonal = np.asarray(basis.reference_diagonal, dtype=float).reshape(-1)
    if reference_diagonal.shape != (int(basis.nb),):
        raise ValueError(
            f"Polshyn reference_diagonal shape {reference_diagonal.shape} does not match nb={basis.nb}"
        )
    reference_matrix = np.diag(reference_diagonal).astype(np.complex128)
    blocks = np.zeros((basis.n_spin, basis.n_eta, basis.nb, basis.nb, basis.nk), dtype=np.complex128)
    for ispin in range(int(basis.n_spin)):
        for ieta in range(int(basis.n_eta)):
            blocks[ispin, ieta, :, :, :] = reference_matrix[:, :, None]
    return _core_flatten_sector_blocks(blocks)

def _validate_polshyn_wang_bundle_shapes(basis: PolshynProjectedBasis, state: PolshynWangHFState) -> None:
    nt = int(basis.n_spin) * int(basis.n_eta) * int(basis.nb)
    nk = int(basis.nk)
    expected_matrix_shape = (nt, nt, nk)
    for name in ("h0", "density", "hamiltonian"):
        arr = np.asarray(getattr(state, name))
        if arr.shape != expected_matrix_shape:
            raise ValueError(f"Polshyn-Wang state.{name} shape {arr.shape} does not match {expected_matrix_shape}")
    energies = np.asarray(state.energies)
    if energies.shape != (nt, nk):
        raise ValueError(f"Polshyn-Wang state.energies shape {energies.shape} does not match {(nt, nk)}")
    expected_h0 = _core_flatten_sector_blocks(np.asarray(basis.h0_blocks, dtype=np.complex128))
    if not np.allclose(np.asarray(state.h0, dtype=np.complex128), expected_h0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("Polshyn-Wang state.h0 does not match flatten_sector_blocks(basis.h0_blocks)")

def _polshyn_projected_basis_contract(
    basis: PolshynProjectedBasis,
    state: PolshynWangHFState,
) -> ContractProjectedBasis:
    if basis.k_grid_frac is None:
        raise ValueError(
            "Polshyn-Wang canonical ProjectedBasis requires basis.k_grid_frac; "
            "the adapter does not reconstruct or guess a k-grid."
        )
    k_grid_frac = np.asarray(basis.k_grid_frac, dtype=float)
    if k_grid_frac.size != int(basis.nk) * 2:
        raise ValueError(f"basis.k_grid_frac shape {k_grid_frac.shape} incompatible with nk={basis.nk}")
    model = _polshyn_single_particle_model(basis)
    lower_folded_count = 2 * sum(1 for index in basis.projected_indices if int(index) < int(basis.target_band_index))
    return ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=np.asarray(basis.kvec, dtype=np.complex128),
        k_grid_frac=k_grid_frac.reshape((int(basis.nk), 2)),
        h0=np.asarray(state.h0, dtype=np.complex128),
        basis_energies=_basis_energies_from_flat_h0(state.h0),
        active_band_indices=_polshyn_active_band_indices(basis),
        active_valence_bands=int(lower_folded_count),
        active_conduction_bands=int(basis.nb - lower_folded_count),
        micro_wavefunctions=np.asarray(basis.wavefunctions, dtype=np.complex128),
        flavor_labels=_polshyn_flavor_labels(basis),
        band_labels=_polshyn_folded_band_labels(basis),
        metadata={
            "projected_basis_source": "PolshynProjectedBasis",
            "wavefunctions_axis_order": "basis,folded_band,valley,k",
            "density_axis_order": "abk",
            "density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "active_band_semantics": "primitive_projected_indices_repeated_over_folds_spin_valley",
            "projected_indices": [int(value) for value in basis.projected_indices],
            "target_band_index": int(basis.target_band_index),
            "supercell": basis.supercell.as_dict(),
            "supercell_reciprocal_vectors_nm_inv": [
                [float(basis.super_b1.real), float(basis.super_b1.imag)],
                [float(basis.super_b2.real), float(basis.super_b2.imag)],
            ],
            "embedding_shape": [int(value) for value in basis.embedding_shape],
            "embedding_origin": [int(value) for value in basis.embedding_origin],
            "supports_crpa": False,
        },
    )

def _round_integer(value: float, *, name: str, atol: float = 1.0e-7) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - float(rounded)) > float(atol):
        raise ValueError(f"{name}={value:.12g} is not integer within atol={atol}")
    return rounded

def _polshyn_wang_density_state(basis: PolshynProjectedBasis, state: PolshynWangHFState) -> ContractDensityState:
    density_delta = np.asarray(state.density, dtype=np.complex128)
    reference = _polshyn_reference_density_flat(basis)
    projector = density_delta + reference
    trace_projector_total = float(np.trace(projector, axis1=0, axis2=1).real.sum())
    n_occupied_total = _round_integer(trace_projector_total, name="Polshyn-Wang projector trace total")
    trace_delta_per_k = np.trace(density_delta, axis1=0, axis2=1).real
    primitive_nu_per_k = np.asarray(trace_delta_per_k, dtype=float) / float(basis.supercell.area_ratio)
    primitive_nu = float(np.mean(primitive_nu_per_k))
    max_nu_deviation = float(np.max(np.abs(primitive_nu_per_k - primitive_nu))) if primitive_nu_per_k.size else 0.0
    return density_state_from_delta(
        density_delta,
        reference,
        reference_scheme="custom",
        filling=primitive_nu,
        n_occupied_total=n_occupied_total,
        reference_metadata={
            "system": "tmbg_polshyn_doubled",
            "raw_density_convention": "stored_delta",
            "density_axis_order": "abk",
            "reference_scheme_source": "PolshynProjectedBasis.reference_diagonal",
            "reference_diagonal": [float(value) for value in np.asarray(basis.reference_diagonal, dtype=float).reshape(-1)],
            "area_ratio": int(basis.supercell.area_ratio),
            "convention": "Polshyn conduction-band filling: lower remote filled, target empty",
        },
        metadata={
            "raw_density_convention": "stored_delta",
            "density_delta_definition": "P_store - R",
            "density_axis_order": "abk",
            "raw_density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "canonical_density_orientation": "stored_abk",
            "adapter": "mean_field.systems.tmbg.polshyn_supercell.polshyn_wang_hf_bundle_to_hf_run_result",
            "primitive_nu_from_density": primitive_nu,
            "primitive_nu_per_k_max_deviation": max_nu_deviation,
        },
    )

def _zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))

def _polshyn_hamiltonian_parts(state: PolshynWangHFState) -> ContractHamiltonianParts:
    h0 = np.asarray(state.h0, dtype=np.complex128)
    total = np.asarray(state.hamiltonian, dtype=np.complex128)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=total - h0,
        hartree=_zero_field_like(h0),
        fock=_zero_field_like(h0),
        total=total,
        density_input_convention="polshyn_wang_stored_delta_collapsed",
        metadata={
            "component_resolution": "collapsed_total_minus_h0",
            "raw_density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "supports_crpa": False,
        },
    )

def _finite_float_or_none(value: object) -> float | None:
    if isinstance(value, bool | np.bool_):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None

def _float_diagnostics(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        finite = _finite_float_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out

def _info_scalar_summary(info: Mapping[str, Any]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in info.items():
        if str(key) == "iteration_history":
            continue
        if value is None or isinstance(value, bool | np.bool_ | str):
            out[str(key)] = None if value is None else (bool(value) if isinstance(value, bool | np.bool_) else str(value))
            continue
        if isinstance(value, int | np.integer):
            out[str(key)] = int(value)
            continue
        finite = _finite_float_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out

def _coerce_iteration_history_value(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    finite = _finite_float_or_none(value)
    if finite is not None:
        return finite
    raise TypeError(f"Unsupported iteration_history value type {type(value).__name__}")

def _iteration_history_from_info(info: Mapping[str, Any]) -> list[dict[str, object]]:
    if "iteration_history" not in info:
        return []
    raw = info["iteration_history"]
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError("info['iteration_history'] must be an explicit sequence of mapping rows")
    history: list[dict[str, object]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"info['iteration_history'][{row_index}] is not a mapping")
        history.append({str(key): _coerce_iteration_history_value(value) for key, value in row.items()})
    return history

def _require_info_key(info: Mapping[str, Any], key: str) -> Any:
    if key not in info:
        raise ValueError(f"Polshyn-Wang canonical adapter requires info[{key!r}]; refusing to fabricate it")
    return info[key]

def _require_info_bool(info: Mapping[str, Any], key: str) -> bool:
    value = _require_info_key(info, key)
    if not isinstance(value, bool | np.bool_):
        raise ValueError(f"Polshyn-Wang info[{key!r}] must be bool, got {type(value).__name__}")
    return bool(value)

def _resolve_polshyn_seed(info: Mapping[str, Any], explicit_seed: int | None) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    for key in ("best_seed", "seed"):
        if key in info:
            return int(info[key])
    raise ValueError(
        "Polshyn-Wang canonical adapter requires an explicit seed or info['seed']; "
        "refusing to invent best_seed"
    )

def polshyn_wang_hf_bundle_to_hf_run_result(
    basis: PolshynProjectedBasis,
    state: PolshynWangHFState,
    info: Mapping[str, Any],
    *,
    seed: int | None = None,
    archive_manifest: Mapping[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap an explicit ``(basis, state, info)`` Polshyn-Wang HF bundle.

    This is a boundary-only canonical I/O adapter.  It preserves the Wang/Xiaoyu
    stored-density orientation used by :class:`PolshynWangHFState`, records that
    orientation in metadata, and never reconstructs missing iteration history or
    full-state archives.  If ``info`` does not contain an explicit
    ``iteration_history`` sequence, the returned history is deliberately empty.
    """

    _validate_polshyn_wang_bundle_shapes(basis, state)
    info_map = dict(info)
    density = _polshyn_wang_density_state(basis, state)
    iteration_history = _iteration_history_from_info(info_map)
    history_source = "info.iteration_history" if "iteration_history" in info_map else "unavailable_in_polshyn_wang_info"
    diagnostics = _float_diagnostics(state.diagnostics)
    diagnostics.update(_float_diagnostics(info_map))
    final_state = ContractHFState(
        basis=_polshyn_projected_basis_contract(basis, state),
        density=density,
        hamiltonian=_polshyn_hamiltonian_parts(state),
        energies=np.asarray(state.energies, dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(state.mu),
        observables={
            "eigenvectors_active_available": False,
            "primitive_nu": float(density.filling),
            "filling_from_density": float(density.filling),
            "iteration_history_available": bool(iteration_history),
            "iteration_history_source": history_source,
            "raw_density_projector_orientation": "wang_xiaoyu_stored_P_star",
            "info_summary": _info_scalar_summary(info_map),
        },
        diagnostics=diagnostics,
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=iteration_history,
        converged=_require_info_bool(info_map, "converged"),
        exit_reason=str(_require_info_key(info_map, "exit_reason")),
        best_seed=_resolve_polshyn_seed(info_map, seed),
        init_mode=str(_require_info_key(info_map, "init_mode")),
        archive_manifest={} if archive_manifest is None else dict(archive_manifest),
    )

__all__ = [name for name in globals() if not name.startswith('__')]
