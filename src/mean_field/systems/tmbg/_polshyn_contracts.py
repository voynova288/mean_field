from __future__ import annotations

from dataclasses import field

from ._polshyn_shared import *  # noqa: F401,F403
from ._polshyn_types import *  # noqa: F401,F403
from ._polshyn_basis import build_polshyn_projected_basis
from ._polshyn_filling import cdw_density_blocks, polshyn_nu_7over2_filling_summary
from ._polshyn_h0 import PolshynH0SubtractionConfig, apply_polshyn_h0_subtraction
from ._polshyn_wang import build_wang_overlap_blocks, run_projected_hf_scf_wang, translation_order_parameters, wang_sector_density_blocks
from .model import TMBGModel


@dataclass(frozen=True)
class PolshynRunHFConfig:
    """Explicit TMBG Polshyn public ``run_hf`` adapter config.

    This adapter deliberately requires system-specific inputs: the projected
    primitive band indices, target primitive band, supercell mesh, interaction
    shifts, and Coulomb parameters.  Generic ``HFConfig -> Polshyn`` inference
    is not implemented because the paper target depends on topology/window
    choices that must be explicit.
    """

    mesh_size: int
    projected_indices: tuple[int, ...]
    target_band_index: int
    shifts: tuple[tuple[int, int], ...] = ()
    v0: float = 0.0
    epsilon_r: float = 10.0
    d_sc_nm: float = 10.0
    max_iter: int = 80
    precision: float = 1.0e-6
    seed: int = 0
    oda_stall_threshold: float = 1.0e-4
    frac_shift: tuple[float, float] = (0.0, 0.0)
    init_mode: str = "bm_wang"
    hartree_scale: float = 1.0
    fock_scale: float = 1.0
    zero_hartree_q0: bool = False
    h0_subtraction: PolshynH0SubtractionConfig = field(default_factory=PolshynH0SubtractionConfig)
    basis: PolshynProjectedBasis | None = None

    def __post_init__(self) -> None:
        if int(self.mesh_size) <= 0:
            raise ValueError(f"mesh_size must be positive, got {self.mesh_size}")
        indices = tuple(int(value) for value in self.projected_indices)
        if not indices:
            raise ValueError("projected_indices must not be empty")
        if int(self.target_band_index) not in indices:
            raise ValueError(
                f"target_band_index={self.target_band_index} is not present in projected_indices={indices}"
            )
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if float(self.precision) <= 0.0:
            raise ValueError("precision must be positive")
        if float(self.oda_stall_threshold) <= 0.0:
            raise ValueError("oda_stall_threshold must be positive")
        if len(self.frac_shift) != 2:
            raise ValueError(f"frac_shift must be a length-2 tuple, got {self.frac_shift}")
        normalized_shifts = tuple((int(m), int(n)) for m, n in self.shifts)
        object.__setattr__(self, "projected_indices", indices)
        object.__setattr__(self, "shifts", normalized_shifts)
        object.__setattr__(self, "frac_shift", (float(self.frac_shift[0]), float(self.frac_shift[1])))
        init_mode = str(self.init_mode)
        if init_mode not in {"bm_wang", "cdw"}:
            raise ValueError("PolshynRunHFConfig.init_mode must be 'bm_wang' or 'cdw'")
        object.__setattr__(self, "init_mode", init_mode)
        h0_subtraction = self.h0_subtraction
        if isinstance(h0_subtraction, str):
            h0_subtraction = PolshynH0SubtractionConfig(mode=h0_subtraction)
        if not isinstance(h0_subtraction, PolshynH0SubtractionConfig):
            raise TypeError(
                "PolshynRunHFConfig.h0_subtraction must be PolshynH0SubtractionConfig or str, "
                f"got {type(h0_subtraction).__name__}"
            )
        object.__setattr__(self, "h0_subtraction", h0_subtraction)

def _polshyn_gvecs_for_shifts(basis: PolshynProjectedBasis, shifts: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.asarray(
        [int(m) * complex(basis.super_b1) + int(n) * complex(basis.super_b2) for m, n in shifts],
        dtype=np.complex128,
    )


def _validate_polshyn_public_hf_config(config: "HFConfig", polshyn_config: PolshynRunHFConfig) -> None:
    mesh = (int(polshyn_config.mesh_size), int(polshyn_config.mesh_size))
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(f"TMBG Polshyn public run_hf requires HFConfig.mesh={mesh}, got {config.mesh}")
    filling = polshyn_nu_7over2_filling_summary(
        polshyn_config.projected_indices,
        target_band_index=int(polshyn_config.target_band_index),
    ).primitive_nu
    if not np.isclose(float(config.filling), float(filling)):
        raise ValueError(f"TMBG Polshyn public run_hf requires HFConfig.filling={filling}, got {config.filling}")
    if int(config.max_iter) != int(polshyn_config.max_iter):
        raise ValueError(
            f"TMBG Polshyn public run_hf requires HFConfig.max_iter={polshyn_config.max_iter}, "
            f"got {config.max_iter}"
        )
    if not np.isclose(float(config.precision), float(polshyn_config.precision)):
        raise ValueError(
            f"TMBG Polshyn public run_hf requires HFConfig.precision={polshyn_config.precision}, "
            f"got {config.precision}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError(
            "TMBG Polshyn Wang HF stores density as a Wang/Xiaoyu stored delta; "
            "set HFConfig.density_convention='stored_delta'"
        )
    if config.interaction_scheme != "average":
        raise ValueError("TMBG Polshyn public run_hf currently supports HFConfig.interaction_scheme='average' only")
    if config.coulomb_kernel != "2d_gate":
        raise ValueError("TMBG Polshyn public run_hf currently supports HFConfig.coulomb_kernel='2d_gate' only")
    if not np.isclose(float(config.epsilon_r), float(polshyn_config.epsilon_r)):
        raise ValueError(
            f"TMBG Polshyn public run_hf requires HFConfig.epsilon_r={polshyn_config.epsilon_r}, "
            f"got {config.epsilon_r}"
        )
    if not np.isclose(float(config.dsc_nm), float(polshyn_config.d_sc_nm)):
        raise ValueError(
            f"TMBG Polshyn public run_hf requires HFConfig.dsc_nm={polshyn_config.d_sc_nm}, got {config.dsc_nm}"
        )
    if config.active_window is not None:
        raise NotImplementedError("TMBG Polshyn public run_hf uses explicit projected_indices; leave active_window unset")
    if config.active_band_indices is not None and tuple(int(v) for v in config.active_band_indices) != polshyn_config.projected_indices:
        raise ValueError(
            "HFConfig.active_band_indices must be unset or match "
            f"polshyn_config.projected_indices={polshyn_config.projected_indices}"
        )


def run_tmbg_polshyn_hf_config_adapter(model: object, config: "HFConfig", **kwargs: Any) -> "HFResult | None":
    """Run TMBG Polshyn-Wang HF from an explicit system config.

    The raw ``(basis, state, info)`` bundle remains the source of truth and is
    wrapped by the existing canonical Polshyn-Wang post-run adapter.  This does
    not infer paper/topology windows from generic ``HFConfig``.
    """

    if not isinstance(model, TMBGModel):
        return None
    if "tmbg_polshyn_config" not in kwargs:
        raise NotImplementedError(
            "Unified run_hf has a TMBG Polshyn adapter only for explicit "
            "tmbg_polshyn_config=PolshynRunHFConfig(...); generic HFConfig -> Polshyn mapping is not implemented"
        )
    polshyn_config = kwargs.pop("tmbg_polshyn_config")
    if not isinstance(polshyn_config, PolshynRunHFConfig):
        raise TypeError(
            f"tmbg_polshyn_config must be PolshynRunHFConfig, got {type(polshyn_config).__name__}"
        )
    if kwargs:
        raise TypeError(f"Unsupported TMBG Polshyn run_hf kwargs: {sorted(kwargs)}")
    _validate_polshyn_public_hf_config(config, polshyn_config)

    basis = polshyn_config.basis
    if basis is None:
        basis = build_polshyn_projected_basis(
            model,
            mesh_size=int(polshyn_config.mesh_size),
            projected_indices=polshyn_config.projected_indices,
            target_band_index=int(polshyn_config.target_band_index),
            frac_shift=polshyn_config.frac_shift,
        )
    elif basis.model is not model and getattr(basis.model, "lattice", None) != getattr(model, "lattice", None):
        raise ValueError("PolshynRunHFConfig.basis does not match the supplied TMBGModel")
    shifts = tuple(polshyn_config.shifts)
    gvecs = _polshyn_gvecs_for_shifts(basis, shifts)
    h0_subtraction_result = None
    overlap_blocks_for_run = None
    if polshyn_config.h0_subtraction.enabled:
        overlap_blocks_for_run = build_wang_overlap_blocks(
            basis,
            basis,
            shifts,
            gvecs,
            epsilon_r=float(polshyn_config.epsilon_r),
            d_sc_nm=float(polshyn_config.d_sc_nm),
            include_hartree=True,
            include_fock=True,
        )
        h0_subtraction_result = apply_polshyn_h0_subtraction(
            basis,
            overlap_blocks_for_run,
            config=polshyn_config.h0_subtraction,
            v0=float(polshyn_config.v0),
        )
        basis = h0_subtraction_result.corrected_basis
    filling = polshyn_nu_7over2_filling_summary(
        polshyn_config.projected_indices,
        target_band_index=int(polshyn_config.target_band_index),
    )
    initial_density = None
    if polshyn_config.init_mode == "cdw":
        initial_density = cdw_density_blocks(
            projected_indices=polshyn_config.projected_indices,
            target_band_index=int(polshyn_config.target_band_index),
            n_spin=basis.n_spin,
            n_eta=basis.n_eta,
            nb=basis.nb,
            nk=basis.nk,
            reference_diagonal=basis.reference_diagonal,
        )
    state, overlap_blocks, info = run_projected_hf_scf_wang(
        basis,
        occupation_counts=filling.occupation_counts,
        shifts=shifts,
        gvecs=gvecs,
        v0=float(polshyn_config.v0),
        epsilon_r=float(polshyn_config.epsilon_r),
        d_sc_nm=float(polshyn_config.d_sc_nm),
        max_iter=int(polshyn_config.max_iter),
        precision=float(polshyn_config.precision),
        initial_density_blocks=initial_density,
        oda_stall_threshold=float(polshyn_config.oda_stall_threshold),
        seed=int(polshyn_config.seed),
        hartree_scale=float(polshyn_config.hartree_scale),
        fock_scale=float(polshyn_config.fock_scale),
        zero_hartree_q0=bool(polshyn_config.zero_hartree_q0),
        overlap_blocks=overlap_blocks_for_run,
    )
    info = dict(info)
    h0_subtraction_diagnostics = (
        polshyn_config.h0_subtraction.to_dict()
        if h0_subtraction_result is None
        else dict(h0_subtraction_result.diagnostics)
    )
    info.update(
        {
            "h0_subtraction_mode": str(h0_subtraction_diagnostics.get("mode", polshyn_config.h0_subtraction.mode)),
            "h0_subtraction_applied_sign": float(h0_subtraction_diagnostics.get("applied_sign", polshyn_config.h0_subtraction.applied_sign)),
            "h0_subtraction_zero_hartree_q0": bool(h0_subtraction_diagnostics.get("zero_hartree_q0", polshyn_config.h0_subtraction.zero_hartree_q0)),
            "h0_subtraction_p0_reference": str(h0_subtraction_diagnostics.get("p0_reference", polshyn_config.h0_subtraction.p0_reference)),
        }
    )
    canonical = polshyn_wang_hf_bundle_to_hf_run_result(
        basis,
        state,
        info,
        seed=int(polshyn_config.seed),
        archive_manifest={
            "source": "fresh_run_hf",
            "adapter": "mean_field.systems.tmbg.polshyn_supercell.run_tmbg_polshyn_hf_config_adapter",
            "projected_indices": [int(value) for value in polshyn_config.projected_indices],
            "target_band_index": int(polshyn_config.target_band_index),
            "shifts": [[int(m), int(n)] for m, n in shifts],
            "h0_subtraction": h0_subtraction_diagnostics,
        },
    )
    density_blocks = wang_sector_density_blocks(state, basis)
    target_order = translation_order_parameters(
        density_blocks,
        projected_indices=polshyn_config.projected_indices,
        target_band_index=int(polshyn_config.target_band_index),
    )
    from pathlib import Path
    from mean_field.api.artifacts import ArtifactManifest, ConventionBundle
    from mean_field.api.hf import HFResult
    from mean_field.api.models import model_record

    record = model_record(model, system_name="tmbg_polshyn")
    observables: dict[str, object] = {
        "public_run_hf_adapter": "mean_field.systems.tmbg.polshyn_supercell.run_tmbg_polshyn_hf_config_adapter",
        "explicit_config_type": "PolshynRunHFConfig",
        "primitive_nu": float(filling.primitive_nu),
        "occupation_counts": filling.occupation_counts.astype(int).tolist(),
        "target_translation_order_x2_mean": float(target_order["target_x2_mean"]),
        "all_translation_order_x2_mean": float(target_order["all_x2_mean"]),
        "h0_subtraction_mode": str(h0_subtraction_diagnostics.get("mode", polshyn_config.h0_subtraction.mode)),
        "h0_subtraction_applied_sign": float(h0_subtraction_diagnostics.get("applied_sign", polshyn_config.h0_subtraction.applied_sign)),
        "h0_subtraction_zero_hartree_q0": bool(h0_subtraction_diagnostics.get("zero_hartree_q0", polshyn_config.h0_subtraction.zero_hartree_q0)),
        "h0_subtraction_p0_reference": str(h0_subtraction_diagnostics.get("p0_reference", polshyn_config.h0_subtraction.p0_reference)),
        **_info_scalar_summary(info),
    }
    return HFResult(
        model=record,
        config=config,
        state={"basis": basis, "state": state, "info": dict(info), "overlap_blocks": overlap_blocks},
        observables=observables,
        artifacts=ArtifactManifest(
            root=Path("."),
            model=record,
            conventions=ConventionBundle(
                energy_unit="eV",
                density_convention="stored_delta",
                density_axis_order="abk",
                hamiltonian_axis_order="abk",
                wavefunction_axis_order="basis,band,flavor,k",
                gauge="tmbg_polshyn_doubled_cell_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": "tmbg.polshyn_wang.explicit_config",
                "system_name": "tmbg_polshyn",
                "adapter": "mean_field.systems.tmbg.polshyn_supercell.run_tmbg_polshyn_hf_config_adapter",
                "canonical_adapter": "mean_field.systems.tmbg.polshyn_supercell.polshyn_wang_hf_bundle_to_hf_run_result",
                "raw_state_type": "PolshynWangHFState",
                "h0_subtraction": h0_subtraction_diagnostics,
            },
        ),
        canonical_run_result=canonical,
    )


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
