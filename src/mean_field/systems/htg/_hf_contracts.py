from __future__ import annotations

from ._hf_types import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403
from ._hf_initialization import *  # noqa: F401,F403
from ._hf_basis import *  # noqa: F401,F403
from ._hf_interaction_path import *  # noqa: F401,F403
from ._hf_runner import *  # noqa: F401,F403

@dataclass(frozen=True)
class HTGRunHFConfig:
    """Explicit primitive-cell HTG public ``run_hf`` adapter config.

    This dataclass mirrors the already-existing :func:`run_htg_hf` runner.  The
    public :class:`mean_field.api.hf.HFConfig` must still match ``nu``,
    ``mesh_size``, iteration controls, density convention, and interaction
    scalars; no generic ``HFConfig -> HTG`` inference is performed here.
    """

    nu: float
    mesh_size: int
    interaction: InteractionParams = field(default_factory=InteractionParams)
    init_mode: str = "flavor"
    seed: int = 1
    beta: float = 1.0
    max_iter: int = 300
    precision: float = 1.0e-6
    oda_stall_threshold: float = 1.0e-3
    g_shells: int | None = None
    projected_band_count: int = 2
    initial_density: np.ndarray | None = None
    use_numba: bool | None = None

    def __post_init__(self) -> None:
        if int(self.mesh_size) <= 0:
            raise ValueError(f"mesh_size must be positive, got {self.mesh_size}")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if float(self.precision) <= 0.0:
            raise ValueError("precision must be positive")
        if float(self.oda_stall_threshold) <= 0.0:
            raise ValueError("oda_stall_threshold must be positive")
        if int(self.projected_band_count) <= 0:
            raise ValueError("projected_band_count must be positive")
        if self.g_shells is not None and int(self.g_shells) < 0:
            raise ValueError("g_shells must be non-negative when provided")


def _validate_htg_public_hf_config(config: "HFConfig", htg_config: HTGRunHFConfig) -> None:
    if not isinstance(htg_config.interaction, InteractionParams):
        raise TypeError(
            f"htg_config.interaction must be InteractionParams, got {type(htg_config.interaction).__name__}"
        )
    mesh = (int(htg_config.mesh_size), int(htg_config.mesh_size))
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(f"HTG public run_hf requires HFConfig.mesh={mesh}, got {config.mesh}")
    if not np.isclose(float(config.filling), float(htg_config.nu)):
        raise ValueError(f"HTG public run_hf requires HFConfig.filling={htg_config.nu}, got {config.filling}")
    if int(config.max_iter) != int(htg_config.max_iter):
        raise ValueError(
            f"HTG public run_hf requires HFConfig.max_iter={htg_config.max_iter}, got {config.max_iter}"
        )
    if not np.isclose(float(config.precision), float(htg_config.precision)):
        raise ValueError(
            f"HTG public run_hf requires HFConfig.precision={htg_config.precision}, got {config.precision}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError(
            "HTG primitive HF stores density as P-R; set HFConfig.density_convention='stored_delta'"
        )
    if config.active_window is not None or config.active_band_indices is not None:
        raise NotImplementedError(
            "HTG public run_hf takes the projected window from htg_config.projected_band_count; "
            "leave HFConfig.active_window/active_band_indices unset for now"
        )
    interaction = htg_config.interaction
    if config.interaction_scheme != interaction.subtraction:
        raise ValueError(
            f"HTG public run_hf requires HFConfig.interaction_scheme={interaction.subtraction!r}, "
            f"got {config.interaction_scheme!r}"
        )
    if config.coulomb_kernel != "2d_gate":
        raise ValueError("HTG public run_hf currently supports HFConfig.coulomb_kernel='2d_gate' only")
    if not np.isclose(float(config.epsilon_r), float(interaction.epsilon_r)):
        raise ValueError(
            f"HTG public run_hf requires HFConfig.epsilon_r={interaction.epsilon_r}, got {config.epsilon_r}"
        )
    if not np.isclose(float(config.dsc_nm), float(interaction.d_sc_nm)):
        raise ValueError(
            f"HTG public run_hf requires HFConfig.dsc_nm={interaction.d_sc_nm}, got {config.dsc_nm}"
        )


def run_htg_hf_config_adapter(model: object, config: "HFConfig", **kwargs: Any) -> "HFResult | None":
    """Run primitive-cell HTG HF from an explicit system config.

    The adapter is intentionally narrow: callers must provide
    ``htg_config=HTGRunHFConfig(...)`` and a matching public ``HFConfig``.  The
    raw :class:`HTGHartreeFockRun` remains the source of truth and is wrapped by
    the existing canonical HTG post-run adapter.
    """

    if not isinstance(model, HTGModel):
        return None
    if "htg_config" in kwargs and "htg_supercell_config" in kwargs:
        raise TypeError("Pass only one of htg_config or htg_supercell_config")
    if "htg_config" not in kwargs:
        if "htg_supercell_config" in kwargs:
            return None
        raise NotImplementedError(
            "Unified run_hf has an HTG primitive adapter only for explicit "
            "htg_config=HTGRunHFConfig(...); generic HFConfig -> HTG runner mapping is not implemented"
        )
    htg_config = kwargs.pop("htg_config")
    if not isinstance(htg_config, HTGRunHFConfig):
        raise TypeError(f"htg_config must be HTGRunHFConfig, got {type(htg_config).__name__}")
    if kwargs:
        raise TypeError(f"Unsupported HTG primitive run_hf kwargs: {sorted(kwargs)}")

    _validate_htg_public_hf_config(config, htg_config)
    raw = run_htg_hf(
        model,
        htg_config.interaction,
        nu=float(htg_config.nu),
        init_mode=str(htg_config.init_mode),
        seed=int(htg_config.seed),
        beta=float(htg_config.beta),
        max_iter=int(htg_config.max_iter),
        precision=float(htg_config.precision),
        oda_stall_threshold=float(htg_config.oda_stall_threshold),
        mesh_size=int(htg_config.mesh_size),
        g_shells=htg_config.g_shells,
        projected_band_count=int(htg_config.projected_band_count),
        initial_density=htg_config.initial_density,
        use_numba=htg_config.use_numba,
    )
    return htg_hf_run_to_hf_result(
        raw,
        config=config,
        observables={
            "public_run_hf_adapter": "mean_field.systems.htg.mean_field_adapter.run_htg_hf_config_adapter",
            "explicit_config_type": "HTGRunHFConfig",
        },
    )

# Canonical post-run contract adapters ---------------------------------------

def _contract_unavailable_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "HTG primitive contract records an already-built projected basis; "
        "use mean_field.systems.htg builders for fresh Hamiltonians."
    )


def _contract_unavailable_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "HTG primitive contract records post-run arrays; "
        "fresh diagonalization is not performed by the adapter."
    )


def _contract_finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _contract_single_particle_model(data: HTGProjectedBasisData) -> ContractSingleParticleModel:
    model = data.model
    params = model.params
    interaction = data.interaction
    metadata: dict[str, object] = {
        "theta_deg": float(model.theta_deg),
        "n_shells": int(model.n_shells),
        "model_name": str(params.model_name),
        "mesh_size": int(data.mesh_size),
        "projected_band_count": int(data.basis.n_band),
        "projected_band_indices": [int(index) for index in data.projected_band_indices],
        "central_band_indices": [int(index) for index in data.central_band_indices],
        "interaction_epsilon_r": float(interaction.epsilon_r),
        "interaction_d_sc_nm": float(interaction.d_sc_nm),
        "interaction_U_ev": float(interaction.U_ev),
        "interaction_subtraction": str(interaction.subtraction),
        "interaction_g_shells": int(interaction.g_shells),
        "finite_zero_limit": bool(interaction.finite_zero_limit),
        "source": "mean_field.systems.htg.mean_field_adapter",
    }
    return ContractSingleParticleModel(
        system="htg",
        lattice=model.lattice,
        params={
            "theta_deg": float(model.theta_deg),
            "n_shells": int(model.n_shells),
            "model_name": str(params.model_name),
            "kappa": float(params.kappa),
            "w_ev": float(params.w_ev),
            "vf_ev_nm": float(params.vf_ev_nm),
        },
        hamiltonian_builder=_contract_unavailable_hamiltonian_builder,
        diagonalizer=_contract_unavailable_diagonalizer,
        metadata=metadata,
    )


def _contract_basis_energies_from_h0(h0: np.ndarray) -> np.ndarray:
    h0_array = np.asarray(h0, dtype=np.complex128)
    out = np.zeros((h0_array.shape[0], h0_array.shape[2]), dtype=float)
    for ik in range(h0_array.shape[2]):
        out[:, ik] = np.linalg.eigvalsh(h0_array[:, :, ik])
    return out


def _contract_flatten_k_grid_frac(data: HTGProjectedBasisData) -> np.ndarray:
    k_grid_frac = np.asarray(data.k_grid_frac, dtype=float)
    if k_grid_frac.size != int(data.nk) * 2:
        raise ValueError(
            "HTG primitive canonical ProjectedBasis requires k_grid_frac with 2 coordinates per k point; "
            f"got shape {k_grid_frac.shape} for nk={data.nk}"
        )
    return k_grid_frac.reshape((int(data.nk), 2))


def _contract_state_index(data: HTGProjectedBasisData) -> np.ndarray:
    return np.arange(int(data.basis.nt), dtype=int).reshape(
        (int(data.basis.n_spin), int(data.basis.n_flavor), int(data.basis.n_band)),
        order="F",
    )


def _contract_active_band_indices(data: HTGProjectedBasisData) -> tuple[int, ...]:
    active = tuple(int(index) for index in data.projected_band_indices)
    if len(active) != int(data.basis.n_band):
        raise ValueError(
            "HTG primitive projected_band_indices must be per projected band; "
            f"got {len(active)} labels for n_band={data.basis.n_band}"
        )
    labels = np.zeros((int(data.basis.nt),), dtype=int)
    state_index = _contract_state_index(data)
    for ispin in range(int(data.basis.n_spin)):
        for ieta in range(int(data.basis.n_flavor)):
            for iband, band_index in enumerate(active):
                labels[int(state_index[ispin, ieta, iband])] = int(band_index)
    return tuple(int(value) for value in labels)


def _contract_flavor_labels(data: HTGProjectedBasisData) -> tuple[str, ...]:
    labels = [""] * int(data.basis.nt)
    state_index = _contract_state_index(data)
    valley_labels = tuple(int(value) for value in VALLEY_SEQUENCE)
    for ispin in range(int(data.basis.n_spin)):
        for ieta in range(int(data.basis.n_flavor)):
            valley = valley_labels[ieta] if ieta < len(valley_labels) else ieta
            for iband in range(int(data.basis.n_band)):
                labels[int(state_index[ispin, ieta, iband])] = f"spin{ispin}_eta{valley}_band{iband}"
    return tuple(labels)


def _contract_band_labels(data: HTGProjectedBasisData) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "active_window_index": int(index),
            "physical_band_index": int(band_index),
            "central_band_index": bool(int(band_index) in {int(value) for value in data.central_band_indices}),
        }
        for index, band_index in enumerate(data.projected_band_indices)
    )


def _contract_reference_scheme(data: HTGProjectedBasisData) -> str:
    reference = htg_band_reference_occupations(int(data.basis.n_band))
    return "average" if np.allclose(reference, 0.5, atol=1.0e-12, rtol=0.0) else "central_average"


def _contract_projected_basis(data: HTGProjectedBasisData) -> ContractProjectedBasis:
    model = _contract_single_particle_model(data)
    n_band = int(data.basis.n_band)
    active_valence = n_band // 2
    return ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=np.asarray(data.kvec, dtype=np.complex128),
        k_grid_frac=_contract_flatten_k_grid_frac(data),
        h0=np.asarray(data.h0, dtype=np.complex128),
        basis_energies=_contract_basis_energies_from_h0(data.h0),
        active_band_indices=_contract_active_band_indices(data),
        active_valence_bands=int(active_valence),
        active_conduction_bands=int(n_band - active_valence),
        micro_wavefunctions=np.asarray(data.basis.wavefunctions, dtype=np.complex128),
        flavor_labels=_contract_flavor_labels(data),
        band_labels=_contract_band_labels(data),
        metadata={
            "projected_basis_source": "HTGProjectedBasisData",
            "wavefunctions_axis_order": "basis,band,flavor,k",
            "density_axis_order": "abk",
            "active_band_semantics": "projected_band_indices_repeated_over_spin_valley",
            "projected_band_indices": [int(index) for index in data.projected_band_indices],
            "projected_band_count": int(data.basis.n_band),
            "central_band_indices": [int(index) for index in data.central_band_indices],
            "reciprocal_grid_shape": [int(value) for value in data.reciprocal_grid_shape],
            "reciprocal_grid_origin": [int(value) for value in data.reciprocal_grid_origin],
            "moire_cell_area_nm2": float(data.moire_cell_area_nm2),
        },
    )


def _contract_reference_density(run: HTGHartreeFockRun) -> np.ndarray:
    state = run.state
    return _htg_reference_density_blocks(
        state.nt,
        state.nk,
        n_spin=state.n_spin,
        n_eta=state.n_eta,
    )


def _contract_density_state(run: HTGHartreeFockRun) -> ContractDensityState:
    data = run.basis_data
    state = run.state
    reference = _contract_reference_density(run)
    return density_state_from_delta(
        state.density,
        reference,
        reference_scheme=_contract_reference_scheme(data),
        filling=float(state.nu),
        n_occupied_total=htg_occupied_state_count(
            state.nu,
            state.nt,
            state.nk,
            n_spin=state.n_spin,
            n_eta=state.n_eta,
        ),
        reference_metadata={
            "system": "htg",
            "raw_density_convention": "stored_delta",
            "density_axis_order": "abk",
            "reference_band_occupations": [
                float(value) for value in htg_band_reference_occupations(int(state.n_band))
            ],
            "reference_scheme_source": "htg_band_reference_occupations",
            "projected_band_count": int(state.n_band),
        },
        metadata={
            "raw_density_convention": "stored_delta",
            "density_delta_definition": "P-R",
            "density_axis_order": "abk",
            "adapter": "mean_field.systems.htg.mean_field_adapter",
            "filling_from_density": float(
                htg_filling_from_density(
                    state.density,
                    n_spin=state.n_spin,
                    n_eta=state.n_eta,
                )
            ),
        },
    )


def _contract_zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))


def _contract_hamiltonian_parts(run: HTGHartreeFockRun) -> ContractHamiltonianParts:
    h0 = np.asarray(run.state.h0, dtype=np.complex128)
    total = np.asarray(run.state.hamiltonian, dtype=np.complex128)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=total - h0,
        hartree=_contract_zero_field_like(h0),
        fock=_contract_zero_field_like(h0),
        total=total,
        density_input_convention="htg_primitive_stored_delta_collapsed",
        metadata={
            "component_resolution": "collapsed_total_minus_h0",
            "supports_crpa": False,
            "interaction_subtraction": str(run.basis_data.interaction.subtraction),
        },
    )


def _contract_float_diagnostics(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        finite = _contract_finite_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out


def _contract_iteration_history(run: HTGHartreeFockRun) -> list[dict[str, Any]]:
    count = max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))
    history: list[dict[str, Any]] = []
    for idx in range(count):
        history.append(
            {
                "iteration": int(idx + 1),
                "energy": float(run.iter_energy[idx]) if idx < len(run.iter_energy) else None,
                "error": float(run.iter_err[idx]) if idx < len(run.iter_err) else None,
                "oda_lambda": float(run.iter_oda[idx]) if idx < len(run.iter_oda) else None,
            }
        )
    return history


def _contract_iteration_count(run: HTGHartreeFockRun) -> int:
    return max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))


def _contract_mesh_from_run(run: HTGHartreeFockRun) -> tuple[int, int]:
    mesh_size = int(run.basis_data.mesh_size)
    if mesh_size > 0:
        return (mesh_size, mesh_size)
    return (int(run.state.nk), 1)


def _contract_default_hf_config_from_run(run: HTGHartreeFockRun) -> "HFConfig":
    from mean_field.api.hf import HFConfig

    data = run.basis_data
    state = run.state
    interaction = data.interaction
    return HFConfig(
        filling=float(state.nu),
        mesh=_contract_mesh_from_run(run),
        active_window=(int(data.basis.n_band // 2), int(data.basis.n_band - data.basis.n_band // 2)),
        active_band_indices=tuple(int(index) for index in data.projected_band_indices),
        interaction_scheme="average",
        density_convention="stored_delta",
        epsilon_r=float(interaction.epsilon_r),
        dsc_nm=float(interaction.d_sc_nm),
        coulomb_kernel="2d_gate",
        max_iter=max(_contract_iteration_count(run), 1),
        precision=float(state.precision),
        seeds=(str(int(run.seed)),),
        metadata={
            "source": "derived_from_HTGHartreeFockRun",
            "max_iter_semantics": "observed_iteration_count_when_original_limit_is_unavailable",
            "init_mode": str(run.init_mode),
            "projected_band_indices": [int(index) for index in data.projected_band_indices],
            "projected_band_count": int(data.basis.n_band),
            "central_band_indices": [int(index) for index in data.central_band_indices],
            "interaction_subtraction": str(interaction.subtraction),
            "interaction_g_shells": int(interaction.g_shells),
            "interaction_n_k": int(interaction.n_k),
        },
    )


def _contract_validate_hf_config_matches_run(config: "HFConfig", run: HTGHartreeFockRun) -> None:
    mesh = _contract_mesh_from_run(run)
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(f"HTG primitive HFResult config.mesh must match raw mesh {mesh}, got {config.mesh}")
    if not np.isclose(float(config.filling), float(run.state.nu)):
        raise ValueError(f"HTG primitive HFResult config.filling={config.filling} does not match raw nu={run.state.nu}")
    if config.density_convention != "stored_delta":
        raise ValueError("HTG primitive raw density is stored as P-R; use HFConfig.density_convention='stored_delta'")


def _contract_result_observables(run: HTGHartreeFockRun) -> dict[str, object]:
    state = run.state
    return {
        "primitive_nu": float(state.nu),
        "filling_from_density": float(
            htg_filling_from_density(
                state.density,
                n_spin=state.n_spin,
                n_eta=state.n_eta,
            )
        ),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "init_mode": str(run.init_mode),
        "seed": int(run.seed),
        "iterations": int(_contract_iteration_count(run)),
        "raw_density_convention": "stored_delta",
        "occupation_counts": None
        if state.occupation_counts is None
        else [int(value) for value in state.occupation_counts],
    }


def htg_hf_run_to_hf_run_result(
    run: HTGHartreeFockRun,
    *,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap a primitive-cell HTG HF run in canonical core contracts.

    The raw :class:`HTGHartreeFockRun` remains the source of truth.  This
    post-run adapter preserves the stored density delta ``P-R`` and creates a
    typed I/O view with collapsed Hamiltonian parts.  It does not recompute HF,
    split Hartree/Fock components, run topology, or touch cRPA.
    """

    state = run.state
    final_state = ContractHFState(
        basis=_contract_projected_basis(run.basis_data),
        density=_contract_density_state(run),
        hamiltonian=_contract_hamiltonian_parts(run),
        energies=np.asarray(state.energies, dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(state.mu),
        observables={
            "eigenvectors_active_available": False,
            "primitive_nu": float(state.nu),
            "filling_from_density": float(
                htg_filling_from_density(
                    state.density,
                    n_spin=state.n_spin,
                    n_eta=state.n_eta,
                )
            ),
            "occupation_counts": None
            if state.occupation_counts is None
            else [int(value) for value in state.occupation_counts],
        },
        diagnostics=_contract_float_diagnostics(state.diagnostics),
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=_contract_iteration_history(run),
        converged=bool(run.converged),
        exit_reason=str(run.exit_reason),
        best_seed=int(run.seed),
        init_mode=str(run.init_mode),
        archive_manifest={} if archive_manifest is None else dict(archive_manifest),
    )


def htg_hf_run_to_hf_result(
    run: HTGHartreeFockRun,
    *,
    config: "HFConfig | None" = None,
    archive_manifest: Mapping[str, Any] | None = None,
    observables: Mapping[str, object] | None = None,
) -> "HFResult":
    """Return a public :class:`HFResult` view of an existing primitive HTG run.

    The raw :class:`HTGHartreeFockRun` remains ``HFResult.state`` and the source
    of truth.  The attached ``canonical_run_result`` is produced by
    :func:`htg_hf_run_to_hf_run_result`; no SCF, interaction, topology, or cRPA
    calculation is rerun here.
    """

    from pathlib import Path

    from mean_field.api.artifacts import ArtifactManifest, ConventionBundle
    from mean_field.api.hf import HFResult
    from mean_field.api.models import model_record

    resolved_config = _contract_default_hf_config_from_run(run) if config is None else config
    _contract_validate_hf_config_matches_run(resolved_config, run)
    canonical = htg_hf_run_to_hf_run_result(
        run,
        archive_manifest=None if archive_manifest is None else dict(archive_manifest),
    )
    result_observables = _contract_result_observables(run)
    if observables is not None:
        result_observables.update(dict(observables))
    record = model_record(run.basis_data.model, system_name="htg")
    return HFResult(
        model=record,
        config=resolved_config,
        state=run,
        observables=result_observables,
        artifacts=ArtifactManifest(
            root=Path("."),
            model=record,
            conventions=ConventionBundle(
                energy_unit="eV",
                density_convention="stored_delta",
                density_axis_order="abk",
                hamiltonian_axis_order="abk",
                wavefunction_axis_order="basis,band,flavor,k",
                gauge="htg_projected_basis_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": "htg.primitive_hf.raw_run_result",
                "system_name": "htg",
                "adapter": "mean_field.systems.htg.mean_field_adapter.htg_hf_run_to_hf_result",
                "canonical_adapter": "mean_field.systems.htg.mean_field_adapter.htg_hf_run_to_hf_run_result",
                "raw_state_type": type(run).__name__,
            },
        ),
        canonical_run_result=canonical,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
