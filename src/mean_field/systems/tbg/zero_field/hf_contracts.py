from __future__ import annotations

"""Canonical mean-field contract adapters for TBG zero-field HF runs.

The functions here are post-run I/O adapters.  They wrap arrays already produced
by the existing TBG zero-field B0/BM workflow and do not change the SCF loop,
interaction contractions, topology, path reconstruction, or cRPA behavior.

A bare :class:`RestrictedHartreeFockRun` is not self-describing enough for the
canonical projected-basis contract: it lacks the k-grid coordinates and BM
micro-wavefunctions.  The safe boundary therefore requires the matching
``grid_solution`` (or a higher-level ``B0HFBenchmarkRun`` that owns it) and
validates that the grid is the current B0 uniform mesh before creating the
canonical view.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np

from mean_field.core.contracts import (
    DensityState as ContractDensityState,
    HFRunResult as ContractHFRunResult,
    HFState as ContractHFState,
    HamiltonianParts as ContractHamiltonianParts,
    ProjectedBasis as ContractProjectedBasis,
    SingleParticleModel as ContractSingleParticleModel,
)
from mean_field.core.hf.contracts_bridge import density_state_from_delta

from .hf import (
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    build_overlap_block_set,
    restricted_filling,
    restricted_occupied_state_count,
    run_restricted_hartree_fock,
)
from .model import BMSolution, TBGZeroFieldBMModel


def _unavailable_hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "TBG zero-field contract records an already-built BM projected basis; "
        "use mean_field.systems.tbg.zero_field.solve_bm_model for fresh Hamiltonians."
    )


def _unavailable_diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "TBG zero-field contract records post-run arrays; "
        "fresh BM diagonalization is not performed by the adapter."
    )


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float_diagnostics(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        finite = _finite_or_none(value)
        if finite is not None:
            out[str(key)] = finite
    return out


def _single_particle_model(solution: BMSolution) -> ContractSingleParticleModel:
    params = solution.params
    return ContractSingleParticleModel(
        system="tbg_zero_field",
        lattice={
            "g1_nm_inv_pair": _complex_pair(params.g1),
            "g2_nm_inv_pair": _complex_pair(params.g2),
            "a1_nm_pair": _complex_pair(params.a1),
            "a2_nm_pair": _complex_pair(params.a2),
            "theta12_rad": float(params.theta12),
            "kt_nm_inv_pair": _complex_pair(params.kt),
            "kb_point_nm_inv_pair": _complex_pair(params.kb_point),
        },
        params={
            "dtheta_rad": float(params.dtheta_rad),
            "convention": str(params.convention),
            "vf": float(params.vf),
            "chemical_potential": float(params.chemical_potential),
            "w0": float(params.w0),
            "w1": float(params.w1),
            "strain": float(params.strain),
            "strain_angle_rad": float(params.strain_angle_rad),
            "poisson": float(params.poisson),
            "beta_g": float(params.beta_g),
            "alpha": float(params.alpha),
            "deformation_potential": float(params.deformation_potential),
        },
        hamiltonian_builder=_unavailable_hamiltonian_builder,
        diagonalizer=_unavailable_diagonalizer,
        metadata={
            "source": "mean_field.systems.tbg.zero_field.BMSolution",
            "model_name": "zero_field_bm",
            "lg": int(solution.lg),
            "nlocal": int(solution.nlocal),
            "n_eta": int(solution.n_eta),
            "n_spin": int(solution.n_spin),
            "nb": int(solution.nb),
            "periodic_g_grid": bool(solution.periodic_g_grid),
            "supports_crpa": False,
        },
    )


def _infer_b0_lk(solution: BMSolution) -> int:
    nk = int(solution.nk)
    side = int(round(math.sqrt(nk)))
    if side * side != nk:
        raise ValueError(
            "TBG zero-field canonical adapter requires a square B0 uniform grid_solution; "
            f"got grid_solution.nk={nk}.  A bare RestrictedHartreeFockRun does not carry enough k-grid metadata."
        )
    lk = side - 1
    if lk <= 0:
        raise ValueError(
            "TBG zero-field canonical adapter requires a B0 uniform grid with lk >= 1; "
            f"got inferred lk={lk} from grid_solution.nk={nk}."
        )
    return lk


def _b0_uniform_k_grid_frac(solution: BMSolution) -> np.ndarray:
    lk = _infer_b0_lk(solution)
    frac = np.arange(lk + 1, dtype=float) / float(lk)
    f1, f2 = np.meshgrid(frac, frac, indexing="ij")
    k_grid_frac = np.stack([np.ravel(f1, order="F"), np.ravel(f2, order="F")], axis=1)
    expected_kvec = np.ravel(
        frac[:, None] * solution.params.g1 + frac[None, :] * solution.params.g2,
        order="F",
    ).astype(np.complex128)
    actual_kvec = np.asarray(solution.lattice_kvec, dtype=np.complex128).reshape(-1)
    if actual_kvec.shape != expected_kvec.shape or not np.allclose(actual_kvec, expected_kvec, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(
            "TBG zero-field canonical adapter requires grid_solution.lattice_kvec to match "
            f"the B0 uniform mesh inferred from nk={solution.nk} (lk={lk}); received a non-uniform or reordered grid."
        )
    return k_grid_frac


def _central_bm_band_indices(solution: BMSolution) -> tuple[int, ...]:
    dim = int(solution.nlocal) * int(solution.lg) * int(solution.lg)
    start = dim // 2 - 1
    return tuple(range(start, start + int(solution.nb)))


def _active_band_indices(solution: BMSolution) -> tuple[int, ...]:
    central = _central_bm_band_indices(solution)
    labels: list[int] = []
    for band_index in central:
        for _ieta in range(int(solution.n_eta)):
            for _ispin in range(int(solution.n_spin)):
                labels.append(int(band_index))
    return tuple(labels)


def _flavor_labels(solution: BMSolution) -> tuple[str, ...]:
    labels: list[str] = []
    valley_labels = ("K", "Kprime")
    for iband in range(int(solution.nb)):
        for ieta in range(int(solution.n_eta)):
            valley = valley_labels[ieta] if ieta < len(valley_labels) else f"eta{ieta}"
            for ispin in range(int(solution.n_spin)):
                labels.append(f"spin{ispin}_{valley}_bm_band{iband}")
    return tuple(labels)


def _band_labels(solution: BMSolution) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "bm_window_index": int(index),
            "full_bm_matrix_band_index": int(band_index),
        }
        for index, band_index in enumerate(_central_bm_band_indices(solution))
    )


def _validate_solution_matches_state(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> None:
    state = run.state
    if int(state.nk) != int(grid_solution.nk):
        raise ValueError(
            "TBG zero-field canonical adapter requires hf_run.state.nk to match grid_solution.nk; "
            f"got {state.nk} and {grid_solution.nk}."
        )
    if int(state.nt) != int(grid_solution.nt):
        raise ValueError(
            "TBG zero-field canonical adapter requires hf_run.state.nt to match grid_solution.nt; "
            f"got {state.nt} and {grid_solution.nt}."
        )
    if int(state.n_spin) != int(grid_solution.n_spin) or int(state.n_eta) != int(grid_solution.n_eta):
        raise ValueError(
            "TBG zero-field canonical adapter requires matching spin/valley dimensions between hf_run and grid_solution; "
            f"got state (spin={state.n_spin}, eta={state.n_eta}) and "
            f"grid_solution (spin={grid_solution.n_spin}, eta={grid_solution.n_eta})."
        )
    if int(state.n_band) != int(grid_solution.nb):
        raise ValueError(
            "TBG zero-field canonical adapter requires hf_run.state.n_band to match grid_solution.nb; "
            f"got {state.n_band} and {grid_solution.nb}."
        )
    expected_uk_shape = (
        int(grid_solution.nlocal) * int(grid_solution.lg) * int(grid_solution.lg),
        int(grid_solution.nb),
        int(grid_solution.n_eta),
        int(grid_solution.nk),
    )
    if np.asarray(grid_solution.uk).shape != expected_uk_shape:
        raise ValueError(
            "TBG zero-field canonical adapter requires grid_solution.uk with raw BM shape "
            f"{expected_uk_shape}; got {np.asarray(grid_solution.uk).shape}."
        )

    flattened = np.asarray(grid_solution.flattened_energies(), dtype=float)
    h0 = np.asarray(state.h0, dtype=np.complex128)
    expected_h0 = np.zeros_like(h0)
    for ik in range(int(grid_solution.nk)):
        np.fill_diagonal(expected_h0[:, :, ik], flattened[:, ik])
    if h0.shape != expected_h0.shape or not np.allclose(h0, expected_h0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError(
            "TBG zero-field canonical adapter cannot safely combine hf_run.state.h0 with grid_solution: "
            "state.h0 is not the diagonal BM h0 built from grid_solution.flattened_energies()."
        )


def _projected_basis(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> ContractProjectedBasis:
    _validate_solution_matches_state(run, grid_solution)
    model = _single_particle_model(grid_solution)
    n_band = int(grid_solution.nb)
    active_valence = n_band // 2
    return ContractProjectedBasis(
        physical_model=model,
        basis_model=model,
        kvec=np.asarray(grid_solution.lattice_kvec, dtype=np.complex128),
        k_grid_frac=_b0_uniform_k_grid_frac(grid_solution),
        h0=np.asarray(run.state.h0, dtype=np.complex128),
        basis_energies=np.asarray(grid_solution.flattened_energies(), dtype=float),
        active_band_indices=_active_band_indices(grid_solution),
        active_valence_bands=int(active_valence),
        active_conduction_bands=int(n_band - active_valence),
        micro_wavefunctions=np.asarray(grid_solution.uk, dtype=np.complex128),
        flavor_labels=_flavor_labels(grid_solution),
        band_labels=_band_labels(grid_solution),
        metadata={
            "projected_basis_source": "BMSolution grid_solution + RestrictedHartreeFockState.h0",
            "k_grid_frac_source": "validated_reconstruction_from_B0_uniform_lk",
            "wavefunctions_axis_order": "bm_micro_basis,bm_band,valley,k",
            "spin_degeneracy_implicit_in_micro_wavefunctions": True,
            "density_axis_order": "abk",
            "hamiltonian_axis_order": "abk",
            "active_state_order": "bm_band,valley,spin with spin fastest",
            "active_band_semantics": "central_full_BM_matrix_band_indices_repeated_over_valley_spin",
            "active_band_indices_per_bm_band": [int(index) for index in _central_bm_band_indices(grid_solution)],
            "lg": int(grid_solution.lg),
            "nlocal": int(grid_solution.nlocal),
            "periodic_g_grid": bool(grid_solution.periodic_g_grid),
            "supports_crpa": False,
        },
    )


def _reference_density(run: RestrictedHartreeFockRun) -> np.ndarray:
    state = run.state
    reference = np.zeros((state.nt, state.nt, state.nk), dtype=np.complex128)
    for ik in range(state.nk):
        np.fill_diagonal(reference[:, :, ik], 0.5)
    return reference


def _density_state(run: RestrictedHartreeFockRun) -> ContractDensityState:
    state = run.state
    reference = _reference_density(run)
    filling_from_density = restricted_filling(state.density)
    return density_state_from_delta(
        state.density,
        reference,
        reference_scheme="average",
        filling=float(state.nu),
        n_occupied_total=restricted_occupied_state_count(state.nu, state.nt, state.nk),
        reference_metadata={
            "system": "tbg_zero_field",
            "raw_density_convention": "stored_delta",
            "density_axis_order": "abk",
            "reference_scheme_source": "0.5 * identity in the projected BM active basis",
            "reference_diagonal": 0.5,
        },
        metadata={
            "raw_density_convention": "stored_delta",
            "density_delta_definition": "P - 0.5 I",
            "density_axis_order": "abk",
            "adapter": "mean_field.systems.tbg.zero_field.hf_contracts",
            "filling_from_density": float(filling_from_density),
        },
    )


def _zero_field_like(template: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(template, dtype=np.complex128))


def _hamiltonian_parts(run: RestrictedHartreeFockRun) -> ContractHamiltonianParts:
    state = run.state
    h0 = np.asarray(state.h0, dtype=np.complex128)
    total = np.asarray(state.hamiltonian, dtype=np.complex128)
    return ContractHamiltonianParts(
        h0=h0,
        fixed=total - h0,
        hartree=_zero_field_like(h0),
        fock=_zero_field_like(h0),
        total=total,
        density_input_convention="tbg_zero_field_stored_delta_collapsed",
        metadata={
            "component_resolution": "collapsed_total_minus_h0",
            "raw_interaction_components_available": False,
            "supports_crpa": False,
            "v0_mev": float(state.v0),
            "beta": _finite_or_none(state.diagnostics.get("beta")),
            "overlap_lg": _finite_or_none(state.diagnostics.get("overlap_lg")),
        },
    )


def _iteration_history(run: RestrictedHartreeFockRun) -> list[dict[str, Any]]:
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


@dataclass(frozen=True)
class TBGZeroFieldRunHFConfig:
    """Explicit public config for TBG zero-field restricted HF dispatch.

    The matching :class:`BMSolution` is required because the canonical TBG
    adapter needs the exact B0 k-grid and BM micro-wavefunctions used by the SCF
    grid.  The public :class:`mean_field.api.hf.HFConfig` is validated as a
    matching contract; it is not translated into hidden BM-grid construction.
    """

    grid_solution: BMSolution
    nu: float
    init_mode: str = "educated"
    seed: int = 1
    beta: float = 1.0
    max_iter: int = 300
    overlap_lg: int | None = None
    precision: float = 1.0e-5
    oda_stall_threshold: float = 1.0e-3
    relative_permittivity: float = 15.0
    screening_lm: float | None = None
    finite_zero_limit: bool = False
    zero_cutoff: float = 1.0e-6

    def __post_init__(self) -> None:
        if not isinstance(self.grid_solution, BMSolution):
            raise TypeError(f"grid_solution must be BMSolution, got {type(self.grid_solution).__name__}")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if float(self.precision) <= 0.0:
            raise ValueError("precision must be positive")
        if float(self.oda_stall_threshold) <= 0.0:
            raise ValueError("oda_stall_threshold must be positive")
        if self.overlap_lg is not None and int(self.overlap_lg) <= 0:
            raise ValueError("overlap_lg must be positive when provided")
        if float(self.relative_permittivity) <= 0.0:
            raise ValueError("relative_permittivity must be positive")
        if self.screening_lm is not None and float(self.screening_lm) <= 0.0:
            raise ValueError("screening_lm must be positive when provided")
        if float(self.zero_cutoff) <= 0.0:
            raise ValueError("zero_cutoff must be positive")


def _tbg_zero_field_grid_side(solution: BMSolution) -> int:
    return int(_infer_b0_lk(solution) + 1)




def _resolved_screening_lm(config: TBGZeroFieldRunHFConfig) -> float:
    if config.screening_lm is not None:
        return float(config.screening_lm)
    params = config.grid_solution.params
    return float(np.sqrt(abs(params.a1) * abs(params.a2)))

def _validate_tbg_zero_field_public_hf_config(config: "HFConfig", tbg_config: TBGZeroFieldRunHFConfig) -> None:
    solution = tbg_config.grid_solution
    side = _tbg_zero_field_grid_side(solution)
    mesh = (side, side)
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(
            "TBG zero-field public run_hf requires HFConfig.mesh to match the B0 grid point count "
            f"{mesh} (lk={side - 1}), got {config.mesh}"
        )
    if not np.isclose(float(config.filling), float(tbg_config.nu)):
        raise ValueError(f"TBG zero-field public run_hf requires HFConfig.filling={tbg_config.nu}, got {config.filling}")
    if int(config.max_iter) != int(tbg_config.max_iter):
        raise ValueError(
            f"TBG zero-field public run_hf requires HFConfig.max_iter={tbg_config.max_iter}, got {config.max_iter}"
        )
    if not np.isclose(float(config.precision), float(tbg_config.precision)):
        raise ValueError(
            f"TBG zero-field public run_hf requires HFConfig.precision={tbg_config.precision}, got {config.precision}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError(
            "TBG zero-field restricted HF stores density as P - 0.5 I; "
            "set HFConfig.density_convention='stored_delta'"
        )
    if config.interaction_scheme != "average":
        raise ValueError("TBG zero-field public run_hf currently requires HFConfig.interaction_scheme='average'")
    if config.coulomb_kernel != "2d_gate":
        raise ValueError("TBG zero-field public run_hf currently requires HFConfig.coulomb_kernel='2d_gate'")
    if not np.isclose(float(config.epsilon_r), float(tbg_config.relative_permittivity)):
        raise ValueError(
            "TBG zero-field public run_hf requires HFConfig.epsilon_r to match "
            f"relative_permittivity={tbg_config.relative_permittivity}, got {config.epsilon_r}"
        )
    screening_lm = _resolved_screening_lm(tbg_config)
    if not np.isclose(float(config.dsc_nm), screening_lm):
        raise ValueError(
            "TBG zero-field public run_hf uses HFConfig.dsc_nm as the resolved screening_lm; "
            f"expected {screening_lm}, got {config.dsc_nm}"
        )
    if config.active_window is not None or config.active_band_indices is not None:
        raise NotImplementedError(
            "TBG zero-field public run_hf takes the active two-band BM window from grid_solution; "
            "leave HFConfig.active_window/active_band_indices unset for now"
        )


def _validate_tbg_zero_field_model_matches_grid(model: TBGZeroFieldBMModel, solution: BMSolution) -> None:
    if int(model.lg) != int(solution.lg):
        raise ValueError(f"TBG zero-field model lg={model.lg} does not match grid_solution.lg={solution.lg}")
    if bool(model.periodic_g_grid) != bool(solution.periodic_g_grid):
        raise ValueError(
            "TBG zero-field model periodic_g_grid does not match grid_solution.periodic_g_grid: "
            f"{model.periodic_g_grid} vs {solution.periodic_g_grid}"
        )
    solution_theta_deg = float(solution.params.dtheta_rad) * 180.0 / math.pi
    if not np.isclose(float(model.theta_deg), solution_theta_deg):
        raise ValueError(
            f"TBG zero-field model theta_deg={model.theta_deg} does not match grid_solution theta_deg={solution_theta_deg}"
        )


def tbg_zero_field_hf_run_to_hf_run_result(
    run: RestrictedHartreeFockRun,
    *,
    grid_solution: BMSolution | None = None,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap a TBG zero-field HF run in canonical core contracts.

    ``RestrictedHartreeFockRun`` itself does not store the k-grid fractional
    coordinates or BM micro-wavefunctions required by ``ProjectedBasis``.  Pass
    the matching B0 ``grid_solution`` (or use
    :func:`b0_hf_benchmark_run_to_hf_run_result`) so the adapter can validate the
    grid and avoid fabricating canonical basis data.
    """

    if grid_solution is None:
        raise ValueError(
            "TBG zero-field canonical HFRunResult adapter requires the matching BMSolution grid_solution; "
            "a bare RestrictedHartreeFockRun has no k-grid coordinates or BM micro-wavefunctions. "
            "Use b0_hf_benchmark_run_to_hf_run_result(result) for benchmark results, or pass "
            "grid_solution=<BMSolution> from the same SCF grid."
        )

    state = run.state
    lk = _infer_b0_lk(grid_solution)
    final_state = ContractHFState(
        basis=_projected_basis(run, grid_solution),
        density=_density_state(run),
        hamiltonian=_hamiltonian_parts(run),
        energies=np.asarray(state.energies, dtype=float),
        eigenvectors_active=np.empty((0,), dtype=np.complex128),
        mu=float(state.mu),
        observables={
            "eigenvectors_active_available": False,
            "grid_solution_available": True,
            "grid_lk": int(lk),
            "bm_lg": int(grid_solution.lg),
            "nu": float(state.nu),
            "filling_from_density": float(restricted_filling(state.density)),
            "micro_wavefunctions_source": "BMSolution.uk",
            "micro_wavefunctions_spin_degeneracy_implicit": True,
        },
        diagnostics=_float_diagnostics(state.diagnostics),
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=_iteration_history(run),
        converged=bool(run.converged),
        exit_reason=str(run.exit_reason),
        best_seed=int(run.seed),
        init_mode=str(run.init_mode),
        archive_manifest={} if archive_manifest is None else dict(archive_manifest),
    )


def b0_hf_benchmark_run_to_hf_run_result(
    result: object,
    *,
    archive_manifest: dict[str, Any] | None = None,
) -> ContractHFRunResult:
    """Wrap a ``B0HFBenchmarkRun``-like result in canonical core contracts.

    This is the preferred safe adapter for TBG zero-field benchmark artifacts
    because the higher-level result carries both ``hf_run`` and the matching
    BM ``grid_solution`` used to build the SCF basis.
    """

    hf_run = getattr(result, "hf_run")
    grid_solution = getattr(result, "grid_solution")
    path_result = getattr(result, "path_result", None)
    inferred_lk = _infer_b0_lk(grid_solution)
    if path_result is not None and int(getattr(path_result, "lk")) != inferred_lk:
        raise ValueError(
            "TBG zero-field canonical adapter requires path_result.lk to match the grid_solution mesh; "
            f"got path_result.lk={getattr(path_result, 'lk')} and inferred grid lk={inferred_lk}."
        )
    return tbg_zero_field_hf_run_to_hf_run_result(
        hf_run,
        grid_solution=grid_solution,
        archive_manifest=archive_manifest,
    )



def _default_hf_config_from_run(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> "HFConfig":
    from mean_field.api.hf import HFConfig

    iteration_count = max(1, len(run.iter_err))
    return HFConfig(
        filling=float(run.state.nu),
        mesh=(_tbg_zero_field_grid_side(grid_solution), _tbg_zero_field_grid_side(grid_solution)),
        interaction_scheme="average",
        density_convention="stored_delta",
        epsilon_r=float(run.state.diagnostics.get("relative_permittivity", 15.0)),
        dsc_nm=float(run.state.diagnostics.get("screening_lm", np.sqrt(abs(grid_solution.params.a1) * abs(grid_solution.params.a2)))),
        coulomb_kernel="2d_gate",
        max_iter=iteration_count,
        precision=float(run.state.precision),
        seeds=(str(int(run.seed)),),
        metadata={
            "source": "derived_from_RestrictedHartreeFockRun_and_BMSolution",
            "max_iter_semantics": "observed_iteration_count_when_original_limit_is_unavailable",
            "init_mode": str(run.init_mode),
            "grid_lk": int(_infer_b0_lk(grid_solution)),
            "bm_lg": int(grid_solution.lg),
        },
    )


def _validate_hf_config_matches_run(config: "HFConfig", run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> None:
    mesh = (_tbg_zero_field_grid_side(grid_solution), _tbg_zero_field_grid_side(grid_solution))
    if (int(config.mesh[0]), int(config.mesh[1])) != mesh:
        raise ValueError(f"TBG zero-field HFResult config.mesh must match B0 grid point count {mesh}, got {config.mesh}")
    if not np.isclose(float(config.filling), float(run.state.nu)):
        raise ValueError(
            f"TBG zero-field HFResult config.filling={config.filling} does not match raw nu={run.state.nu}"
        )
    if config.density_convention != "stored_delta":
        raise ValueError(
            "TBG zero-field raw density is stored as P - 0.5 I; use HFConfig.density_convention='stored_delta'"
        )


def _result_observables(run: RestrictedHartreeFockRun, grid_solution: BMSolution) -> dict[str, object]:
    state = run.state
    return {
        "nu": float(state.nu),
        "filling_from_density": float(restricted_filling(state.density)),
        "converged": bool(run.converged),
        "exit_reason": str(run.exit_reason),
        "init_mode": str(run.init_mode),
        "seed": int(run.seed),
        "iterations": int(max(len(run.iter_energy), len(run.iter_err), len(run.iter_oda))),
        "raw_density_convention": "stored_delta",
        "grid_lk": int(_infer_b0_lk(grid_solution)),
        "bm_lg": int(grid_solution.lg),
    }


def tbg_zero_field_hf_run_to_hf_result(
    run: RestrictedHartreeFockRun,
    *,
    grid_solution: BMSolution,
    config: "HFConfig | None" = None,
    archive_manifest: Mapping[str, Any] | None = None,
    observables: Mapping[str, object] | None = None,
) -> "HFResult":
    """Return a public :class:`HFResult` view of an existing TBG zero-field HF run.

    The matching ``grid_solution`` is required for the same reason as the
    canonical adapter: the raw HF run does not carry BM micro-wavefunctions or
    fractional k-grid coordinates.
    """

    from mean_field.api.artifacts import ArtifactManifest, ConventionBundle
    from mean_field.api.hf import HFResult
    from mean_field.api.models import model_record

    resolved_config = _default_hf_config_from_run(run, grid_solution) if config is None else config
    _validate_hf_config_matches_run(resolved_config, run, grid_solution)
    canonical = tbg_zero_field_hf_run_to_hf_run_result(
        run,
        grid_solution=grid_solution,
        archive_manifest=None if archive_manifest is None else dict(archive_manifest),
    )
    result_observables = _result_observables(run, grid_solution)
    if observables is not None:
        result_observables.update(dict(observables))
    record = model_record(TBGZeroFieldBMModel.from_config(grid_solution.params.dtheta_rad * 180.0 / math.pi, lg=grid_solution.lg, params=grid_solution.params), system_name="tbg_zero_field")
    return HFResult(
        model=record,
        config=resolved_config,
        state=run,
        observables=result_observables,
        artifacts=ArtifactManifest(
            root=Path("."),
            model=record,
            conventions=ConventionBundle(
                energy_unit="meV",
                density_convention="stored_delta",
                density_axis_order="abk",
                hamiltonian_axis_order="abk",
                wavefunction_axis_order="bm_micro_basis,bm_band,valley,k",
                gauge="tbg_zero_field_bm_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": "tbg.zero_field.restricted_hf.raw_run_result",
                "system_name": "tbg_zero_field",
                "adapter": "mean_field.systems.tbg.zero_field.hf_contracts.tbg_zero_field_hf_run_to_hf_result",
                "canonical_adapter": "mean_field.systems.tbg.zero_field.hf_contracts.tbg_zero_field_hf_run_to_hf_run_result",
                "raw_state_type": type(run).__name__,
            },
        ),
        canonical_run_result=canonical,
    )


def run_tbg_zero_field_hf_config_adapter(model: object, config: "HFConfig", **kwargs: Any) -> "HFResult | None":
    """Run TBG zero-field restricted HF from an explicit grid-owning config."""

    if not isinstance(model, TBGZeroFieldBMModel):
        return None
    if "tbg_zero_field_config" not in kwargs:
        raise NotImplementedError(
            "Unified run_hf has a TBG zero-field adapter only for explicit "
            "tbg_zero_field_config=TBGZeroFieldRunHFConfig(grid_solution=...); "
            "generic HFConfig -> BMSolution/grid workflow mapping is not implemented"
        )
    tbg_config = kwargs.pop("tbg_zero_field_config")
    if not isinstance(tbg_config, TBGZeroFieldRunHFConfig):
        raise TypeError(f"tbg_zero_field_config must be TBGZeroFieldRunHFConfig, got {type(tbg_config).__name__}")
    if kwargs:
        raise TypeError(f"Unsupported TBG zero-field run_hf kwargs: {sorted(kwargs)}")

    _validate_tbg_zero_field_model_matches_grid(model, tbg_config.grid_solution)
    _validate_tbg_zero_field_public_hf_config(config, tbg_config)
    overlap_lg = int(tbg_config.grid_solution.lg if tbg_config.overlap_lg is None else tbg_config.overlap_lg)
    state = RestrictedHartreeFockState.from_bm_solution(
        tbg_config.grid_solution,
        nu=float(tbg_config.nu),
        precision=float(tbg_config.precision),
    )
    state.diagnostics["relative_permittivity"] = float(tbg_config.relative_permittivity)
    state.diagnostics["screening_lm"] = float(_resolved_screening_lm(tbg_config))
    state.diagnostics["finite_zero_limit"] = float(bool(tbg_config.finite_zero_limit))
    state.diagnostics["zero_cutoff"] = float(tbg_config.zero_cutoff)
    overlap_blocks = build_overlap_block_set(
        tbg_config.grid_solution,
        lg=overlap_lg,
        relative_permittivity=float(tbg_config.relative_permittivity),
        screening_lm=tbg_config.screening_lm,
        finite_zero_limit=bool(tbg_config.finite_zero_limit),
        zero_cutoff=float(tbg_config.zero_cutoff),
    )
    raw = run_restricted_hartree_fock(
        state,
        overlap_blocks,
        tbg_config.grid_solution.lattice_kvec,
        tbg_config.grid_solution.params,
        init_mode=str(tbg_config.init_mode),
        seed=int(tbg_config.seed),
        beta=float(tbg_config.beta),
        max_iter=int(tbg_config.max_iter),
        oda_stall_threshold=float(tbg_config.oda_stall_threshold),
    )
    return tbg_zero_field_hf_run_to_hf_result(
        raw,
        grid_solution=tbg_config.grid_solution,
        config=config,
        observables={
            "public_run_hf_adapter": "mean_field.systems.tbg.zero_field.hf_contracts.run_tbg_zero_field_hf_config_adapter",
            "explicit_config_type": "TBGZeroFieldRunHFConfig",
        },
    )


__all__ = [
    "TBGZeroFieldRunHFConfig",
    "b0_hf_benchmark_run_to_hf_run_result",
    "run_tbg_zero_field_hf_config_adapter",
    "tbg_zero_field_hf_run_to_hf_result",
    "tbg_zero_field_hf_run_to_hf_run_result",
]
