from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mean_field.core.io import write_json_artifact

from .artifacts import ArtifactManifest, ConventionBundle, ModelRecord, write_contract_artifacts


DensityConventionName = Literal["projector", "stored_delta", "half_shifted"]
InteractionSchemeName = Literal["average", "cn", "zhang_crpa_split"]
CoulombKernelName = Literal["2d_gate", "3d_layered", "crpa", "onsite_intersite"]


@dataclass(frozen=True)
class HFConfig:
    filling: float
    mesh: tuple[int, int]
    active_window: tuple[int, int] | None = None
    active_band_indices: tuple[int, ...] | None = None
    interaction_scheme: InteractionSchemeName = "average"
    density_convention: DensityConventionName = "stored_delta"
    epsilon_r: float = 10.0
    dsc_nm: float = 10.0
    coulomb_kernel: CoulombKernelName = "2d_gate"
    max_iter: int = 300
    precision: float = 1.0e-8
    seeds: tuple[str, ...] = ("random",)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.mesh) != 2 or int(self.mesh[0]) <= 0 or int(self.mesh[1]) <= 0:
            raise ValueError(f"mesh must be positive (n1, n2), got {self.mesh}")
        if self.active_window is not None and len(self.active_window) != 2:
            raise ValueError(f"active_window must be (n_valence, n_conduction), got {self.active_window}")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.precision <= 0.0:
            raise ValueError("precision must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "filling": float(self.filling),
            "mesh": [int(self.mesh[0]), int(self.mesh[1])],
            "active_window": None if self.active_window is None else list(self.active_window),
            "active_band_indices": None if self.active_band_indices is None else list(self.active_band_indices),
            "interaction_scheme": self.interaction_scheme,
            "density_convention": self.density_convention,
            "epsilon_r": float(self.epsilon_r),
            "dsc_nm": float(self.dsc_nm),
            "coulomb_kernel": self.coulomb_kernel,
            "max_iter": int(self.max_iter),
            "precision": float(self.precision),
            "seeds": list(self.seeds),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class HFState:
    density: np.ndarray
    hamiltonian: np.ndarray | None = None
    h0: np.ndarray | None = None
    energies: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WavefunctionBundle:
    k: np.ndarray
    wavefunctions: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)
    convention: ConventionBundle = field(default_factory=ConventionBundle)


_SIDECAR_SEQUENCE_INLINE_LIMIT = 16


def _finite_float(value: object, *, path: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Non-finite value at {path}: {out!r}")
    return out


def _canonical_sidecar_array_summary(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "kind": "array_summary",
        "shape": [int(axis) for axis in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }


def _canonical_sidecar_value(value: object, *, path: str) -> object:
    """Return a strict-JSON-safe, metadata-only representation.

    Dense arrays are summarized rather than serialized.  Non-finite numbers and
    complex scalars are rejected so public JSON sidecars remain portable and do
    not hide physics/diagnostic failures behind Python-specific JSON tokens.
    """

    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _canonical_sidecar_array_summary(value)
    if isinstance(value, np.complexfloating) or isinstance(value, complex):
        raise TypeError(f"Complex scalar is not allowed in canonical HF sidecar metadata at {path}")
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _finite_float(value.item(), path=path)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _finite_float(value, path=path)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_sidecar_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        if len(value) > _SIDECAR_SEQUENCE_INLINE_LIMIT:
            return {
                "kind": "sequence_summary",
                "length": int(len(value)),
                "python_type": type(value).__name__,
            }
        return [
            _canonical_sidecar_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"Object of type {type(value).__name__} is not allowed in canonical HF sidecar at {path}")


def _canonical_sidecar_mapping(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return dict(_canonical_sidecar_value(value, path=path))


def _array_shape(value: object) -> list[int]:
    return [int(axis) for axis in np.asarray(value).shape]


def _mapping_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value)
    return []


def _canonical_hf_run_result_sidecar(canonical_run_result: Any) -> dict[str, object]:
    final_state = canonical_run_result.final_state
    basis = final_state.basis
    density = final_state.density
    reference = density.reference
    hamiltonian = final_state.hamiltonian
    iteration_history = list(canonical_run_result.iteration_history)
    last_iteration = (
        _canonical_sidecar_value(dict(iteration_history[-1]), path="iteration_history.last")
        if iteration_history
        else None
    )
    return {
        "schema_version": 1,
        "contract_type": "mean_field.core.contracts.HFRunResult",
        "converged": bool(canonical_run_result.converged),
        "exit_reason": str(canonical_run_result.exit_reason),
        "best_seed": int(canonical_run_result.best_seed),
        "init_mode": str(canonical_run_result.init_mode),
        "iteration_history": {
            "count": len(iteration_history),
            "fields": sorted(str(key) for row in iteration_history for key in row),
            "last": last_iteration,
        },
        "final_state": {
            "contract_type": "mean_field.core.contracts.HFState",
            "mu": _finite_float(final_state.mu, path="final_state.mu"),
            "energies_shape": _array_shape(final_state.energies),
            "eigenvectors_active_shape": _array_shape(final_state.eigenvectors_active),
            "observables_keys": _mapping_keys(final_state.observables),
            "diagnostics_keys": _mapping_keys(final_state.diagnostics),
            "basis": {
                "contract_type": "mean_field.core.contracts.ProjectedBasis",
                "system": str(basis.physical_model.system),
                "k_count": int(np.asarray(basis.kvec).size),
                "k_grid_frac_shape": _array_shape(basis.k_grid_frac),
                "h0_shape": _array_shape(basis.h0),
                "basis_energies_shape": _array_shape(basis.basis_energies),
                "micro_wavefunctions_shape": _array_shape(basis.micro_wavefunctions),
                "active_state_count": len(tuple(basis.active_band_indices)),
                "active_valence_bands": int(basis.active_valence_bands),
                "active_conduction_bands": int(basis.active_conduction_bands),
                "metadata": _canonical_sidecar_mapping(basis.metadata, path="final_state.basis.metadata"),
            },
            "density": {
                "contract_type": "mean_field.core.contracts.DensityState",
                "convention": str(density.convention),
                "density_delta_definition": "P-R",
                "density_delta_shape": _array_shape(density.density_delta),
                "reference_shape": _array_shape(reference.reference),
                "reference_scheme": str(reference.scheme),
                "filling": _finite_float(density.filling, path="final_state.density.filling"),
                "n_occupied_total": int(density.n_occupied_total),
                "metadata": _canonical_sidecar_mapping(density.metadata, path="final_state.density.metadata"),
                "reference_metadata": _canonical_sidecar_mapping(
                    reference.metadata,
                    path="final_state.density.reference_metadata",
                ),
            },
            "hamiltonian": {
                "contract_type": "mean_field.core.contracts.HamiltonianParts",
                "h0_shape": _array_shape(hamiltonian.h0),
                "fixed_shape": _array_shape(hamiltonian.fixed),
                "hartree_shape": _array_shape(hamiltonian.hartree),
                "fock_shape": _array_shape(hamiltonian.fock),
                "total_shape": _array_shape(hamiltonian.total),
                "density_input_convention": str(hamiltonian.density_input_convention),
                "metadata": _canonical_sidecar_mapping(
                    hamiltonian.metadata,
                    path="final_state.hamiltonian.metadata",
                ),
            },
        },
        "archive_manifest_keys": sorted(str(key) for key in dict(canonical_run_result.archive_manifest)),
    }


@dataclass(frozen=True)
class HFResult:
    model: ModelRecord
    config: HFConfig
    state: HFState | Any
    observables: dict[str, object] = field(default_factory=dict)
    artifacts: ArtifactManifest | None = None
    canonical_run_result: Any | None = None

    def quasiparticle_bands(self, path: Any) -> Any:
        if hasattr(self.state, "quasiparticle_bands"):
            return self.state.quasiparticle_bands(path)
        raise NotImplementedError("HFResult.quasiparticle_bands needs a system adapter for this result")

    def reconstruct_micro_wavefunctions(self) -> WavefunctionBundle:
        if hasattr(self.state, "reconstruct_micro_wavefunctions"):
            return self.state.reconstruct_micro_wavefunctions()
        raise NotImplementedError(
            "Micro-wavefunction reconstruction is a required public API, but this system adapter has not exposed it yet"
        )

    def save(self, output_dir: str | Path) -> Path:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest_files: dict[str, object] = {}
        manifest_metadata: dict[str, object] = {}
        conventions: ConventionBundle | dict[str, object] = ConventionBundle(
            density_convention=str(self.config.density_convention)
        )
        if self.artifacts is not None:
            manifest_files.update(dict(self.artifacts.files))
            manifest_metadata.update(dict(self.artifacts.metadata))
            conventions = self.artifacts.conventions
        if self.canonical_run_result is not None:
            sidecar = _canonical_hf_run_result_sidecar(self.canonical_run_result)
            write_json_artifact(sidecar, root / "canonical_hf_run_result.json")
            manifest_files["canonical_hf_run_result"] = "canonical_hf_run_result.json"
            manifest_metadata["canonical_hf_run_result"] = {
                "schema_version": sidecar["schema_version"],
                "contract_type": sidecar["contract_type"],
                "state_contract_type": sidecar["final_state"]["contract_type"],
            }
        paths = write_contract_artifacts(
            root,
            workflow="hf.result",
            system_name=self.model.system_name,
            model=self.model,
            config=self.config.to_dict(),
            conventions=conventions,
            validation={},
            observables=dict(self.observables),
            files=manifest_files,
            metadata=manifest_metadata,
        )
        return paths["manifest.json"]


def _run_tdbg_hf_if_explicit(model: object, config: HFConfig, kwargs: dict[str, Any]) -> HFResult | None:
    from mean_field.systems.tdbg import (
        TDBGModel,
        TDBGProjectedHFConfig,
        TDBGProjectedHFResult,
        build_tdbg_projected_hf_data,
        run_tdbg_projected_hf,
        tdbg_projected_hf_result_to_hf_run_result,
    )

    if not isinstance(model, TDBGModel):
        return None

    tdbg_config = kwargs.pop("tdbg_config", kwargs.pop("projected_config", None))
    if tdbg_config is None:
        raise NotImplementedError(
            "Unified run_hf has a TDBG adapter only for explicit "
            "tdbg_config=TDBGProjectedHFConfig(...) plus init_mode=...; "
            "generic HFConfig -> TDBGProjectedHFConfig mapping is not implemented"
        )
    if not isinstance(tdbg_config, TDBGProjectedHFConfig):
        raise TypeError(f"tdbg_config must be TDBGProjectedHFConfig, got {type(tdbg_config).__name__}")
    init_mode = kwargs.pop("init_mode", None)
    if init_mode is None:
        raise TypeError("TDBG public run_hf adapter requires explicit init_mode=...")
    seed = int(kwargs.pop("seed", 1))
    if kwargs:
        raise TypeError(f"Unsupported TDBG run_hf kwargs: {sorted(kwargs)}")

    _validate_tdbg_public_hf_config(model, config, tdbg_config)
    data = build_tdbg_projected_hf_data(tdbg_config)
    raw = run_tdbg_projected_hf(data, init_mode=str(init_mode), seed=seed)

    from .models import model_record

    record = model_record(model, system_name="tdbg")
    summary = raw.to_summary_dict() if hasattr(raw, "to_summary_dict") else {}
    canonical_run_result = (
        tdbg_projected_hf_result_to_hf_run_result(raw)
        if isinstance(raw, TDBGProjectedHFResult)
        else None
    )
    return HFResult(
        model=record,
        config=config,
        state=raw,
        observables=dict(summary),
        artifacts=ArtifactManifest(
            root=Path("."),
            model=record,
            conventions=ConventionBundle(
                energy_unit="eV",
                density_convention="projector",
                density_axis_order="abk",
                gauge="tdbg_projected_hf_system_defined",
            ),
            metadata={
                "schema_version": 1,
                "workflow": "tdbg.projected_hf.explicit_config",
                "system_name": "tdbg",
                "adapter": "mean_field.api.run_hf",
            },
        ),
        canonical_run_result=canonical_run_result,
    )


def _validate_tdbg_public_hf_config(model: object, config: HFConfig, tdbg_config: Any) -> None:
    if int(config.mesh[0]) != int(config.mesh[1]) or int(config.mesh[0]) != int(tdbg_config.mesh_size):
        raise ValueError(
            "TDBG public run_hf requires HFConfig.mesh=(mesh_size, mesh_size) matching "
            f"tdbg_config.mesh_size={tdbg_config.mesh_size}, got {config.mesh}"
        )
    if float(config.filling) != float(int(tdbg_config.filling)):
        raise ValueError(
            f"TDBG public run_hf requires HFConfig.filling={tdbg_config.filling}, got {config.filling}"
        )
    if config.max_iter != int(tdbg_config.max_iter):
        raise ValueError(
            f"TDBG public run_hf requires HFConfig.max_iter={tdbg_config.max_iter}, got {config.max_iter}"
        )
    if not np.isclose(float(config.precision), float(tdbg_config.precision)):
        raise ValueError(
            f"TDBG public run_hf requires HFConfig.precision={tdbg_config.precision}, got {config.precision}"
        )
    if config.density_convention != "projector":
        raise ValueError(
            "TDBG projected HF stores an absolute projector density; set "
            "HFConfig.density_convention='projector' for this explicit adapter"
        )
    if config.active_window is not None or config.active_band_indices is not None:
        raise NotImplementedError(
            "TDBG public run_hf takes the projected window from tdbg_config.window; "
            "leave HFConfig.active_window/active_band_indices unset for now"
        )
    model_theta = getattr(model, "theta_deg", None)
    model_cut = getattr(model, "cut", None)
    if model_theta is not None and not np.isclose(float(model_theta), float(tdbg_config.theta_deg)):
        raise ValueError(
            f"TDBG model theta_deg={model_theta} does not match tdbg_config.theta_deg={tdbg_config.theta_deg}"
        )
    if model_cut is not None and not np.isclose(float(model_cut), float(tdbg_config.cut)):
        raise ValueError(f"TDBG model cut={model_cut} does not match tdbg_config.cut={tdbg_config.cut}")


def run_hf(model: object, config: HFConfig, **kwargs: Any) -> HFResult:
    """Run HF through a system-provided public hook.

    Phase 1 intentionally does not rewrite existing HF runners.  Systems should
    later expose a `run_hf(config, **kwargs)` adapter that returns or can be
    wrapped as an `HFResult`.
    """

    if hasattr(model, "run_hf"):
        raw = model.run_hf(config, **kwargs)  # type: ignore[attr-defined]
        if isinstance(raw, HFResult):
            return raw
        from .models import model_record

        return HFResult(model=model_record(model), config=config, state=raw)

    explicit_result = _run_tdbg_hf_if_explicit(model, config, dict(kwargs))
    if explicit_result is not None:
        return explicit_result

    raise NotImplementedError(
        "Unified run_hf is frozen at the API level, but this model has no run_hf(config) adapter yet"
    )


__all__ = [
    "CoulombKernelName",
    "DensityConventionName",
    "HFConfig",
    "HFResult",
    "HFState",
    "InteractionSchemeName",
    "WavefunctionBundle",
    "run_hf",
]
