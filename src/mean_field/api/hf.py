from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

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
        manifest_files: dict[str, object] = {}
        manifest_metadata: dict[str, object] = {}
        conventions: ConventionBundle | dict[str, object] = ConventionBundle(
            density_convention=str(self.config.density_convention)
        )
        if self.artifacts is not None:
            manifest_files.update(dict(self.artifacts.files))
            manifest_metadata.update(dict(self.artifacts.metadata))
            conventions = self.artifacts.conventions
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
