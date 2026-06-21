from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403

from .artifacts import ArtifactManifest, ConventionBundle, ModelRecord, ResultDirectory, load_result, write_contract_artifacts


DensityConventionName = Literal["projector", "stored_delta", "half_shifted"]
InteractionSchemeName = Literal["average", "cn", "zhang_crpa_split"]
CoulombKernelName = Literal["2d_gate", "3d_layered", "crpa", "onsite_intersite"]
HFAdapterType = Literal["run_hf", "hf_result", "canonical_hf_run_result"]


@dataclass(frozen=True)
class HFAdapterInfo:
    """Public descriptor for a safe HF boundary adapter.

    ``supports_run_hf_config`` is intentionally separate from registration:
    most stable adapters are post-run canonical I/O converters, not config-to-run
    solvers. Registering them here makes the stable public surface discoverable
    without inventing missing ``HFConfig -> system runner`` logic.  The
    ``run_hf_config_reason`` records either the explicit config contract for a
    supported run adapter or why a converter remains post-run-only.
    """

    name: str
    system_name: str
    adapter_type: HFAdapterType
    import_path: str
    description: str
    supports_run_hf_config: bool = False
    requires_explicit_inputs: tuple[str, ...] = ()
    run_hf_config_reason: str = ""

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

__all__ = [name for name in globals() if not name.startswith('__')]
