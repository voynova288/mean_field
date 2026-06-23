from __future__ import annotations
from mean_field.core.hf.reconstruction import reconstruct_projected_micro_wavefunctions
from ._hf_shared import *  # noqa: F401,F403
from ._hf_types import HFConfig, HFState, WavefunctionBundle
from ._hf_sidecars import _canonical_hf_run_result_sidecar, _write_canonical_hf_array_payload
_REQUIRED_RECONSTRUCTION_ARRAYS = "canonical_run_result.final_state.basis.micro_wavefunctions and final_state.eigenvectors_active"
_CANONICAL_MICRO_AXES = "k,microscopic_basis,active_basis"
def _micro_reconstruction_unavailable(reason: str) -> None:
    raise NotImplementedError("HFResult.reconstruct_micro_wavefunctions requires either a state adapter or canonical dense arrays: " f"{_REQUIRED_RECONSTRUCTION_ARRAYS} must be present and non-empty. {reason}")
def _required_attr(parent: Any, attr: str, *, path: str) -> Any:
    if parent is None or not hasattr(parent, attr): _micro_reconstruction_unavailable(f"{path} is missing.")
    return getattr(parent, attr)
def _required_nonempty_array(value: Any, *, path: str) -> np.ndarray:
    if value is None: _micro_reconstruction_unavailable(f"{path} is missing.")
    array = np.asarray(value)
    if array.size == 0: _micro_reconstruction_unavailable(f"{path} is empty.")
    return array
def _metadata_mapping(value: Any) -> dict[str, object]: return dict(value) if isinstance(value, Mapping) else {}
def _require_canonical_micro_basis(array: np.ndarray, metadata: Mapping[str, object]) -> None:
    if array.ndim != 3: _micro_reconstruction_unavailable(f"canonical fallback requires micro_wavefunctions with rank 3 ({_CANONICAL_MICRO_AXES}), got shape {array.shape}.")
    if str(metadata.get("wavefunctions_axis_order", "")) != _CANONICAL_MICRO_AXES: _micro_reconstruction_unavailable("canonical fallback requires basis.metadata['wavefunctions_axis_order']='k,microscopic_basis,active_basis'.")
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
        if self.canonical_run_result is None:
            _micro_reconstruction_unavailable("canonical_run_result is not attached to this HFResult.")
        final_state = _required_attr(self.canonical_run_result, "final_state", path="canonical_run_result.final_state")
        basis = _required_attr(final_state, "basis", path="canonical_run_result.final_state.basis")
        micro_wavefunctions = _required_nonempty_array(_required_attr(basis, "micro_wavefunctions", path="canonical_run_result.final_state.basis.micro_wavefunctions"), path="canonical_run_result.final_state.basis.micro_wavefunctions")
        eigenvectors_active = _required_nonempty_array(_required_attr(final_state, "eigenvectors_active", path="canonical_run_result.final_state.eigenvectors_active"), path="canonical_run_result.final_state.eigenvectors_active")
        kvec = _required_nonempty_array(_required_attr(basis, "kvec", path="canonical_run_result.final_state.basis.kvec"), path="canonical_run_result.final_state.basis.kvec")
        basis_metadata = _metadata_mapping(getattr(basis, "metadata", None))
        _require_canonical_micro_basis(micro_wavefunctions, basis_metadata)
        basis_metadata["hf_result_reconstruction"] = "canonical_dense_array_fallback"
        reconstructed = reconstruct_projected_micro_wavefunctions(micro_wavefunctions, eigenvectors_active, kvec=kvec, k_grid_frac=getattr(basis, "k_grid_frac", None), basis_metadata=basis_metadata)
        metadata = dict(reconstructed.basis_metadata)
        metadata.update({"source": reconstructed.source, "reconstruction_path": "HFResult.canonical_dense_array_fallback"})
        return WavefunctionBundle(k=reconstructed.kvec, wavefunctions=reconstructed.psi_micro, metadata=metadata, convention=ConventionBundle(density_convention=str(self.config.density_convention), wavefunction_axis_order=str(metadata.get("psi_micro_axis_order", "k,microscopic_basis,hf_state"))))
    def save(
        self,
        output_dir: str | Path,
        *,
        canonical_payload: Literal["metadata_only", "arrays"] = "metadata_only",
    ) -> Path:
        if canonical_payload not in {"metadata_only", "arrays"}:
            raise ValueError(f"Unsupported canonical_payload={canonical_payload!r}; expected 'metadata_only' or 'arrays'")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest_files: dict[str, object] = {}
        manifest_metadata: dict[str, object] = {}
        array_files: tuple[str | Path, ...] = ()
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
            canonical_metadata = {
                "schema_version": sidecar["schema_version"],
                "contract_type": sidecar["contract_type"],
                "state_contract_type": sidecar["final_state"]["contract_type"],
            }
            if canonical_payload == "arrays":
                arrays_path, schema_path, archive_metadata = _write_canonical_hf_array_payload(
                    root,
                    self.canonical_run_result,
                )
                manifest_files["canonical_hf_arrays_schema"] = schema_path.name
                manifest_files["canonical_hf_arrays"] = arrays_path.name
                canonical_metadata.update(
                    {
                        "payload_mode": "arrays_npz",
                        "arrays_key": "canonical_hf_arrays",
                        "arrays_schema_key": "canonical_hf_arrays_schema",
                    }
                )
                manifest_metadata["canonical_hf_archive"] = archive_metadata
                array_files = (arrays_path,)
            manifest_metadata["canonical_hf_run_result"] = canonical_metadata
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
            array_files=array_files,
        )
        return paths["manifest.json"]
__all__ = [name for name in globals() if not name.startswith('__')]
