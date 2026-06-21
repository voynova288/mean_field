from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403
from ._hf_types import HFConfig, HFState, WavefunctionBundle
from ._hf_sidecars import _canonical_hf_run_result_sidecar, _write_canonical_hf_array_payload

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
