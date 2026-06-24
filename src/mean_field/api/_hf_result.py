from __future__ import annotations
from collections.abc import Iterable as _Iterable
import inspect as _inspect
from mean_field.core.hf.reconstruction import reconstruct_projected_micro_wavefunctions
from ._hf_shared import *  # noqa: F401,F403
from ._hf_types import HFConfig, HFState, WavefunctionBundle
from ._hf_sidecars import _canonical_hf_run_result_sidecar, _write_canonical_hf_array_payload
_REQUIRED_RECONSTRUCTION_ARRAYS = "canonical_run_result.final_state.basis.micro_wavefunctions and final_state.eigenvectors_active"
_CANONICAL_MICRO_AXES = "k,microscopic_basis,active_basis"
_CANONICAL_RECONSTRUCTION_UNITARITY_ATOL = 1.0e-8
_MAX_DENSE_ELEMENTS_UNSET = object()
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

def _requested_reconstruction_kwargs(
    *,
    state_indices: int | _Iterable[int] | None,
    band_indices: int | _Iterable[int] | None,
    max_dense_elements: object,
) -> dict[str, object]:
    if state_indices is not None and band_indices is not None:
        raise ValueError("Pass only one of state_indices or band_indices for HFResult.reconstruct_micro_wavefunctions")
    kwargs: dict[str, object] = {}
    if state_indices is not None:
        kwargs["state_indices"] = state_indices
    if band_indices is not None:
        kwargs["band_indices"] = band_indices
    if max_dense_elements is not _MAX_DENSE_ELEMENTS_UNSET:
        kwargs["max_dense_elements"] = max_dense_elements
    return kwargs

def _call_state_reconstruction_adapter(adapter: Any, kwargs: Mapping[str, object]) -> Any:
    if not kwargs:
        return adapter()
    try:
        signature = _inspect.signature(adapter)
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            "HFResult state adapter signature is not inspectable; refusing to ignore selected-state "
            "reconstruction keywords or guess a canonical fallback while a state adapter is attached."
        ) from exc
    parameters = signature.parameters
    accepts_var_keywords = any(param.kind == _inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    unsupported: list[str] = []
    for name in kwargs:
        parameter = parameters.get(name)
        if parameter is None:
            if not accepts_var_keywords:
                unsupported.append(name)
        elif parameter.kind == _inspect.Parameter.POSITIONAL_ONLY:
            unsupported.append(name)
    if unsupported:
        raise NotImplementedError(
            "HFResult state adapter does not support selected-state reconstruction keyword(s): "
            f"{', '.join(unsupported)}. Refusing to ignore them or guess a canonical fallback while "
            "a state adapter is attached."
        )
    return adapter(**dict(kwargs))

def _normalize_canonical_reconstruction_state_indices(
    *,
    state_indices: int | _Iterable[int] | None,
    band_indices: int | _Iterable[int] | None,
    n_state: int,
) -> tuple[tuple[int, ...], str]:
    if state_indices is not None and band_indices is not None:
        raise ValueError("Pass only one of state_indices or band_indices for HFResult.reconstruct_micro_wavefunctions")
    source = "all"
    raw_indices: int | _Iterable[int] | None = None
    if state_indices is not None:
        source = "state_indices"
        raw_indices = state_indices
    elif band_indices is not None:
        source = "band_indices"
        raw_indices = band_indices
    if raw_indices is None:
        selected = tuple(range(int(n_state)))
    elif isinstance(raw_indices, (str, bytes)):
        raise TypeError("state_indices/band_indices must be an integer, an iterable of integers, or None")
    elif isinstance(raw_indices, (int, np.integer)):
        selected = (int(raw_indices),)
    else:
        selected = tuple(int(index) for index in raw_indices)
    if not selected:
        raise ValueError("HFResult canonical reconstruction requires at least one selected HF state")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate HFResult reconstruction state indices {selected}")
    invalid = [int(index) for index in selected if int(index) < 0 or int(index) >= int(n_state)]
    if invalid:
        raise ValueError(f"HFResult reconstruction state indices {invalid} are outside [0, {int(n_state)})")
    return selected, source

def _validate_canonical_reconstruction_size(
    *,
    n_k: int,
    microscopic_basis_dim: int,
    n_selected: int,
    max_dense_elements: object,
) -> int:
    dense_elements = int(n_k) * int(microscopic_basis_dim) * int(n_selected)
    if max_dense_elements is _MAX_DENSE_ELEMENTS_UNSET or max_dense_elements is None:
        return dense_elements
    max_elements = int(max_dense_elements)
    if max_elements < 0:
        raise ValueError("max_dense_elements must be non-negative or None")
    if dense_elements > max_elements:
        raise ValueError(
            "HFResult canonical dense-array fallback would exceed the explicit size guard: "
            f"estimated {dense_elements} complex output elements for {int(n_selected)} selected HF states "
            f"> max_dense_elements={max_elements}. Pass selected state_indices/band_indices or increase "
            "max_dense_elements only for an intentional reconstruction call."
        )
    return dense_elements

def _selected_active_eigenvector_unitarity_residual(coeffs: np.ndarray, selected: tuple[int, ...]) -> float:
    selected_coeffs = coeffs[:, np.asarray(selected, dtype=int), :]
    gram = np.einsum("ahk,amk->hmk", selected_coeffs.conjugate(), selected_coeffs, optimize=True)
    identity = np.eye(len(selected), dtype=np.complex128)[:, :, None]
    return float(np.max(np.abs(gram - identity))) if gram.size else 0.0

def _canonical_fallback_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    out = dict(metadata)
    out["hf_result_reconstruction"] = "canonical_dense_array_fallback"
    out["topology_eligible"] = False
    out.setdefault("topology_ineligible_reason", "HFResult canonical dense-array fallback is algebraic only; no system sewing/grid topology adapter is attached")
    raw_paths = out.get("evidence_paths", ())
    if isinstance(raw_paths, (str, bytes)):
        evidence_paths: list[object] = [raw_paths]
    else:
        try:
            evidence_paths = list(raw_paths)  # type: ignore[arg-type]
        except TypeError:
            evidence_paths = []
    for path in ("src/mean_field/api/_hf_result.py", "src/mean_field/core/hf/reconstruction.py"):
        if path not in evidence_paths:
            evidence_paths.append(path)
    out["evidence_paths"] = evidence_paths
    out.setdefault(
        "uncertainty",
        "Canonical dense-array fallback performs algebraic projected-basis contraction only; "
        "system-specific sewing/topology eligibility is not inferred by HFResult.",
    )
    return out
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
    def reconstruct_micro_wavefunctions(
        self,
        *,
        state_indices: int | _Iterable[int] | None = None,
        band_indices: int | _Iterable[int] | None = None,
        max_dense_elements: object = _MAX_DENSE_ELEMENTS_UNSET,
    ) -> WavefunctionBundle:
        requested_kwargs = _requested_reconstruction_kwargs(
            state_indices=state_indices,
            band_indices=band_indices,
            max_dense_elements=max_dense_elements,
        )
        if hasattr(self.state, "reconstruct_micro_wavefunctions"):
            return _call_state_reconstruction_adapter(self.state.reconstruct_micro_wavefunctions, requested_kwargs)
        if self.canonical_run_result is None:
            _micro_reconstruction_unavailable("canonical_run_result is not attached to this HFResult.")
        final_state = _required_attr(self.canonical_run_result, "final_state", path="canonical_run_result.final_state")
        basis = _required_attr(final_state, "basis", path="canonical_run_result.final_state.basis")
        micro_wavefunctions = _required_nonempty_array(_required_attr(basis, "micro_wavefunctions", path="canonical_run_result.final_state.basis.micro_wavefunctions"), path="canonical_run_result.final_state.basis.micro_wavefunctions")
        eigenvectors_active = _required_nonempty_array(_required_attr(final_state, "eigenvectors_active", path="canonical_run_result.final_state.eigenvectors_active"), path="canonical_run_result.final_state.eigenvectors_active")
        kvec = _required_nonempty_array(_required_attr(basis, "kvec", path="canonical_run_result.final_state.basis.kvec"), path="canonical_run_result.final_state.basis.kvec")
        basis_metadata = _canonical_fallback_metadata(_metadata_mapping(getattr(basis, "metadata", None)))
        _require_canonical_micro_basis(micro_wavefunctions, basis_metadata)
        micro_basis = np.asarray(micro_wavefunctions, dtype=np.complex128)
        coeffs = np.asarray(eigenvectors_active, dtype=np.complex128)
        n_k, microscopic_basis_dim, n_active = (int(value) for value in micro_basis.shape)
        if coeffs.shape != (n_active, n_active, n_k):
            raise ValueError(f"canonical fallback active_eigenvectors must have shape ({n_active}, {n_active}, {n_k}), got {coeffs.shape}")
        kvec_arr = np.asarray(kvec, dtype=np.complex128).reshape(-1)
        if kvec_arr.shape != (n_k,):
            raise ValueError(f"kvec must have shape ({n_k},), got {kvec_arr.shape}")
        k_grid_frac = getattr(basis, "k_grid_frac", None)
        if k_grid_frac is not None and np.asarray(k_grid_frac, dtype=float).shape != (n_k, 2):
            raise ValueError(f"k_grid_frac must have shape ({n_k}, 2), got {np.asarray(k_grid_frac).shape}")
        selected, selection_source = _normalize_canonical_reconstruction_state_indices(
            state_indices=state_indices,
            band_indices=band_indices,
            n_state=n_active,
        )
        dense_elements = _validate_canonical_reconstruction_size(
            n_k=n_k,
            microscopic_basis_dim=microscopic_basis_dim,
            n_selected=len(selected),
            max_dense_elements=max_dense_elements,
        )
        if selected == tuple(range(n_active)):
            reconstructed = reconstruct_projected_micro_wavefunctions(
                micro_basis,
                coeffs,
                kvec=kvec_arr,
                k_grid_frac=k_grid_frac,
                basis_metadata=basis_metadata,
                unitarity_atol=_CANONICAL_RECONSTRUCTION_UNITARITY_ATOL,
            )
            metadata = dict(reconstructed.basis_metadata)
            psi_micro = reconstructed.psi_micro
            source = reconstructed.source
        else:
            residual = _selected_active_eigenvector_unitarity_residual(coeffs, selected)
            if residual > _CANONICAL_RECONSTRUCTION_UNITARITY_ATOL:
                raise ValueError(
                    "canonical fallback selected active_eigenvectors must be unitary at each k point; "
                    f"max column-Gram residual {residual:.6e} exceeds {_CANONICAL_RECONSTRUCTION_UNITARITY_ATOL:.6e}"
                )
            selected_array = np.asarray(selected, dtype=int)
            psi_micro = np.einsum("kba,ahk->kbh", micro_basis, coeffs[:, selected_array, :], optimize=True)
            metadata = dict(basis_metadata)
            metadata.update(
                {
                    "micro_basis_axis_order": "k,microscopic_basis,active_basis",
                    "input_micro_basis_axes": {"k_axis": 0, "microscopic_basis_axis": 1, "active_axis": 2},
                    "active_eigenvectors_axis_order": "active_basis,hf_state,k",
                    "psi_micro_axis_order": "k,microscopic_basis,hf_state",
                    "n_k": n_k,
                    "microscopic_basis_dim": microscopic_basis_dim,
                    "n_active": n_active,
                    "state_labels": tuple({"hf_state_index": int(index)} for index in selected),
                    "kvec_provided": True,
                    "active_eigenvectors_unitarity_residual": residual,
                }
            )
            if k_grid_frac is not None:
                metadata["k_grid_frac_shape"] = [n_k, 2]
            source = "hf_reconstructed"
        metadata.update(
            {
                "source": source,
                "reconstruction_path": "HFResult.canonical_dense_array_fallback",
                "projected_hf_reconstruction": "explicit_selected_dense_opt_in",
                "dense_reconstruction_estimated_elements": int(dense_elements),
                "dense_reconstruction_size_policy": "counts selected output psi_micro elements; all-state psi_micro is materialized only when all HF states are selected",
                "max_dense_elements": None if max_dense_elements is _MAX_DENSE_ELEMENTS_UNSET or max_dense_elements is None else int(max_dense_elements),
                "selection_argument": selection_source,
                "selected_hf_state_indices": [int(index) for index in selected],
                "selected_hf_band_indices": [int(index) for index in selected],
                "band_indices_argument_meaning": "HF eigenstate indices/columns of canonical_run_result.final_state.eigenvectors_active, not system noninteracting band labels",
                "all_hf_state_count": int(n_active),
                "n_reconstructed_states": int(len(selected)),
            }
        )
        return WavefunctionBundle(k=kvec_arr, wavefunctions=psi_micro, metadata=metadata, convention=ConventionBundle(density_convention=str(self.config.density_convention), wavefunction_axis_order=str(metadata.get("psi_micro_axis_order", "k,microscopic_basis,hf_state"))))
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
