from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403


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
            "fields": sorted({str(key) for row in iteration_history for key in row}),
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


_CANONICAL_HF_RUN_RESULT_CONTRACT = "mean_field.core.contracts.HFRunResult"
_CANONICAL_SINGLE_PARTICLE_MODEL_CONTRACT = "mean_field.core.contracts.SingleParticleModel"
_CANONICAL_ARRAYS_FILE = "canonical_hf_arrays.npz"
_CANONICAL_ARRAYS_SCHEMA_FILE = "canonical_hf_arrays.schema.json"


def _json_safe_payload(value: object, *, path: str) -> object:
    """Return a strict-JSON payload for the opt-in canonical array schema.

    Unlike the metadata sidecar helper, this function rejects ndarrays instead
    of summarizing them: dense arrays belong exclusively in the NPZ payload.
    """

    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        raise TypeError(f"Dense arrays are not allowed in canonical HF JSON schema at {path}")
    if isinstance(value, np.complexfloating) or isinstance(value, complex):
        raise TypeError(f"Complex scalar is not allowed in canonical HF JSON schema at {path}")
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
            str(key): _json_safe_payload(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [
            _json_safe_payload(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"Object of type {type(value).__name__} is not allowed in canonical HF JSON schema at {path}")


def _schema_array_entry(array: np.ndarray, *, key: str) -> dict[str, object]:
    arr = np.asarray(array)
    return {"key": key, "shape": [int(axis) for axis in arr.shape], "dtype": str(arr.dtype)}


def _schema_alias_entry(array: np.ndarray, *, alias_of: str) -> dict[str, object]:
    arr = np.asarray(array)
    return {"alias_of": alias_of, "shape": [int(axis) for axis in arr.shape], "dtype": str(arr.dtype)}


def _add_canonical_array(
    arrays: dict[str, np.ndarray],
    schema_arrays: dict[str, dict[str, object]],
    path: str,
    key: str,
    value: object,
) -> None:
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        raise TypeError(f"Canonical HF array {path!r} has object dtype and cannot be archived without pickle")
    arrays[key] = arr
    schema_arrays[path] = _schema_array_entry(arr, key=key)


def _single_particle_model_schema(model: Any, *, role: str) -> dict[str, object]:
    return {
        "contract_type": _CANONICAL_SINGLE_PARTICLE_MODEL_CONTRACT,
        "system": str(model.system),
        "lattice": _json_safe_payload(model.lattice, path=f"basis.{role}.lattice"),
        "params": _json_safe_payload(model.params, path=f"basis.{role}.params"),
        "metadata": _json_safe_payload(dict(model.metadata), path=f"basis.{role}.metadata"),
        "callables": "unavailable_without_resolver",
    }


def _canonical_hf_array_payload(canonical_run_result: Any) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    final_state = canonical_run_result.final_state
    basis = final_state.basis
    density = final_state.density
    reference = density.reference
    hamiltonian = final_state.hamiltonian

    basis_h0 = np.asarray(basis.h0)
    hamiltonian_h0 = np.asarray(hamiltonian.h0)
    if not np.array_equal(basis_h0, hamiltonian_h0):
        raise ValueError(
            "Canonical HF dense-array payload requires final_state.basis.h0 and "
            "final_state.hamiltonian.h0 to be identical for aliasing"
        )

    arrays: dict[str, np.ndarray] = {}
    schema_arrays: dict[str, dict[str, object]] = {}
    _add_canonical_array(arrays, schema_arrays, "final_state.basis.kvec", "basis__kvec", basis.kvec)
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.basis.k_grid_frac",
        "basis__k_grid_frac",
        basis.k_grid_frac,
    )
    _add_canonical_array(arrays, schema_arrays, "final_state.basis.h0", "basis__h0", basis_h0)
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.basis.basis_energies",
        "basis__basis_energies",
        basis.basis_energies,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.basis.micro_wavefunctions",
        "basis__micro_wavefunctions",
        basis.micro_wavefunctions,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.density.density_delta",
        "density__density_delta",
        density.density_delta,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.density.reference.reference",
        "density__reference",
        reference.reference,
    )
    schema_arrays["final_state.hamiltonian.h0"] = _schema_alias_entry(
        basis_h0,
        alias_of="final_state.basis.h0",
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.hamiltonian.fixed",
        "hamiltonian__fixed",
        hamiltonian.fixed,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.hamiltonian.hartree",
        "hamiltonian__hartree",
        hamiltonian.hartree,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.hamiltonian.fock",
        "hamiltonian__fock",
        hamiltonian.fock,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.hamiltonian.total",
        "hamiltonian__total",
        hamiltonian.total,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.energies",
        "final_state__energies",
        final_state.energies,
    )
    _add_canonical_array(
        arrays,
        schema_arrays,
        "final_state.eigenvectors_active",
        "final_state__eigenvectors_active",
        final_state.eigenvectors_active,
    )

    schema: dict[str, object] = {
        "schema_version": 1,
        "payload_version": 1,
        "contract_type": _CANONICAL_HF_RUN_RESULT_CONTRACT,
        "arrays_file": _CANONICAL_ARRAYS_FILE,
        "array_storage": {"format": "npz", "allow_pickle": False, "compressed": False},
        "scalars": {
            "run": {
                "converged": bool(canonical_run_result.converged),
                "exit_reason": str(canonical_run_result.exit_reason),
                "best_seed": int(canonical_run_result.best_seed),
                "init_mode": str(canonical_run_result.init_mode),
                "iteration_history": _json_safe_payload(
                    list(canonical_run_result.iteration_history),
                    path="run.iteration_history",
                ),
                "archive_manifest": _json_safe_payload(
                    dict(canonical_run_result.archive_manifest),
                    path="run.archive_manifest",
                ),
            },
            "final_state": {
                "mu": _finite_float(final_state.mu, path="final_state.mu"),
                "observables": _json_safe_payload(dict(final_state.observables), path="final_state.observables"),
                "diagnostics": _json_safe_payload(dict(final_state.diagnostics), path="final_state.diagnostics"),
            },
            "basis": {
                "physical_model": _single_particle_model_schema(basis.physical_model, role="physical_model"),
                "basis_model": _single_particle_model_schema(basis.basis_model, role="basis_model"),
                "active_band_indices": [int(index) for index in basis.active_band_indices],
                "active_valence_bands": int(basis.active_valence_bands),
                "active_conduction_bands": int(basis.active_conduction_bands),
                "flavor_labels": _json_safe_payload(list(basis.flavor_labels), path="basis.flavor_labels"),
                "band_labels": _json_safe_payload(list(basis.band_labels), path="basis.band_labels"),
                "metadata": _json_safe_payload(dict(basis.metadata), path="basis.metadata"),
            },
            "density": {
                "convention": str(density.convention),
                "filling": _finite_float(density.filling, path="density.filling"),
                "n_occupied_total": int(density.n_occupied_total),
                "metadata": _json_safe_payload(dict(density.metadata), path="density.metadata"),
                "reference_scheme": str(reference.scheme),
                "reference_metadata": _json_safe_payload(dict(reference.metadata), path="density.reference.metadata"),
            },
            "hamiltonian": {
                "density_input_convention": str(hamiltonian.density_input_convention),
                "metadata": _json_safe_payload(dict(hamiltonian.metadata), path="hamiltonian.metadata"),
            },
        },
        "arrays": schema_arrays,
    }
    return arrays, schema


def _write_canonical_hf_array_payload(root: Path, canonical_run_result: Any) -> tuple[Path, Path, dict[str, object]]:
    arrays, schema = _canonical_hf_array_payload(canonical_run_result)
    arrays_path = write_npz_artifact(arrays, root / _CANONICAL_ARRAYS_FILE)
    schema_path = write_json_artifact(schema, root / _CANONICAL_ARRAYS_SCHEMA_FILE)
    metadata = {
        "schema_version": 1,
        "contract_type": _CANONICAL_HF_RUN_RESULT_CONTRACT,
        "payload_version": int(schema["payload_version"]),
        "sidecar_key": "canonical_hf_run_result",
        "arrays_key": "canonical_hf_arrays",
        "arrays_schema_key": "canonical_hf_arrays_schema",
        "large_array_policy": "npz_only_no_dense_arrays_in_json",
        "loader": "mean_field.api.reconstruct_canonical_hf_run_result",
    }
    return arrays_path, schema_path, metadata


def _reject_schema_json_constant(token: str) -> None:
    raise ValueError(f"Non-standard JSON numeric token is not allowed in canonical HF schema: {token}")


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_schema_json_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _result_artifact_path(result: ResultDirectory, key: str) -> Path:
    files = result.manifest.get("files", {}) if isinstance(result.manifest, Mapping) else {}
    if not isinstance(files, Mapping) or key not in files:
        raise ValueError(
            "Canonical HF array payload unavailable (metadata-only archive): "
            f"manifest is missing {key!r}"
        )
    raw_path = files[key]
    if not isinstance(raw_path, str):
        raise ValueError(f"Manifest file entry {key!r} must be a relative path string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest file entry {key!r} must stay within the result root: {raw_path!r}")
    root = result.root
    resolved_root = root.resolve()
    resolved_path = (root / relative).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Manifest file entry {key!r} escapes the result root: {raw_path!r}") from exc
    if not resolved_path.exists():
        raise FileNotFoundError(f"Manifest references missing canonical HF array payload file {key!r}: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Manifest canonical HF array payload entry {key!r} is not a file: {resolved_path}")
    return resolved_path


def _require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at {path}")
    return value


def _validate_canonical_array_schema_header(schema: Mapping[str, Any], schema_path: Path) -> None:
    if schema.get("schema_version") != 1:
        raise ValueError(f"Invalid canonical HF array schema_version in {schema_path}")
    if schema.get("payload_version") != 1:
        raise ValueError(f"Invalid canonical HF array payload_version in {schema_path}")
    if schema.get("contract_type") != _CANONICAL_HF_RUN_RESULT_CONTRACT:
        raise ValueError(f"Invalid canonical HF array schema contract_type in {schema_path}")
    storage = _require_mapping(schema.get("array_storage"), path="schema.array_storage")
    if storage.get("format") != "npz" or storage.get("allow_pickle") is not False:
        raise ValueError(f"Canonical HF array schema must declare npz storage with allow_pickle=false in {schema_path}")


def _validate_array_against_schema(array: np.ndarray, spec: Mapping[str, Any], *, path: str) -> None:
    if "shape" in spec:
        shape_value = spec["shape"]
        if not isinstance(shape_value, list | tuple):
            raise ValueError(f"Schema array entry {path!r} has invalid shape declaration")
        declared_shape = tuple(int(axis) for axis in shape_value)
        if array.shape != declared_shape:
            raise ValueError(
                f"Schema/NPZ mismatch for {path}: shape {array.shape} does not match declared {declared_shape}"
            )
    if "dtype" in spec:
        declared_dtype = str(np.dtype(str(spec["dtype"])))
        if str(array.dtype) != declared_dtype:
            raise ValueError(
                f"Schema/NPZ mismatch for {path}: dtype {array.dtype} does not match declared {declared_dtype}"
            )


def _load_canonical_schema_arrays(
    schema: Mapping[str, Any],
    payload: Any,
    *,
    validate: bool,
) -> dict[str, np.ndarray]:
    specs = _require_mapping(schema.get("arrays"), path="schema.arrays")
    loaded: dict[str, np.ndarray] = {}

    def resolve(path: str, stack: tuple[str, ...] = ()) -> np.ndarray:
        if path in loaded:
            return loaded[path]
        if path in stack:
            raise ValueError(f"Canonical HF array schema has an alias cycle at {path!r}")
        spec = _require_mapping(specs.get(path), path=f"schema.arrays.{path}")
        alias_of = spec.get("alias_of")
        if alias_of is not None:
            if not isinstance(alias_of, str):
                raise ValueError(f"Canonical HF array alias for {path!r} must be a string")
            if alias_of not in specs:
                raise ValueError(f"Canonical HF array alias for {path!r} points to missing {alias_of!r}")
            array = resolve(alias_of, stack + (path,))
            if validate:
                _validate_array_against_schema(array, spec, path=path)
            loaded[path] = array
            return array

        key = spec.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Canonical HF array schema entry {path!r} must declare a non-empty NPZ key")
        if key not in payload.files:
            raise ValueError(f"Canonical HF NPZ payload is missing key {key!r} for schema entry {path!r}")
        array = np.asarray(payload[key])
        if validate:
            _validate_array_against_schema(array, spec, path=path)
        loaded[path] = array
        return array

    for path in specs:
        resolve(str(path))
    return loaded


def _placeholder_single_particle_model(record: Mapping[str, Any], *, role: str) -> ContractSingleParticleModel:
    def hamiltonian_builder(_kvec: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            f"SingleParticleModel {role} was reconstructed from a canonical HF dense-array payload "
            "without model_resolver; hamiltonian_builder is unavailable"
        )

    def diagonalizer(_kvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError(
            f"SingleParticleModel {role} was reconstructed from a canonical HF dense-array payload "
            "without model_resolver; diagonalizer is unavailable"
        )

    metadata = record.get("metadata", {})
    return ContractSingleParticleModel(
        system=str(record.get("system", "")),
        lattice=record.get("lattice"),
        params=record.get("params", {}),
        hamiltonian_builder=hamiltonian_builder,
        diagonalizer=diagonalizer,
        metadata=dict(_require_mapping(metadata, path=f"schema.scalars.basis.{role}.metadata")),
    )


def _resolve_single_particle_model(
    record: Mapping[str, Any],
    *,
    role: str,
    model_resolver: Callable[[dict[str, Any], str], ContractSingleParticleModel] | None,
) -> ContractSingleParticleModel:
    if model_resolver is not None:
        model = model_resolver(dict(record), role)
        if not isinstance(model, ContractSingleParticleModel):
            raise TypeError(
                f"model_resolver returned {type(model).__name__} for {role}; "
                "expected mean_field.core.contracts.SingleParticleModel"
            )
        return model
    return _placeholder_single_particle_model(record, role=role)


def _list_of_mappings(value: object, *, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"Expected list at {path}")
    return [dict(_require_mapping(item, path=f"{path}[{index}]")) for index, item in enumerate(value)]


def reconstruct_canonical_hf_run_result(
    path_or_result: str | Path | ResultDirectory,
    *,
    model_resolver: Callable[[dict[str, Any], str], ContractSingleParticleModel] | None = None,
    validate: bool = True,
) -> ContractHFRunResult:
    """Reconstruct a canonical HF run result from the opt-in dense-array payload.

    ``load_result(...)`` intentionally stays metadata-only.  This helper is the
    explicit opt-in path that opens ``canonical_hf_arrays.npz`` with
    ``allow_pickle=False`` and rebuilds the structural dataclasses.
    """

    result = path_or_result if isinstance(path_or_result, ResultDirectory) else load_result(path_or_result)
    schema_path = _result_artifact_path(result, "canonical_hf_arrays_schema")
    arrays_path = _result_artifact_path(result, "canonical_hf_arrays")
    schema = _read_json_object(schema_path)
    if validate:
        _validate_canonical_array_schema_header(schema, schema_path)
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = _load_canonical_schema_arrays(schema, payload, validate=validate)

    scalars = _require_mapping(schema.get("scalars"), path="schema.scalars")
    run_scalars = _require_mapping(scalars.get("run"), path="schema.scalars.run")
    final_state_scalars = _require_mapping(scalars.get("final_state"), path="schema.scalars.final_state")
    basis_scalars = _require_mapping(scalars.get("basis"), path="schema.scalars.basis")
    density_scalars = _require_mapping(scalars.get("density"), path="schema.scalars.density")
    hamiltonian_scalars = _require_mapping(scalars.get("hamiltonian"), path="schema.scalars.hamiltonian")

    physical_model = _resolve_single_particle_model(
        _require_mapping(basis_scalars.get("physical_model"), path="schema.scalars.basis.physical_model"),
        role="physical_model",
        model_resolver=model_resolver,
    )
    basis_model = _resolve_single_particle_model(
        _require_mapping(basis_scalars.get("basis_model"), path="schema.scalars.basis.basis_model"),
        role="basis_model",
        model_resolver=model_resolver,
    )
    basis = ContractProjectedBasis(
        physical_model=physical_model,
        basis_model=basis_model,
        kvec=arrays["final_state.basis.kvec"],
        k_grid_frac=arrays["final_state.basis.k_grid_frac"],
        h0=arrays["final_state.basis.h0"],
        basis_energies=arrays["final_state.basis.basis_energies"],
        active_band_indices=tuple(int(index) for index in basis_scalars.get("active_band_indices", ())),
        active_valence_bands=int(basis_scalars.get("active_valence_bands", 0)),
        active_conduction_bands=int(basis_scalars.get("active_conduction_bands", 0)),
        micro_wavefunctions=arrays["final_state.basis.micro_wavefunctions"],
        flavor_labels=tuple(basis_scalars.get("flavor_labels", ())),
        band_labels=tuple(basis_scalars.get("band_labels", ())),
        metadata=dict(_require_mapping(basis_scalars.get("metadata", {}), path="schema.scalars.basis.metadata")),
    )
    reference = ContractReferenceDensity(
        scheme=str(density_scalars.get("reference_scheme", "custom")),
        reference=arrays["final_state.density.reference.reference"],
        metadata=dict(
            _require_mapping(
                density_scalars.get("reference_metadata", {}),
                path="schema.scalars.density.reference_metadata",
            )
        ),
    )
    density = ContractDensityState(
        density_delta=arrays["final_state.density.density_delta"],
        reference=reference,
        filling=float(density_scalars.get("filling", 0.0)),
        n_occupied_total=int(density_scalars.get("n_occupied_total", 0)),
        convention=str(density_scalars.get("convention", "delta")),
        metadata=dict(_require_mapping(density_scalars.get("metadata", {}), path="schema.scalars.density.metadata")),
    )
    hamiltonian = ContractHamiltonianParts(
        h0=arrays["final_state.hamiltonian.h0"],
        fixed=arrays["final_state.hamiltonian.fixed"],
        hartree=arrays["final_state.hamiltonian.hartree"],
        fock=arrays["final_state.hamiltonian.fock"],
        total=arrays["final_state.hamiltonian.total"],
        density_input_convention=str(hamiltonian_scalars.get("density_input_convention", "delta")),
        metadata=dict(
            _require_mapping(hamiltonian_scalars.get("metadata", {}), path="schema.scalars.hamiltonian.metadata")
        ),
    )
    final_state = ContractHFState(
        basis=basis,
        density=density,
        hamiltonian=hamiltonian,
        energies=arrays["final_state.energies"],
        eigenvectors_active=arrays["final_state.eigenvectors_active"],
        mu=float(final_state_scalars.get("mu", 0.0)),
        observables=dict(_require_mapping(final_state_scalars.get("observables", {}), path="schema.scalars.final_state.observables")),
        diagnostics=dict(_require_mapping(final_state_scalars.get("diagnostics", {}), path="schema.scalars.final_state.diagnostics")),
    )
    return ContractHFRunResult(
        final_state=final_state,
        iteration_history=_list_of_mappings(run_scalars.get("iteration_history", []), path="schema.scalars.run.iteration_history"),
        converged=bool(run_scalars.get("converged", False)),
        exit_reason=str(run_scalars.get("exit_reason", "")),
        best_seed=int(run_scalars.get("best_seed", 0)),
        init_mode=str(run_scalars.get("init_mode", "")),
        archive_manifest=dict(
            _require_mapping(run_scalars.get("archive_manifest", {}), path="schema.scalars.run.archive_manifest")
        ),
    )

__all__ = [name for name in globals() if not name.startswith('__')]
