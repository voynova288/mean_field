# Model API contract

The stable model entrypoint is:

```python
from mean_field.api import make_model, model_record, component_group_records
```

`make_model(system_name, **kwargs)` normalizes public names and aliases through a small model adapter registry, then delegates to existing system model constructors.  It does not move Hamiltonian logic out of system modules.  Registry helpers are available as `list_model_adapters()`, `get_model_adapter_info(name)`, and `resolve_model_adapter(name)`.

## Supported public names

Current façade names in the tracked core profile:

- `htg`
- `rlg_hbn`
- `tbg` (`variant="zero_field_bm"` only; BM single-particle bands)
- `tdbg`
- `tmbg`

ATMG and HTQG are archived optional exploratory systems under `local_archive/optional_features/` for the 35k core-profile cleanup. Public workflows should use the façade spelling for retained systems.

## ContinuumModel protocol

A system model should converge toward this public shape:

```python
class ContinuumModel(Protocol):
    system_name: str
    params: object
    lattice: object

    @property
    def matrix_dim(self) -> int: ...

    def build_hamiltonian(self, k_tilde: complex, **kwargs) -> np.ndarray: ...
    def diagonalize(self, k_tilde: complex, **kwargs) -> tuple[np.ndarray, np.ndarray | None]: ...
    def lattice_summary(self) -> dict[str, object]: ...
    def component_groups(self) -> tuple[ComponentGroup, ...]: ...
```

`component_groups()` is how systems expose physical labels such as layers, sublattices, valleys, or orbital subsets.  `component_group_records(model)` converts those declarations to JSON-serializable `{name, indices}` records for artifact metadata, preserving optional `index_space` and `description` fields when a system needs to document a non-local basis convention.  Core HF/analysis code should not infer physical component meanings from array dimensions.

Current component-group status:

- `rlg_hbn`: declares `layer_0`, `layer_1`, ... over the two-sublattice local layer blocks.
- `tmbg`: declares `layer_bottom`, `layer_middle`, and `layer_top` for the local six-orbital block `(A_b, B_b, A_m, B_m, A_t, B_t)`.
- `htg`: intentionally left unset while HTG work is owned elsewhere.
- `tdbg`: declares sector/layer/sublattice groups through a dedicated adapter. Public model records use the full q-site-major Hamiltonian-basis index `4*q_site + alpha` with `alpha=(A1,B1,A2,B2)` and carry `index_space="tdbg_full_hamiltonian_basis"`; projected-HF overlap helpers use a separate embedded eight-component local basis `4*sector + alpha`.

## Optional topology convenience

Models with reviewed thin topology wrappers may expose:

```python
def topology_on_grid(mesh_size: int, band_indices, **kwargs) -> analysis.topology.TopologyResult: ...
```

This is an optional convenience, not a required `ContinuumModel` protocol method. Current tracked coverage is TMBG, TDBG, RLG-hBN, and HTG. At the model/wrapper/grid-result layer, `band_indices` means the system/grid-result band labels, normally absolute Hamiltonian band indices, and the common topology adapter maps them to returned eigenvector columns. HTG `topology_on_grid(...)` requests the contiguous absolute band window needed by its scipy diagonalizer, then lets the common grid-result adapter select the requested labels. Only the low-level `compute_topology_from_eigenvectors(...)` entrypoint uses raw eigenvector-column indices.

`topology_on_grid(...)` diagonalizes a 2D grid with eigenvectors and is therefore a numerical job for realistic meshes. Use `endpoint=False`; paper-level topology validation still requires explicit provenance and Slurm-scale validation.

## ModelRecord

`model_record(model)` creates a serializable record for artifacts.  It is intentionally lossy: it captures public summary metadata, not enough data to reconstruct all caches.  Reconstructable workflow inputs belong in `config.yaml` and system-specific result metadata.

## Compatibility rule

If an existing system lacks one of the target protocol methods, do not invent physics in the façade.  Add an explicit system adapter or raise a clear `NotImplementedError`.
