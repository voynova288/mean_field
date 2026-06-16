# Model API contract

The stable model entrypoint is:

```python
from mean_field.api import make_model, model_record, component_group_records
```

`make_model(system_name, **kwargs)` normalizes public names and aliases, then delegates to existing system model constructors.  It does not move Hamiltonian logic out of system modules.

## Supported public names

Current façade names:

- `htg`
- `rlg_hbn`
- `tdbg`
- `tmbg`
- `atmg`

Each system may keep historical internal class names and paths.  Public workflows should use the façade spelling.

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

`component_groups()` is how systems expose physical labels such as layers, sublattices, valleys, or orbital subsets.  `component_group_records(model)` converts those declarations to JSON-serializable `{name, indices}` records for artifact metadata.  Core HF/analysis code should not infer physical component meanings from array dimensions.

Current component-group status:

- `rlg_hbn`: declares `layer_0`, `layer_1`, ... over the two-sublattice local layer blocks.
- `tmbg`: declares `layer_bottom`, `layer_middle`, and `layer_top` for the local six-orbital block `(A_b, B_b, A_m, B_m, A_t, B_t)`.
- `atmg`: declares `layer_0`, `layer_1`, ... over the two-sublattice local layer blocks.
- `htg`: intentionally left unset while HTG work is owned elsewhere.
- `tdbg`: intentionally left unset until a dedicated adapter labels q-site/sector/layer indices without guessing from array dimensions.

## ModelRecord

`model_record(model)` creates a serializable record for artifacts.  It is intentionally lossy: it captures public summary metadata, not enough data to reconstruct all caches.  Reconstructable workflow inputs belong in `config.yaml` and system-specific result metadata.

## Compatibility rule

If an existing system lacks one of the target protocol methods, do not invent physics in the façade.  Add an explicit system adapter or raise a clear `NotImplementedError`.
