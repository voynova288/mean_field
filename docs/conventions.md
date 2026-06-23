# Mean Field conventions

This document freezes public conventions used by the stable `mean_field.api` façade and by future workflow artifacts.  It is a contract document: changing it requires an explicit migration note and compatibility bridge.

## Units

- Energies in public metadata are in `meV` unless a field explicitly says `ev`.
- Lengths are in `nm`.
- Momenta are in `nm^-1`.
- Complex momenta use `k = k_x + i k_y` with real part along Cartesian `x` and imaginary part along Cartesian `y`.

## Labels

- Valley labels: `K = +1`, `Kprime = -1`.
- Spin labels in metadata: `up = 0`, `down = 1`.
- Systems may keep historical internal flavor orderings, but artifacts must record the ordering in `conventions.json` or result metadata.

## Array axis order

- HF density arrays use `axis_order = "abk"`, shape `(n_state, n_state, n_k)`.
- HF Hamiltonian arrays use the same `abk` convention.
- Public path-band eigenvectors use `k_basis_band` when exported through `BandBundle`.
- System-local arrays with a different order must be converted or explicitly documented before crossing into `mean_field.api` or an artifact manifest.

## Density convention

The canonical definitions live in `mean_field.core.hf.density`.

- `projector`: physical occupied-state projector `P` in stored orientation `P_ab = <c_a^† c_b>`.
- `stored_delta`: stored HF delta `D = P - P_ref`.
- `half_shifted`: the special delta `D = P - 1/2 I`.

Use `DensityBundle.as_projector()` or `density_to_projector(...)` instead of open-coded additions/transposes.  For ket-space wavefunction contractions, request `orientation="ket"`, which transposes the stored matrix axes.

## Gauge and reciprocal-cell conventions

- A model owns its plane-wave basis, reciprocal-cell labels, and boundary sewing convention.
- `analysis.topology` owns the minimal system-independent FHS link/plaquette/Chern formulas and projector QGT/quantum-metric formulas. System or workflow artifacts that use it must record whether wavefunctions are periodic, sewn at the boundary, or represented in a physical Cartesian mBZ; concrete system wrappers remain separate reviewed surfaces. Current concrete wrapper coverage is limited to thin TMBG/TDBG/ATMG/RLG-hBN topology delegation. At the system wrapper/model/grid-result layer, `band_indices` are system/grid band labels (normally absolute Hamiltonian band indices); the common topology adapter maps them to eigenvector columns and records `absolute_band_indices`, `column_indices`, and `grid_result_band_indices` metadata. Topology `orientation_sign` is a `+1/-1` convention only: `-1` conjugates returned FHS links/Berry connection so connection, curvature, and Chern sign stay self-consistent. TMBG/ATMG/TDBG/RLG-hBN thin wrappers own their boundary-sewing choices; TMBG/ATMG use G-index reciprocal-basis relabeling on torus wraps, while system-specific paper validation remains separate from this software/API convention.

## Artifact metadata

Every workflow result should eventually contain:

```text
manifest.json
model.json
config.yaml
conventions.json
environment.json
validation.json
observables.json
```

Large arrays may live in `hf_state.npz`, `bands_path.npz`, or method-specific NPZ files.  The manifest records their relative paths.
