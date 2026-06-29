# Minimal FHS topology framework

`src/analysis/topology` has one topology computation chain:

```text
FHSState / eigenstate grid
  -> generic boundary sewing from state/basis metadata
  -> normalized Fukui-Hatsugai-Suzuki/Wilson links
  -> Berry plaquette flux
  -> Chern number
```

No projector-QGT finite differences, Fubini-Study metric, paper-target registry,
saved-result validator, wavefunction-layout adapter, or system-side topology
calculator belongs in this surface.

## Public API

```python
from analysis.topology import (
    FHSState,
    BlockSewingSpec,
    fhs_state_from_wavefunctions,
    fhs_state_from_grid_result,
    compute_lattice_topology,
    compute_link_variables,
    berry_curvature_from_links,
    chern_number_from_berry_curvature,
)
```

The intended public call is:

```python
state = fhs_state_from_grid_result(grid, band_indices, basis_sewing=sewing_spec)
result = compute_lattice_topology(state)
```

`compute_lattice_topology(...)` is the only convenience wrapper that returns
links, Berry connection phases, plaquette flux, and Chern number.

## Canonical state and indices

Topology state selection should be expressed in terms of band and flavor labels.
Current code stores this as selected state columns plus metadata:

- `state_indices`: selected columns in the eigenvector state axis;
- `reported_indices`: physical band labels when a grid object exposes
  `grid_result.band_indices`;
- `metadata`: flavor labels such as spin/valley when present.

Systems should converge toward this common state representation rather than
adding system-specific topology APIs.

## Generic boundary sewing

Boundary sewing is common once the eigenstate basis is block-major and labelled.
Use `BlockSewingSpec` to describe the basis:

- `block_coordinates`: integer/fractional reciprocal block coordinates;
- `local_block_size`: number of internal orbitals per block;
- `translations`: the two reciprocal torus seam shifts;
- `block_labels`: optional labels that must match under sewing, e.g. q-site
  sector/flavor embedded in the basis.

`compute_lattice_topology(FHSState(..., basis_sewing=spec))` generates the
actual target-side seam maps in the common layer. System modules should not keep
private seam-transform implementations for topology.

## System rule

System modules may:

- construct Hamiltonian/eigenvector grids;
- expose `fhs_state_*` builders that package the grid into `FHSState`;
- provide basis metadata needed to instantiate `BlockSewingSpec`;
- attach system/band/flavor metadata.

System modules must not:

- export `compute_topology_*` / `topology_on_grid` Chern calculators;
- reimplement normalized links, determinant links, plaquette loops, or Chern
  integration;
- keep system-private topology sewing code once the basis is representable by
  `BlockSewingSpec`.

Current state builders route TMBG, TDBG, RLG-hBN, ATMG, and HTQG states into the
common FHS core. All five system topology adapters use generic `BlockSewingSpec`
metadata for seam sewing; the common topology layer generates the actual seam
maps.

## Berry curvature convention

The array named `berry_curvature` is a plaquette flux in radians. It is not a
continuum density unless a caller explicitly divides by a cell area outside this
minimal module. The Chern number is always

```text
C = sum(berry_curvature) / (2π)
```

## Actual-case validation

Tracked topology tests compute Chern numbers only through
`analysis.topology.compute_lattice_topology(FHSState)`.

- TMBG: Park Fig. 2 current-code checks on `test001` use the state-only path
  `TMBGModel.fhs_state_on_grid(...) -> compute_lattice_topology(state)` with
  generic TMBG basis sewing.  At `theta=1.21`, `n_shells=5`, mesh 9, valley K,
  the observed Chern numbers match the saved paper oracle: δ=0 central pair
  `(326,327)` has `C=-1`; δ=+60 meV valence/conduction have `C=-2,+1`;
  δ=-40 meV valence/conduction have `C=+1,-2`.
- TDBG: AB-BA high-D single-valley conduction band state gives the Liu-2022
  single-valley Chern signs on the sewn torus.
- ATMG: tracked tests exercise an actual reduced chiral L3 central two-band
  subspace through the common pipeline with nonsingular links.  This is a
  reduced code-path validation, not a paper-local ATMG topology claim; L3+
  single-band scans with near-zero link magnitude must not be reported as
  physical Chern numbers.
- RLG-hBN: tracked tests exercise an actual reduced non-HF central pair and the
  physical valley-signed plane-wave block sewing through the common pipeline.
  This is not the Fig. 6 HF paper gate.  The xi1 saved HF sector remains a
  positive control, but xi0 paper-local `C=0` still requires a fresh explicit
  flavor-sector Slurm rerun and current state-only FHS postprocess.
- HTQG: actual Fujimoto-style checkpoint in `tests/test_htqg_model.py` uses
  realistic parameters, shell-6 plane-wave basis, mesh-9 FHS grid, generic
  HTQG basis sewing, and `reports/htqg_fig1_chern_comparison_20260611.md`:
  αβγ K-valley valence `C=-2`, conduction `C=0`.  This validates integrated
  FHS Chern only; paper-local Ω(k) morphology requires a separate target,
  normalization, and convergence check.

## Out of scope

The following are intentionally not part of `analysis.topology`:

- projector-QGT finite-difference curvature;
- Fubini-Study metric / quantum metric;
- paper-normalized Berry/QGT maps;
- saved-result validators and dated paper targets;
- generic wavefunction layout canonicalizers beyond the canonical FHS state;
- system topology adapter factories or system-private Chern calculators.

If one of those capabilities is needed again, restore it as a separate optional
module/package with its own validation gates rather than expanding this minimal
FHS directory.
