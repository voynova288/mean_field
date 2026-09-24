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
    link_variable_from_overlap_matrix,
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
links, Berry connection phases, plaquette flux, and Chern number. It accepts an
`FHSState` only; callers must not bypass state/basis metadata with raw-array
arguments.

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
private seam-transform implementations for topology. For a finite reciprocal
cutoff, missing translated blocks are zero-filled; every resulting link must
remain nonsingular, and cutoff convergence is required before physical use.

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
common FHS core. TMBG, ATMG, TDBG, RLG-hBN, and HTQG use generic
`BlockSewingSpec` for seam sewing.

## Physical overlap adapters

When endpoint states live in a non-Euclidean or k-dependent basis, the owning
physical adapter must assemble the endpoint overlap with its validated metric or
operator convention.  It then passes the square overlap matrix to
`link_variable_from_overlap_matrix(...)`, which uses the same common single-band
or polar/determinant FHS normalization as `compute_link_variables(...)`.
Scalar-zero, singular-subspace, and non-finite overlaps fail closed before any
plaquette flux is formed. No identity regularizer is added because
it would violate independent endpoint frame covariance. Plaquette flux and
Chern integration remain in this common module.

## Berry curvature convention

The array named `berry_curvature` is a principal-branch plaquette flux in
`(-π, π]`, measured in radians. It is not a continuum density unless a caller
explicitly divides by a cell area outside this minimal module. The Chern number
is always

```text
C = sum(berry_curvature) / (2π)
```

`LatticeTopologyResult.minimum_plaquette_branch_margin` reports
`π - max(abs(berry_curvature))`. Production callers must choose and record a
strictly positive safety margin appropriate to their calculation; an
integer-looking Chern number alone is not an admissibility certificate.

## Direct sums and principal-branch aliases

For an exact orthogonal direct sum `P = P_a ⊕ P_b`, the physical first Chern
class obeys `c1(P) = c1(P_a) + c1(P_b)`. Normalized determinant links obey the
same product law. Principal-branch plaquette fluxes, however, obey it only
modulo `2π` at each plaquette:

```text
F_a + F_b - F_union = 2π n_plaquette
```

Consequently, separately wrapped coarse-grid FHS sums can differ by an integer
when one representation lies close to or crosses the `±π` branch cut. Do not
turn raw equality of separately principal-wrapped Chern sums into an
unconditional gate. A direct-sum validation must instead:

1. verify the exact block/projector decomposition and fixed occupied rank;
2. compare link products and plaquette residuals modulo `2π`;
3. record the integer branch-winding field;
4. require non-singular physical overlaps, local `U(N)` covariance, and an
   explicit plaquette branch margin for every result promoted to authority.

When an exact conserved-sector decomposition is available and each sector is
admissible, the sector-summed Chern is a valid same-mesh discriminator for the
union even if the composite principal-branch FHS field aliases. This does not by
itself establish continuum/off-mesh or basis-cutoff convergence.

## Actual-case validation

Tracked topology tests compute Chern numbers only through
`analysis.topology.compute_lattice_topology(FHSState)`.

- TDBG: AB-BA high-D single-valley conduction band state gives the Liu-2022
  single-valley Chern signs on the sewn torus.
- HTQG: actual Fujimoto-style checkpoint in `tests/test_htqg_model.py` uses
  realistic parameters, shell-6 plane-wave basis, mesh-9 FHS grid, generic
  HTQG basis sewing, and `reports/htqg_fig1_chern_comparison_20260611.md`:
  αβγ K-valley valence `C=-2`, conduction `C=0`.
- RLG-hBN: common generic block-sewing tests verify the physical valley-signed
  plane-wave block relabeling used by the state builder; full paper-local RLG
  topology claims still require the separate saved-target/Slurm validation gate.

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
