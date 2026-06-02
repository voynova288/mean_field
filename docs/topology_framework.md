# Unified topology / Berry-geometry framework

Date: 2026-05-28

This is the durable project note for future Berry connection, Berry curvature / plaquette flux, and Chern-number work.

## Core principle

For every system in this repository, the numerical topology calculation is the same once the wavefunctions are known:

1. Put cell-periodic wavefunctions on a two-dimensional momentum mesh as
   `psi[i, j, basis, state]`.
2. Select one state column or a multi-state subspace.
3. Record what those selected columns mean physically: band index, central-pair index, Chern-sublattice label, spin/flavor/valley, system name, etc.
4. If crossing a Brillouin-zone boundary changes the plane-wave basis representation, apply the appropriate sewing / transition function before taking overlaps.
5. Build Fukui-Hatsugai-Suzuki links, Berry-connection phases, plaquette Berry flux, and the integrated Chern number.

Therefore, **system-specific topology code should not reimplement the Berry formula**.  System-specific code should only:

- generate the wavefunction mesh;
- choose and label the state/subspace indices;
- provide optional boundary sewing transforms;
- map the unified result back to that system's historical result dataclass/API.

## Implementation location

Canonical framework:

```text
src/analysis/topology/
  core.py
  __init__.py
  README.md
  validate_existing_results.py
```

Main public objects:

- `WavefunctionIndex`: immutable metadata for the selected wavefunction columns.
- `compute_lattice_topology`: one-stop FHS calculation for Berry connection, Berry flux, and Chern number.
- `compute_link_variables`: builds normalized U(1) link variables for selected line bundles or subspaces.
- `berry_curvature_from_links`: computes plaquette flux from links.
- `chern_number_from_berry_curvature`: integrates saved or freshly computed Berry flux.
- `matrix_sewing_transform`: convenience helper for matrix-valued boundary transition functions.

The framework intentionally lives under `src/analysis/`, not `mean_field.core`, because it is currently a local analysis layer that unifies system workflows without committing to a stable package API.

## Mathematical convention

For selected orthonormal columns `Psi(k)` with shape `(basis_dim, n_selected)`, the link in direction `mu` is

```text
U_mu(k) = phase det[ Psi(k)^† S_mu Psi(k + mu) ]
```

where `S_mu` is identity away from the torus boundary and the system-specific sewing transform at the boundary.  For a single band this reduces to the overlap phase.

The plaquette flux is

```text
F_12(k) = arg[ U_1(k) U_2(k + e1) conj(U_1(k + e2)) conj(U_2(k)) ]
```

and

```text
C = sum_k F_12(k) / (2 pi).
```

`berry_connection` in the result is the pair of link phases `arg U_1`, `arg U_2` with shape `(2, mesh_1, mesh_2)`.  `berry_curvature` is the dimensionless plaquette flux, not divided by physical plaquette area.

## Boundary sewing rule

Many continuum-model plane-wave bases are not literally periodic under `k -> k + G_M`.  At the torus boundary the same physical Bloch state may be represented by relabeling reciprocal vectors, for example `G -> G + G_Mi`.  In that case a periodic raw array comparison is wrong even if it gives an integer.

Always ask these questions before computing Chern numbers:

1. What basis labels define `basis_dim`?
2. Is `H(k + b_i)` identical to `H(k)` in this basis, or only identical after a basis relabeling / gauge transform?
3. If a transform is needed, does it act on a vector `(basis_dim,)` and a subspace `(basis_dim, n_selected)`?
4. Is the reported sign in raw fractional orientation or physical reciprocal-space orientation?

`compute_lattice_topology(..., sewing_transforms=(S1, S2))` applies `S1` only on links that wrap from `i=mesh_1-1` to `0`, and `S2` only on links that wrap from `j=mesh_2-1` to `0`.

## Current adapters

The following system modules now delegate their Berry geometry to `analysis.topology`:

- `mean_field.systems.tmbg.topology`
- `mean_field.systems.tdbg.topology`
- `mean_field.systems.atmg.topology`
- `mean_field.systems.RnG_hBN.topology`
- `mean_field.systems.tmbg.topology_sewn`
- `mean_field.systems.htg.topology` for link/Chern construction with HTG boundary sewing

The historical API is preserved: each system still returns its existing `TopologyResult` or `ChernBasisResult` shape where applicable.  New fields such as `berry_connection`, `min_link_magnitude`, and `index_metadata` are added to compatible dataclasses when practical.

## Validation status

Slurm validation job:

```text
job id: 132090
wrapper: historical scripts/validate_unified_topology.sbatch (retired; use scripts/submit_mean_field.sbatch with pytest or saved-result validator)
node: test001
result: 37 passed in 3.09s
saved-result validation: status=pass, failures=0, known_gaps=1
```

Artifacts:

```text
logs/topo_framework_val_132090.out
logs/topo_framework_val_132090.err
results/topology_framework_validation_20260528/saved_result_validation.md
results/topology_framework_validation_20260528/saved_result_validation.json
```

The targeted validation covers:

- QWZ single-band Chern;
- full two-band trivial subspace;
- invariance under local single-band phase changes;
- invariance under local multi-band unitary frame rotations;
- explicit boundary sewing on a toy line bundle;
- tMBG sewn topology smoke test;
- compatibility wrappers for tMBG, TDBG, ATMG, and RnG/hBN;
- HTG Chern-basis link calculations;
- saved tMBG Berry-flux integration against saved Chern values;
- saved TDBG Fig. 3 Chern summary;
- saved HTG Fig. 2b/3b Chern-basis values.

The one recorded `known_gap` is not a framework failure: existing RnG/hBN Fig. 6 `xi0_V064meV` saved artifacts do not match the paper's expected `|C|=0`, while the `xi1_V064meV` saved artifact does match `|C|=1`.  See `results/reproduction_inventory_20260528.md`.

A broader non-slow test job (`132093`) produced `253 passed, 7 failed, 1 deselected`; the failures were in old B0 HF / tMBG Hamiltonian-validation tests and are not topology-framework regressions.

## Saved-result validator

Use the file-only validator when checking old artifacts without rerunning Hamiltonian solves:

```bash
python -m analysis.topology.validate_existing_results \
  --root /data/home/ziyuzhu/Mean_Field \
  --output-dir /data/home/ziyuzhu/Mean_Field/results/topology_framework_validation_YYYYMMDD
```

This reads existing JSON/NPZ artifacts and performs consistency checks.  It does not generate missing wavefunctions.  For old tMBG/TDBG outputs, saved Chern numbers and Berry flux may exist even when full topology-grid eigenvectors were not persisted; exact recomputation with the new framework then requires regenerating eigenvectors from the model.

## Rules for future topology code

When adding or modifying topology calculations:

1. Do not duplicate `_unit_link`, `_subspace_link`, or plaquette loops in system modules.  Add capability to `analysis.topology` if the common framework is insufficient.
2. Always attach a `WavefunctionIndex` or equivalent metadata.  A Chern number without band/subspace/flavor/valley labels is not a reusable result.
3. Treat boundary sewing as part of the physical convention, not a numerical detail.
4. Store enough information for reproducibility: selected indices, labels, valley, mesh size, orientation convention, min link magnitude / singular value, and whether boundary sewing was used.
5. For paper claims, distinguish:
   - framework validation;
   - saved-result consistency;
   - full numerical recomputation;
   - visual/paper-overlay reproduction.
6. Run topology-grid recomputations and any BLAS/eigensolver-heavy validation through Slurm, not on login nodes.

## Minimal example

```python
from analysis.topology import WavefunctionIndex, compute_lattice_topology

result = compute_lattice_topology(
    eigenvectors,                       # (mesh_1, mesh_2, basis_dim, n_states)
    state_indices=(valence_index,),
    index=WavefunctionIndex(
        indices=(valence_index,),
        role="band",
        labels=("valence",),
        system="my_system",
        valley=1,
    ),
    k_grid_frac=k_grid_frac,
    sewing_transforms=(sew_b1, sew_b2),  # or None if truly periodic
)

print(result.chern_number, result.rounded_chern_number)
print(result.berry_connection.shape)  # (2, mesh_1, mesh_2)
print(result.berry_curvature.shape)   # (mesh_1, mesh_2)
```
