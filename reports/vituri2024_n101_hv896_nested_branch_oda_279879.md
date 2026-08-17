# Vituri 2024 N101/Hv896 nested degenerate-shell ODA result

## Status

Slurm job `279879` completed on `node034` with exit code `0:0` in `00:58:05`. The sealed runroot is:

`/data/home/ziyuzhu/.runs/Mean_Field_a397fa0_vituri_hf_n101_hv896_nested_branch_oda_v2_20260816`

The source calculation commit is `a397fa070e117811aa0809fdc0c44c25a7d58ab0`. The external sentinel is `COMPLETE_279879.json` with SHA256 `e9e0d25a752e48118af8cce64abf669de881af964cbd061d595f1208ca70a4e9`.

A machine-readable terminal-state and artifact-hash attestation is stored in `reports/data/vituri2024_n101_hv896_nested_branch_oda_279879_attestation.json`.

## Branch-tree result

The run replayed each child from the original `half_metal_sz_plus`, seed-0 initializer and consumed two ordered exact-trigger choices:

- `p1_c0`: parent branch 1, child shell state `35499`;
- `p1_c1`: parent branch 1, child shell state `35903`;
- `p4_c0`: parent branch 4, child shell state `35503`;
- `p4_c1`: parent branch 4, child shell state `35907`.

All four paths converged after eight SCF iterations and passed the fixed-rank stationarity gates. No generation-3 unresolved shell appeared. Together with the four stationary inherited root leaves `r0`, `r2`, `r3`, and `r5`, this closes the declared coordinate tree with eight stationary coordinate leaves and zero unresolved frontier leaves.

This is an exhaustive coordinate-path statement, not an assertion that all eight leaves are physically distinct.

## Endpoint coalescence

The sibling paths coalesce pairwise at the exact final-density level:

| Paths | Final-density SHA256 | Holes by valley | Independent energy (eV) |
|---|---|---:|---:|
| `p1_c0`, `p1_c1` | `5808983ec8f2ce155fd00a3b9214b20fbf2ab284e3e3d6bd8d610f90457d679b` | `888 / 904` | `20379.958970065687` |
| `p4_c0`, `p4_c1` | `ed53bf1466b9fe58a2cab1408bb6fbb75172b506f6a8cd32009f90333c1ed9bf` | `904 / 888` | `20379.958970065687` |

Thus the eight coordinate leaves produce six distinct final-density hashes. The two child endpoint classes are related by opposite finite-volume valley imbalance. Within each sibling pair, final Hamiltonian, eigenvalue-array, and trajectory hashes are not byte-identical, so only final-density and scalar-energy coalescence is established.

## Energy comparison

The child endpoint energy is lower than each inherited stationary root leaf by approximately `0.70662243 meV` in this finite N101/Hv896 calculation. The full stationary-energy spread is

`0.0007066224425216205 eV`.

This is a same-model finite-volume variational comparison. It is not a global-ground-state proof, thermodynamic-limit result, or absolute paper-energy interpretation.

## Stationarity and replay gates

Every child endpoint has:

- `converged=true`, `iterations=8`;
- final raw norm `0`;
- raw and mixed projector defect `0`;
- raw and mixed particle number `39012`;
- raw and mixed Fock commutator `0`;
- independent/engine energy residual at most `5.4569682106375694e-11 eV`;
- fixed-rank linearized-energy excess at most `2.9103830456733704e-11 eV`.

Postflight reran all four paths and obtained byte-exact arrays with maximum residual `0`.

## Physical classification

All child endpoints are fully spin-polarized, share a common spin axis, and are intervalley-incoherent. They are **not** valid homogeneous half-metal candidates under the implemented diagnostic because their valley occupations are `888/904` or `904/888`, giving valley-balance residual `16` rather than the required tolerance `1e-7`.

Therefore the result does not authorize a valley-balanced homogeneous-half-metal claim.

## Authority limits

The sealed result supports only an independent finite-volume coordinate branch-tree qualifier. It does not establish:

- author-exact numerical policy;
- local HF stability or Hessian positivity;
- global ground state or phase identification;
- regulator convergence or thermodynamic-limit physics;
- stationary-source or TDHF source promotion;
- production readiness;
- paper reproduction.

The runroot seal is tamper-evident and hash-bound, but it is not administrator-immutable or cryptographically signed. This Git-tracked report and JSON attestation provide an external repository anchor for the observed terminal Slurm state and sealed artifact hashes.
