# Benchmark Strategy

## Principle

The Python rewrite should be validated at multiple levels. Comparing only the final band plot is too weak. A solver can produce a superficially similar final figure while already drifting in intermediate quantities.

Physical logic must be checked before plotting logic. A plot is only a diagnostic view of some computed object; it is not the object itself.

In particular for HF path bands:

- first identify the physical quantity that Julia is computing
- then verify that Python is constructing the same quantity
- only after that should the rendered plot be used as a visual check
- if the upstream Julia workflow never constructs that post-SCF quantity, do not invent it in Python and call it a benchmark observable

A plotting change must never silently replace one physical quantity with another. If two plots differ, the first question is not "which rendering is nicer" but "are these two curves built from the same Hamiltonian and the same k-space object".

For the original author repository
`/data/home/ziyuzhu/TBG_HartreeFock/TBG_HartreeFock`,
the checked-in zero-field `B0` analysis scripts do not perform a post-SCF high-symmetry-path HF reconstruction after convergence. The original scripts inspect:

- the converged SCF grid itself
- contour / density maps on that grid
- or simple cuts extracted from the SCF grid

This matters for parity work. A post-SCF dense path reconstruction may be a useful auxiliary diagnostic in some workflows, but it is not automatically the same physical object as the original author's HF output.

## Benchmark levels

### Level 0: provenance and source parity

- every benchmark case records its upstream generator within the active `B0` workflow
- the Python port follows the checked-in `B0` benchmark bundle by default
- `B0k` is out of the default workflow and should be ignored unless explicitly requested

### Level 1: project and data integrity

- benchmark manifest loads
- case directories are discoverable
- summary metadata is consistent

### Level 2: geometry and path construction

These are the first intermediate variables to lock down because they are:

- deterministic
- cheap to test
- used by every later zero-field workflow

Current geometry-level checks:

- derived `Params` quantities from the Julia formulas
- selected adjacent `M` point
- `M-K-Γ-M` path node coordinates
- node indices and cumulative path distances

### Level 3: non-interacting spectrum

- zero-field BM band energies along the benchmark path
- flavor ordering conventions where applicable
- benchmark BM gauge references when HF parity depends on the two-band eigenvector convention rather than only on energies

### Level 4: Hartree-Fock state

- chemical potential
- representative occupied state selection
- energy decomposition
- selected order parameters and flavor-resolved diagnostics

### Level 5: performance and scaling

- wall time for full BM and HF benchmark cases
- strong-scaling behavior over the allocated cluster CPU cores
- recorded process/thread configuration, including BLAS and JIT settings
- separated cold-start and steady-state timings when `Numba` or another compiler is used

## Current benchmark sources

- `benchmarks/b0/benchmark_manifest.tsv`: checked-in zero-field HF case list; this is the active `B0` benchmark baseline
- `benchmarks/b0/parameter_reference.tsv`: Julia-generated parameter snapshots for the benchmark angles
- `benchmarks/b0/bm_inputs/unstrained_path/`: checked-in BM helper benchmark files bundled with the active `B0` benchmark tree
- `benchmarks/b0/bm_inputs/unstrained_path/overlap_reference_path.tsv`: authoritative compact overlap benchmark currently used by Python
- `benchmarks/b0/bm_inputs/bm_theta_*_lk*_lg*_uk_reference.tsv`: optional Julia-exported benchmark-grid BM eigenvectors used when a gauge-sensitive HF audit needs the Julia basis itself rather than only the derived overlaps
- `TBG_HartreeFock/B0/proj/build_b0_python_benchmark.py`: active `B0` benchmark builder
- `TBG_HartreeFock/B0/proj/run_fig6_b0_hf_benchmark_case.py`: rerun helper that records runtime and parity data for the accepted zero-field HF cases
- `TBG_HartreeFock/B0/proj/run_fig6_b0_hf_path.jl`: active `B0` HF path-band exporter
- `TBG_HartreeFock/B0/proj/run_fig6_b0_overlap.jl`: `B0`-side BM overlap / benchmark input generator for the shared BM inputs used by the HF reruns
- `TBG_HartreeFock/B0/proj/build_b0_bm_unstrained_benchmark.py`: optional BM-only packager retained for audit and standalone BM exports
- `benchmarks/b0/cases/*/reference_nodes.tsv`: Julia-generated HF path nodes in the checked-in snapshot

## Policy

- Use `B0` as the benchmark truth for the checked-in zero-field HF bundle.
- Use the BM helper files bundled under `benchmarks/b0/bm_inputs/unstrained_path/` as the active zero-field BM benchmark inputs for Python.
- Ignore `B0k` in the default workflow unless the user explicitly asks for comparison or audit work.
- Runtime benchmarks and any nontrivial compute-side validation must be run on Slurm CPU allocations, not on login nodes.
- `login001` and `login002` are submission / inspection entry points only. They must not be used to run numerical tests, benchmark commands, SCF solves, eigensolvers, BLAS-heavy scripts, or other compute work.
- The short-task route for such compute work is `login002 -> test001`, and the recorded metadata must include the allocated CPU count plus thread settings.
- `long` is also a valid CPU partition and may be used when the benchmark step does not fit comfortably inside `test001` or when `test001` is unavailable.
- Formal runs should first be considered as Slurm jobs rather than as interactive-node work.
- On the shared cluster, Slurm submissions should stay conservative: absent explicit user instruction, use at most 5 nodes and prefer serial per-node task lists when running multiple cases.
- For single-node CPU benchmark runs, the default policy is to fill the node as much as practical: choose `cpus-per-task` to match the node class in use, typically `56` on common CPU nodes and `64` on the larger ones, instead of keeping a stale `28-core` split by default.
- Memory requests should be sized with the same intent: prefer a near-full-node allocation when the benchmark is the only task on that node, rather than leaving large unused headroom without reason.
- Add an intermediate benchmark whenever a bug could hide for a long time while the final plot still looks reasonable.
- Prefer quantities that are deterministic and cheap to regenerate.
- Prefer benchmark data exported from Julia rather than benchmark values re-derived inside Python.
- For HF, treat the path Hamiltonian, density matrix, overlap contractions, and flavor-resolved spectrum as the primary physics objects. The plot is a downstream visualization only.
- Never promote an auxiliary diagnostic plot into the default benchmark artifact unless its physical definition matches the Julia reference quantity.
- For zero-field HF, do not treat post-SCF path reconstruction as a rigorously validated default observable unless the upstream Julia workflow explicitly uses it for that case family.
- For the original `TBG_HartreeFock/TBG_HartreeFock` `B0` workflow, the conservative policy is:
  use exact SCF-grid quantities as the physical benchmark object first, and treat any later path reconstruction as non-authoritative unless separately justified.

## Kmesh and Path Advisory

For SCF-grid diagnostics, the chosen high-symmetry path must be checked against the actual `kmesh` before interpreting the plot.

- High-symmetry points must be chosen inside the sampled `kmesh` region. Do not place a path node in an equivalent Brillouin-zone image that sits outside the actually sampled cell.
- For SCF-grid diagnostics, the advisor must use `K/M/Γ` points from the moire Brillouin-zone geometry itself, not the BM/HF-internal `params.kt` valley-center convention.
- The default construction should enumerate symmetry-equivalent representatives of the requested path *within the sampled cell* and choose the representative that maximizes exact on-path SCF points while minimizing node miss distance.
- The current `M-K-Γ-M` advisor uses the in-cell `30-60-90` right-triangle family with the right angle at `M`, so all three path segments can carry exact SCF-grid points on the sampled parallelogram.
- A path should be ranked using geometry-level metrics first:
  exact on-path SCF point count, high-symmetry-node hit count, mean nearest distance from the dense path samples to the SCF grid, and the corresponding max distance.
- The helper script
  `scripts/inspect_b0_kmesh_path_advisor.py`
  writes both summary tables and `kmesh + path` overlay plots for this purpose, including the sampled cell boundary and the moire Brillouin-zone hexagon.
- For `theta = 1.20°`, the corrected in-cell right-triangle construction yields exact on-path SCF counts `19, 23, 24, 32` for `lk = 19, 23, 24, 32`, respectively.
- `lk = 24` is especially clean because the corresponding in-cell `M/K/Γ` representatives are all exact grid hits.
- This advisory logic should generalize to other requested high-symmetry paths by:
  enumerating symmetry-equivalent node representatives inside the sampled cell,
  scoring each candidate path geometrically,
  and selecting the path that keeps the nodes inside the sampled region and captures the largest number of exact SCF points.
