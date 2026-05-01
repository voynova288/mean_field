# B0 Julia Port Audit

This note tracks how the active `TBG_HartreeFock/B0/` Julia code maps into `Mean_Field`.

Scope rules:

- Treat `B0` as the zero-field benchmark truth.
- Ignore `B0k` here unless a later audit explicitly asks for comparison-only additions.
- `tMBG` is out of scope for this note.

## Core benchmark path: migrated

| Julia source | Status | Python landing zone | Notes |
| --- | --- | --- | --- |
| `B0/libs/Parameters_mod.jl` | migrated | `src/mean_field/systems/tbg/params.py` | Parameter construction and derived moire quantities are in Python. |
| `B0/libs/Lattice_mod.jl` | migrated | `src/mean_field/core/lattice.py`, `src/mean_field/systems/tbg/zero_field/path.py` | Lattice/path responsibilities are split into reusable and TBG-specific pieces. |
| `B0/libs/BM_mod.jl` | migrated | `src/mean_field/systems/tbg/zero_field/model.py` | BM Hamiltonian, eigenvectors, `sigma_z`, and uniform B0 lattice solve are present. |
| `B0/libs/HF_mod.jl` | migrated in core workflow | `src/mean_field/core/hf/`, `src/mean_field/systems/tbg/zero_field/hf.py` | Generic HF helpers are now split from the TBG-specific SCF engine. |
| `B0/libs/HF_mod_fig6_restricted.jl` | migrated in benchmark workflow | `src/mean_field/systems/tbg/zero_field/hf.py`, `hf_runners.py`, `runners.py` | Restricted initialization, flavor conventions, path evaluation, and benchmark export exist in Python. |
| `B0/proj/run_fig6_b0_overlap.jl` | migrated | `src/mean_field/systems/tbg/zero_field/overlap.py`, `runners.py`, checked-in `benchmarks/b0/bm_inputs/` | Python consumes and audits the same overlap products. |
| `B0/proj/run_fig6_b0_hf_path.jl` | migrated | `src/mean_field/systems/tbg/zero_field/hf_runners.py` | Path Hamiltonian reconstruction and parity comparison exist. |
| `B0/proj/plot_fig6_b0_hf_path.py` | migrated | `src/mean_field/systems/tbg/zero_field/plotting.py` | Path-band plotting is present. |
| `B0/proj/plot_fig6_b0_unstrained_path.py` | migrated | `src/mean_field/systems/tbg/zero_field/plotting.py` | BM path plotting is present. |
| `B0/proj/build_b0_python_benchmark.py` | migrated | `src/mean_field/benchmarks.py`, `src/mean_field/systems/tbg/zero_field/runners.py` | Python benchmark bundle loading and artifact writing are in place. |
| `B0/proj/build_b0_bm_unstrained_benchmark.py` | migrated | `src/mean_field/systems/tbg/zero_field/runners.py` | BM-only benchmark run/export exists. |
| `B0/proj/run_fig6_b0_hf_benchmark_case.py` | migrated | `src/mean_field/systems/tbg/zero_field/runners.py`, `src/mean_field/cli.py` | Benchmark case execution, summaries, and CLI dispatch exist. |

## Partially migrated or absorbed into other modules

| Julia source | Status | Python landing zone | Gap that remains |
| --- | --- | --- | --- |
| `B0/libs/helpers.jl` | partial | split across `params.py`, `path.py`, `runners.py` | No single helper module mirrors the Julia utility surface. |
| `B0/libs/plot_helpers.jl` | partial | `plotting.py`, `runners.py` | Common plotting/report helpers are present, but not all helper entry points are mirrored one-to-one. |
| `B0/libs/HF_mod_test.jl` | partial | `tests/test_b0_hf_helpers.py`, `tests/test_b0_hf_benchmark_runner.py` | Convention and runner coverage exists, but not as a direct Julia test-port suite. |
| `B0/libs/DensityMatrix_reduction.jl` | partial | `hf.py`, `devtools/compare_b0_hf_*`, `devtools/inspect_b0_hf_*` | Some density diagnostics exist, but there is no dedicated reduction module yet. |
| `B0/proj/run_hf.jl` | partial | `src/mean_field/systems/tbg/zero_field/hf.py` | The full-HF solver exists, but the surrounding user-facing workflow is not mirrored as a separate runner script. |
| `B0/proj/run_fig6_b0_hf_full.jl` | partial | `hf.py`, `runners.py` | Full-HF dispatch is supported, but the Julia script-level workflow and follow-up analysis surface are not matched one-to-one. |
| `B0/proj/run_fig6_b0_hf_scf_points_case.jl` | partial | `hf_runners.py`, `plotting.py` | SCF-grid path extraction exists, but not as an exact script clone. |
| `B0/proj/diagnose_fig6_b0_hf_case.jl` | partial | `src/mean_field/devtools/compare_b0_hf_*`, `inspect_b0_hf_*` | Diagnostics are present but remain fragmented across several devtools scripts. |
| `B0/proj/hf_order_parameter_analysis.jl` | partial | `hf.py` diagnostics, benchmark summaries, devtools | Order-parameter quantities are exposed only in limited form. |
| `B0/proj/run_hf_analysis.jl` | partial | `runners.py`, devtools scripts | Summary output exists, but not the full Julia analysis script. |
| `B0/proj/summarize_fig6_b0_hf_restricted.jl` | partial | `write_b0_hf_suite_summary`, `write_hf_path_summary` | Similar summaries exist, but not the exact Julia post-processing workflow. |
| `B0/proj/summarize_fig6_b0_hf_full.jl` | partial | `runners.py` summaries | Same limitation as the restricted case. |
| `B0/proj/summarize_fig6_b0_convergence.py` | partial | benchmark/runtime summaries | No dedicated convergence-summary module yet. |

## Not yet migrated

These Julia pieces do not have a real Python counterpart in the TBG B0 workflow today.

| Julia source | Missing area |
| --- | --- |
| `B0/libs/BMChern_mod.jl` | BM Chern workflow |
| `B0/libs/HFChern_mod.jl` | HF Chern workflow |
| `B0/libs/ParametersChern_mod.jl` | Chern-specific parameter layer |
| `B0/libs/hybridWannier_mod.jl` | hybrid Wannier analysis |
| `B0/libs/BM_mod_legacy.jl` | intentionally skipped legacy path |
| `B0/proj/run_bm_chern.jl` | BM Chern runner |
| `B0/proj/run_hfChern.jl` | HF Chern runner |
| `B0/proj/run_hfChern_analysis.jl` | HF Chern analysis |
| `B0/proj/run_fig6_b0_hf_chern.jl` | Fig. 6 HF Chern workflow |
| `B0/proj/run_fig6_b0_hf_occupied_chern.jl` | occupied-band Chern workflow |
| `B0/proj/run_DensityMat_analysis.jl` | dedicated density-matrix analysis workflow |
| `B0/proj/run_fig6_b0_hf_restart.jl` | restart workflow |
| `B0/proj/run_fig6_b0_hf_perturb_restart.jl` | perturb-and-restart workflow |
| `B0/proj/run_fig6_b0_hf_restricted.jl` | standalone Julia-style restricted runner script surface |
| `B0/proj/submit_*.sbatch`, `submit_*.sh` | one-to-one Slurm submission wrappers; Python should add generic Slurm wrappers instead of copying every script verbatim |

## Recommended next migration order

1. Consolidate the fragmented HF diagnostics into one Python analysis surface before adding new physics. The Julia gaps around density-matrix reduction and order-parameter analysis are more urgent than copying more plotting scripts.
2. Decide whether Chern workflows belong in the near-term TBG scope. If yes, port `ParametersChern_mod.jl`, `BMChern_mod.jl`, and `HFChern_mod.jl` together as one coherent slice instead of ad hoc script copies.
3. Add a restart-capable Python runner only after the current benchmark and analysis surfaces are stable. Restart logic should be workflow-level code, not embedded deeper into the generic `core/hf/` layer.
