# Julia to Python module map

This is a working map, not a claim that file names will stay identical.

## Benchmark-facing zero-field pieces

Julia sources currently relevant to the first port:

- active benchmark baseline in Julia:
  - `B0/libs/BM_mod.jl`
  - `B0/libs/Lattice_mod.jl`
  - `B0/libs/Parameters_mod.jl`
  - `B0/libs/HF_mod.jl`
  - `B0/libs/HF_mod_fig6_restricted.jl`
  - `B0/proj/run_fig6_b0_overlap.jl`
  - `B0/proj/run_fig6_b0_hf_path.jl`
  - `B0/proj/plot_fig6_b0_hf_path.py`
  - `B0/proj/build_b0_python_benchmark.py`
- `B0k` is intentionally omitted from the default map and should be ignored unless the user explicitly asks for a comparison workflow.

## Proposed Python targets

- `src/mean_field/systems/tbg/params.py`
  Maps parameter structs and derived quantities.

- `src/mean_field/core/lattice.py`
  Owns reciprocal-space grids and path discretization helpers.

- `src/mean_field/core/hf/{flavors,occupations}.py`
  Own reusable flavor/block conventions, band labeling, occupation selection, and SCF convergence helpers that should remain system-agnostic.

- `src/mean_field/systems/tbg/zero_field/model.py`
  Builds the zero-field continuum/BM Hamiltonian.

- `src/mean_field/systems/tbg/zero_field/overlap.py`
  Handles overlap/form-factor data needed by HF.

- `src/mean_field/systems/tbg/zero_field/hf.py`
  Implements TBG zero-field SCF, initialization modes, Hartree/Fock updates, and energy bookkeeping on top of the reusable `core/hf` helpers.

- `src/mean_field/systems/tbg/zero_field/hf_runners.py`
  Owns post-SCF path reconstruction and parity helpers that are specific to the B0 benchmark workflow.

- `src/mean_field/systems/tbg/zero_field/runners.py`
  High-level orchestration for benchmark runs and exported outputs.

- `src/mean_field/benchmarks.py`
  Loads benchmark metadata and file paths.

## Design note

The mapping is semantic, not textual. Python modules should correspond to responsibilities, not to one Julia file each.

The active split for the HF port is:

- Julia `HF_mod*.jl` state-independent helper logic -> Python `core/hf/`
- Julia `HF_mod*.jl` TBG/B0-specific mean-field construction -> Python `systems/tbg/zero_field/hf.py`
- Julia `proj/*.jl` benchmark or plotting scripts -> Python `systems/tbg/zero_field/{hf_runners,runners,plotting}.py`

The Python code should follow the checked-in `B0` benchmark truth by default. `B0k` stays out of scope unless the user explicitly asks for comparison or audit work.
