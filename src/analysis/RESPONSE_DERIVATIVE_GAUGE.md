# Gauge-safe response derivative tool

This note records the local derivative module introduced for response calculations involving Berry connections, generalized derivatives, shift vectors, and phase derivatives.

## Local implementation

```text
src/analysis/response_derivative_gauge.py
```

The higher-level reusable shift-current API that consumes this derivative layer is:

```text
src/analysis/shift_current/
```

Main derivative API:

```python
hamiltonian_gauge_data(...)
covariant_derivative_matrix(...)
wannierberri_matrix_gen_derivative_ln(...)
wannierberri_matrix_gen_derivative_nn(...)
berry_connection_generalized_derivative(...)
berry_connection_generalized_derivative_pair(...)
shift_integrand_from_generalized_derivative(...)
shift_integrand_from_pair_generalized_derivative(...)
shift_vector_from_generalized_derivative(...)
shift_vector_from_pair_generalized_derivative(...)
link_shift_vector(...)

wannierberri_shift_current_internal_imn(...)
wannierberri_shift_current_group_trace(...)

degenerate_band_groups(...)
random_block_unitary(...)
apply_band_gauge_to_matrix(...)
apply_band_gauge_to_axis_matrix(...)
trace_subspace(...)
```

## Fixed reference

Primary reference is the local WannierBerri copy:

```text
reference/upstream/wannier-berri/wannierberri/data_K/data_K.py::D_H, get_A_H, dEig_inv, covariant
reference/upstream/wannier-berri/wannierberri/formula/formula.py::Matrix_GenDer_ln
reference/upstream/wannier-berri/wannierberri/formula/elementary.py::Dcov, DerDcov
reference/upstream/wannier-berri/wannierberri/calculators/dynamic.py::ShiftCurrentFormula
```

Conventions mirrored:

```text
D_H = -Xbar('Ham', 1) * dEig_inv[..., None]
A_H = 1j * D_H                         # external_terms=False / continuum local basis
A_H = 1j * D_H + Xbar('AA')             # when external position terms exist
A_{;d} = partial_d A - D^d A + A D^d
```

WannierBerri's dynamic shift-current calculator also uses a principal-value regularizer for intermediate denominators:

```text
D_H_Pval = -V_H * (DeltaE)/(DeltaE^2 + sc_eta^2)
```

See `calculators/dynamic.py::ShiftCurrentFormula`.  The local generalized-derivative helpers therefore accept optional `principal_value_eta`; default `None` keeps the exact phase-derivative sum rule used for Wilson-link checks.

For exact line-by-line comparison with WannierBerri's block API, use:

```python
wannierberri_matrix_gen_derivative_ln(...)
wannierberri_matrix_gen_derivative_nn(...)
```

For full nondegenerate band matrices, use:

```python
covariant_derivative_matrix(...)
```

## Gauge/subspace rule from WannierBerri

WannierBerri dynamic calculators group nearly degenerate bands and trace covariant formulas over subspaces, rather than trusting arbitrary phases/vectors inside the degenerate manifold.  Local helpers mirror this validation pattern:

```python
groups = degenerate_band_groups(energies, threshold=...)
G = random_block_unitary(groups, nb, rng)
X_g = apply_band_gauge_to_matrix(X, G)       # X -> G† X G
trace_subspace(X_g, group) == trace_subspace(X, group)
```

Use these for gauge-randomization tests.  If an active band window cuts a degenerate group, expand the window or exclude the point.

## Phase derivative rule

Do not differentiate raw eigenvector phases or raw `np.angle(A_mn)` values.  Use either:

1. analytic/generalized derivative route, e.g. `berry_connection_generalized_derivative`; or
2. Wilson-link finite difference, e.g. `link_shift_vector`, which parallel-transports `A_mn(k+dk)` back to the gauge at `k`.

Near zeros of the optical matrix element, compare the gauge-invariant product

```text
Im[A_mn (A_nm)_;]
```

rather than the shift vector alone.

## Validation

Focused validation lives in:

```text
tests/test_response_derivative_gauge.py
tests/test_shift_current_generic.py
```

The generic shift-current module adds system-facing helpers for component parsing, named WannierBerri/Joya conventions, Fermi occupations, Lorentzian conventions, transition tables, heatmap accumulation, and selected-pair/full-tensor transition weights. Paper-specific workspaces should call `analysis.shift_current` rather than reimplementing these pieces.

Current validation checks:

1. Loads the actual upstream WannierBerri `formula.py` source with minimal stubs and verifies our `wannierberri_matrix_gen_derivative_ln/nn` against `Matrix_GenDer_ln` numerically.
2. Verifies `berry_connection_generalized_derivative` against the common `analysis.shift_current` sum-rule implementation on a nonlinear two-band toy model with nonzero `d2H/dkdk`.
3. Verifies the optimized selected-pair generalized derivative against the full tensor and pair integrand/shift-vector helpers against their full-tensor forms.
4. Verifies the optional WannierBerri/Wannier90 principal-value regularized selected-pair derivative against the full tensor.
5. Random U(1) gauge test: `Im[A_mn(A_nm)_;]` is invariant under eigenvector phase rotations.
6. Random block-unitary subspace test: covariant derivatives transform as `G† X G`, and traces over degenerate groups are invariant.
7. Wilson-link independent-gauge test: `link_shift_vector` is invariant under independent U(1) phase choices at `k` and `k+dk`.
8. Wilson-link phase derivative test: `link_shift_vector` agrees with the covariant derivative shift vector on a smooth nondegenerate point.
9. Ported WannierBerri `ShiftCurrentFormula` internal-term integrand matches the existing SLG reference-formula audit and exposes group-trace helpers.
10. Historical Chaudhary b0 and hTG legacy wrapper gates were retired with those paper-audit surfaces; their durable convention lessons are kept in the common API and this note.
11. Generic `analysis.shift_current` gates: `JOYA_EQ7_GEOMETRIC_CONVENTION` matches the ordered pair integrand/no-`1/pi` Lorentzian, `WANNIERBERRI_INTERNAL_IMN_CONVENTION` matches upstream internal `Imn`, and selected-pair kernels agree with their full-tensor forms.

Command used:

```bash
PYTHONPATH=src pytest -q tests/test_tdbg_shift_current_adapter.py tests/test_shift_current_generic.py tests/test_response_derivative_gauge.py
```

Result:

```text
24 passed
```
