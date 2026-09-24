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
projected_basis_connection(...)
projected_basis_covariant_hamiltonian_derivative(...)
projected_basis_covariant_velocity(...)
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

## k-dependent projected-basis derivative (P2.2)

Let the orthonormal columns of a possibly rectangular matrix `U(k)` define the
projected basis and let

```text
H_proj = U† H_full U
A_basis,a = i U† partial_a U.
```

If the range of `U` is invariant under `H_full` (including an exactly
reconstructed projected operator), differentiating `H_proj` gives

```text
partial_a H_proj = U† (partial_a H_full) U + i [A_basis,a, H_proj],
D_a H_proj = partial_a H_proj - i [A_basis,a, H_proj]
           = U† (partial_a H_full) U,
v_a,cov = D_a H_proj / hbar.
```

The sign is fixed by `A_basis = i U† partial U`; it is not a tunable response
convention. Under a k-dependent projected-frame change `U -> U W`,

```text
H_proj -> W† H_proj W,
A_basis,a -> W† A_basis,a W + i W† partial_a W,
D_a H_proj -> W† (D_a H_proj) W.
```

The current HTQG projected-HF adapter uses the same sign implicitly. For a
neighbor basis `U(k+dk)`, its overlap/polar transporter is
`q = polar[U(k)† U(k+dk)] = I + dk U† partial U + O(dk^2)`. Therefore
`q H_proj(k+dk) q†` has derivative
`partial H_proj - i[A_basis,H_proj]` in the base frame.

Use:

```python
A_basis = projected_basis_connection(U, dU)
dH_cov = projected_basis_covariant_hamiltonian_derivative(H_proj, dH_proj, A_basis)
v_cov = projected_basis_covariant_velocity(H_proj, dH_proj, A_basis, hbar=...)
```

Do not identify `A_basis` automatically with WannierBerri's Hamiltonian-gauge
`A_H = i D_H + Xbar('AA')`. `A_basis` is the connection of the active/projected
frame, and the response decomposition must be chosen consistently. In the
Hamiltonian eigenbasis, for `n != m`,

```text
(D_a H_proj)_nm = (partial_a H_proj)_nm
                  + i(E_n-E_m)(A_basis,a)_nm,
-i(D_a H_proj)_nm/(E_n-E_m)
  = -i(partial_a H_proj)_nm/(E_n-E_m) + (A_basis,a)_nm.
```

Thus, if `hamiltonian_gauge_data` receives the already covariant `D H_proj`, its
internal `i D_H` already contains the off-diagonal projected-basis connection.
Adding the same rotated `A_basis` again as `external_connection` would double
count it. The WannierBerri-like alternative is to use the ordinary `partial
H_proj` and add the consistently rotated basis/position connection externally.
The diagonal connection and generalized derivative still require a deliberate,
consistent treatment; velocity matrix elements alone do not determine them.
That response-level integration is a later audit gate.

If the projected range is not invariant under `H_full`, the identity above has
additional leakage terms involving `(1-UU†) H_full U`. The common helper does
not estimate or hide them; a full-continuum comparison is required.

## Gauge/subspace rule from WannierBerri

WannierBerri dynamic calculators identify band groups, evaluate one common
spectral/Fermi factor per group pair, and call
`ShiftCurrentFormula.trace_ln`, which sums the complete initial/final pair
block. For strictly exact degeneracy, the local product represented by that
sum is

```text
Tr_I[(A^c_;a)_IF A^b_FI] + (b <-> c).
```

Under independent `U_I` and `U_F` rotations, the rectangular factors transform
as `G_IF -> U_I† G_IF U_F` and `A_FI -> U_F† A_FI U_I`; their product changes by
similarity and the trace is invariant. Individual labeled `(n,m)` summands are
not invariant. The reusable response-level implementation is
`analysis.shift_current.component_group_trace_amplitude`, with strict
energy/occupation validation in
`exact_degenerate_group_transition_weight`.

Local matrix helpers also support subspace-covariance tests:

```python
groups = degenerate_band_groups(energies, threshold=...)
G = random_block_unitary(groups, nb, rng)
X_g = apply_band_gauge_to_matrix(X, G)       # X -> G† X G
trace_subspace(X_g, group) == trace_subspace(X, group)
```

A numerical grouping threshold is useful for finding candidate manifolds but
does not turn distinct energies into an exact gauge freedom. This milestone
demands arbitrary U(N) covariance only for strictly exactly-degenerate groups.
The near-degenerate WannierBerri mean-group-energy prescription is an
additional approximation; a local near-degenerate cluster spectrum requires a
separate matrix-valued spectral/occupation contract and is not claimed here.
If an active window cuts an exact degenerate group, expand it or exclude the
point.

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

The P2.2 tests in `tests/test_response_derivative_gauge.py` are exact,
finite-dimensional analytic constructions. They check:

1. a moving rectangular three-dimensional projected basis inside a
   four-dimensional full space;
2. the sign-discriminating identity
   `partial H_proj - i[A_basis,H_proj] = U†(partial H_full)U`;
3. the corresponding explicit `1/hbar` velocity identity;
4. the transformation law of `A_basis` under a k-dependent frame change;
5. covariance and trace invariance under a seeded random `U(2)` rotation inside
   an exactly degenerate doublet.

`tests/test_shift_current_generic.py` separately covers the reusable
shift-current API and named WannierBerri/Joya response conventions. Its P2.1
exact-degenerate spectrum gate applies independent random `U(2)` rotations to
occupied and empty doublets at every sampled k point, proves that ordinary
pair-resolved weights change, and checks max/L2/peak/integrated residuals of the
group-trace spectrum. The broader historical derivative tests (upstream
`Matrix_GenDer_ln` parity, Wilson-link phase checks, and the old hTG wrapper
comparison) were pruned from the public snapshot; their earlier pass record
must not be mistaken for a current run. The new P2.1 gate remains authored but
not numerically executed until it runs on an allowed compute/test node.

Cluster policy forbids numerical `pytest` on login nodes. Run the current tests
on an allowed test/compute node before marking the executable milestone
numerically passed, for example:

```bash
PYTHONPATH=src pytest -q \
  tests/test_response_derivative_gauge.py \
  tests/test_shift_current_generic.py
```
