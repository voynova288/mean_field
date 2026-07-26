# RLG/hBN Track-C affine-support no-go

## Scope

This note concerns only finite raw-plane-wave supports on the 12x12 torus. It
does not claim that exact C3 is impossible in the infinite continuum model.
It explains why Track C cannot be implemented as an ordinary constant-size,
single-representative raw-G replacement of the current 19-RLV parent while
retaining `G=0` and the same local layer/sublattice multiplicity.

## Fixed-sector affine action

For a stored torus representative

```text
k' = C3 k + R,
C3(m,n) = (-n,m-n),
```

a raw reciprocal label in valley `eta=+/-1` transforms as

```text
G_raw' = C3 G_raw - eta R.
```

The two obstructed fixed sectors on mesh 12 are

```text
pair (4,8): R=(1,1)
pair (8,4): R=(1,0)
```

Closing the 19-point shell under this affine action produces 27 points for
each fixed pair and valley. The resulting parent Hamiltonians commute with
the explicit affine C3 unitary to about `7e-13 meV` in the focused C1 gate.
This proves the fixed-fiber construction, but not a global HF regulator.

## Constant-support-size obstruction

Let a finite support be closed under its local C3 action.

At Gamma, `R=0`. The only integer fixed point of `G -> C3 G` is `G=0`,
because `I-C3` has nonzero determinant. Therefore any C3-invariant support
that contains the low-momentum state `G=0` consists of one fixed point plus
three-cycles:

```text
N_G(Gamma) = 1 mod 3.
```

At either nonzero fixed sector, an affine fixed label would solve

```text
(I-C3) G = -eta R.
```

For `R=(1,1)` and `R=(1,0)` the solutions are fractional, not integer.
Hence the affine action has no fixed raw-G label; every orbit has length
three:

```text
N_G(K_fixed) = 0 mod 3.
```

No finite support cardinality `N_G` can be both `1 mod 3` and `0 mod 3`.
With the same nonzero local multiplicity `2*L`, unequal `N_G` also means
unequal parent Hilbert-space dimensions. Consequently:

```text
There is no constant-size, single-representative finite raw-G support that
is exactly C3 closed at both Gamma and the nonzero fixed torus sectors while
retaining G=0 and representing C3 by a permutation of raw plane waves.
```

For the current shell this is visible directly:

```text
Gamma 19-point support: one 1-cycle + six 3-cycles
fixed affine 27-point support: nine 3-cycles
```

## Consequences for `c3_affine_ws_v1`

Using 27 points only at `(4,8)` and `(8,4)` changes the parent Hilbert-space
dimension at isolated k points. It does not by itself define:

- a constant-dimension parent/projected vector bundle;
- the number/reference weight of filled remote bands;
- a continuous scalar HF energy;
- mixed 19/27 density vertices;
- the finite-q ph/hp lift and its pairing adjoint.

Using a nearest-27 physical-momentum rule everywhere does not solve the
problem. At Gamma the 27th radial threshold cuts a C3 orbit; including all
exact ties gives 31 points. Selecting only part of the tie breaks C3, while
averaging tied representatives defines a quotient or weighted regulator,
not an ordinary single parent Hamiltonian.

Under the stated assumptions—mesh 12, retention of `G=0`, fixed local
layer/sublattice multiplicity, and a single-label raw-plane-wave permutation
representation—Track C requires one of the following explicitly new objects:

1. a variable-rank fiber theory with a derived scalar/reference and
   rectangular inter-fiber density vertices;
2. a multi-representative or weighted boundary-tie quotient with its lift
   and pairing adjoint derived from one scalar;
3. a converged-cutoff sequence where C3 is restored only in the cutoff
   limit; or
4. another finite representation whose C3 action is not a permutation of
   raw plane waves.

The current local evidence does not select one of these. In particular, it
does not uniquely define the proposed `c3_affine_ws_v1` interaction shell,
remote-valence h0, or finite-q Hessian. No fresh Track-C HF/TDHF production
run should be launched until that choice is explicit and typed.

## Implemented safe boundary

The code currently provides only:

- valley-aware fixed-sector affine support closure;
- exact support provenance and fail-closed validation;
- a transient 27-point fixed parent;
- an explicit C3 unitary and strict parent-Hamiltonian covariance gate.

These are geometry/one-body prerequisites. They must not be interpreted as
a complete Track-C provider or as a Fig. S45 reproduction.
