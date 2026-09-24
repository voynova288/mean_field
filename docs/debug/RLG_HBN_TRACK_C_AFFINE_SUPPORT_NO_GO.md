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

## Post-full-mesh regulator assessment (2026-08-01)

The user-authorized shell-5 and shell-6 full `12x12` intraflavor meshes add a
direct **parent-only** cutoff test: the parent cutoff was increased while the
interaction cutoff remained fixed at `3|q1|`. Shells 4, 5, and 6 have the
same raw signed classification at every one of the 144 q labels:

```text
parent shell                    4           5           6
stable sectors                109         109         109
complex sectors                35          35          35
real-negative-only sectors      0           0           0
max |Im Omega| (meV)        2.971       3.097       3.012
paper stable-energy RMSE     0.559       0.524       0.634
```

This classification robustness does not establish restoration of C3. The
same four C3 orbits have nonuniform `static_negative OR complex` patterns at
all three cutoffs. Moreover,

```text
max stable lowest-branch C3 spread (meV)   0.4834  0.5034  0.4601
max Hstat-minimum C3 spread (meV)           0.5961  0.6041  0.6001
```

The defect is therefore non-monotonic and essentially unchanged through
shell 6. The energies also fail the proposed `0.02 meV`/`1%` convergence
gate. These facts rule out treating shell 5 or 6 as an approximately exact-C3
finite regulator.

The four candidate regulator classes can now be assessed more sharply:

1. **Variable-rank fibers:** mathematically possible only after specifying a
   global direct-sum Fock space, rectangular density vertices
   `rho(k+q,k): H_k -> H_(k+q)`, the remote reference at unequal ranks, and
   the induced finite-q tangent metric. None is fixed by the 19-to-27 support
   closure.
2. **Weighted/multi-representative quotient:** requires an explicit lift
   `L_k`, metric `L_k^dagger L_k`, scalar energy, and pairing adjoint. The
   already tested branch quotient is internally covariant but changes the
   response Hilbert space, fixed-point weight, and stiffness; it is a distinct
   regulator prediction rather than evidence for the paper's finite-cutoff
   calculation.
3. **Cutoff-limit sequence:** is the only option directly representable by the
   unchanged ordinary single-representative parent path, but it is not selected
   as the paper's physical regulator. Section III C and Appendix B.3 of
   `2312.11617v1` fix radial parent and interaction cutoffs at `4|q1|` and
   `3|q1|`: the average-scheme remote Fock term converges slowly, while states
   approaching the eV scale lie outside the continuum model's stated validity;
   Appendix C.3 compares a larger-cutoff HF phase diagram. An infinite-cutoff
   extrapolation is therefore not a paper-authorized physical regulator
   removal. The shell-4/5/6 parent-only data do not show C3 or energy
   convergence, but they do not exclude every possible joint parent/interaction
   cutoff sequence.
4. **Alternative non-permutation representation:** remains unspecified and
   would require new basis functions plus newly derived interaction vertices
   and remote reference. There is no current paper or code evidence selecting
   one.

Accordingly, no option is presently justified as the corrected physical
functional. The safe decision is:

```text
Track P  = the paper-declared finite-cutoff hypothesis test;
Track C  = an unselected regulator research program, not a production model.
```

`c3_affine_ws_v1` must remain fail-closed. A future selection requires either
author-supplied finite-rank/boundary conventions or an explicitly new UV
completion with its own physical interpretation. Shell-7/8 parent-only
extrapolation by itself would not resolve that missing UV choice and is not
authorized.

## Implemented safe boundary

The code currently provides only:

- valley-aware fixed-sector affine support closure;
- exact support provenance and fail-closed validation;
- a transient 27-point fixed parent;
- an explicit C3 unitary and strict parent-Hamiltonian covariance gate.

These are geometry/one-body prerequisites. They must not be interpreted as
a complete Track-C provider or as a Fig. S45 reproduction.
