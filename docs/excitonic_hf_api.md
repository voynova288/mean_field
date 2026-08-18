# Generic excitonic matrix-HF API

The public front door is `mean_field.api.excitonic` and is also re-exported
from `mean_field.api`. The implementation lives in
`mean_field.core.hf.excitonic`; physical-system adapters remain under
`mean_field.systems.<system>`.

## Mathematical contract

The API uses an ordinary-electron, ket-oriented density matrix

\[
P_{ab}(k)=\langle c^\dagger_{kb}c_{ka}\rangle,
\qquad 0\le P_k\le I,
\]

with array shape `(n, n, nk)` and explicit positive quadrature weights `w_k`.
The SCF variable is always

\[
D(k)=P(k)-P_{\rm ref}(k),\qquad
H[D](k)=H_0(k)+\Sigma[D](k).
\]

`P_ref` is an explicit input; the solver never silently replaces it by the
noninteracting Fermi state. `absolute_density_builder(H)` returns the absolute
ordinary-electron density `P(H)`, and only the generic solver forms
`D=P-P_ref`.

`LinearSelfEnergyFunctional` declares that `Sigma` is linear and self-adjoint
under the weighted ket-oriented trace pairing. Construction requires an
accepted executable `LinearSelfEnergyCertificate`; the convenient
`LinearSelfEnergyFunctional.from_probes(...)` constructor runs two independent
Hermitian probes and rejects failed linearity or weighted-self-adjointness.
System adapters must bind the certificate to the concrete operator fingerprint
and remain responsible for broader production-scale certification

\[
\langle A,B\rangle_w=\sum_k w_k\,\mathrm{Tr}(A_k B_k).
\]

Under this contract,

\[
\Delta U=\sum_k w_k\mathrm{Tr}(H_{0,k}D_k)
 +\frac12\sum_k w_k\mathrm{Tr}(\Sigma[D]_kD_k).
\]

A component builder may expose, for example, separate Hartree and Fock terms,
but their sum must close numerically to the total self-energy.

Finite-temperature occupation uses `thermal_energy = k_B T` in the same
energy units as `H`. Every solver input is a typed
`ThermodynamicDensityBuilder` binding that thermal scale and its constraint.
`FixedChemicalPotential(mu)` evaluates one immutable ordinary-electron
chemical potential and performs no number or neutrality root.
`FixedOccupation(n)` solves only the weighted total occupation. System-specific
neutrality constraints may wrap their own map in a typed builder with the same
thermal attestation.

For a grand-canonical run,

\[
\Delta\Omega=\Delta U-(k_BT)\Delta(S/k_B)-\mu\Delta N.
\]

The chemical potential in `ReferenceSubtractedHFConfig.grand_canonical_mu` is
used only in this Legendre transform; it is not changed by the solver. A grand
potential is rejected unless the typed density builder carries the same
immutable fixed chemical potential. Final raw fixed-point residual is the
single convergence authority: an iteration-level pass is demoted to
`final_raw_residual` if the recomputed final state misses the declared gate.

## Electron-hole diagnostics

`ElectronHoleSubspaces` requires explicit disjoint index sets. It never infers
band identity from energy order or assumes equal electron/hole ranks. For an
ordinary-electron basis it reports

\[
n_e=\sum_k w_k\operatorname{Tr}(Q_eP_k),\qquad
n_h=\sum_k w_k\operatorname{Tr}[Q_h(I-P_k)],
\]

and singular values of explicitly distinguished blocks such as `D_eh` and
`Sigma_eh`. In an excitonic calculation the interaction-induced order
parameter is normally the self-energy block `Sigma_eh`; a pre-existing bare
`H0_eh` is not silently relabeled as excitonic order.

## Minimal example

```python
import numpy as np
from mean_field.api import (
    ElectronHoleSubspaces,
    FixedChemicalPotential,
    LinearSelfEnergyFunctional,
    ReferenceSubtractedHFConfig,
    make_fermi_density_builder,
    run_reference_subtracted_hf,
)

h0 = np.zeros((2, 2, 1), complex)
weights = np.ones(1)
p_ref = 0.5 * np.eye(2)[:, :, None]
thermal_energy = 0.2
coupling = 8 * thermal_energy * np.arctanh(0.5)

density_builder = make_fermi_density_builder(
    weights,
    thermal_energy=thermal_energy,
    ensemble=FixedChemicalPotential(0.0),
)
seed = np.array([[0.0, 0.25], [0.25, 0.0]], complex)[:, :, None]
probe2 = np.array([[0.1, 0.03j], [-0.03j, -0.1]], complex)[:, :, None]
interaction = LinearSelfEnergyFunctional.from_probes(
    lambda density_delta: -coupling * density_delta,
    seed,
    probe2,
    weights,
    validation_label="exact two-level analytic oracle",
    component_builder=lambda density_delta: {
        "exchange": -coupling * density_delta,
    },
    label="exchange",
)

result = run_reference_subtracted_hf(
    h0,
    weights,
    p_ref,
    absolute_density_builder=density_builder,
    interaction=interaction,
    config=ReferenceSubtractedHFConfig(
        thermal_energy=thermal_energy,
        search_mode="seeded_ei",
        mixing=1.0,
        precision=1e-12,
        grand_canonical_mu=0.0,
    ),
    initial_density_delta=seed,
    electron_hole_subspaces=ElectronHoleSubspaces((0,), (1,)),
)
```

This analytic one-k model obeys

\[
x=\frac12\tanh\frac{gx}{2k_BT}
\]

with the exact seeded solution `x=1/4` and interaction energy `-g*x**2`.

## System boundary

The generic module does **not** choose:

- a physical Hamiltonian or band/projector window;
- E1/H1, layer, orbital, spin, or valley identities;
- Coulomb screening, dielectric boundaries, singular-cell quadrature, or
  microscopic form factors;
- the source or electrostatic ensemble of a fixed chemical potential;
- a normal-ordering reference appropriate to a particular device.

The InAs/GaSb adapter demonstrates the intended split: it retains Kane bundle
fingerprints, E1/H1 conventions, microscopic carrier projectors, interaction
attestation, and the E1-empty/H1-filled vacuum while delegating the generic SCF
and thermodynamic bookkeeping to this API.

## Validation boundary

Unit tests cover fixed-mu energy-zero covariance, nonuniform weighted-number
roots, complex off-diagonal density orientation, unequal-rank E/H subspaces,
linear/self-adjoint interaction checks, an exact finite-temperature excitonic
fixed point, and the interaction-off reference limit. A nonzero microscopic
InAs/GaSb toy agrees with the pre-refactor SCF arrays and trajectories to
`2e-14` absolute tolerance. The pre-refactor nonzero-Fock arrays, generator,
and relevant source snapshots are stored as a hash-attested test fixture. These checks establish API and reduced-path
parity; they do not establish radial, angular, ultraviolet, or material-model
convergence for any production system.
