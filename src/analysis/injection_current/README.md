# Generic injection-current / CPGE module

This package is the common formula API for injection-current and CPGE calculations. New
optical workflows should usually enter through `analysis.optical` and choose
`kind="injection_current"` or `kind="cpge"`; import this package directly only
when injection-current specific controls are needed.

It is intentionally small: it reuses the Hamiltonian-gauge velocity and Berry
connection ingredients already used by `analysis.shift_current` and adds only
the length-gauge injection kernel

```text
D_mn^{c,ab} = Delta_mn^c r_nm^b r_mn^a,
Delta_mn^c = v_mm^c - v_nn^c.
```

For an occupied initial band `n` and empty final band `m`, use:

```python
from analysis.injection_current import (
    positive_injection_transition_terms,
    precompute_injection_current_tensors,
)

tensors = precompute_injection_current_tensors(energies, evecs, dhdk)
transitions, weights = positive_injection_transition_terms(tensors, "y;yy")
```

The returned weights are complex. The real part corresponds to the linear
injection channel; the imaginary part corresponds to circular injection / CPGE.
The optional spectrum helper uses the toy-unit Lin--Hsu prefactor `-2*pi` with
`e = hbar = tau = 1`. Production SI prefactors, scattering time, spin/valley
factors, and paper colorbar normalization are workflow-layer choices.

## Reuse boundary

- Reused from `analysis.shift_current`: component parsing, Fermi occupations,
  Hamiltonian-gauge velocity matrices, Berry connection, positive transitions,
  Lorentzian accumulation.
- New here: injection kernel `Delta v * r * r`, complex linear/CPGE split, and
  toy-model checkpoint helpers.  The Haldane helpers include optional C3
  symmetrized quadrature to reduce finite-grid artifacts without smoothing or
  changing the response formula.
- Not here: paper figure orchestration, Slurm sweeps, Parker velocity-gauge
  extraction, or moire-system HF response conventions.

## Haldane toy-model oracle

`toy_models/haldane.py` retains the Lin--Hsu 2025 Haldane Hamiltonian, analytic
first/second derivatives, primitive-BZ quadrature, spectrum helpers, and local
quantum-metric/injection/shift kernels as independent formula and symmetry
oracles. The Checkpoint-A-specific zone-average scan, CLI, plotting, and
artifact-writing workflow are archived rather than maintained inside this
generic analysis package. Unit tests validate software/numerical invariants;
they do not establish paper-level reproduction.
