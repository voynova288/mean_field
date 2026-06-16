# TDHF API contract

The stable TDHF/RPA façade is:

```python
from mean_field.api import TDHFConfig, run_tdhf
```

`run_tdhf(hf_result_or_archive, config)` freezes the public call shape.  Existing dense q=0 pilots and system-specific archive loaders remain in their current modules until adapters expose a uniform `run_tdhf(config)` hook.

## TDHFConfig

`TDHFConfig` records:

- collective momentum sector, e.g. `q0` or integer grid shift;
- flavor/spin/valley channel selection;
- pair-count and dense-memory safety limits;
- assembly route (`auto`, `generic`, `vectorized`, or system-specific choices);
- extra metadata.

## Required upstream state

TDHF uses a converged HF basis/eigenvalue set and HF-basis two-body matrix elements.  It should not use an unrelated one-body Hamiltonian in place of the HF spectrum.  If the system needs microscopic form factors or layer-resolved Coulomb tensors, the HF result/archive adapter must expose them explicitly.

## Current status

The façade raises `NotImplementedError` unless the input object supplies `run_tdhf(config)`.  This prevents scripts from silently treating a partially migrated dense pilot as a complete public TDHF implementation.
