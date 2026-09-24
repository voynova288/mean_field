# Du2017 atomistic finite-BZ parent candidate audit

**Date:** 2026-08-07
**Scope:** completed external-model audit retained for provenance only
**Verdict:** **USER-DISALLOWED for the Du2017 task; no follow-up permitted**

> **User scope correction (2026-08-07):** Du2017 reproduction must use only the model family in the Du paper, supplement, thesis, and author/senior-code lineage. Tan/Uptight, atomistic, Wannier, and other external parents may not replace it. The audit below is historical provenance, not an active plan.

## Why this audit was attempted

The fixed-ordinary-electron-`mu_el` radial carrier-Fock branch fails momentum-window closure at `kmax=0.24 nm^-1`. Its interaction-induced component remains large at the endpoint, so neither enlarging the unregularized continuum cutoff nor absorbing only the normal-carrier Fock term repairs the model. Earlier source-bound audits also found generic square-lattice finite-BZ completions edge dominated and stencil dependent.

This audit initially tested whether an independent microscopic parent could address that pathology. The user subsequently ruled this route out for the Du2017 reproduction: the correct continuation is instead to reconstruct any undocumented cutoff or numerical convention inside the Du model itself.

## Public candidates

### 1. Uptight plus Tan-2016 parameters — viable engineering candidate

Public source:

- [tiberlab/uptight](https://github.com/tiberlab/uptight), LGPL-3.0;
- pinned commit `72d0d82361575e3b48d13b7442d575424cf5a0b3`;
- [Tan et al., PRB 94, 045311 (2016)](https://doi.org/10.1103/PhysRevB.94.045311), [arXiv:1603.05266](https://arxiv.org/abs/1603.05266).

The repository contains first-neighbour, spin-orbit-coupled, strain-aware `sp3d5s*` parameter files for all three central binary materials:

```text
parameters/Tan/InAs.etb  sha256 8d180ef4ef763a8feaf96c33f475559c0387e90532394d0c84fcc7c0718f6b05
parameters/Tan/GaSb.etb  sha256 3ce375ca7ed2f60189180398f5a9eb0031d6c6ec994a4f04eb09274079532f1f
parameters/Tan/AlSb.etb  sha256 dde19b15b755600271b0ccf209daf922f18167111a63a0f133a3634ecdf83b45
LICENSE.txt              sha256 e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118
```

Tan et al. explicitly validate the transferable model against HSE06 for four- and eight-atomic-layer InAs/GaSb superlattices, including the no-common-cation/no-common-anion interface, and also report InAs/AlSb superlattice gaps. This is materially more relevant than a generic lattice regularization.

Uptight's typed supercell data support per-atom material labels, zincblende geometry, spin-orbit coupling, strain, `xy`/`xyz` periodicity, sparse eigensolvers, and an externally supplied static potential file. These are sufficient in principle for a finite-BZ noninteracting heterostructure benchmark.

### 2. NanoTB-sp3d5s* — not source-complete

[yyyu200/NanoTB-sp3d5s_star](https://github.com/yyyu200/NanoTB-sp3d5s_star) describes a relevant atomistic model but publishes only a static executable archive and states that executable/input examples are available by contacting the author. It is not an independently auditable source authority for this project.

### 3. Historical InAs/GaSb ETB papers — method evidence, not executable authority

[Wei and Razeghi, PRB 69, 085316 (2004)](https://doi.org/10.1103/PhysRevB.69.085316) models type-II InAs/GaSb superlattices using empirical `sp3s*` tight binding and interface engineering. It confirms that atomistic interface chemistry matters, but no complete public code/input bundle was identified.

The APS abstract by Wu, Soluyanov, and Troyer on an ab-initio/Wannier InAs/GaSb quantum-well investigation did not lead to a public reproducible Hamiltonian or parameter archive in the search performed here.

## Isolated engineering gate

A clean shared-home clone of Uptight was pinned to the commit above and compiled only on `test001`; no login-node numerical work or Slurm job was used.

The upstream README says to run `./src/configure` from the repository root, but that places `make.common` in the root while `src/lib_uptight/Makefile` requires `../make.common`, i.e. `src/make.common`. The working build sequence is:

```text
cd src
./configure
cd lib_uptight
make uptight.a
```

Result:

```text
uptight.a size       892448 bytes
uptight.a sha256     077308fe7d403e6ad3e93f3dbfb1ca4a1c50b416582dac3a6328732565c239f1
```

`test001` has versioned LAPACK/BLAS runtime libraries but no unversioned development symlinks, so the default `-llapack -lblas` link fails. Without installing anything, the example executable links by explicitly supplying the available SONAME paths:

```text
make PARAMETERIZER LIBMKLIB="/lib64/liblapack.so.3 /lib64/libblas.so.3"
PARAMETERIZER sha256 529a432e32e2245409d8afec025b20420060da5cbb647cb31d739a560d1a251e
```

This validates source retrieval and compilation only.

## Bulk-Gamma source replay

A source-bound two-atom zincblende Gamma calculation was then run for the unmodified current Uptight/Tan parameter files. The source lattice constants, spin-orbit coupling, zero explicit strain, and exact 2025-imported parameter bytes were used.

```text
material  computed Eg_Gamma  computed Delta_SO  Tan-2016 ETB table
InAs       0.346469 eV        0.392618 eV        0.348 / 0.391 eV
GaSb       0.731409 eV        0.682822 eV        0.703 / 0.714 eV
AlSb       2.286577 eV        0.690157 eV        2.225 / 0.642 eV
```

InAs replays within `1.6 meV`, but GaSb and AlSb miss the source-table direct gaps by `+28.409/+61.577 meV` and the spin-orbit splittings by `-31.178/+48.157 meV`. The direct gaps are closer to the room-temperature reference values, suggesting that the 2025 TiberCAD-imported files may encode a later device-oriented convention rather than the exact 2016 table Hamiltonian. This is not yet uniquely established.

All selected states remain Kramers paired. The single-k path coordinate written by `PARAMETERIZER` is `NaN` because the upstream example divides by zero path length; the finite energy column is independently parsed. Full receipts are under `fixed_mu_full_radial_scf/atomistic_finite_bz_candidate_preflight/`.

The antimonide mismatch fails the source-parity gate before any interface or quantum-well construction. No flag or parameter scan is authorized to fit it away.

## Why this is not yet a physical GO

The public atomistic code and bulk parameters do not uniquely specify the Du2017 device:

1. **Atomic layer sequence and interface termination.** The 11.5/8 nm nominal wells do not uniquely determine integer monolayer counts, InSb-like versus GaAs-like no-common-atom interfaces, segregation/intermixing, or the full Al(Ga)Sb cap/barrier stack.
2. **Low-temperature offsets.** Tan's parameterization contains offset corrections and a `delta_d` convention distinguishing room-temperature experimental targets from zero-temperature HSE06 targets. A 1.4 K choice must be declared and validated rather than selected by agreement with Fig. 2.
3. **Electrostatics.** Uptight accepts a static potential file but the inspected source tree does not provide the missing Du-specific self-consistent gate/Poisson boundary data.
4. **Interaction completion.** A full atomistic Coulomb normal-ordering reference, dielectric environment, ionic/background charge, and exchange/pairing window must be derived anew. The continuum radial interaction tensor cannot be reused.
5. **Model identity.** A Tan/Uptight calculation is an independent microscopic benchmark, not the unpublished Du2017 Hamiltonian and not evidence for the paper's hidden Fig. 2 conventions.

## Historical staged path — cancelled by scope correction

The sequence below is retained only to explain what was considered. None of these stages may be continued for the Du2017 task:

1. **Source gate:** archive the exact external commit, license, parameter bytes, compiler, and build receipt.
2. **Bulk gate:** reproduce source-paper InAs/GaSb/AlSb bulk band edges, spin-orbit splittings, and effective masses with both zero-temperature and room-temperature offset conventions kept separate.
3. **Interface gate:** reproduce Tan's published four-/eight-layer InAs/GaSb superlattice checkpoints and compare both interface terminations without selecting by the Du target.
4. **Geometry gate:** enumerate integer-monolayer realizations compatible with the reported 11.5/8 nm wells and explicitly list unresolved cap/barrier/interface choices.
5. **Low-energy parity gate:** compare the atomistic E1/H1 quartet against the N95 Kane-Poisson parent only in a predeclared near-Gamma window, using energies and subspace overlaps. No fitting or energy alignment may hide disagreement.
6. **Finite-BZ charge gate:** define occupied-density and ionic/background subtraction over the complete lattice BZ before any Poisson iteration.
7. **Interaction gate:** derive atomistic Coulomb matrix elements and normal ordering in the same microscopic basis before projected HF/BCS.

## Decision

This route is closed by user scope, independently of its technical quality. No Tan/Uptight parameter follow-up, interface model, or quantum-well construction will be performed for Du2017.

The active path is Du-only source forensics: test undocumented or result-selected continuum momentum cutoffs, retained-subband windows, density/reference choices, quadrature, chemical-potential semantics, and diagonal-Fock conventions. Any unsupported convention that reproduces the figure must remain explicitly labelled forensic/uncontrolled, not predictive physics.
