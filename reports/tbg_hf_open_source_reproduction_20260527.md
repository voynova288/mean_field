# TBG-HF Open-Source Reproduction Gate, 2026-05-27

## Scope

This note records a direct run of the public `TBG-HF` code cloned at:

```text
reference/TBG-HF
commit 0d2a3d742aa901fa45ce46690c1385887165f58c
```

The code accompanies:

```text
Y. H. Kwan, Z. Wang, G. Wagner, N. Bultinck, S. H. Simon,
S. A. Parameswaran, "Mean-field Modelling of Moire Materials:
A User's Guide with Selected Applications to Twisted Bilayer Graphene",
arXiv:2511.21683.
```

It is also a simplified public version of code used in Kwan et al.,
PRX 11, 041063 (2021) and Wagner et al., PRL 128, 156401 (2022).

This is not Zhang/Lu/Liu PRL 128, 247402 HF+cRPA code.  It has no public
remote-band cRPA dielectric pipeline.  The test below only checks whether a
public TBG HF code can reproduce its own documented examples.

## Working Copies

The original clone was not modified.  Runs used copied work directories:

- `results/TBG_HF_cRPA/open_source_repro_20260527/TBG-HF_default_run`
- `results/TBG_HF_cRPA/open_source_repro_20260527/TBG-HF_example2_run`

## Default End-to-End Gate

Slurm job:

```text
131554 tbg_hf_open_gate COMPLETED 0:0 elapsed 00:00:06
```

Command sequence:

```text
python test_singleParticle.py
python test_mainProgram.py
```

Result:

- `test_singleParticle.py` completed and wrote the BM/interactions cache.
- `test_mainProgram.py` converged at HF iteration 47.
- Final printed observables:
  - energy: `1.7700e+01`
  - gap: `9.7176e-04 eV`
  - difference: `9.9912e-09`
  - valley polarization: `1.8750e-01`
  - IVC: `3.7088e-14`
  - spin polarization: `-1.6250e+00`
  - C2T break: `4.8023e-14`

This confirms that the public code runs cleanly in this environment for its
default HF example.

## Example 2 Gate

The README says `example2` should generate a plot similar to Kwan et al. PRX
11, 041063 (2021), Fig. 3a, showing a KIVC-to-IKS transition versus strain.

Initial sequential job:

```text
131570 tbg_hf_ex2_gate CANCELLED after 00:10:22
```

Reason for cancellation:

- The upstream example is sequential over `7 strains x 2 boosts x 10 seeds`.
- The test partition limit is 30 minutes.
- At the observed sequential rate, completion was at risk.
- Completed seed directories with `output.json`, `P.npy`, and `HFeigs.npy` were kept.

Replacement resume/parallel job:

```text
131577 tbg_hf_ex2_par COMPLETED 0:0 elapsed 00:03:28
```

The resume runner only changed outer scheduling:

- Core HF computation still calls the public `mainProgram.mainP`.
- Completed cases were skipped.
- Missing or incomplete cases were run in parallel with 8 workers.
- Thread variables were set to one thread per worker.

Completion:

```text
cases = 140
done = 102
skipped = 38
missing output.json = 0
```

Generated output:

- `results/TBG_HF_cRPA/open_source_repro_20260527/TBG-HF_example2_run/example2.png`
- `results/TBG_HF_cRPA/open_source_repro_20260527/TBG-HF_example2_run/example2_lowest_energy_summary.csv`

## Lowest-Energy Summary

The lowest-energy solution at each strain is:

| Strain (%) | Boost | Seed | Energy | Gap (meV) | IVC | T break | T' break |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.000` | 0 | 3 | `41.788914833937` | `14.116340` | `0.496957` | `3.984052` | `0.008458` |
| `0.050` | 0 | 9 | `41.684656446157` | `11.414240` | `0.368569` | `2.948552` | `0.000000` |
| `0.100` | 0 | 9 | `41.413027063641` | `6.962922` | `0.149407` | `1.195256` | `0.000000` |
| `0.150` | 4 | 2 | `41.041237875337` | `17.648101` | `0.429997` | `0.000244` | `3.440186` |
| `0.200` | 4 | 4 | `40.606415408978` | `16.202678` | `0.421638` | `0.000000` | `3.373104` |
| `0.250` | 4 | 6 | `40.158523487021` | `14.664772` | `0.412382` | `0.000000` | `3.299060` |
| `0.300` | 4 | 4 | `39.698520323327` | `13.071357` | `0.402267` | `0.000000` | `3.218134` |

Interpretation:

- The lowest-energy boost changes from `boost=0` to `boost=4` between
  `0.10%` and `0.15%` strain.
- This matches the README statement that the KIVC-to-IKS transition is between
  `0.1%` and `0.15%` strain for the bundled example parameters.
- The generated `example2.png` was overwritten by the local run at
  `2026-05-27 15:58:35 +0800`.

## Relevance To The Zhang HF+cRPA Debug

This open-source run is useful but limited:

1. It confirms that a public continuum TBG HF implementation reproduces its own
   documented mean-field example in this environment.
2. It is a useful external anchor for BM basis, periodic form factors,
   `P - P_ref`, and Hartree/Fock contraction conventions.
3. It does not reproduce Zhang PRL 128, 247402 Fig. 4 because it does not
   include Zhang's remote-band cRPA dielectric construction or the HF+cRPA
   screened-interaction insertion.
4. Therefore it cannot replace the current Zhang-specific cRPA audit.  The
   next Zhang-specific source question remains the constrained-polarizability
   target-subspace definition and screened-interaction normalization.
