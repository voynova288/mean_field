# RLG/hBN Track-P smallest-q cutoff evidence inventory

Date: 2026-08-09
Status: **INTERSPIN CUTOFF EXTRAPOLATION NOT AVAILABLE FROM EXISTING SHELL-5/6 FULL MESHES**

## Core finding

The previously proposed task “extract the shell-5/6 smallest-q spin-stiffness cutoff extrapolation from the existing full-mesh artifacts without rerunning” is not executable as stated.

The shell-5 and shell-6 full-mesh summaries contain only the **intraflavor** channel:

```text
scope = full 12x12 independently assembled intraflavor mesh
q_count = 144
```

The Goldstone spin stiffness belongs to the **interspin** channel. The existing shell-5/6 full meshes therefore cannot be reinterpreted as spin-wave data. Doing so would mix different pair inventories and different collective branches.

This is a data-inventory conclusion, not a numerical failure and not a downgrade of the Fig. S45 reproduction assessment.

## Bound artifacts

```text
shell-4 intervalley/interspin full mesh
results/RnG_hBN/tdhf_m2_pilot/paper_finite_cutoff_hypothesis_v1_20260724/
  track_p_flavorflip_fullmesh_144q_summary.json
SHA256 2fd12c9850c1278ee650ba1ec63707b0cf134373023e836f4779c1864208f3de

shell-5 intraflavor full mesh
results/RnG_hBN/tdhf_m2_pilot/track_p_parent_cutoff_v1_20260726/shell5/fullmesh/
  track_p_shell5_fullmesh_144q_summary.json
SHA256 72f344711c02d8ce5714285418d07ac1deedac17a8bfc9dbc7e5938ca9088d7a

shell-6 intraflavor full mesh
results/RnG_hBN/tdhf_m2_pilot/track_p_parent_cutoff_v1_20260726/shell6/fullmesh/
  track_p_shell6_fullmesh_144q_summary.json
SHA256 d7478e4f4609f57bb21e943387471f753666015c0e128478784821156fa26dbd
```

## Existing shell-4 interspin checkpoint

At the first reciprocal-mesh shell,

```text
|q|   = 0.0520429 nm^-1
|q|^2 = 0.002708459705957871 nm^-2
```

The three independently assembled C3-related interspin points give

```text
Omega(q) = 0.13345586367003584,
           0.13370750924551700,
           0.13620334580704610 meV

D = Omega/|q|^2 = 49.273712057251345,
                  49.366623011373220,
                  50.288119667217440 meV nm^2
```

The digitized paper checkpoint is

```text
Omega_paper = 0.13092446015825487 meV
D_paper     = 48.33908360174485 meV nm^2
```

Thus the existing fresh shell-4 Track-P interspin result is paper-close at the first shell. Here `D=Omega/|q|^2` is a first-shell effective stiffness estimator; one mesh shell alone does not establish a continuum q→0 fit.

## Why the shell-5/6 smallest-q rows cannot substitute

The shell-5/6 summaries do contain q labels `(1,0)`, `(0,1)`, and `(-1,-1)`, but those rows are intraflavor modes:

```text
shell 5: 3.5190343151, 3.5208745005, 3.5480564847 meV
shell 6: 3.5668315896, 3.5676585667, 3.5950642315 meV
```

Their opposite signed representatives are also finite near `3.30--3.39 meV`. These are not the q→0 SU(2) magnon branch and `Omega/|q|^2` must not be labeled a spin stiffness.

## Extrapolation boundary

No shell-5 or shell-6 interspin first-shell energies are currently archived. Consequently:

```text
shell-5 interspin stiffness: unmeasured
shell-6 interspin stiffness: unmeasured
cutoff extrapolation:        not defined
```

Even after obtaining three cutoff values, a continuum-cutoff extrapolation would require a preregistered cutoff variable and fit law. The observed Track-P intraflavor and frozen-qualifier energies vary nonmonotonically across parent shells; interspin shell-5/6 behavior remains unmeasured. A post-hoc linear fit in `1/shell_count` would therefore not be justified automatically.

## Minimal future calculation, only if requested

A narrow qualifier would reuse the already converged shell-5/6 HF archives and q=0 Ward gates, then independently assemble the interspin signed sectors for the three first-shell C3 representatives:

```text
(1,0), (0,1), (-1,-1)
```

Each signed calculation must retain its independently built q and -q matrices and provenance. No full 144-q rerun is needed. This extra cutoff qualifier is optional under the current benchmark scope; existing artifacts do not authorize presenting it as already measured.
