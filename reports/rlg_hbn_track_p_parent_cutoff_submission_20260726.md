# RLG/hBN Fig. S45 Track-P parent-cutoff qualifier submission

Date: 2026-07-26
Status: **RUNNING — no shell-5/6 spectrum result is claimed in this report**

## Purpose

The current Track-P 12×12 calculation is already close to the published Fig. S45 bottom row in the intervalley and interspin channels. The remaining visually important disagreement is concentrated in the intraflavor panel:

- paper-white regions remain real-stable in a small set of calculated sectors;
- several darkest paper markers have small but visibly higher calculated real energies.

The displayed paper panel is raster evidence rather than author raw numerical data. In particular, a paper-white marker establishes `static-negative OR complex`, not uniquely an imaginary frequency, and the darkest colored markers do not prove exact zero energy. Repeated-zone marker counts also differ from the number of independent torus q sectors.

The purpose of this qualifier is to test whether the discrepancy is caused by the finite parent plane-wave cutoff while leaving the physical interaction-transfer shell unchanged.

## Baseline

The validated Track-P baseline uses

```text
parent shell_count              4
parent N_G                     19
single-valley parent dimension 190
interaction_cutoff_q1           3
physical interaction G shifts 13
active interaction             fixed_g_torus_single_representative_v1
physical shift policy           fixed_abs_g_shell_v1
frozen remote h0 policy         actual_node_ws_c3_fixed_copy_average_v1
```

The current 144-q map is

```text
reports/figures/
  rlg_hbn_figs45_track_p_paper_finite_cutoff_hypothesis_20260724.png
```

Current quantitative comparison:

```text
intraflavor stable-energy RMSE  0.5594800270508348 meV
intervalley RMSE                0.1302131317036424 meV
interspin RMSE                  0.0499155294095046 meV
intraflavor complex sectors    35
max intraflavor |Im Omega|      2.970758949601387 meV
paper-white TP/FN/FP/TN        35/10/0/60
```

The ten independent paper-white/current-real-stable sectors are

```text
(0,3), (0,-3), (3,0), (-3,0), (3,3), (-3,-3),
(4,5), (-4,-5), (5,4), (-5,-4)
```

Their shell-4 values are not numerical zero modes: the selected positive real energies are approximately `0.944--1.684 meV`, the static-Hessian minima are positive, and `max |Im Omega|` is only numerical noise (`~1e-13 meV`). Their C3 closure contains 18 torus labels and is retained as a cutoff qualifier because finite Track P is not exact C3.

A second conditional qualifier tracks the nine torus sectors producing the twelve darkest displayed raster markers. This remains explicitly raster-derived; their digitized paper energies are finite (`~0.752`, `0.902`, and `1.168 meV`).

## Controlled cutoff experiment

Two fresh parent calculations were submitted:

| shell_count | N_G | parent cutoff (nm^-1) | single-valley dimension |
|---:|---:|---:|---:|
| 5 | 31 | 1.8028176966 | 310 |
| 6 | 43 | 2.1633812359 | 430 |

Controlled quantities:

```text
layer_count                    5
xi                             1
theta                          0.77 deg
physical displacement field   64 meV
screening epsilon_r            5
gate distance                 10 nm
active valence/conduction      0/4
k mesh                         12x12
interaction_cutoff_q1          3 (13 shifts; unchanged)
nu                             1
occupation_counts              (1,0,0,0)
initialization                 flavor, seed=1
active quotient                disabled
```

For each shell, screening, projected basis, frozen-remote `h0`, overlap cache, Track-P provider, and HF source are rebuilt. The converged shell-4 density is **not** inserted into the new active coordinates. Shell 5 and 6 start independently from the same explicit flavor sector.

Before HF, the runner embeds shell-4 and target-shell wavefunctions on their common raw reciprocal grid and computes all four-band active-subspace singular values. The run fails closed if the minimum singular value is below `0.5` or any value is nonfinite.

## Slurm submission

```text
array job              194628
array tasks             shell 5 and shell 6, throttle 2
account                 hmt03
partition               regular6430
resources per task      1 full node, 64 CPU, exclusive, mem=0
wall time               7 days
excluded node           gpuh2002
shell 5 node            node020
shell 6 node            node025
```

At the latest check both tasks were running, the warm-source hash/config/provider/source-closure preflight had passed, and both stderr files were empty. No q=0 or finite-q shell-5/6 result existed yet.

Logs:

```text
logs/rlg_tp_cut56_hf_194628_5.out
logs/rlg_tp_cut56_hf_194628_5.err
logs/rlg_tp_cut56_hf_194628_6.out
logs/rlg_tp_cut56_hf_194628_6.err
```

Outputs:

```text
results/RnG_hBN/tdhf_m2_pilot/track_p_parent_cutoff_v1_20260726/
  shell5/hf/
  shell6/hf/
```

Immutable source snapshot:

```text
/data/home/ziyuzhu/.runs/Mean_Field_6703203_cutoff56_20260726
base commit: 6703203
```

Runner and qualifier evidence:

```text
runner:
  tmp/tdhf/run_rlg_hbn_track_p_parent_cutoff_hf_20260726.py
  SHA256 5e680e1673a925eae2b001eb36257c8429f6d8266e1e505360cc95fd6d2e04ff

qualifier manifest:
  tmp/tdhf/rlg_hbn_track_p_parent_cutoff_qualifier_manifest_20260726.json
  SHA256 ee1c8d2563a3c1f28fafbcd355f9cf17f74d052988d69ee0777cba85bffbff2f
```

Warm shell-4 source:

```text
results/RnG_hBN/tdhf_m2_pilot/a_v64_single_rep_hf_v1_192734/
  checkpoint_A_average_V64_hf/xi1_V064meV/runs/flavor_seed1/
    hf_run_state.npz

SHA256 44b27d3e676ddb466c7f2451bf669c918b449cee29b20c963713370b993473fd
provider fingerprint f0885cadbcc2ae6e86ea94ea8a595d389194e0180f2575ebdc5bb0825b604635
```

## Post-HF gates

For each successful fresh source:

1. require convergence, typed provider integrity, saved-H closure, projector idempotency, and source stationarity;
2. run q=0 interspin Goldstone/Ward and intervalley wavefunction gates;
3. independently assemble the signed-q intraflavor qualifier, including exact-M, the C3 closure of the paper-white mismatch, and the darkest-raster anchors;
4. save raw eigenvalues, eta norms, solver residuals, signed static-Hessian inertia, classifications, and `|Im Omega|`;
5. compare shell 4/5/6 without C3 orbit copying, matrix averaging, Hermitization, energy fitting, or mask repair.

The planned qualifier has 19 requested labels. Each non-self-conjugate request must save independently assembled `+q` and `-q`, so the resulting coverage is larger than 19 unique signed sectors.

A shell-6 144-q full mesh is authorized only if shell 5 and 6 select the same HF branch, pass q=0 gates, and give stable qualifier classifications/margins. If the ten shell-4 false negatives remain clearly real-stable and cutoff-converged, the result will be reported as strengthened Track-P nonreproduction rather than hidden by a full-mesh plot.

## Interpretation boundary

This study changes the parent plane-wave cutoff only. It does not define the blocked exact-C3 Track-C regulator and does not convert Track P into an exact-C3 calculation. The final claims must remain separated:

```text
Track P: finite-cutoff paper-hypothesis diagnostic
Track C: exact-C3 regulator prediction (still undefined/blocked)
```
