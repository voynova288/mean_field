# RLG/hBN Fig. 6 Chern mismatch status

Date: 2026-05-28

## What matches the paper so far

Completed-task Chern aggregation:

- `results/RnG_hBN/fig6_completed_task_chern_20260528_1112/completed_task_chern_report.md`
- `results/RnG_hBN/fig6_completed_task_sector_chern_20260528_1120/all_sector_occupied_conduction_chern_report.md`

Current completed pure-sector result that matches Fig. 6:

| task | panel | occupied-like sector | err | C | |C| | paper | match |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `task_16_xi1_V064meV_flavor_seed1` | xi1 | spin=0, eta=0 | `2.1859e-5` | -1 | 1 | 1 | yes |

All completed xi0 occupied-like sectors still disagree with the paper expectation `|C|=0`.

## Important new finding: flavor seeds did not scan flavors

The Fig. 6 array intended to run `flavor:1`, `flavor:2`, `flavor:3`, `flavor:4`.  However, current initialization code ignores `seed` for `init_mode="flavor"`:

```python
counts = rlg_hbn_flavor_occupation_counts_for_init_mode("flavor", ...)
```

and `rlg_hbn_flavor_occupation_counts_for_init_mode` always fills positive `nu=1` in the fixed order:

```text
(spin, eta) = (0,0), (0,1), (1,0), (1,1)
```

So for `nu=1` all four flavor seeds initialize the same extra conduction flavor `(spin=0, eta=0)`.  This is also visible in the completed outputs: `task_0`, `task_1`, `task_2`, `task_3` have identical energy/error/Chern and the all-sector table gives the occupied-like sector `(0,0)` for all of them.

Consequence: the current array has not actually tested the alternative valley/spin polarized xi0 states that could realize the paper's trivial `|C|=0` sector.

## All-sector postprocess evidence

The all-sector postprocess computes the Chern of each spin/eta sector's conduction band and its saved-density occupation.  For the pure xi0 flavor/perturbed states, the only occupied-like sector is always `(spin=0, eta=0)` and has `C=-1`, while the empty sectors often have `|C|=0`.

Representative rows from `all_sector_occupied_conduction_chern.tsv`:

```text
task_0 xi0 flavor:1: occupied sector (0,0), occ=0.999999, C=-1, paper |C|=0 -> mismatch
                        empty eta/spin sectors have |C|=0 but are not occupied
task_8 xi0 perturbed:1: occupied sector (0,0), occ=0.999999, C=-1 -> mismatch
task_16 xi1 flavor:1: occupied sector (0,0), occ=1.000000, C=-1, paper |C|=1 -> match
```

The random/bm xi0 states are not clean pure-sector states; the existing per-sector Chern comparison is therefore not a reliable topological label for their actual occupied HF subspace.

## Previous suspected causes: test status

| Suspect | Test/status | Result so far |
| --- | --- | --- |
| xi0 convergence only | completed xi0 strict random `task_13`, `err=8.7e-7`; flavor/perturbed max-iter states around `4.6e-4`; all still not paper-trivial in occupied-like sectors | convergence alone is not enough |
| q=0 Fock convention | Slurm `131526` running, output `results/RnG_hBN/diag_xi0_q0fock_zero_20260527_151212`; dependent Chern `131528` pending | not completed yet |
| `(4+4)` active window | Slurm `131527` is held to preserve slots; dependent Chern `131530` pending | not run yet |
| active-window isolation | static diagnostic already done: `min_upper_gap_u_mev = 0.0720` for xi0 `(3+3)` screened basis | confirms fragility but not causal proof |
| q=0 Hartree double count | not yet run | pending |
| hBN moire phase/layer/sign | not yet rerun against paper topology checkpoints | pending |

## Current most likely explanation

At this point, the strongest concrete issue is not generic HF contraction but **incomplete/biased flavor-sector search**:

1. The code/array labels suggest four flavor seeds, but all flavor seeds actually initialize the same occupied flavor.
2. The xi0 mismatch is tied to the occupied `(spin=0, eta=0)` sector, while some unoccupied sectors are trivial.
3. Therefore the present completed xi0 comparison does not prove that the implementation cannot produce the paper's `|C|=0`; it proves that the currently searched occupied sector gives `|C|=1`.

The next decisive test is to rerun xi0 with explicit occupation counts for each flavor, e.g. for `(active_valence=3, n_band=6, nu=1)`:

```text
(spin0,eta0): (4,3,3,3)  # current behavior
(spin0,eta1): (3,4,3,3)
(spin1,eta0): (3,3,4,3)
(spin1,eta1): (3,3,3,4)
```

If one of the alternative occupied flavors converges to lower energy and `|C|=0`, the mismatch is mainly a flavor-initialization/search bug.  If all explicit flavors still converge to topological/mixed nontrivial states, then the remaining suspects are active-window fragility and hBN/q=0 conventions.
