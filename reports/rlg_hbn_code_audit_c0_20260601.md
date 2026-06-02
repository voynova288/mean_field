# RLG/hBN C=0 reproduction code audit

Date: 2026-06-01

## Main code issue found

The Fig. 6 `flavor:1..4` jobs did **not** actually scan four flavor sectors.  In the previous code path, `init_mode="flavor"` ignored `seed`, and `rlg_hbn_flavor_occupation_counts_for_init_mode()` always used the fixed flavor order

```text
(spin, eta) = (0,0), (0,1), (1,0), (1,1)
```

For `nu=1`, this always made the extra conduction electron occupy `(spin=0, eta=0)`, so `flavor:1`, `flavor:2`, `flavor:3`, and `flavor:4` were duplicates.  This is visible in the completed outputs: `task_0..task_3` have identical energies, errors, and Chern numbers.

This is a plausible reason the xi0 `|C|=0` state was not found: the current completed xi0 flavor scan only tested the `(0,0)` sector, whose occupied conduction band has `C=-1`, while some empty sectors in the postprocess have `|C|=0`.

## Patch applied

File changed:

```text
src/mean_field/systems/RnG_hBN/hf.py
```

Change:

- Added optional `seed` handling to `rlg_hbn_flavor_occupation_counts_for_init_mode()`.
- For `init_mode="flavor"`, seeds now rotate the occupied flavor order:

```text
seed=1 -> (4,3,3,3)  # extra electron in spin0, eta0
seed=2 -> (3,4,3,3)  # extra electron in spin0, eta1
seed=3 -> (3,3,4,3)  # extra electron in spin1, eta0
seed=4 -> (3,3,3,4)  # extra electron in spin1, eta1
```

For `perturbed`, `bm`, and `random`, behavior remains unchanged in the occupation-count resolver (`None` counts / global Aufbau where applicable).

Validation run on login node only:

```bash
PYTHONPATH=src python -m py_compile src/mean_field/systems/RnG_hBN/hf.py
PYTHONPATH=src python - <<'PY'
from mean_field.systems.RnG_hBN.hf import rlg_hbn_flavor_occupation_counts_for_init_mode
for seed in [1,2,3,4,5]:
    print(seed, rlg_hbn_flavor_occupation_counts_for_init_mode('flavor', nu=1, active_valence_bands=3, n_spin=2, n_eta=2, n_band=6, seed=seed))
print('perturbed', rlg_hbn_flavor_occupation_counts_for_init_mode('perturbed', nu=1, active_valence_bands=3, n_spin=2, n_eta=2, n_band=6, seed=2))
PY
```

Output:

```text
1 (4, 3, 3, 3)
2 (3, 4, 3, 3)
3 (3, 3, 4, 3)
4 (3, 3, 3, 4)
5 (4, 3, 3, 3)
perturbed None
```

## Caveat for existing jobs

Already running jobs will not see this code change.  Held/requeued flavor jobs that already have `hf_checkpoint_latest.npz` may resume from old duplicated-flavor checkpoints unless their checkpoint/run directory is reset.  Therefore a clean decisive test of xi0 `C=0` should submit fresh explicit flavor-sector runs or delete/reset the affected old flavor checkpoints.

## Slurm release action

After the code audit/patch, the held Fig. 6 array tasks were released on `login002`:

```text
127851_7, 127851_11, 127851_12, 127851_15, 127851_19, ..., 127851_25
```

They are no longer `JobHeldUser`; pending array tasks are limited by the existing `ArrayTaskThrottle=3` until current array tasks finish.
