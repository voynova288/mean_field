# RLG/hBN Fig. 6 reduce flavor tasks to two seeds

Date: 2026-06-01

## Code/submission changes

Default Fig. 6 flavor seeds are reduced from four to two:

```text
flavor:1, flavor:2
```

The following files were updated:

- `src/mean_field/devtools/run_rlg_hbn_paper_hf.py`
  - default Fig. 6 run specs now use `flavor:1, flavor:2, bm:1, perturbed:1..4, random:1..4`.
- `scripts/submit_rlg_hbn_paper_hf_array.sbatch`
- `scripts/submit_rlg_hbn_paper_hf_packed_array.sbatch`
- `scripts/submit_rlg_hbn_paper_hf_selected.sbatch`
- `scripts/submit_rlg_hbn_paper_hf_panel_array.sbatch`
  - default Fig. 6 specs/lists now include only `flavor:1, flavor:2`.
- `scripts/submit_rlg_hbn_fig6_existing_resume_exclusive.sbatch`
  - kept the old 26-array-index mapping for compatibility with the already-submitted array `127851`, but added an early skip for old-layout `flavor` seeds `3` and `4`:

```bash
if [[ "${INIT_MODE}" == "flavor" && "${SEED}" -gt 2 ]]; then
  echo "[skip] flavor seed ${SEED} is no longer part of the Fig. 6 production set; keeping only flavor:1,2"
  exit 0
fi
```

Validation:

```bash
PYTHONPATH=src python -m py_compile src/mean_field/devtools/run_rlg_hbn_paper_hf.py src/mean_field/systems/RnG_hBN/hf.py
bash -n scripts/submit_rlg_hbn_paper_hf_array.sbatch \
       scripts/submit_rlg_hbn_paper_hf_packed_array.sbatch \
       scripts/submit_rlg_hbn_paper_hf_selected.sbatch \
       scripts/submit_rlg_hbn_paper_hf_panel_array.sbatch \
       scripts/submit_rlg_hbn_fig6_existing_resume_exclusive.sbatch
```

## Queue cleanup

Actions on `login002`:

- `scontrol requeue 127851_16`
  - This was old-layout Fig. 6 `xi1 flavor:4`, which is no longer needed.
  - Requeue was used instead of `scancel` so the array dependency can still finish cleanly; with the updated wrapper it should skip quickly when scheduled.
- `scancel 131528`
  - This was `rlg_xi0_q0_chern`, stuck at `DependencyNeverSatisfied` after q0-Fock job `131526` timed out.

Current no-longer-needed old-layout task still pending:

- `127851_15` = old-layout `xi1 flavor:3`; it should skip immediately when it gets an array slot because of the wrapper change.
- `127851_16` = old-layout `xi1 flavor:4`; requeued to pending and will also skip.

Useful running RLG tasks remain:

- `131527` active `(4+4)` xi0 diagnostic.
- `127851_17`, `127851_18`, `127851_19` current Fig. 6 array tasks that are not flavor seed 3/4.
