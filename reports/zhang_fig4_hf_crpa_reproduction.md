# Zhang Supplementary Fig. 4 HF+cRPA Reproduction

Status: in progress, not accepted.

## Screening Convention

- Fig. 1(e) diagnostic quantity: `eps_total(q) = eps_BN * eps_crpa(q)`.
- `eps_BN = 4`.
- HF Fock production convention: `V_screened(q) = V_bare_with_BN(q) / eps_crpa(q)`.
- `eps_total` is plotting-only and must not be used as the HF divisor when `V_bare_with_BN` already contains hBN screening.
- Current cRPA Hartree implementation uses the full non-diagonal `V_cRPA(qe=0)` matrix in the Q basis.

## Accepted Dielectric Input

- Accepted Fig. 1(e) window artifact:
  `/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q9_zhang_appendix_fig4_merged/epsilon_fig1e_window.png`
- Accepted diagnostic summary:
  `/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q9_zhang_appendix_fig4_merged/crpa_epsilon_diagnostics_summary.md`

The q9 dielectric table reproduces the paper plotting window, but exact HF matrix-diagonal lookup on the endpoint-including `lk=24` B0 HF mesh with `overlap_lg=9` requires `q_lg >= 11`.

## q9 HF Smoke Outcome

Short smoke job: `94853`.

Outcome: failed intentionally before production HF acceptance.

Reason:

- `q_lookup_failures = 1872689`
- `q_lookup_fallbacks = 0`
- `max_q_reconstruction_residual_nm_inv = 1.009e-15`
- Representative missing exact lookup shift: `shift_key=(-5,-4)`

Interpretation: the q decomposition is numerically precise, but the stored q9 Q table is too small for exact HF. This is not a Fig. 1(e) dielectric failure. It is an HF integration cutoff requirement.

## Code Guardrails Added

- Production cRPA Fock lookup defaults to `matrix_diagonal`.
- The HF runner now refuses exact Fig. 4 cRPA when `q_lg` is too small for the current HF overlap shell.
- Per-run outputs now include:
  - `hf_band_nu_<nu>_crpa.png`
  - `hf_band_nu_<nu>_crpa.csv`
  - `hf_summary_nu_<nu>_crpa.json`
  - `density_matrix_final_nu_<nu>.npz`
  - `order_parameters_nu_<nu>.json`
- Per-run JSON records include the screening convention, gap estimates, energy decomposition, order proxies, q-lookup diagnostics, and the cRPA table metadata.

## Active q11 cRPA Prerequisite

Submitted from `login002` on 2026-05-10:

- BM cache job: `94874`, completed.
- q11 chunk array: `94875`, running with array concurrency 2.
- q11 merge job: `94876`, pending on `afterok:94875`.
- Chunk root:
  `/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_appendix_fig4_chunks`
- Merged target:
  `/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_appendix_fig4_merged`
- Submission manifest:
  `/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/submission_jobs_crpa_lk24_lg9_q11_zhang_appendix_fig4_20260509.tsv`

The q11 chunk submission excludes `node023,node024`, matching the failure diagnosis from the stale q9 array.

## Pending Validation Steps

1. Wait for q11 chunk array `94875` and merge `94876`.
2. Rerun the HF smoke gate against:
   `/data/home/ziyuzhu/Mean_Field/results/TBG_HF_cRPA/crpa_lk24_lg9_q11_zhang_appendix_fig4_merged`
3. Require:
   - `q_lookup_failures = 0`
   - `q_lookup_fallbacks = 0`
   - `max_q_reconstruction_residual_nm_inv < 1e-10`
4. Only after the smoke passes, submit the seven-filling Fig. 4 HF+cRPA benchmark.
5. Compare the seven-filling outputs against Zhang Supplementary Fig. 4 gap/order/Chern targets.

## Acceptance

Do not use the final acceptance statement yet.

Required final statement, only after all checks pass:

`The cRPA dielectric is validated, and the HF+cRPA integration passes the Zhang Supplementary Fig. 4 benchmark.`
