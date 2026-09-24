# Detached postflight audit — physical-operator hole crosscheck v3 job 517203

- **Reviewed capsule:** `results/ptse2_openmx_screened_hf/source_data/ptse2_fractional_fillings_v1/7p340993_epsilon5_15_fillings_v1/physical_operator_hole_crosscheck_v3`
- **Producer job:** `517203`
- **Independent numerical sources:** published `crosscheck_arrays.npz` and `coefficients.csv`
- **Independent execution host:** `test001` (Python 3.11, NumPy 1.26.4; no production module imported)
- **Scope:** Q0 plus six shell-1 determinant-three TSB channels; saved fixed-rank local branch; finite seven-mode reconstruction only

## Decision

**Pass within the declared scope.** Terminal accounting, the immutable oracle checkpoint and crosscheck output closure, the two coefficient identities, Q0 count/spin values, exact reference zeros, spin-major `+y` convention, and the finite-seven-mode minimum are internally consistent. No blocker was found.

This does **not** classify the raw residuals with a newly invented tolerance, establish pointwise positivity, promote seven modes to a complete Fourier representation, prove the pinned B1 branch is a global ground state, or grant authority outside Q0 plus the six retained shell-1 TSB channels.

## Terminal accounting

A fresh `sacct` query records:

- parent `517203`: `COMPLETED`, exit `0:0`, elapsed `00:06:07`, account `hmt03`, partition `regular256`, 64 CPUs, node `node039`;
- batch step: `COMPLETED 0:0`, `MaxRSS=2566456K`;
- MPI proxy steps `.0` and `.1`: both `COMPLETED 0:0`, with `MaxRSS=115315908K` and `133227056K`;
- start/end: `2026-09-23T04:04:43` / `2026-09-23T04:10:50` in scheduler time.

Exact rows are in `SACCT_JOB517203.txt`. The completed record had aged out of live `scontrol` by the publication query; `SCONTROL_JOB517203.txt` records that explicit `Invalid job id` result. This does not conflict with persistent terminal `sacct`. The Slurm stderr file is zero bytes (SHA-256 `e3b0c442...b855`).

## Oracle checkpoint and output closure

The checker recursively hashed every manifest payload and inspected modes without modifying the capsule.

### Oracle checkpoint

- Exact namespace: **70 files** = 68 manifest payloads + `OUTPUT_SHA256.json` + sentinel `ORACLE_COMPLETE`; no missing/unexpected files.
- All 68 payload hashes match; all payloads are regular nonsymlink files at mode `0400`.
- Checkpoint root and `vertex/` directory modes are `0700`; manifest and sentinel modes are `0400`.
- Checkpoint output-manifest SHA-256: `8ccaf6996c3a2fbfe1a969a10b6f88f582c6ebe62fb154add8667c6e3db0878e`.
- `ORACLE_COMPLETE` SHA-256: `ce1e8b34817f4eda04a6932d6d587885272d76df1451ef7a193ae5cb72c4fc57`.
- `oracle_manifest.json` SHA-256: `053631ae3e29536f729d73a8f11240721ed918e50e86ad3557c2e27da1e36c87`.
- Sentinel-to-output-manifest and sentinel-to-oracle-manifest bindings both match.
- The hidden `.oracle_publication.517203` path is a two-file receipt pair, not a payload mirror; its `oracle_manifest.json` and `ORACLE_COMPLETE` are byte- and mode-identical to the checkpoint copies.

### Crosscheck output

- Exact namespace: **6 files** = four manifest payloads + `OUTPUT_SHA256.json` + sentinel `COMPLETE`; no missing/unexpected files.
- All four payload hashes match; payload, manifest, and sentinel modes are `0400`; output directory mode is `0700`.
- Output-manifest SHA-256: `b60335dbf1f60124c8afce28d6fdeebd33d6f3e26700a013f4f40234e29f81d7`.
- `COMPLETE` SHA-256: `1b0d2b6bffd2dcc31c3984901ae25fad9c5956e826ef9cb18efc3db6639be391`.
- Payload hashes: `summary.json` `a1d2973a...a0a9c9`; `coefficients.csv` `fd5269f6...f4478d`; `crosscheck_arrays.npz` `42ca0291...9615f`; `REPORT.md` `5d83d2e9...1790b`.
- `COMPLETE` correctly binds the output manifest, summary, oracle sentinel, producer job `517203`, and source-frozen digest `6b238a7d...419272`.
- All six files in `.crosscheck_publication.517203` are byte- and mode-identical to final output.

Full per-file hashes, sizes, modes, and closure booleans are in `RECOMPUTE.json`.

## Independent NPZ/CSV recomputation

The detached checker did not import `formula_core.py`, `run_crosscheck_mpi.py`, or any capsule/production module. It independently parsed all 168 CSV rows, mapped the four projectors, four axes, and seven physical labels, and separately loaded the NPZ arrays. CSV and NPZ coefficients and saved residual arrays agree exactly (`max difference = 0`).

From the coefficient arrays it formed

- `electron + direct-hole - full-reference`;
- `direct-hole - complement-hole`.

Both identities have the same raw maximum absolute residual:

```text
1.844897597698432e-10
```

The maximum occurs in `Sy/hbar` at physical label `(1,0)`, with value

```text
-1.713199304642643e-10 + 6.845402023003544e-11 i.
```

The values formed independently from NPZ coefficients and CSV coefficients agree exactly with each other, with both saved residual arrays, and with `summary.json`. This is a raw residual report, not a tolerance-based classification.

## Q0 counts and full-reference spin

Independently parsed CSV values are

```text
N_e             = 21.999999799616788 - 2.282016826623343e-26 i
N_h direct      =  2.0000000150679575 - 1.629050360008404e-26 i
N_h complement  =  2.000000015067958  + 2.282016826623343e-26 i
N_ref           = 23.99999981468476   + 0 i
```

against nominal ranks `(22,2,2,24)`. The corresponding raw real deviations are `-2.0038321224546962e-7`, `+1.506795754835366e-8`, `+1.506795799244287e-8`, and `-1.8531524048626125e-7`.

The Q0 full-reference spin is

```text
(Sx,Sy,Sz)/hbar = (-6.713888704249863e-13,
                   -2.4733502283306543e-12,
                   -2.1856590611453913e-13)
```

with zero stored imaginary parts.

## Exact TSB reference coefficients

For the six nonzero physical labels

```text
(-1,0), (0,-1), (1,-1), (1,0), (0,1), (-1,1),
```

all 24 full-reference entries (six labels × charge/Sx/Sy/Sz) are **exact complex zero** in both CSV and NPZ (`max_abs = 0.0`), not merely below a threshold.

## Spin-major `+y` convention

The detached check constructed `S_a/hbar = sigma_a/2` directly and evaluated `chi=(1,i)/sqrt(2)`. It obtained

```text
<Sx>/hbar = 0
<Sy>/hbar = +0.4999999999999999
<Sz>/hbar = 0
```

confirming the declared spin-major block sign `My=[[0,-iM],[iM,0]]` and the `+y` convention.

## Finite-seven-mode Pauli diagnostic

Using only CSV direct-hole coefficients, area `21.809962662515687 nm^2`, physical labels `(m,n)`, and the declared convention

```text
f(x,y) = A^-1 sum_(m,n) f_(m,n) exp[-2 pi i (m x + n y)],
```

the independent `192x192` reconstruction gives

```text
min_r [ n_h(r)/2 - |S_h(r)|/hbar ] = -0.035043673571072964 nm^-2.
```

The independently reconstructed density, all three spin grids, and margin match their NPZ arrays exactly (`max difference = 0`). The largest residual imaginary field component is `2.290697920083893e-17 nm^-2`.

The negative minimum is reported as a **finite-seven-mode diagnostic only**. It is neither clipped nor smoothed, and it is not evidence of a complete-field positivity violation or a positivity proof.

## Scope and no-overclaim verdict

The runtime report and summary keep the correct boundary:

- seven channels only: Q0 plus six shell-1 determinant-three TSB channels;
- raw identity residuals, with `classification=null` and `coefficient_thresholds=null`;
- direct top-two rank-two projector comparison, allowing internal rotation at exact degeneracy;
- finite seven-mode real-space diagnostic only;
- pinned local fixed-rank B1 branch, not a global-ground-state proof.

**No overclaim was found.** This postflight verifies internal consistency and immutable publication closure only; it does not rerun OpenMX, reconstruct the coefficient contraction from raw oracle shards, extend the Fourier inventory, or establish new physical authority.

## Reproducibility and mutation boundary

- Independent script: `recompute_npz_csv.py`.
- Machine-readable results: `RECOMPUTE.json`.
- Runtime record: `RECOMPUTE_PROVENANCE.txt`.
- Fresh terminal accounting: `SACCT_JOB517203.txt`; aged-out live-control query: `SCONTROL_JOB517203.txt`.
- Numerical recomputation ran on `test001`; no numerical work ran on a login node.
- No v3 source, control, checkpoint, staging, publication, output, Slurm log, sentinel, or metadata byte was modified. Only this detached postflight directory was created.
