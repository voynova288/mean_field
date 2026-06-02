# RnG/hBN HF vs R5G-AHC comparison notes

Date: 2026-05-27

## Sources checked

- Current code:
  - `src/mean_field/systems/RnG_hBN/hf.py`
  - `src/mean_field/systems/RnG_hBN/interaction.py`
  - `src/mean_field/systems/RnG_hBN/screening.py`
  - `src/mean_field/systems/RnG_hBN/hamiltonian.py`
- Local project docs:
  - `AGENTS.md`
  - `plan/RnG_hBN_hf工作文档.md`
  - `plan/RnG+hBN无相互作用工作文档.md`
- External reference:
  - `reference/R5G-AHC` from https://github.com/zybbigpy/R5G-AHC at `cabc0b9`
  - Main files: `hf_scf.py`, `moire_hamk.py`, `moire_lattice.py`

## Important limitation

`R5G-AHC` is **not** an implementation of the Kwan et al. RnG/hBN MFCI-III HF problem. It is a half-metal HF code for rhombohedral multilayer graphene with a simplified moire/supercell setup. It is useful for tensor and layer-index sanity checks, but not authoritative for:

- hBN moire potential convention;
- screened-basis projection;
- average vs CN reference scheme;
- remote-band average correction;
- MFCI-III q=0 internal screening.

## Things that look consistent

### Basis ordering

`R5G-AHC` reshapes eigenvectors as

```python
psi0.reshape((num_kpt, num_G, Nl, 2, ntarget))
```

so the basis order is `(G, layer, sublattice)`. Current code uses

```python
basis_index = (g_index * layer_count + layer) * 2 + sublattice
```

which is the same ordering.

### Fock momentum argument

`R5G-AHC` uses

```python
q = kmesh[ikp] - kmesh[ik] + glist[iQ]
```

for the exchange kernel. Current code uses

```python
qvals = source_k - target_k + G
```

which matches this convention.

### Hartree/Fock normalization scale

`R5G-AHC` builds a dimensionless Coulomb kernel and multiplies by

```python
Uvalue = 14.4 * 2*pi / Omega0
param = (Hartree - Fock) / num_kpt
```

Current code uses `V(q)` in `meV nm^2`, multiplies by `v0 = 1 / A_moire`, and divides by `nk`. This is the same dimensional structure: `V(q)/A/Nk`.

### Layer form factor structure

`R5G-AHC` forms layer-resolved Lambdas, e.g.

```python
Lamk_fock ~ conj(D[p, G]) * D[k, G'] * trans[Q]
```

Current code computes per-layer overlap blocks and then contracts them with the density. The spin/flavor block-diagonal overlap plus off-diagonal density contraction is compatible with intervalley-coherent Fock terms.

## Possible issues / things to test next

### 1. q=0 exchange treatment is ambiguous and worth an ablation

`R5G-AHC` explicitly drops the entire `q=0` Coulomb value:

```python
if qnorm > small:
    ...
return 0.0
```

Current `interaction.py` instead returns the MFCI-III finite q=0 interlayer capacitive kernel

```python
V_ll'(0) = - e^2 |z_l-z_l'| / (2 eps0 eps_r)
```

and this kernel enters both Hartree and Fock through `build_rlg_hbn_interaction_components` when `shift=(0,0)` and `ik_source == ik_target`.

The q=0 Hartree part is definitely needed for internal electrostatic screening. The open question is whether the same q=0 interlayer value should also be included in the exchange contraction, or whether the interaction sum in the projected HF code should exclude the literal `q+G=0` exchange contribution after normal ordering/background handling.

**Suggested focused test:** without changing production code first, run a small ablation or branch that zeros only

```text
fock_layer_coulomb[(0,0)][ik, ik, :, :]
```

while keeping q=0 Hartree/screening intact, then compare the xi0 Fig. 6 Chern and band plot. This term is suppressed by the explicit `1/Nk` in the exchange sum, so it may be numerically small, but it is a clean convention difference exposed by `R5G-AHC`.

### 2. xi0 Fig. 6 active window is extremely fragile

Existing diagnostics show the xi0 Fig. 6 screened-basis active window has a near collision with the upper remote band:

```text
results/RnG_hBN/diagnostics_xi0_active_window_gamma_mprime_20260525/active_window_summary.json
min_upper_gap_u_mev = 0.07203289078856301
active_band_indices = [92, 93, 94, 95, 96, 97]
upper_neighbor = 98
screened_u_mev = 40.348153722057404
```

The HF work doc says `(3+3)` is allowed for Fig. 6 but must pass projection-band occupancy/window checks. A 0.07 meV upper gap at Gamma is a red flag: a tiny convention or remote correction error can swap topology in xi0. Current latest result has

```text
xi0 occupied conduction C = -1, paper expects |C| = 0
xi1 occupied conduction C = -1, paper expects |C| = 1
```

So the mismatch is localized to the subtle xi0 case. Before changing broad HF logic, test xi0 with `(4+4)` active bands and/or inspect the band-98 mixing around Gamma/M'. If `(4+4)` restores xi0 |C|=0 or changes the gap qualitatively, the issue is projection-window fragility rather than generic HF failure.

### 3. hBN moire convention remains a likely non-HF source of xi0 mismatch

`R5G-AHC` has no hBN moire potential, so it cannot validate the xi-dependent moire phase/sign convention. Since xi1 currently matches the expected |C|=1 while xi0 incorrectly gives |C|=1, the noninteracting xi0 moire convention should remain on the suspect list:

- sign of `moire_phase_rad` for xi0;
- assignment of xi=0 vs xi=1 table entries;
- whether the moire potential acts on the correct physical hBN-adjacent layer after the layer-order convention is fixed;
- high-symmetry path / valley label convention used for the paper comparison.

The current code follows the local work doc formula, but this is still not cross-checked by `R5G-AHC`.

### 4. screened-basis q=0 Hartree double-counting should be audited

The code now builds

```text
H_HF = projected H_sp(V) in H_sp(U) basis
     + active HF[P_active - P_ref]
     + remote average HF[P_remote_phys - P_remote_ref]
```

This matches the project doc. However, the q=0 layer Hartree field also enters the screened-basis construction through `U(V)`. The implementation must avoid accidentally counting the same remote-valence q=0 polarization twice: once in the screened basis and once in the fixed remote-average correction. `R5G-AHC` cannot settle this because it has no screened basis or remote average scheme.

**Suggested check:** isolate the q=0 Hartree contribution from `_remote_average_hamiltonian_from_source` and from active `density_delta` at the final xi0 state; compare its layer-slope with the screening solver's `hartree.interlayer_slope_mev`.

### 5. R5G-AHC's simplified interlayer potential should not replace the current one

`R5G-AHC` uses

```python
same layer: tanh(q d_gate)/q
interlayer: exp(-q d0 |l-l'|)/q
q=0: 0
```

The current code uses the full dual-gate Dirichlet layer Green's function, reducing to the paper's q=0 expression. This is closer to MFCI-III and should not be replaced by the R5G-AHC expression.

## Recommended next order

1. q=0 Fock ablation for xi0 only.
2. xi0 `(4+4)` active-window diagnostic/HF smoke comparison.
3. q=0 Hartree double-count audit between screening and remote correction.
4. noninteracting xi0 moire convention/topology checkpoint against `2312.11617v1.pdf` Fig. 2 / MFCI-II appendix.
