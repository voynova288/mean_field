# Du2017 numerical source-lineage table

**Date:** 2026-08-07
**Scope:** Du main paper, Supplementary Notes 2/6, Du thesis, cited Li–Yang–Chang method, preserved TEIB/senior code, and current Mean_Field reconstructions
**Purpose:** keep author statements, inherited executable defaults, local regulator choices, and rejected forensic hypotheses separate

**Strict integrity rule:** archived senior/reference paper-vector, hard-fit, and calibrated branches are documentary evidence only and must never be executed, plotted, or presented as calculations. Paper digitization and target-derived reconstruction outputs were deleted on 2026-08-07. If a paper panel is needed, use a direct PDF crop.

## Evidence labels

- **P — published:** stated in the Du paper, supplement, thesis, or cited method.
- **A — archived lineage:** present in preserved author/senior-associated code or execution receipts, but not necessarily published.
- **R — local reconstruction:** explicitly chosen and validated in this repository; not an author setting.
- **F — forensic/rejected:** useful for lineage diagnosis only; cannot support predictive physics.
- **U — unresolved:** absent or algebraically ambiguous in available sources.

No value acquires publication authority merely because it appears in inherited code.

## 1. Published and cited sources

| lineage | normal dispersion / retained model | in-plane momentum domain and grid | growth-direction / subband window | chemical potential and ensemble | interaction, quadrature, and reference | authority and unresolved items |
|---|---|---|---|---|---|---|
| Du main text Fig. 2 two-band model | Generic single-particle electron/hole energies `E_k^e,h`; point-layer Coulomb model. Reports `m_e≈0.032 m0`, `m_h≈0.136 m0`, `epsilon≈15`, `d≈10 nm`, but uses the masses explicitly for a Bohr-radius estimate rather than certifying the numerical Fig. 2 bands. | No numerical `kmax`, nodes, weights, or momentum conversion. Fig. 2a displays approximately `0–0.035 nm^-1`. | Two effective bands only; no connection to a specific Kane E1/H1 window is stated. | No numerical pair `mu` or number equation in the main text. Experimental CNP estimate is `5.5e10 cm^-2`, not a documented solver constraint. | `V_ee=V_hh=e^2/(2 epsilon q)`, `V_eh=V_ee exp(-qd)`. No singular-cell or background prescription. | **P/U.** Physical inputs are published; the actual Fig. 2 scalar bands and closure are not. See `docs/du2017_paper_logic.md`, Sec. 1A. |
| Supplementary Note 2 generalized BCS | Generic `epsilon_a(k), epsilon_b(k)` with self-consistent `Delta`, diagonal exchange in `xi`, and imbalance channel `eta`. | Momentum sums/integrals are symbolic. No cutoff, radial grid, angular rule, or singular-cell treatment. | Two-band scalar model; no Kane wavefunctions or projected vertex are supplied. | Printed Eq. (1) implies `xi_bare=epsilon_P-2mu`, `eta_bare=epsilon_M`; later equations subtract one `mu` from both `epsilon_P` and `epsilon_M`. No unique mapping to the Kane Fermi level. | Same- and opposite-species channels are printed, but filled-sea subtraction, background, and whether diagonal Fock is already absorbed are unstated. | **P/U.** The diagonal-exchange structure is authoritative; the numerical ensemble and `mu` algebra remain underdetermined. |
| Du thesis Littlewood/Zhu ancestor and Fig. 6.22 | Isotropic parabolic electron/hole two-band model. For the reported masses, the intrinsic pair curvature is `1.4707651566 eV nm^2`; this is not an extra fit coefficient. | Formal continuum momentum sums; no numerical cutoff/grid for the displayed Du curve. The independently reproduced Zhu benchmark uses the source mapping `k=tan(beta)` over an infinite domain, but that is a benchmark implementation rather than a recovered Du grid. | Scalar two-band model. Fig. 6.22 caption credits the calculation to Kai Chang but gives no arrays or retained subbands. | Thesis `mu` is a pair-density Lagrange multiplier. It is not Kane–Poisson `mu_el=113.8688 meV`. | Printed zero-temperature gap plus diagonal-exchange equations; no numerical singular-cell/background convention for Fig. 6.22. | **P/U.** Establishes the equation family and pair-`mu` role, not the hidden numerical implementation. |
| Supplementary Note 6 Kane–Poisson | Full 8-band Kane Hamiltonian with Table-1 material parameters. No post-Kane scalar `C*k^2`, `c4*k^4`, Gaussian/ring term, or pairing vertex is printed. | In-plane `d k_parallel` integrals are unbounded-looking; no numerical in-plane cutoff or grid. | Plane waves `m=-N...N`, but no numerical `N` or `L`. Density sums over subbands `s`; the exact retained set is not specified. | One ordinary-electron Fermi level is fixed through integrated electron-hole charge neutrality. No mapping to Note-2 pair `mu`. | Poisson equation and component-resolved electron/hole densities are printed. Boundary, reference sea/background, gate feedback, and UV subtraction are not numerically closed. | **P/U.** Used for Fig. 1g/Fig. 4f; publication never states that its E1/H1 bands feed Fig. 2. GaSb `gamma2=8.18` is printed; local `0.08` remains inferred and not author-confirmed. |
| Li–Yang–Chang cited Kane–Poisson method | Eight-band Kane/Poisson with modified parameters and `E_P=22.5 eV`. | Example calculations use approximately `N=25`, but this is not a Du2017 setting. No transferable Du in-plane cutoff is documented. | Includes subbands showing anticrossing behavior and notes that not all subbands are included; the exact Du retained set is absent. | Common Fermi level / charge-neutral Kane–Poisson logic. | Supplies method equations, not the Du device boundary/background data. | **P/U.** Supports method structure only. It cannot fill Du-specific `N`, `kmax`, subband, or gate inputs. |

## 2. Preserved TEIB and senior-code lineage

| executable lineage | parent grid / band window | BCS radial/angular grid | closure/reference | interaction/scaling | dispersion/vertex additions | classification |
|---|---|---|---|---|---|---|
| July-20 TEIB Kane–Poisson default | `N=7`, `nk=81`, `nz=256`, `kmax=0.12 nm^-1`; final mixed `Ve*`; pre-excitonic E1/H1 branches rebuilt on `nk=241`, same `kmax`. | Legacy Kane production preset `nk=241`, `ntheta=192`, `diagonal_subcells=64`; uniform interpolation over the saved source interval. | Target density inherited from the supplied normal NPZ; dynamic density/neutrality multipliers in the legacy scalar solver. | `epsilon_r=15`, `d=10 nm`, default global interaction scale `1`. | No separate `pair_curvature` in `reference/teib_20260720_unpacked/teib/src/excitonic_bcs.py` (SHA-256 `929ccdaa...`). | **A/F.** Reproducible legacy diagnostic, not z-converged and not a published author contract. Source: July-20 README lines 6–44 and `run_bcs.py` presets. |
| July-20 TEIB `paper` mode | No Hamiltonian-generated normal dispersion; shape-preserving interpolation of PDF vector anchors. | `paper_nk=3501` production; analytic radial JDOS with display regularization `0.02 meV`. | Curve reconstruction, not a thermodynamic ensemble. | Not a predictive interaction calculation. | The published vector curves themselves are inputs. | **F.** Explicit `paper-calibrated reconstruction`, `not_ab_initio=true`; never physics evidence. |
| July-20 TEIB `disclosed` mode | Paper-mass parabolic model and external Naveh–Laikhtman finite-well equations; intrinsic mass curvature only. | Production `nk=241`, `ntheta=192`, `diagonal_subcells=64`; `K_MAX_NM_INV` is a user control, not recovered Du authority. | Separate disclosed-parameter model. | Reported `epsilon_r`, layer widths, and unit interaction unless changed. | No senior `16.60946` term. | **A/R.** Transparent independent model, but not a demonstrated Du generator. |
| Preserved legacy Kane, scale 1 | July-20 N7 Kane parent. | Same reduced scalar grid family. | Dynamic two-channel chemical potentials / fixed-density proxy. | Unit global scale. | Raw converted Kane detuning. | **F.** Converged `E0/Emin/Delta=8.448/3.578/3.701 meV`; disagrees with Fig. 2. |
| July-28 `current_uncalibrated_fixed_density` execution | Logged source hashes exist, but exact executed source bytes are absent. Validation records `source_npw=25`, `source_nk=81`, `source_nz=512`; natural Kane–Poisson density `0.00268823 nm^-2`. | Compactified Gauss–Legendre radial grid, `nk_bcs=281`, quadrature cutoff `0.35 nm^-1`, `ntheta=192`, `diagonal_subcells=64`. | Pair-number and charge-balance multipliers enforce `n_e=n_h=0.00055 nm^-2`; this is an explicit closure, not publication authority. | Unit bare Coulomb, `epsilon_r=15`, `d=10 nm`. | No calibrated parameters according to validation; direct full Kane nonparabolicity retained. | **A/R.** Logged SHA for executed `src/excitonic_bcs.py` is `fbb8e3f6...`; result `8.605/3.627/3.741 meV`. Exact source cannot be cold replayed from the restored archive. |
| Restored historical calibrated senior route | Source normal NPZ reports `npw=25`, `nk=81`, `nz=512`; continuous E1/H1 subspace rebuilt from `Ve*`. | Node route `nk_bcs=71`, `kmax=0.35 nm^-1`; production `ntheta=192`, `diagonal_subcells=64`; separate refined continuum audit. | Low-density branch seed `9.4e9 cm^-2`, then fixed pair `mu` (calibrated default `-37.61102 meV`); density becomes output. | Intralayer `4.2`, interlayer `2.09`, diagonal cell `0.782`. | Extra `16.60946 k^2 eV`, Gaussian `0.01840197 eV`/width `0.02869179 nm^-1`, ring `-0.00015 eV` at `0.025/0.008 nm^-1`, and pairing vertex `0.38`/`0.019 nm^-1`. | **A/F.** Explicit multi-parameter calibrated reconstruction. None of these additions is source-backed by Du. Active `src/excitonic_bcs.py` SHA is `91cb9a79...`. |
| Restored archive manifests/README | `SOURCE_SHA256SUMS.txt`, README, active source, and current/calibrated execution logs describe different chronology states. | Defaults differ (`run_nonint.py` restored default `kmax=0.30`, older README command `0.12`). | Mixed. | Mixed. | Mixed. | **A/U.** Top-level manifest is stale; authority is receipt-bound to each execution log and source hash. Do not infer one coherent “senior default.” |

## 3. Current controlled Mean_Field reconstructions

| reconstruction | numerical domain | model/ensemble/reference | verified use | authority limit |
|---|---|---|---|---|
| Native Zhu1995 benchmark | Infinite `k=tan(beta)` mapped domain with Gaussian quadrature and cell-integrated angular logarithmic singularity. | Source Littlewood/Zhu fixed-density equations. | Reproduces public Zhu textual checkpoints; validates factors and measure. | **R.** Solver benchmark only; not the hidden Du grid. |
| Du-only hard-cutoff scalar ladder, job `208255` | `kmax/kF=infinity,6,4,3,2,1.5,1.25,1.10,1.02`; `nbeta=192`, `nphi=384`. | Paper masses, `epsilon_r=15`, `d=10 nm`, fixed density, unit Coulomb, intrinsic `c=1.470765`. | Tests undocumented cutoff as a forensic hypothesis. | **F.** Gap-like cutoffs retain active boundaries, miss `E(0)`, and truncate the paper domain. |
| Du-only fixed-pair-`mu`/cutoff matrix, job `208259` | Four cutoff families × 15 predeclared `mu` shifts; `nbeta=160`, `nphi=320`. | Pair `mu` fixed, density output; otherwise same unit-Coulomb scalar model. | Tests the only remaining scalar closure printed but numerically unspecified. | **F.** Wrong-direction `E(0)`/gap tradeoff; does not reproduce Fig. 2. |
| Split-zero Löwdin trace-channel audit | Saved 61-point radial grid `0–0.12 nm^-1`; nested fits through `0.06 nm^-1`. | U(2)-invariant E1/H1 block traces; no EI interaction or target. | Finds source-only `c_xi=1.283–1.290 eV nm^2` and subleading Kane `k^4`. | **R.** Diagnoses direct Kane nonparabolicity already present in the parent; cannot be added back as a correction. |
| Canonical N95 Kane–Poisson normal state | `N=95`, `Lz=65.5 nm`, `Lambda_z=9.1130168577 nm^-1`; radial source through `0.24 nm^-1`. | Ordinary-electron neutrality root, `mu_el=113.8688022886634 meV`, periodic source model. | Local z-regulator and normal-state checkpoint. | **R.** Stability-selected, not recovered Du settings; N79→N95 component drift and in-plane UV closure remain open. |
| N95 fixed-`mu_el` full radial matrix HF | `nr=160`, `nphi=512`, `kmax=0.24 nm^-1`; exact saved-grid bands. | Fixed ordinary-electron `mu_el`; `D=P-P_vac`, E1 empty/H1 filled; unit Coulomb; grand potential. | Five seeds, restart equivalence, coupled-harmonic oracle. | **R/F.** Numerically converged but cutoff-open and electrostatically frozen; physical phase/JDOS/optics blocked by `STOP_GATE.json`. |

## 4. Convention disposition matrix

| convention/hypothesis | source status | numerical result | disposition |
|---|---|---|---|
| Physical mass curvature `1.470765 eV nm^2` | Derived from reported `m_e,m_h`; masses are not unambiguously certified as Fig. 2 numerical bands. | Native scalar model tested. | Allowed transparent model input; not an extra knob. |
| Direct Kane `c2+c4+...` nonparabolicity | Derived from reconstructed Note-6 parent. | Full no-scale route and trace audit tested. | Already included when direct Kane bands are used; adding a fitted polynomial back is double counting. |
| BHZ common `C-Dk^2` | No BHZ parameters in Du lineage. Algebraically affects `eta`, not `xi_pair`. | No EI scan needed. | Cannot be relabelled as a pair-curvature fix. |
| Senior `16.60946 eV nm^2` plus Gaussian/ring | Restored calibrated code only; no Du-source coefficient. | Helps historical calibration only with several other knobs. | Forbidden for predictive work. |
| Global Coulomb scale `0.6` / effective `epsilon_r=25` | Explicit energy-matched sensitivity, not published. | Reproduces some Fig. 2 energy scales but not source authority. | Forensic only; forbidden. |
| `k/sqrt(2*pi)` display coordinate | No publication/code authority; motivated by inverse-spacing coincidence. | Moves ring momentum near the paper. | Forensic coordinate hypothesis only; cannot rescale physical outputs. |
| Hard `kmax` | Publication omits it. | Jobs `208255/208259` reject cutoff and cutoff+pair-`mu` as sufficient. | Forensic tests complete; no result-selected predictive cutoff. |
| Fixed Du pair `mu` | Printed role exists, value absent. | Fixed-`mu` scan fails as sufficient. | Distinct valid ensemble concept, but author value/closure unresolved. |
| Fixed ordinary-electron `mu_el=113.8688 meV` | Local N95 Kane–Poisson result, not Du Note-2 `mu`. | Matrix HF numerically converged but UV-open. | Keep fixed only inside its typed local contract. |
| Pairing-only BCS | Contradicts printed simultaneous diagonal exchange unless exchange is already absorbed. | Produces different branches. | Not source-faithful without author confirmation. |
| External Tan/Uptight/Wannier/generic finite BZ | Outside Du model family and explicitly user-disallowed. | Historical preflight closed. | No follow-up. |

## 5. Remaining irreducible unknowns

Only author/source clarification can fill the following cells:

1. actual Fig. 2 `epsilon_a(k), epsilon_b(k)` arrays and their energy zero;
2. numerical Note-2 pair `mu`, density constraint, and resolution of the printed `xi/eta` algebra;
3. whether diagonal `V_aa/V_bb` Fock was iterated or preabsorbed, including its sea/background subtraction;
4. radial nodes, `kmax`, angular quadrature, and singular-cell treatment;
5. whether Note 6 supplied Fig. 2 bands, and if so the E1/H1 basis/downfolding map;
6. Note-6 numerical `N`, `L`, in-plane domain, retained subbands, and reference/background;
7. device electrostatic boundary, fixed charge, gate work function, and whether Poisson was updated after excitonic occupations changed;
8. exact Fig. 2b JDOS versus optical-weight formula and broadening.

These questions are now explicit in `results/du2017_inas_gasb_excitonic_insulator/AUTHOR_DATA_REQUEST_DRAFT.md`.

## 6. Operational consequence

The source audit is exhausted for locally available material. Allowed work is limited to preserving forensic receipts and obtaining author data. No further result-selected cutoff, `mu`, curvature, Gaussian/ring, Coulomb-scale, coordinate, or JDOS-weight scan can promote the current reconstruction to predictive physics. Heavy HF, exact-off-axis response, or optical production remains blocked by the existing stop gate.
