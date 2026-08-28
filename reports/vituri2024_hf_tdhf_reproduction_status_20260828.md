# Vituri 2024 ABC 三层石墨烯 HF/TDHF 复现状态

更新时间：2026-08-28

分支：`debug/vituri-scalar-hessian`

本文是该分支当前唯一的 Vituri 结果状态报告，取代
`vituri2024_hf_tdhf_reproduction_status_20260821.md` 以及更早的 N101
简并分支和 panel-c 初步报告。过时的 N101 active-report JSON/图已删除；历史
证据仍保存在 sealed runroots 与 Git 历史中。物理结论以本文为准。

## 一句话结论

**采用预先固定的合理数值 contract（`N=179`, `H_v=769`,
`a0*Delta k=0.004`, `d=369 Angstrom`, `epsilon=8`, `Delta1=28 meV`），
精确 no-wrap FFT 在 homogeneous fixed-half-metal sector 内完成全自洽 SCF，
并独立复现 Fig. 4(c) 的双峰、中心浅谷和两个费米交点；能量尺度也与论文
panel 高度一致。该结论是
`independent chosen-contract Fig. 4(c) reproduction`；它不等同于作者 exact
cutoff、UV limit、unrestricted ground state 或 TDHF source authority。**

## 1. Sealed 计算与证据边界

本报告包含两条互补证据链：job `461276` 是 full cutoff ladder；job
`462560` 加 zero-science recovery `462719` 是 chosen-contract Fig. 4(c)
复算与图形发布。

Full-ladder 权威计算 job `461276`：

- source commit：`ec45b19768a19ce500be9171dd929376ee7c9fa1`；
- backend：exact finite-square zero-padded no-wrap FFT convolution；
- partition/node：`regular6430/node018`，64 CPU；
- 10 个 case 使用 10 个独立进程，每个进程固定 6 CPU，余下 4 CPU 留给
  controller；
- completion sentinel：
  `/data/home/ziyuzhu/.runs/Mean_Field_ec45b19_vituri_panel_c_fft_full_scf_ladder_v3_20260828/COMPLETE_461276.json`；
- postflight：`PASS_ALL_ENDPOINTS_STATIONARY_BFS_CLOSED`。

外部 `sacct` 在完成后显示 `COMPLETED`, exit `0:0`, elapsed `00:03:10`，外部
stderr 文件当时为零字节；这些是后验 scheduler/log 观察，不属于 sealed output
manifest 的科学 claim。

Sentinel、summary、postflight 和 31 个 manifest 文件的 SHA256/size 均已重新
核对；最终目录为 `0555`，文件为 `0444`，没有 missing 或 hash drift。

机器可读摘要：

- full ladder：`reports/data/vituri2024_panel_c_fft_full_scf_ladder_461276_attestation.json`，
  SHA256 `e34af3ef6ab1c4eca4b01a6585ac930bd8e743f5c871246005e6af1e5049e9f8`；
- chosen-contract Fig. 4(c)：
  `reports/data/vituri2024_fig4c_chosen_contract_462560_attestation.json`。

最新 Fig. 4(c) 复算使用 source commit
`09075cd22d47edbb4738229d98b52a9650154ec5` 与公开 fixed-sector BFS API。
原 job `462560` 的数值 step 和独立 postflight 均 `COMPLETED 0:0`，batch 仅因
compute node 缺失 login-node gnuplot target 而在绘图阶段 `FAILED 1:0`；
zero-science recovery job `462719` 用保存的 raw branch CSV 生成图并完成原子
发布，没有重跑或修改 SCF。最终 sentinel：
`/data/home/ziyuzhu/.runs/Mean_Field_09075cd_vituri_fig4c_chosen_contract_v2_20260828/COMPLETE_462560.json`。

两个前序 capsule 均只留下预数值工程失败 provenance；它们不是 job
`461276` 的科学输出：

- job `456071`：`llvmlite/binding/libllvmlite.so` 等 runtime 库未进入 v1
  inventory；证据 manifest
  `prior_failure/job_456071/EVIDENCE_SHA256SUMS.txt` 的 SHA256 为
  `90d6806830da52e4c3401767d5fc87b91f8c11696ba2b401d3984683feee985a`；
- job `461208`：runner 的 ProcessPool import 比 standalone preflight 多加载
  `_multiprocessing` 和 `_queue`；对应 manifest SHA256 为
  `27bf8ec787b2eabc79a2fd88e650f060e16434806bb91cca647fba609e401c9f`；
- 两者均无 staging/output/sentinel，也没有执行 SCF；批准均已消耗且没有
  复用。

v3 在计算前要求 standalone、runner parent 和 fork worker 的 exact
120-entry runtime inventory 一致，并在 N81 prepare + 单次 FFT Fock/energy
action 前后验证没有新增映射。

## 2. 物理与数值 contract

目标参数：

- ABC trilayer graphene 六带模型，active band 为第三低能带；
- `Delta1=28 meV`；
- hole density `1.03e12 cm^-2`；
- flavors `0,2` 满占据，flavors `1,3` 各含固定 holes；
- normal-order reference：active-electron `R=0`；
- `epsilon=8`, `qTF=0.04/a0`；
- odd endpoint-inclusive Cartesian square，有限方形域，不做 momentum wrap；
- 只使用保存 SCF 网格上的 exact `ky=0` 点。

fixed-density ladder 使用 `H_v=768`, `d=250 Angstrom`,
N=`81,101,121,141,161,179,181,201`。两个 paper-profile discriminator 使用
N179、`H_v=769`、显式 `a0*Delta k=0.004`，分别取 `d=369` 与
`250 Angstrom`。

每个 case 内的所有 branch 从该 case 同一种构造得到的 mirror-symmetric mixed
`h0` initializer 重放；不同 mesh 的 `h0` 数组当然不同。所有 exact fixed-rank
shells 枚举全部纯 coordinate choices；每个 choice 绑定 exact Fock、
previous density、generation 和 SHA256。禁止用能量、图形、crossing 或诊断
选择 branch。正的 sub-tolerance splitting 必须拒绝，不能 stable-sort 穿过。

## 3. Full-SCF fixed-density ladder

以下取 flavor 3 的 exact `ky=0` cut；flavor 1 是镜像。中心与 crossing 指标在
同一 case 的所有 stationary endpoints 间一致（paper `d=369` 的一个 root
仅有约 `3e-16` 的浮点差）。

| N | `a0*k_axial` | `E(0)-mu` (meV) | crossing 数 | `kx*a0` crossings |
|---:|---:|---:|---:|---|
| 81  | 0.160102 | -5.019549 | 4 | `[-0.077507,-0.030430,0.025467,0.063213]` |
| 101 | 0.200128 | -3.853657 | 4 | `[-0.075578,-0.026824,0.022413,0.062085]` |
| 121 | 0.240154 | -2.551096 | 4 | `[-0.073737,-0.021935,0.018452,0.061241]` |
| 141 | 0.280179 | -1.621432 | 4 | `[-0.071596,-0.017780,0.014870,0.060363]` |
| 161 | 0.320205 | -0.642795 | 4 | `[-0.069530,-0.011269,0.009610,0.059463]` |
| 179 | 0.356228 | +0.475365 | 2 | `[-0.068404,0.059275]` |
| 181 | 0.360231 | +0.614860 | 2 | `[-0.068393,0.059335]` |
| 201 | 0.400256 | +1.590246 | 2 | `[-0.067446,0.059580]` |

结论：

1. 中心随 square domain 扩大从 `-5.02` 单调漂移到 `+1.59 meV`；
2. 四交点到两交点的 Lifshitz pattern 在 N161 与 N179 之间出现；
3. N179–N201 中心仍变化约 `+1.115 meV`，因此 sampled N81–N201 区间
   没有观察到 plateau；这不排除更大或不同 domain 才出现 plateau；
4. 在该固定 contract 下，N101 的 `-3.853657 meV` 在全部 stationary
   endpoints 间一致，因而不是本次 branch postselection 或未收敛造成；本计算
   不排除共享的 interaction/occupation convention 错误；
5. N179 的正中心与两交点不能把该 cutoff 提升为作者 cutoff 或 UV limit。

先前 one-shot job `432973` 只做初始 Fock action，不能替代这次 full-SCF
ladder；它现在仅作为历史域敏感性证据。

## 4. N179 paper-profile discriminators

显式 `a0*Delta k=0.004`, `H_v=769`：

| `d` | `E(0)-mu` (meV) | flavor-3 crossings | exact-grid local maxima `(kx*a0, meV)` |
|---:|---:|---|---|
| 369 Å | +0.534126 | `[-0.0684586,0.0592840]` | `[(-0.048,5.66960),(0.040,8.33828)]` |
| 250 Å | +0.486123 | `[-0.0684584,0.0592859]` | `[(-0.048,5.62570),(0.040,8.29401)]` |

在已计算的 N179/Hv769/`a0*Delta k=0.004` case 中，将 `d=250` 改为条件
推断的 `369 Angstrom` 只使中心上移约 `0.0480 meV`，crossings 几乎不变；
该受控差异远小于 sampled cutoff ladder 的漂移。它没有直接测试 N101 或 UV
limit 的 gate-distance 效应，因此不能作更广泛的排除。

fixed-density N179/Hv768/d250 的中心为 `+0.475365 meV`，与显式 spacing
N179/Hv769/d250 只差约 `0.010758 meV`。因此在这个有限域附近，整数 regulator
和两种 spacing construction 不是中心符号翻转的主因。

## 5. Branch closure 与 stationarity

十个 case 全部满足：

- independent canonical BFS closure；
- `unconsumed_frontier_count=0`；
- terminal rejection `=0`；
- normal gate rejection `=0`；
- 所有 normal endpoints stationary；
- fresh-H fixed-rank raw projector 与 engine final raw projector byte-exact；
- independent postflight rebuild 再次 byte-exact；
- projector、commutator、off-diagonal coherence、population 和 final raw norm
  均为零；
- E/F residual 通过预声明的 relative gate。

Branch 数量与 array coalescence：

| case | replayed paths | stationary endpoints | exact array groups |
|---|---:|---:|---:|
| N81 | 5 | 4 | 4 |
| N101 | 5 | 4 | 1 |
| N121 | 1 | 1 | 1 |
| N141 | 5 | 4 | 4 |
| N161 | 21 | 16 | 4 |
| N179/Hv768 | 5 | 4 | 1 |
| N181 | 5 | 4 | 1 |
| N201 | 5 | 4 | 4 |
| N179/Hv769/d369 | 5 | 4 | 4 |
| N179/Hv769/d250 | 5 | 4 | 4 |

因此不能笼统声称所有 coordinate paths 都 byte-coalesce。若有多个 exact
array groups，它们全部保留且全部通过 stationarity；中心、cut 和 crossings
是 branch-consistent observables，而不是通过 postselection 选择的某一 leaf。

## 6. Fig. 4(c) chosen-contract 复现

新复算的 contract 在 job `462560` 前预先声明；运行中没有调参、拟合、缩放、
branch 选择或 visual pass gate，也没有用旧 scientific arrays 或目标 observables
作 pass gate。`N=179` 与 `d=369 Angstrom` 来自此前 cutoff/paper-profile 审计，
不是 blind author-exact 输入。四个 exact-shell BFS endpoints 全部 stationary；
所有 branch 均保留。其 `ky=0` cut 的最大 branch spread 仅
`1.78e-12 meV`，中心 spread 为零。

独立计算得到：

- `E(0)-mu = +0.534126 meV`；
- crossings：`[-0.0684586, 0.0592840] / a0`；
- exact-grid maxima：`(-0.048, 5.66960 meV)` 与
  `(0.040, 8.33828 meV)`；
- common `mu_mid = 4.966436164427063 eV`。

计算图：

- `figures/vituri2024_fig4c_chosen_contract_calculated_20260828.png`；
- `figures/vituri2024_fig4c_chosen_contract_calculated_20260828.pdf`；
- 与论文 direct crop 的 display-only 并排图：
  `figures/vituri2024_fig4c_chosen_contract_vs_paper_display_20260828.png`。

直接并排检查显示，计算与论文在两个交点、左/右峰位置与高度、中心浅谷及整体
非对称曲率上均强一致。计算冻结后，又对论文 embedded raster 做了独立 held-out
像素标定；该步骤没有回流 solver、参数、branch、convergence 或 pass gate。
固定 published axes 后提取 402 个 unsmoothed paper samples，得到：

- 全样本 RMSE `0.964 meV`、MAE `0.657 meV`；
- 去掉 6 个明确贴近 frame、受边界裁切影响的端点后 RMSE `0.887 meV`；
- 中心：paper `0.400 +/- 0.053 meV`，calculation `0.534 meV`；
- crossings：paper `[-0.06981, 0.06113]`，calculation
  `[-0.06846, 0.05928]`；
- paper raster peaks 约为 `(-0.04849,5.067 meV)`、
  `(0.04264,7.947 meV)`，对应 calculation
  `(-0.048,5.670 meV)`、`(0.040,8.338 meV)`。

因此本报告把 Fig. 4(c) 标记为 **independent chosen-contract reproduction**。
论文 raster digitization 只是 calculation 冻结后的评价证据，并与独立计算数组
分开保存；没有拟合、缩放、平滑或 branch postselection。定量诊断图：
`figures/vituri2024_fig4c_heldout_raster_evaluation_20260828.png`；原始像素、paper
samples 与 comparison CSV 位于 `reports/data/`。可解析的源 figure 与 direct crop
分别保存为 `reports/reference/vituri2024_SC_fig_4_2408.10309v1.pdf` 和
`reports/reference/vituri2024_fig4c_paper_crop_2408.10309v1.png`。

作者没有公开 raw numerical array，因此上述 residual 受 raster resolution、
line thickness 与边界裁切限制；这限制 `author-exact/full-paper` 标签，但不再
阻塞本次 chosen-contract panel 复现。旧 N101 mismatch 图已从 active reports
删除，避免继续代表当前最佳结果。

## 7. TDHF/scalar-Hessian authority

generic/reduced TDHF algebra 与候选 provider/replay 层次来自该分支此前的独立
测试和 artifact，不是 HF-only jobs `461276` 或 `462560` 的结论，也不授予
production TDHF authority。这两条 HF 证据链都没有证明：

- unrestricted/coherent global HF ground state；
- local HF Hessian positivity 或 finite-q stability；
- 作者相同 background/reference/cutoff；
- full-projector TDHF source promotion；
- Fig. 3 susceptibility、collective modes 或 pairing kernel reproduction。

因此当前仍是 **Vituri HF/TDHF debug 与资格审计分支**，不是 production
TDHF 结果分支。

## 8. 当前结论与后续边界

Fig. 4(c) 的 chosen-contract HF panel 复现已经闭合，不再等待作者 exact
numerical policy。后续若继续推进 TDHF，才需要独立检查 unrestricted/coherent
basins、local Hessian 与 finite-q stability；这些不回溯否定本次固定 sector 的
panel 复现。

当前 authority：

```text
independent_chosen_contract_fig4c_reproduction = true
independent_finite_volume_fixed_sector_fft_full_scf_discriminator = true
uv_plateau_established = false
author_cutoff_identified = false
unrestricted_ground_state_established = false
full_paper_reproduction_verified = false
local_hf_stability_proved = false
tdhf_authority = false
production_authority = false
visual_similarity_used_as_pass_gate = false
```

这里 `full_paper_reproduction_verified=false` 至少表示未复现论文全部 phase
diagram、susceptibility、collective modes 与 pairing calculation，也没有作者
exact contract 或 paper raw array 的 pointwise parity；它不否定 Fig. 4(c) 的
chosen-contract reproduction。
