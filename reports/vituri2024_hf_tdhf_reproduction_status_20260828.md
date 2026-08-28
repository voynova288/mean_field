# Vituri 2024 ABC 三层石墨烯 HF/TDHF 复现状态

更新时间：2026-08-28

分支：`debug/vituri-scalar-hessian`

本文是该分支当前唯一的 Vituri 结果状态报告，取代
`vituri2024_hf_tdhf_reproduction_status_20260821.md` 以及更早的 N101
简并分支和 panel-c 初步报告。旧 JSON、图和 runroot 仍作为历史证据保留；
物理结论以本文为准。

## 一句话结论

**精确 no-wrap FFT 全 SCF 已在受限 homogeneous half-metal sector 的 N179
有限-cutoff case 中得到正中心与两个交点；但是 N81–N201 sampled
fixed-density ladder 持续明显漂移，在该区间没有观察或建立 UV plateau。
该结果只证明 finite-volume profile discriminator，不证明作者 cutoff、
unrestricted ground state、完整论文复现或 TDHF source authority。**

## 1. Sealed 计算与证据边界

权威计算为 Slurm job `461276`：

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

- `reports/data/vituri2024_panel_c_fft_full_scf_ladder_461276_attestation.json`；
- SHA256：`e34af3ef6ab1c4eca4b01a6585ac930bd8e743f5c871246005e6af1e5049e9f8`。

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

## 6. 与论文 Fig. 4(c) 的关系

计算得到的 N179 finite-cutoff profile 具有正中心、两个 crossings 和报告表中
列出的两个 exact-grid maxima。它与论文 panel 的相似性目前只是未量化的视觉
观察；paper raster 没有进入 sealed numerical/postflight 路径，因此该观察不授予
reproduction authority。

仍不能标记 `full_paper_reproduction_verified=true`，原因是：

1. 作者没有公布 panel 的 raw `kx*a0,E-mu` 数组；
2. 作者 momentum-domain 形状、cutoff、endpoint/wrap、quadrature 未知；
3. sampled N81–N201 ladder 没有观察或建立 UV plateau；
4. panel 是否来自 unrestricted mBZ solver 或另一个 constrained calculation
   仍未知；
5. background/reference 与 exact finite-grid occupation policy 未获作者确认。

现有旧图
`figures/vituri2024_hf_band_panel_c_matched_density_comparison_20260818.png`
只显示历史 N101 mismatch，不再代表当前最佳 finite-cutoff profile。论文图从未
用于求解、branch 选择、拟合、平滑或缩放。

## 7. TDHF/scalar-Hessian authority

generic/reduced TDHF algebra 与候选 provider/replay 层次来自该分支此前的独立
测试和 artifact，不是 HF-only job `461276` 的结论，也不授予 production
TDHF authority。job `461276` 没有证明：

- unrestricted/coherent global HF ground state；
- local HF Hessian positivity 或 finite-q stability；
- 作者相同 background/reference/cutoff；
- full-projector TDHF source promotion；
- Fig. 3 susceptibility、collective modes 或 pairing kernel reproduction。

因此当前仍是 **Vituri HF/TDHF debug 与资格审计分支**，不是 production
TDHF 结果分支。

## 8. 后续最小闭环

1. 向作者索取或确认 momentum domain/cutoff、finite-grid occupation、
   background/reference、gate distance、panel source state 和 raw Fig. 4(c)
   数据；
2. 在作者 policy 已知后，按同一 immutable FFT/full-SCF 路径复算；
3. 独立搜索 unrestricted/coherent/finite-q basins，并做 local Hessian；
4. 只有 HF source functional、basin 与 stability 都闭合后，才允许 TDHF
   source promotion。

Sealed sentinel/postflight 的 authority object 原样保持：

```text
independent_finite_volume_fixed_sector_fft_full_scf_discriminator = true
uv_plateau_established = false
author_cutoff_identified = false
unrestricted_ground_state_established = false
full_paper_reproduction_verified = false
tdhf_authority = false
production_authority = false
visual_match_promotes_authority = false
```

此外，尚未独立建立 `local_hf_stability_proved`、
`author_exact_numerical_policy` 或 `tdhf_source_promotion_authorized`；这些是报告
中的附加资格缺口，不是上述 sealed authority schema 的字段。
