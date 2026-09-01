# Vituri 2024 ABC 三层石墨烯 HF/TDHF 复现状态

更新时间：2026-08-31

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

本报告包含三条互补证据链：job `461276` 是 full cutoff ladder；job
`462560` 加 zero-science recovery `462719` 是 chosen-contract Fig. 4(c)
复算与图形发布；job `466196` 则证明同一 chosen contract 已由新的通用
exact-grid curve API 完整接管，且与旧胶囊数值输出保持 parity。

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

承担 chosen-contract 科学复现 authority 的 Fig. 4(c) 复算使用 source commit
`09075cd22d47edbb4738229d98b52a9650154ec5` 与公开 fixed-sector BFS API；
后续 job `466196` 是同一 contract 的 generic-API engineering parity 重跑。
原 job `462560` 的数值 step 和独立 postflight 均 `COMPLETED 0:0`，batch 仅因
compute node 缺失 login-node gnuplot target 而在绘图阶段 `FAILED 1:0`；
zero-science recovery job `462719` 用保存的 raw branch CSV 生成图并完成原子
发布，没有重跑或修改 SCF。最终 sentinel：
`/data/home/ziyuzhu/.runs/Mean_Field_09075cd_vituri_fig4c_chosen_contract_v2_20260828/COMPLETE_462560.json`。

### 通用 API 替换闭合（job 466196）

source commit `7f884f81fed8e24c6735dab95133d62320e12345` 上的 job `466196`
重新执行同一个 N179/Hv769/d369 chosen contract，并只通过公开
`run_vituri2024_fixed_sector_bfs`、Vituri curve adapter 和
`mean_field.core.curve_workflow` 生成产物。外部 completion 后的 `sacct`
观察为 `COMPLETED 0:0`、64 CPU、`regular256/node048`；这些 scheduler
字段不是 sealed artifact manifest 内的科学证据。完整 runroot：

`/data/home/ziyuzhu/.runs/Mean_Field_7f884f8_vituri_fig4c_generic_api_replacement_v5_20260830`

该执行先写出并完整 reload compute-only JSON/NPZ/CSV，再写入
`CALCULATION_FREEZE.json`；只有计算冻结后才读取旧 job `462560` 输出和论文
raster。随后 full-workflow artifact 又经过一次 loader 重建和 comparison
重算。最终 sentinel SHA256 为
`dcdf1d7b3e5a791c9dfd2d4af0ee08e52034ee18bfc10410609159e9448e973b`，
bundle fingerprint 为
`12f201a6549e4bf6038f65049f26db46bd5eb4008820f4907f58f27513c3aca3`。

与 job `462560` 的四条 branch 逐路径比较得到：

- 所有 transformed-y 数组 exact equal；
- 最大 x 差 `5.55e-17`，最大 crossing 差 `1.39e-17`；
- center、maxima energy 和 branch-spread 差均为零；
- 没有 alignment、fit、rescale 或 branch postselection；
- 预声明 `1e-10` replacement tolerance 内没有任何超限项。

新的 raster 比较保持 `postfreeze nonblind evidence only`，不是 held-out claim：
402 samples，RMSE `0.963557800 meV`，MAE `0.656716052 meV`。generic residual
采用 calculation-minus-paper，因此 mean error `-0.121829872 meV`；旧报告采用
paper-minus-calculation，所以符号相反但数值一致。机器可读 attestation：
`reports/data/vituri2024_fig4c_generic_api_replacement_466196_attestation.json`，
SHA256 `b7d4a5f24743121da3f21d0c7a583b4c6813e6323ea1708706b4edbaed46cc81`。

这关闭了“通用 API 完全替代旧 panel-specific 胶囊”的工程问题，但不改变科学
权限边界：chosen-contract reproduction 仍由 job `462560/462719` 支持；job
`466196` 只增加 generic-API replacement/parity authority，不建立作者 cutoff、
UV plateau、unrestricted ground state、Hessian、TDHF 或 production authority。

前序 v4 job `466166` 在任何数值构造前因只读 source 缺少可写 Numba cache
locator 而失败；v5 使用 node-local `NUMBA_CACHE_DIR` 修复，v4 没有 output、
sentinel 或科学结果。

两个更早的 ladder capsule 均只留下预数值工程失败 provenance；它们不是 job
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
非对称曲率上均强一致。计算冻结后，又对论文 embedded raster 做了 postfreeze
nonblind、evidence-only 像素标定；chosen contract 的历史选择并非 blind，因此
这里不再使用 held-out claim。该步骤没有回流 solver、参数、branch、convergence
或 pass gate。
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

## 7. Fig. 2 单-q IVC spiral 候选进展

在 Fig. 4(c) 闭合之后，本分支新增了体系层
`vituri2024_hf_spiral.py`。它实现 SM 中 `G=0` 的 shifted-valley 基底
`p_(tau,k)=k+tau*q/2`，复用既有 exact no-wrap `E/F/dF` 与通用
`run_hartree_fock_problem`；没有复制 SCF loop。选定自旋的两个 valley 只固定
总 rank `2*Nk-2*H_v`，允许 valley redistribution 与 IVC coherence；另一自旋
保持全满。显示基底的 B3 分量被独立固定为正实数，但由于论文 `B3/psi_6`
文字冲突，它不是 author-gauge authority。`G=0` 也不能表示三-q IVC crystal。

代码里 dense 与 FFT 在非零 q 的 interaction action、energy、Fock 与 dF 已通过
小网格 parity；q=0 identity gauge 则逐字节退化为旧 homogeneous functional。
提交 `e12486e`、`4fc36f4` 与 `8e96b6e` 分别加入候选 spiral API、FFT backend
和 hash-bound saved-density/full-step continuation。相关 focused/API 测试为
`62 passed`；完整 non-slow suite 另有一个与本改动无关的历史 TBG artifact
fingerprint mismatch，因此未把该单项失败归因于 Vituri。

首个 N81、`a0*Delta k=0.004`、`d=369 Angstrom`、`Delta1=28 meV` 的 q scout
job `468711` 覆盖 `H_v=635,747`、`qa0=0,0.02,...,0.07` 和 normal/IVC 两个
initializer，共 28 次尝试。它按预注册 gate 正确发布
`DIAGNOSTIC_INCOMPLETE`：6 个 normal endpoint stationary，8 个 normal 尝试在
全局 selected-spin occupation boundary 的 exact zero gap 上 fail closed；14 个
IVC 候选保持非零 `|phi|/n_h`，但 13 个因 ODA 选择精确 `lambda=0` 而 stall，
一个达到 max iteration，故当时没有授权能量比较。

后续 job `468737` 先对每个 density 中 `|phi|` 最大的 exact-lambda-zero 候选做
两个 preregistered full-step discriminator；两者分别在 34、14 步收敛。随后 job
`468739` 对全部 14 个保存的 IVC density 做 exact hash-bound continuation，结果
**14/14 converged 且 fresh-map stationary**：

- full-step iterations：`10--63`；
- fresh-map max residual：`3.685e-9--4.653e-9`；
- commutator residual：`3.384e-11--5.930e-11 eV`；
- idempotency residual 最大约 `1.55e-15`；
- occupation gap：`6.49e-7--3.53e-5 eV`；
- `|phi|/n_h` 的最大 absolute drift 约 `6.33e-8`，没有候选塌缩到
  `phi=0` normal branch。

这说明在本次声明的 14 个有限域候选里，原 ODA `lambda=0` 终止不能解释为
“没有邻近 IVC fixed point”；强制相同 HF projector map 取 full step 后均闭合。
它不证明 full-step 一般优于 ODA，也不授予 optimal-q 或能量下降 authority。

随后提交 `2dbb809` 与 `38441da` 加入 normal exact-shell BFS、共同 initializer
重放和 exact-h0 seed lineage。历史 normal initializer 在部分参数上位于 exact
noninteracting h0 shell；这里只把其 canonical C-order coordinate choice 绑定为
job `468711` 的共同 pre-SCF seed，不把它提升成物理 shell selector。任何实际
applied SCF Fock map 上的 exact frontier 仍穷举全部 coordinate projectors。

Normal closure jobs `469823` 与 recovery `469847` 的 hash-bound merge 已闭合
全部 8 个预声明执行 inventory：

- 7 个参数点得到共 `1088/1088` 个 fresh-map stationary normal endpoints，全部
  terminal replay、full-step 和 branch-tree exhaustion 通过；
- `(H_v,q a0)=(635,0.03)` 的 16 条 exhaustive paths 全部得到 typed
  `positive_subtolerance_splitting_rejection`，所以本协议没有授权该点的 normal
  comparator；这不证明协议之外不存在 normal stationary solution；
- `(747,0.07)` 的扩大-cap recovery 穷举 `2051` 个节点，得到 `1028/1028`
  stationary endpoints，分成 8 个 density groups。

针对 `(635,0.03)`，job `474291` 又对覆盖全部 16 条 rejection paths 的两个
unique density classes 做了独立 O(Nk^2) dense-Fock oracle。每个 class 中，保存的
Hamiltonian 与重新计算的 FFT Fock **逐数组完全相等**；FFT boundary gap 均为
`2.220446049250313e-16 eV`，在绑定的 `1e-12 eV` roundoff envelope 内，因此被
原规则分类为 positive-subtolerance。独立 dense action 则在两个 density 上都给出
逐浮点 exact `gap=0`。dense/FFT 的完整 Fock 最大差仅
`6.661365969895318e-16 eV`，scalar-energy 差为
`3.637978807091713e-12 eV`，均通过预注册 parity gate。

所以这里的正 gap 已定位为 **相对 dense float64 oracle 的 FFT-backend-specific
arithmetic splitting**，不是已证明的物理有限域 lifting。但该 oracle 只诊断原因，
没有改写已执行的 applied-map 轨迹，因而仍未凭自身恢复 normal comparator。下一步
必须做 dense-oracle-bound exact-shell replay，或先证明一个 symmetry-preserving FFT
map correction；不能简单放宽 tolerance 或把该 rejection 后处理成 endpoint。

仅对 7 个具有 normal endpoint 的相同 `(H_v,q)` 对，使用相同
base/functional/gauge physical-map fingerprints 计算
`1000*(E_IVC-E_normal)`。以下数值是固定 N81 finite square 的 raw **total**
scalar-energy difference（meV），没有按 hole、particle、cell 或 area 归一化，
也不是论文 Fig. 2 的报告尺度：

| holes/valley | q a0 | normal leaves | Delta E range (meV) |
|---:|---:|---:|---:|
| 635 | 0.02 | 6 | -660.5420 to -660.4199 |
| 635 | 0.03 | 0 | unavailable: 16 typed positive-subtolerance rejections |
| 635 | 0.05 | 2 | -449.4190 to -449.4190 |
| 635 | 0.06 | 8 | -270.2656 to -269.9346 |
| 635 | 0.07 | 36 | -147.6193 to 387.2849 |
| 747 | 0.04 | 2 | -449.2970 to -449.2970 |
| 747 | 0.05 | 6 | -477.4558 to -477.4314 |
| 747 | 0.07 | 1028 | -143.7555 to 160.3716 |

所有 normal leaves 都保留，未按能量、symmetry 或 diagnostics postselect；因此
两个 `q a0=0.07` 点的 sign 是 branch-dependent，而不是唯一 phase-ordering
结论。由于 `(635,0.03)` 没有合法 comparator、normal branch 也未证明唯一，完整
Fig. 2 curve、最佳 q、cross-q 排序和 paper reproduction 仍未授权；cross-q 比较
还受 shifted finite-domain UV 边界影响。

机器可读证据：

- `reports/data/vituri2024_fig2_g0_spiral_candidate_progress_468711_468739.json`
  （SHA256 `cee4b03b6d506080d28899d1cc67d4876f08360143b446b7d5781fccce19f773`）；
- `reports/data/vituri2024_fig2_normal_exact_shell_closure_469823_469847_merge_v8.json`
  （SHA256 `3fa3311c645bd6fa9ab4bc200ce85a15857092533cfeda1c72fabb73599e0ea5`）；
- `reports/data/vituri2024_fig2_q003_dense_boundary_oracle_474291_attestation.json`
  （SHA256 `09daa514f91246698269a132828add8591fbf4a11e508f81985b0a1b3c13b0e5`）。

## 8. TDHF/scalar-Hessian authority

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

## 9. 当前结论与后续边界

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
selected_spin_g0_ivc_saved_endpoint_stationary_closure = true
normal_exact_shell_execution_inventory_closed = true
stationary_normal_comparator_case_count = 7
same_q_raw_total_energy_comparison_case_count = 7
all_eight_stationary_normal_comparators_available = false
q003_positive_subtolerance_root_cause_localized_to_fft_backend_arithmetic = true
q003_normal_comparator_recovered = false
matched_normal_exact_shell_closure = false
same_q_phase_ordering_established = false
fig2_reproduction_verified = false
local_hf_stability_proved = false
tdhf_authority = false
production_authority = false
visual_similarity_used_as_pass_gate = false
```

这里 `full_paper_reproduction_verified=false` 至少表示未复现论文全部 phase
diagram、susceptibility、collective modes 与 pairing calculation，也没有作者
exact contract 或 paper raw array 的 pointwise parity；它不否定 Fig. 4(c) 的
chosen-contract reproduction。
