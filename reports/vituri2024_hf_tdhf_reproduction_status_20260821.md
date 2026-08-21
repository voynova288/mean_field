# Vituri 2024 ABC 三层石墨烯 HF/TDHF 复现状态

更新时间：2026-08-21

分支：`debug/vituri-scalar-hessian`

本文是该分支当前唯一的 Vituri 结果状态报告，取代较早的 N101
简并分支报告和 2026-08-18 panel-c 初步比较。旧报告中的可核查 JSON
证据仍保留；物理结论以本文为准。

## 一句话结论

**已声明的 fixed-sector finite-volume stationarity、简并壳分支门和保存
结果重放均已通过精确检查，但更广义的 HF functional/source authority
没有闭合，论文主文 Fig. 4(c) 也尚未复现；因此现有 Vituri
TDHF/scalar-Hessian 代码不能以该 HF 态作为论文等价的生产源。**

主要差异是：论文的 `k=0` 中心最低点在费米能之上并只有两个外侧
交点，而独立 fixed-sector HF 得到
`E(0)-mu=-3.8536573 meV` 和四个交点。

![论文 Fig. 4(c) 与独立 fixed-sector HF 比较](figures/vituri2024_hf_band_panel_c_matched_density_comparison_20260818.png)

图左是 arXiv:2408.10309v1 主文 Fig. 4(c) 的原始栅格；图右是保存的
SCF 网格 `ky=0` 数据。论文图没有被数字化、拟合、缩放或作为求解器
输入。

## 1. 论文目标与来源

目标参数为：

- 模型：ABC trilayer graphene，第三低能 active band；
- `Delta1=28 meV`；
- signed carrier density `n=-1.03e12 cm^-2`；
- spin-polarized half-metal；
- 比较量：电子 HF 本征值 `E(k)-mu` 的 `ky=0` 切线。

本地论文：

- PDF：`/data/home/ziyuzhu/Mean_Field/reference/2408.10309v1.pdf`；
- 主文源码：`/tmp/arxiv_2408.10309v1_audit/source/main.tex`；
- 补充材料源码：`/tmp/arxiv_2408.10309v1_audit/source/SM.tex`；
- Fig. 4 原始图：`/tmp/arxiv_2408.10309v1_audit/source/figs/SC_fig_4.pdf`。

PDF SHA256：
`ec761a2b494a8e5983ff3fb6cfb842e114526cc0ba8b3e7cdc7c128f5d204bc8`。

这些绝对路径只是生成机器上的定位信息，不是可移植的仓库接口；
尤其 `/tmp` 解包源码可能被系统清理。长期公开来源是
<https://arxiv.org/abs/2408.10309>，报告中的源码判断绑定上述 PDF/source
hash，而不依赖读者机器存在相同路径。

## 2. 我们实际怎么算

### 2.1 单粒子 Hamiltonian

实现逐项使用补充材料的六带 Hamiltonian：

- 基底 `(A1,B3,B1,A2,B2,A3)`；
- hopping/on-site 参数为论文表格值；
- `pi=tau*kx+i*ky`；
- active band 是 zero-based index `2`，即第三低能带；
- 每个动量直接对角化六带矩阵并保存 active-band projector。

对 Hamiltonian、active-band index 和 `Delta1` 符号的重新核对没有发现
与论文的差异。补充材料后文把 B3 写成 `psi_6`，与其显示基底冲突；
这只影响所述 gauge 标签，不改变 homogeneous projector 或谱。

### 2.2 投影相互作用

独立计算使用论文形式

```text
V0(q)  = 2*pi*e^2*tanh(q*d)/(epsilon*q)
VTF(q) = V0(q) / [1 + qTF*(epsilon/(2*pi*e^2))*V0(q)]
```

以及：

- `epsilon=8`；
- `qTF=0.04/a0`；
- active-electron `R=0` 绝对密度；
- 解析保留有限的 dual-gate `q=0` 极限；
- 运行时独立选择 `d=250 Angstrom`；
- 二维权重为 `1/A=(Delta k/2*pi)^2`。

补充材料正文没有给出 `d`。对补充 Fig. 5 的矢量曲线和对数轴进行
条件反推：无 Thomas-Fermi 屏蔽与有屏蔽的两条独立 `q=0` 曲线都给出

```text
d ≈ 150*a0 ≈ 369 Angstrom.
```

这是基于论文显示公式的图形推断，不是作者文字或代码确认；本报告
不利用这一推断宣称 panel-c 的 gate-distance 敏感度界限。

### 2.3 固定密度与有限体积

计算不把连续密度四舍五入为一个任意占据，而使用相邻两个
finite-volume regulators：

- `H_v=768` holes per valley；
- `H_v=770` holes per valley。

对于给定整数 holes 和目标密度，物理面积固定，进而
`Delta k=2*pi/sqrt(A)`。N101 使用 odd Cartesian square labels。`H_v=768`
时 `a0*Delta k=0.00400256119`，与论文公布的 HF extraction spacing
`0.004` 匹配。

化学势不是后处理拟合值，而是保存 Fock 在固定 rank 边界上下能级的
中点。图只使用精确保存的 `ky=0` SCF 点。

### 2.4 简并壳与 ODA

若固定 rank 边界正好穿过简并壳，代码不会 stable-sort 选一个任意
态，而是：

1. 枚举所有纯 coordinate projectors；
2. 每次选择绑定 exact Fock 和 previous-density SHA256；
3. nested child 从共同 initializer 重放，而不是使用未经授权的中间态；
4. 正的 sub-tolerance splitting 作为 nonexact rejection，不作为分支许可；
5. 所有分支必须达到 projector、particle number、commutator、线性化能量
   和 byte-exact replay 门。

job `279879` 关闭了原 N101/Hv896 的完整 nested tree：八个 stationary
coordinate leaves、六个不同 endpoint、没有未解析 frontier。四个 nested
child `p1_c0/p1_c1/p4_c0/p4_c1` 分别消费 shell-state
`35499/35903/35503/35907`，均在八步后收敛。每个 child 的 final raw norm、
projector defect 和 Fock commutator 为零，particle number 为 `39012`；
independent/engine energy residual 不超过 `5.46e-11 eV`，线性化能量 excess
不超过 `2.91e-11 eV`。Sibling final densities 两两合并，但保存的 final
Hamiltonian、eigenvalue array 和 trajectory 并非 byte-identical。Child
能量比 inherited root 低约 `0.70662243 meV`，但有 `888/904` 或
`904/888` 的有限体积 valley imbalance，因此不能作为 homogeneous
half-metal 候选，也不是 global-ground-state 证据。

panel-c fixed half-metal 的 job `303156` 对 `H_v=768,770` 各枚举四条
mirror-shell coordinate paths。八条路径均在三步后 stationary，并在每个 regulator 内精确合并为同一
final density、Hamiltonian、scalar energy 和 band cut。

## 3. HF 结果

### 3.1 Panel-c 中心与交点

job `303156`：

| regulator | `E(0)-mu` | `kx*a0` 线性插值交点 |
|---|---:|---|
| `H_v=768` | `-3.8536573012 meV` | `[-0.0755785,-0.0268241,0.0224130,0.0620849]` |
| `H_v=770` | `-3.8575364761 meV` | `[-0.0755739,-0.0268538,0.0224279,0.0620746]` |

两个 regulator 的中心只差 `0.00388 meV`，branch envelopes 宽度严格为
零。因此 tested rank bracket 和 coordinate branch 选择不能解释中心
符号或交点数差异。匹配论文的显示窗口之外，deep negative tail 在两个
regulator 间相差约 `6.82 meV`；所以这不是 full-cut 或 thermodynamic
convergence 结论。

### 3.2 `h0` 与 HF 自能分解

`H_v=768` stationary endpoint：

```text
bare h0 center relative to sector mu       -4.7426265 meV
centered HF nonidentity correction         +0.8889692 meV
final HF center relative to mu             -3.8536573 meV
```

所以论文要求的上移方向是正确的，但当前 HF 修正不足。若保持 projector
不变而只缩放 centered interaction，需要大于 `5.33497` 的强度因子才能
使中心过零；这只是诊断，不能作为拟合许可。

### 3.3 Frozen-Fock global Aufbau 检查

对 job `303156` 保存 Fock 在全部四个 flavors 上重新排序：

- `H_v=768`：零占据变化，global gap `0.0396596 meV`；
- `H_v=770`：零占据变化，global gap `0.0607894 meV`。

因此简单放开固定 flavor rank 的 diagonal resort 不改变状态。这不排除
unrestricted coherent、finite-q 或其他 SCF basin。

## 4. UV/domain 与 reference 检查

### 4.1 Sealed one-shot domain ladder

job `432973` 固定 `H_v=768`、密度、面积、`Delta k`、权重、相互作用和
共同标签上的初始 projector，只扩大 square domain；每个点只做一次
initial Fock action，没有 SCF update，也没有 Fock-derived occupation：

| mesh | `a0*k_axial` | `E(0)-mu` |
|---:|---:|---:|
| N81 | `0.16010245` | `-5.1196213 meV` |
| N101 | `0.20012806` | `-4.3351942 meV` |
| N121 | `0.24015367` | `-3.5285999 meV` |

总上移 `1.5910214 meV`，小于 stationary mismatch `3.8536573 meV`，而且
没有 UV plateau。该区间不足以解释论文，但也没有建立 UV 收敛。

### 4.2 完全填满 active-band reference 的冻结候选

对当前 homogeneous、flavor-diagonal 保存态，同一 valley 的另一个 flavor
完全填满，并共享 `h0` 和总 Hartree field。去除 `k`-independent Hartree
常数后，可以构造有限范围的冻结诊断：

```text
F_h(R=I) = F_h(R=0) + h0 - F_full(R=0).
```

这不是一般恒等式，也不是论文授权的 reference。结果：

- stationary center：`-3.85366 -> -1.79513 meV`；
- 仍有四个交点；
- frozen N81/N101/N121 ladder 在共同 hole labels 和不变 boundary ordering
  下均为 `-2.2268972 meV`，精确到 roundoff。

在上述 candidate 和全部假设下，tested ladder 的 `R=0` 漂移会代数
消除；这不证明真实因果归属，也不能由此声称更大/不同 domain 或 SCF
后已经 UV 收敛。论文显示的是 raw active quartic 加可吸收到 `mu` 的
`Delta mu*N`，没有明确授权 `R=I`。

### 4.3 论文 Hartree 指标问题

从论文先给出的 projected quartic 严格收缩，Hartree 内侧应为 `u_jj(k')`。
论文后面的 HF 方程却打印为 `u_ii(k')`，同时保留一个未使用的 `sum_j`。
在 homogeneous `deltaG=0,q=0` 才会按正确 `jj` 收缩化为 identity。

job-303156 的 `d=250 Angstrom` 冻结 counterfactual：

- 保留错误 same-`i` 但删除无用 `sum_j`：`+32.17 meV`；
- 完全按打印式保留 factor 6：`+225.49 meV`。

两者都远离论文中略高于零的中心，且都不是 SCF。最合理解释是打印
指标错误，但作者实现不可见。

## 5. 与论文 Fig. 4(c) 的区别

一致之处：

- 总体 asymmetric double-hump 形状；
- 外侧动量尺度相近；
- HF 相互作用相对 bare `h0` 把中心向上移动。

不一致之处：

1. 论文中心 minimum 在 `E=0` 以上；计算为约 `-3.85 meV`；
2. 论文显示两个外侧 crossing；计算有四个 crossing；
3. 论文描述的是接近出现 inner pocket 的 simply connected hole pocket；
   计算已经有中心 occupied-electron island，对应 annular hole FS；
4. 论文没有给出原始数值数组，因此没有对 paper curve 指定伪精确的
   digitized center 或 roots。

这里不存在 electron/hole sign 翻转：若把电子谱反号，论文中心 local
minimum 会变成 maximum，与其 inner-electron-pocket Lifshitz 解释矛盾。

## 6. 当前 TDHF/scalar-Hessian 代码实际证明了什么

该分支包含：

- system-agnostic scalar-curvature approval/certificate；
- Vituri restricted finite-orbital actual-vertex oracle；
- full-projector factorized `E/F/dF` functional；
- immutable provider-candidate schema 和 replay bridge；
- signed `A/B` readiness 与严格 authority flags。

已经验证的层次是：

1. generic scalar finite-difference/algebra contract；
2. reduced finite-orbital Vituri vertex/C9 parity；
3. factorized full-space algebra与小维 rank-four oracle parity；
4. candidate saved-array schema、hash 和 selected-representative replay parity。

尚未证明：

- job `303156` 是作者相同 HF functional 的 unrestricted ground state；
- current source 的 local Hessian 为正；
- complete full-projector provider 对真实 source 获得 scalar-Hessian authority；
- panel-c HF state 可以提升为 TDHF source；
- 论文 Fig. 3 susceptibility、collective modes 或 pairing kernel 被复现。

因此本分支目前是 **TDHF/scalar-Hessian debug 与资格审计分支**，不是
Vituri production TDHF 结果分支。没有可信的 panel-c TDHF spectrum 可以
在本报告中展示。

## 7. 尚未解决的问题

需要作者或独立来源确认：

1. HF momentum-domain 形状、cutoff、点数、endpoint/wrap 和 quadrature；
2. Fig. 4(c) 是 `a0*Delta k=0.004` 原网格还是 `0.0016` 插值切线；
3. finite-grid exact-rank occupation 和简并边界处理；
4. active quartic 使用 raw density，还是未打印的 filled-band/reference
   subtraction；
5. `q=0` background/normal-order 约定以及 `d=150*a0` 图形推断的确认；
6. Fig. 4(c) 来自 unrestricted mBZ solver 收敛到零 IVC/CDW，还是单独
   constrained homogeneous half-metal；
7. panel-c 原始 `kx*a0,E-mu` 数据或 converged density/Fock checkpoint；
8. unrestricted/coherent 与 finite-q local stability；
9. 在上述 HF source authority 闭合后，真实 full-projector TDHF Hessian、
   susceptibility 和 collective-mode paper parity。

在这些信息缺失时，盲目增加 N161/N201 只会引入新的独立 cutoff 选择，
不能称为 paper discriminator。

## 8. 证据与可复查路径

当前报告的机器可读证据：

- `reports/data/vituri2024_hf_band_panel_c_matched_density_metrics_20260818.json`；
- `reports/data/vituri2024_hf_band_panel_c_matched_density_provenance_20260818.json`；
- `reports/data/vituri2024_n101_hv896_nested_branch_oda_279879_attestation.json`；
- `reports/data/vituri2024_hf_band_panel_c_discrepancy_followup_20260820.json`。

下列 runroot 绝对路径同样只是生成机器上的非可移植定位信息；仓库中
可移植的证据是上列 JSON、figure 和其内记录的 hashes。

主要 sealed runroots：

- nested branch job `279879`：
  `/data/home/ziyuzhu/.runs/Mean_Field_a397fa0_vituri_hf_n101_hv896_nested_branch_oda_v2_20260816`；
- panel-c stationary job `303156`：
  `/data/home/ziyuzhu/.runs/Mean_Field_a397fa0_vituri_hf_band_panel_c_n103_mirror_branch_v2_20260818`；
- UV one-shot job `432973`：
  `/data/home/ziyuzhu/.runs/Mean_Field_a397fa0_vituri_hf_band_panel_c_uv_one_shot_v2_20260818`。

Authority flags remain：

```text
unrestricted_global_aufbau_solution = false
local_hf_stability_proved = false
paper_reproduction_verified = false
author_exact_numerical_policy = false
tdhf_source_promotion_authorized = false
```
