# Du2017 InAs/GaSb Kane–Poisson debug report

**分支**：`debug/du2017-kane-poisson-uv-20260723`
**基线**：`origin/main@0ed1b9f3d435bd06cde58257045752f31b234cda`
**日期**：2026-07-23
**状态**：debug checkpoint；**不是完整Kane–Poisson认证，也不允许启动HF/BCS**

## 1. 目标和边界

目标是独立复核Du et al., *Nature Communications* **8**, 1971 (2017) Supplementary Note 6中的InAs/GaSb 8-band Kane–Poisson normal state，再决定是否允许后续多带BCS/HF。

当前坚持的物理闭合是

\[
U\rightarrow H_{\rm Kane}[U]\rightarrow E_s,\Phi_s
\rightarrow \mu:\ n_e(\mu,U)=n_h(\mu,U)
\rightarrow n_e(z),n_h(z)\rightarrow U_{\rm Poisson}.
\]

以下操作均被禁止：

- 把实验密度作为Note-6输入；
- 从独立电子/空穴根拼出共同化学势；
- 为匹配论文调能量、动量或密度比例；
- 放松pair-overlap/assignment门继续SCF；
- 在缺少来源时自行引入参考海、背景扣除或任意UV cutoff；
- 在normal parent未认证时启动HF/BCS。

## 2. 最重要的根因修正：旧pair不是物理CB1/VB1

旧canonical cutoff ladder在`zres=0.5 nm`、candidate-28父空间中使用Gamma候选位置

```text
E: (14,15), energy ~ +5.864 meV
H: (12,13), energy ~ -11.215 meV
```

该低能quartet的数值输运和replay可以复现，但它不是Du2017的最低InAs conduction subband与最高GaSb valence subband。因此旧ladder不能用于物理E1/H1密度、broken-gap overlap或cutoff结论。

独立TEIB orbital/layer oracle与完整波函数投影共同识别出同一kdotpy父模型中的物理pair：

```text
E1/CB1: candidate positions (20,21), E = 89.4323 meV
H1/VB1: candidate positions (22,23), E = 125.3401 meV
bare overlap H1-E1 = 35.9078 meV
```

选择依据是Gamma6/valence成分、InAs/GaSb层局域性及rank-two full-wavefunction projector，不是论文目标能量。

### 旧结果的处理

旧ladder的以下数值只保留为wrong-pair forensic records：

- candidate `28→32`在固定cutoff下密度变化`5.28e-16`；
- `kmax=0.15→0.18 nm^-1`密度变化`45.10%`；
- 化学势变化约`0.200 meV`；
- 旧`kmax=0.18`密度`7.0404e10 cm^-2`。

这些值不能认证或否定物理CB1/VB1 Kane–Poisson链。

## 3. 生长方向z离散：来源方法收敛，但无限FDM refinement不收敛

### 3.1 来源一致的plane-wave结果

Supplementary Note 6沿z使用有限plane-wave展开。使用物理E1/H1 projector输运得到：

| plane-wave N | E1-H1 (meV) |
|---:|---:|
| 47 | -35.4002 |
| 63 | -35.4844 |
| 79 | -35.5291 |
| 95 | -35.5639 |

最后两级变化为`0.0447`和`0.0347 meV`，相邻N的E1/H1 projector principal values在N=95约为`0.999985/0.999999`。因此裸Gamma父模型在该来源方法下给出稳定broken-gap overlap：

\[
\delta_{\rm BG}^{\rm bare}(\Gamma)\simeq35.56\ {\rm meV}>0.
\]

### 3.2 FDM细网格的高-kz谱污染

把有限差分步长不断减小并不等价于上述来源方法。细网格允许越来越大的非物理`kz`模式进入8-band连续模型：

- 粗网格单一E1 Kramers pair与TEIB projector重叠约`0.995`；
- 细网格时E1投影分裂到两个相邻Kramers pairs；
- 两pair合并子空间仍约`0.9985`稳定；
- 离散梯度和高频谱权重显示邻近态是高-kz污染/混合，而不是已认证的新低能物理pair。

结论：生产z闭合必须保留来源有限plane-wave cutoff或另有物理正则化；不能用`dz→0`的无限带宽FDM宣称收敛。

## 4. 正确pair的径向预检通过

在`kmax=0.24 nm^-1`、96个径向点、candidate 28下，正确Gamma anchors `(20,21)/(22,23)`给出：

```text
minimum E/H radial principal values  ~0.99218 / 0.99216
minimum E/H assignment margins       ~0.48442 / 0.48440
minimum candidate-edge gap            9.10 meV
maximum eigen residual                7.55e-12 meV
```

因此裸CB1/VB1在该范围内可被可靠识别，且确实存在电子—空穴能带交叠。问题不是“没有broken gap”。

## 5. Corrected-pair canonical Poisson：固定网格SCF收敛，但in-plane cutoff不收敛

正确pair的canonical-periodic结果为：

| kmax (nm^-1) | nr | identity evidence | mu (meV) | n_e=n_h (cm^-2) |
|---:|---:|---|---:|---:|
| 0.12 | 32 | separate E/H replay | 116.1927 | 1.3933e11 |
| 0.18 | 72 | separate E/H replay | 111.5731 | 2.6138e11 |
| 0.24 | 128 | quartet-only diagnostic | 113.5205 | 3.1622e11 |
| 0.30 | 200 | quartet-only diagnostic | 119.5144 | 4.3018e11 |

各完成run在自己的固定离散上满足：

- canonical common-mu neutrality root；
- fixed-point residual约`1.7–1.9e-5 meV`；
- periodic Poisson Gauss residual约`1e-13 meV nm^-2`；
- typed numeric archive exact reload；
- warm reconstruction和两个独立cold builders replay。

但是successive density changes约为：

```text
0.12 -> 0.18 : 87.6%
0.18 -> 0.24 : 21.0%
0.24 -> 0.30 : 36.0%
```

`kmax=0.30 nm^-1`最外10%径向网格仍贡献约`22.3%`的hole orbital-transfer density。因此固定网格SCF收敛不等于物理cutoff收敛。

自洽结果中的局域character broken-gap overlap仍约`23.7–30.1 meV`，即Poisson没有消除交叠；但该数值随未闭合的UV cutoff变化，不能作为最终预测。

## 6. E/H rank-two identity与quartet charge之间的冲突

严格separate-pair `kmax=0.24` run在`ik=73` fail closed：

```text
H1 previous-U assignment margin = 7.001e-4
required margin                  = 1.000e-3
```

没有放松阈值，也没有使用previous-U homotopy继续生产SCF。

同一物理四维quartet在整个路径上稳定，且Note-6 orbital charge对quartet内部U(2)xU(2)基底旋转具有规范不变性；但E/H rank-two partition在反交叉附近可能出现路径依赖。quartet-only runs仅用于区分“电荷公式是否可计算”和“单独E/H标签是否已认证”，不能替代用户要求的separate-projector生产门。

此前`zres=0.25`的另一低能wrong quartet也出现过类似现象：quartet principal values为1，但不同输运路径得到不兼容的E/H rank-two projectors。candidate `28→32`不能消除冲突。这证明问题不是简单索引或候选窗口不足。

## 7. in-plane UV问题

### 7.1 UV在这里的含义

UV指高动量、短波长自由度，不是紫外光。Note-6密度含二维动量积分

\[
n\sim\int d^2k\,w(k)=2\pi\int^{k_{\max}}k\,dk\,w(k).
\]

对典型两带Kane反交叉

\[
H(k)=\begin{pmatrix}
\alpha_e k^2 & Ak\\
A^*k & -\alpha_h k^2
\end{pmatrix},
\]

错误轨道成分渐近为

\[
w_{\rm mix}(k)\sim\frac{|A|^2}{(\alpha_e+\alpha_h)^2k^2},
\]

因而二维正定orbital-transfer density一般含

\[
\int^K\frac{dk}{k}\sim\log K
\]

的UV敏感性。若当前范围仍处于强混合/近简并区，pre-asymptotic增长可以更强。

### 7.2 完整带空间不自动抵消

令`P_c`为Gamma6 projector，`P_v=I-P_c`，`F`为占据projector：

\[
n_e(k)=\operatorname{Tr}(P_cF),\qquad
n_h(k)=\operatorname{Tr}[P_v(I-F)].
\]

完整性在适当rank filling下可证明`n_e(k)=n_h(k)`，但两者都是正定转移量，不能彼此抵消。完整1056-state FDM parent的代数一致性因此不能解决物理UV；其旧结果`mu≈1059.8 meV`、`n≈2.17e12 cm^-2`正是UV dominated diagnostic。

## 8. 来源缺口

### 8.1 Du2017 Supplementary Note 6

补充材料说明：

- Kane Hamiltonian只在Gamma附近成立；
- z方向使用plane waves；
- 电子密度使用Gamma6 components 1–2；
- 空穴密度使用valence components 3–8和`1-f`；
- Fermi level由电中性决定。

但没有提供：

- in-plane积分范围或网格；
- 精确保留的`s`子带列表；
- 参考海、离子背景或normal-ordering subtraction；
- 完整BZ/lattice completion；
- electrostatic device boundary。

### 8.2 Li–Yang–Chang method

PRB 80, 035303 (2009)明确写道：

> The summation Sigma_s includes all the subbands which show anticrossing behavior.

该文在讨论体系中只考虑最低conduction与最高valence subbands的反交叉。这支持物理CB1/VB1 anticrossing family，而不提供in-plane cutoff。其footnote 34还承认`sum_s`没有遍历全部子带，导致Fermi level与精确值略有差异。

### 8.3 TEIB和公开仓库

- TEIB默认`kmax=0.12 nm^-1`和固定等电子/空穴密度；这是代码作者明确标注的透明近似，不是Du cutoff证据。
- Rice DSpace公开item的ORIGINAL bundle只有主文PDF，没有数值数组、源码或隐藏supplementary code。

## 9. 目前不能采用的“修复”

| 候选 | 决定 |
|---|---|
| 不断增加`kmax` | 拒绝；会进入Kane无效范围且不保证消除log tail |
| 在密度接近实验`n0`处停止 | 拒绝；target-driven |
| 使用TEIB固定Fermi disk | 只作forensic fixed-density control |
| 改成整支band occupation counting | 新模型，不是Note-6 component formula |
| 减去`U=0`或decoupled-layer参考海 | 来源未给出，未授权 |
| 使用全部FDM states | 代数完整但UV dominated |
| 打开kdotpy lattice regularization | 新的lattice model；Du2017未声明，也缺BZ积分约定 |
| 放松pair assignment threshold | 拒绝 |

## 10. electrostatic ensemble和材料不确定性

当前Poisson结果只能标记为

```text
canonical_periodic_window_root_not_fixed_gate
```

它不是Wafer-B fixed-gate device calculation。缺失的信息包括gate stack、work function、固定界面/表面电荷、掺杂及电压零点。

此外GaSb `gamma2=0.08`是根据Supplementary Table 1可能的`8.18`排版错误、bulk Luttinger参数和inverse Kane renormalization推断的opt-in修正。它有source-hash attestation，但未获作者确认。所有上述broken-gap结果必须带此限定。

## 11. 当前认证矩阵

| Gate | 状态 |
|---|---|
| split=0/source-pinned Hamiltonian | PASS |
| inferred gamma2 attestation | PASS，但未获作者确认 |
| physical bare CB1/VB1 identity | PASS |
| bare Gamma broken-gap overlap | PASS |
| source-method z plane-wave convergence | PASS |
| candidate/radial preflight | PASS |
| fixed-discretization SCF residual | PASS |
| warm + two independent cold replay | PASS for completed runs |
| separate E/H identity through kmax=0.24 SCF | FAIL CLOSED |
| in-plane momentum cutoff convergence | FAIL |
| source-backed UV/reference closure | MISSING |
| fixed-gate device boundary | MISSING |
| temperature convergence | BLOCKED |
| HF/BCS/JDOS release | BLOCKED |

## 12. 下一步，只能按以下顺序

1. 向作者索取Note-6实际in-plane k grid/range、`sum_s`列表、reference/background、z-plane-wave N、Poisson boundary和GaSb gamma2说明。
2. 若获得来源闭合，先实现typed cutoff/reference policy，并使archive明确记录其来源和hash。
3. 从正确CB1/VB1重新做candidate、radial、cutoff、z、temperature ladder。
4. 要求所有生产结果通过separate E/H projectors、warm replay和两个真正独立cold replay。
5. 只有normal parent全门通过后才启动HF/BCS；不得把历史scalar/TEIB/presentation plots升级为预测结果。

## 13. 本地证据路径

以下为本地ignored evidence，不包含在本report-only Git branch中：

```text
results/inas_gasb_matrix_ei_experiment/runs/
  phase0_du2017_teib_bare_gamma_n_convergence_v2_20260723/
  phase0_du2017_teib_kdotpy_e1h1_projector_oracle_v2_20260723_roughness/
  phase0_physical_cb1_vb1_pair_preflight_kmax024_20260723/
  phase0_physical_cb1_vb1_canonical_nr32_kmax012_20260723/
  phase0_physical_cb1_vb1_canonical_nr72_kmax018_20260723/
  phase0_physical_cb1_vb1_canonical_nr128_kmax024_20260723/
  phase0_physical_cb1_vb1_quartet_canonical_nr128_kmax024_20260723/
  phase0_physical_cb1_vb1_quartet_canonical_nr200_kmax030_20260723/
  phase0_physical_cb1_vb1_canonical_cutoff_audit_v2_20260723/
  phase0_kane_poisson_physical_pair_correction_20260723/
  phase0_du2017_inplane_uv_source_audit_20260723/
```

关键报告：

```text
phase0_kane_poisson_physical_pair_correction_20260723/
  KANE_POISSON_CONVERGENCE_STATUS.md

phase0_du2017_inplane_uv_source_audit_20260723/
  INPLANE_UV_SOURCE_AUDIT.md
```

## 14. 分支内容说明

该分支故意只提交本debug报告。原工作树包含大量其他体系、topology、optical和HF并行修改；为避免污染，本分支从`origin/main`独立创建，没有stash、clean、reset或打包这些改动。InAs/GaSb实现和大型结果目前仍是本地工作区/ignored evidence，后续如需推送代码，应另做一次按依赖闭包审计的isolated commit。

## 15. 结论

当前已经确认两件此前混淆的事实：

1. 物理CB1/VB1确实是broken-gap，裸Gamma overlap稳定在约`35.56 meV`；
2. 正确pair并不会自动使Note-6 canonical density cutoff收敛。

剩余问题不是通过调图、换pair或加密网格可以解决的普通数值误差，而是连续8-band orbital-transfer density缺少公开的in-plane UV/reference定义。获得该定义前，最科学的状态是：保留固定离散SCF和broken-gap证据，明确撤销wrong-pair物理解读，并继续冻结完整Kane–Poisson、HF和BCS声明。
