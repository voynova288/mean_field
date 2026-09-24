# 1T-TiSe₂ 无相互作用低能能带

依据 C. Monney *et al.*, “Spontaneous exciton condensation in 1T-TiSe₂: a BCS-like approach”, [arXiv:0809.1930](https://arxiv.org/abs/0809.1930) 的 Sec. II A 与文末参数说明绘制。

## 使用的裸色散

令 `q` 分别表示相对 Γ 点（价带）和三个 L 点（导带）的动量：

\[
\epsilon_v(\mathbf q)=\frac{\hbar^2(q_x^2+q_y^2)}{2m_v}
+t_v\cos(\pi q_z/k_{\Gamma A})+\epsilon_v^0,
\]

\[
\epsilon_c^{(i)}(\mathbf q)=
\frac{\hbar^2q_{\parallel,i}^2}{2m_L}
+\frac{\hbar^2q_{\perp,i}^2}{2m_S}
+t_c\cos(\pi q_z/k_{\Gamma A})+\epsilon_c^0.
\]

取论文参数：

- `epsilon_v^0 = -0.03 eV`, `m_v = -0.23 m_e`, `t_v = 0.06 eV`
- `epsilon_c^0 = -0.01 eV`, `m_L = 5.5 m_e`, `m_S = 2.2 m_e`, `t_c = 0.03 eV`
- 激子序参量 `Delta = 0`，因此四条裸带互不杂化。

默认切线为折叠低能描述中的 `q || ΓM`、`q_z=0`。`L_1` 椭圆长轴平行该切线；另两个谷相对旋转 `±120°`，故在此切线上 `c_2=c_3`。

> 这是论文在各带极值附近采用的有效质量模型，不是适合跨越整个布里渊区的 DFT/Wannier 全能带；因此图只画局域低能窗口。

## 运行

### 局域折叠能带

```bash
python plot_noninteracting_bands.py
```

输出：

- `tise2_noninteracting_bands.png`
- `tise2_noninteracting_bands.svg`
- `tise2_noninteracting_bands.csv`

### 二维投影 K–Γ–M–K 路径

```bash
python plot_k_gamma_m_k.py
```

输出：

- `tise2_K_Gamma_M_K_noninteracting.png`
- `tise2_K_Gamma_M_K_noninteracting.svg`
- `tise2_K_Gamma_M_K_noninteracting.csv`

这里把体材料的 `L₁` 口袋投影到基面 `M` 点：`E_v` 在 Γ 附近展开，`E_c`（`c₁`）在 `M` 附近展开。实线表示局域有效模型窗口，淡色点线是为了连接完整高对称路径而显示的形式外推，不能当作论文给出的全布里渊区预测。路径坐标换算默认采用可修改的 `a = 3.54 Å`。

### M–Γ–−M 等价性检查

```bash
python plot_m_gamma_minus_m.py
```

输出 `tise2_M_Gamma_minusM_equivalence.{png,svg,csv}`。路径在两个 M 点外各延伸 30%，并分别使用以 `M₊`、`M₋` 为中心的倒格矢等价局域坐标。由于 `M₋ = M₊ - (b₁+b₂)`，两端导带能量严格相同；若错误地把单个 `M₊` 局域抛物线外推到 `M₋`，会得到没有物理意义的不等价结果。

### 三维高对称点与有限 Q 选择

```bash
python plot_3d_bz_finite_q.py
```

输出：

- `tise2_3d_high_symmetry_finite_Q.png`
- `tise2_3d_high_symmetry_finite_Q.svg`
- `tise2_3d_finite_Q_vectors.csv`

该图同时给出三维六方 BZ、基面投影、侧视图和论文中的相关带边。体材料的三个序矢是 `Q_i = ΓL_i`；`ΓM_i` 只是其面内投影。图中采用一个 C₃ 对称的倒格分数坐标代表集：`Q₁=(1/2,0,1/2)`、`Q₂=(0,1/2,1/2)`、`Q₃=(-1/2,-1/2,1/2)`。每个 `Q_i` 都满足 `-Q_i ≡ Q_i (mod G)`，三者生成完整的八扇区 `2×2×2` 倒格商。

## 当前完整四扇区 HF：`4 × 4 → 16 × 16`

当前目标实现位于 `folded_hf.py`。内部空间保留完整的四维

\[
H_0(\mathbf k)\in\mathbb C^{4\times4},
\]

并在四个固定动量代表元

\[
Q_0=0,\quad Q_1,\quad Q_2,\quad Q_3
\]

上构造扇区优先的 16 维基底 \(|s,\alpha;\mathbf p\rangle\)，其中
`s=0,...,3`、`alpha=0,...,3`。裸 Hamiltonian 是

\[
H_{0,\mathrm{fold}}(\mathbf p)
=\bigoplus_{s=0}^3 H_0(\mathbf p+\mathbf Q_s).
\]

相互作用采用内部空间单位 form factor，并对完整 `16 × 16` 密度矩阵收缩；代码没有价带–导带、导带–导带、对角块或 bright/dark block mask。固定代表元模型只保留严格满足

\[
Q_s+Q_u-Q_t-Q_v=0
\]

的 28 条 `G=0` 扇区路线。Umklapp 不能通过“模倒格矢相等”或平均两个不等价 Coulomb 转移来猜测；若要加入，必须另行提供 `G` 分辨的 form factor 与 cutoff。

当前参考约定把输入 `H0` 视为裸/无相互作用 Hamiltonian：

\[
H_{\rm HF}[P]=H_0+\Sigma_H[P-P_{\rm ref}]+\Sigma_F[P].
\]

为了复用通用 ODA/SCF 框架，实际写成

\[
h_{0,\rm eff}=H_0+\Sigma_F[P_{\rm ref}],\qquad
L[D]=\Sigma_H[D]+\Sigma_F[D],\quad D=P-P_{\rm ref}.
\]

`q=0` Hartree 模由中性背景移除；有限 `q` Hartree 和完整 Fock 保留。占据使用一个全局有限温化学势，目标为每个 reduced momentum 平均 4 个电子，而不是强制每个 k 点恰好占据 4 条带。所有 16 个本征值、本征态和占据均保存。k 点权重一致进入粒子数、Hartree/Fock、能量和熵。

```python
from mean_field.systems.tise2 import (
    FoldedHFConfig,
    build_folded_hf_state,
    run_folded_hf,
)

config = FoldedHFConfig(
    epsilon_r=8.0,
    temperature_K=65.0,
    inplane_shells=4,
    inplane_spacing_Ainv=0.04,
    z_shells=3,
    z_spacing_Ainv=0.05,
)
state = build_folded_hf_state(config)
result = run_folded_hf(state, init_mode="random", seed=0)
```

这仍然不是材料级全 BZ 模型：Monney 有效质量色散被外推到四个扇区，常数介电、动量域、有限温度和 `G=0` 路线政策都是显式外部输入。生产 SCF、范围收敛和能带图必须通过 Slurm；只有保存的精确 SCF 点可以进入默认 HF 图。

当前源码的聚焦验证由 Slurm job `511955` 完成（`34 passed`）。固定范围 `N=4→6` 的 normal-branch 诊断由 job `511951` 完成，但共同精确路径点上的最大 HF 谱漂移仍为 `0.219861484 eV`，且 `N=6` 边界完整自能最大值仍为 `2.254450372 eV`，所以网格/范围门控未通过，禁止在范围外接裸带。job `511954` 仅用 13 个精确保存的 N=6 路径点绘制全部 16 条裸/HF 谱；它是 normal-branch 实现诊断，不是收敛的激子 TiSe₂ 能带。

## 局域 unrestricted 完整 `4 × 4` HF

`local_full_hf.py` 保留 Monney 共同局域动量基底

\[
\Psi_{\mathbf p}=(v_{\Gamma+\mathbf p},c_{L_1+\mathbf p},c_{L_2+\mathbf p},c_{L_3+\mathbf p}),
\]

但不同于历史 `hf.py`，它对完整 `4 × 4` 密度矩阵作用 Hartree 与 Fock，未预设任何外部矩阵块为零。单位 form factor 下仍只保留 `physical_momentum_routes()` 定义的 28 条精确固定代表元 `G=0` 路线；这是动量守恒输入，而不是价带/导带 block mask。

Monney 参数是拟合后的正常态局域色散，因此该适配器采用完整参考减除

\[
D=P-P_{\rm ref},\qquad
H_{\rm HF}[D]=H_0+\Sigma_H[D]+\Sigma_F[D].
\]

这样 `D=0` 精确恢复输入 `H0`，避免再次加入完整正常态交换。`q=0` Hartree 由中性背景移除，有限动量 Hartree 保留；一个全局有限温化学势固定每个局域动量平均一个电子。所有四个本征值、本征态和有限温占据均保存。`symmetric_exciton` 和 `random_full` 只是初始场；SCF 迭代不施加对称投影或矩阵块约束。

```python
from mean_field.systems.tise2 import (
    LocalFullHFConfig,
    build_local_full_hf_state,
    run_local_full_hf,
)

config = LocalFullHFConfig(
    epsilon_r=10.0,
    temperature_K=65.0,
    inplane_shells=6,
    inplane_spacing_Ainv=0.16 / 6.0,
    z_shells=5,
    z_spacing_Ainv=0.15 / 5.0,
    mixing_policy="fixed",
    fixed_mixing=0.20,
)
state = build_local_full_hf_state(config)
result = run_local_full_hf(state, init_mode="symmetric_exciton", seed=0)
```

当前聚焦测试由 Slurm job `512114` 完成（`67 passed`）。`epsilon_r=10`、`T=65 K`、`nk=1397` 的 source-bound fixed-mixing job `512115` 中，`symmetric_exciton` 分支经 313 次迭代通过 `1e-8` raw fixed-point 门控（`final_raw_norm=8.93e-9`）；其完整 `4 × 4` 密度、自能和 Hamiltonian 的所有矩阵位置都由收缩产生非零值。相同作业中的 `random_full_seed0` 在 1600 次后 residual 仍为 `2.86e-5`，因此被拒绝。这个结果只授权一个已收敛的分支，不证明其为全局最低自由能，也不证明 cutoff/mesh 收敛。

最初的 render job `512149` 错误地把同一组 common-`p` 本征值在 `x=p_x` 与 `x=Q_{1x}+p_x` 下各画一次；这种横坐标平移不能冒充物理 Γ 与 `L₁/M₁` 点分辨的能带，相关输出已经撤销并保留为 `*_coordinate_embedding_REVOKED_diagnostic.*`。

修正后的 Slurm render job `512163` 只读取该分支保存的 13 个精确 `(m,0,0)` SCF 网格点。Γ 面板的裸带只使用 `H0[0,0]=E_v(Γ+p)`，`L₁/M₁` 面板的裸带只使用 `H0[1,1]=E_{c1}(L₁+p)`，二者不是复制关系。HF 的四个本征值是折叠问题共有的极点；物理点分辨信息分别由归档本征态的原始 `|U_{0n}|²` 与 `|U_{1n}|²` 给出，并仅通过线性 marker opacity 表示。没有权重阈值、展宽、平滑、off-grid 重构或额外能量对齐。该图是 sector-resolved spectral-pole 图；当前局域模型并未提供 Γ 和 M 处所有远程材料能带。

稳定输出：

- `results/tise2/TiSe2_local_full4_HF_Gamma_L.png`
- `results/tise2/TiSe2_local_full4_HF_Gamma_L.pdf`

这仍是常数介电、单位 form factor、局域 cutoff 下的模型结果，不是材料级 TiSe₂ 全 BZ 能带或无参数论文复现。

## 历史受限四口袋 canary（非当前目标）

`monney.py` 和 `hf.py` 实现论文采用的受限低能截断

\[
\{0,Q_1,Q_2,Q_3\}\equiv
\{v_\Gamma,c_{L_1},c_{L_2},c_{L_3}\}.
\]

它是四个低能口袋的截断，而不是完整八扇区倒格商。这个历史适配器只保留三个 `v-L_i` 异常 Fock 块，因此不能验证上面的完整 `16 × 16` HF。既有 canary、screening scan 和相关图只用于受限模型诊断，不是当前模型结果。平均场 Hamiltonian 采用

\[
H(\mathbf p)=
\begin{pmatrix}
\epsilon_v&-\Delta_1&-\Delta_2&-\Delta_3\\
-\Delta_1^*&\epsilon_c^1&0&0\\
-\Delta_2^*&0&\epsilon_c^2&0\\
-\Delta_3^*&0&0&\epsilon_c^3
\end{pmatrix},
\qquad
\Delta_i(\mathbf p)=\int\frac{d^3p'}{(2\pi)^3}
\frac{4\pi e^2}{\epsilon|\mathbf p-\mathbf p'|^2}
\langle b_i^\dagger a\rangle_{\mathbf p'}.
\]

体系层只实现裸 Hamiltonian、C₃ 闭合局域网格、库仑卷积、占据/参考密度和能量泛函；SCF 循环复用 `mean_field.core.hf.problem.HartreeFockProblem`。

### 必须明确的论文外输入

论文只写出 `V_c(q)=4πe²/[ε(q)q²]` 和形式上的序参量定义；实际谱图使用从实验估计的常数 `Delta`，没有数值求解自洽方程。以下量在论文中没有闭合给出，因此本实现全部要求显式记录：

- `epsilon_r` 或更完整的 `epsilon(q)`；
- 三维积分域/低能截止与网格；
- `q=0` 奇异胞元的处理；
- 温度以及固定粒子数或固定化学势系综；
- 自旋简并、Bloch form factor 和远带修正。

当前数值 canary 使用每个自旋副本平均每个局域动量一个电子、有限温固定粒子数、C₃ 闭合六角柱截止，并用等体积球解析积分 `q=0` 胞元；它不是无参数论文复现。

论文原文的导带 `k_z` 项按 `+t_c cos(...)`、`t_c>0` 实现为默认 `conduction_z_convention="paper_literal"`。`"l_minimum_diagnostic"` 只用于符号敏感性诊断，不能冒充论文公式。

### Python API

```python
from mean_field.systems.tise2 import (
    MonneyHFConfig,
    build_monney_hf_state,
    linearized_monney_gap,
    run_monney_hf,
)

config = MonneyHFConfig(
    epsilon_r=8.0,
    temperature_K=65.0,
    inplane_shells=4,
    inplane_spacing_Ainv=0.04,
    z_shells=3,
    z_spacing_Ainv=0.05,
)
state = build_monney_hf_state(config)
linear = linearized_monney_gap(state)
result = run_monney_hf(state, init_mode="symmetric")
```

真实 SCF 和网格收敛测试必须通过 Slurm；上述代码块只说明 API，不应在 login 节点直接执行。
