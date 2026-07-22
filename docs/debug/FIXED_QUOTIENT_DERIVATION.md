# RLG/hBN fixed quotient：q=0 HF-response/Hessian derivation

**日期**：2026-07-16
**状态**：q=0 derivation contract；generic finite-q all-copy normalization 尚未推导
**公式来源**：arXiv:2312.11617 Appendix D1/D12/D18/D19；本地 `d18_d19_formula_mapping_20260626.md`；corrected HF implementation `_hf_c3_quotient.py`

## 1. 目标

建立一个不依赖 canonical q anchor 的 q=0 response oracle：

\[
\mathcal K_0[\delta D]
=\left.\frac{d\Sigma[D_0+t\delta D]}{dt}\right|_{t=0},
\]

其中 `Sigma` 必须是 actual HF run 使用的

```text
actual_node_ws_fixed_variational_copy_v2
```

同一个 interaction builder。该 oracle先验证 density orientation、ordinary D18/D19 mapping、fixed-copy derivative和 finite-difference identity；它不宣称已经解决非零 q copy-pair lift。

## 2. 代码中的 stored density convention

`build_rlg_hbn_density_from_hamiltonian()` 保存：

```python
density = occupied_vecs.conjugate() @ occupied_vecs.T - reference_density
```

若单粒子本征矢列为 `u_n[a]`，则

\[
\Delta D_{ab}=\langle c_a^\dagger c_b\rangle-R_{ab}
=\sum_{n\in occ}u_{n,a}^*u_{n,b}-R_{ab}.
\]

因此这里的 `ΔD` 是索引顺序 `(creation basis index, annihilation basis index)`；相对于常写作 `P_{ab}=u_a u_b^*` 的 projector，它是 transpose/conjugate representation。response tangent 必须遵循这个 stored convention，不能直接放 `|p><h|` 的普通矩阵表示。

reference density 与 response derivative 无关，因为它是常数；`delta D` 直接进入 linear interaction builder。

## 3. X/ph tangent

论文 D19 的 X operator：

\[
d_{p,k}^\dagger d_{h,k}\qquad(q=0).
\]

对 stored density `ΔD_ab=<c_a† c_b>-R_ab`，与该 particle-hole ket variation对应的 tangent 是

\[
(\delta D^{X_j})_{ab}
=u^*_{h_j,a}(k_j)u_{p_j,b}(k_j).
\]

代码表示：

```python
blocks[:, :, k_j] = np.outer(np.conj(u_h_j), u_p_j)
```

也就是 orbital basis 中唯一非零元素

```text
D[h_j, p_j] = 1.
```

令 `K_Xj = K[delta D_Xj]`，则 interaction contribution

\[
A^{\rm int}_{ij}
=\langle p_i,k_i|K_Xj(k_i)|h_i,k_i\rangle.
\]

把 `D[h_j,p_j]=1` 代入 HF derivative，得到

\[
A^{\rm int}_{ij}
=\mathcal V[p_i,h_j;h_i,p_j]
-\mathcal V[p_i,h_j;p_j,h_i],
\]

正是 Appendix D12/D18 在 q=0 的 A-direct + A-exchange。

## 4. Y/hp tangent

D19 的 Y partner在 q=0 是反向 coherence。stored tangent：

\[
(\delta D^{Y_j})_{ab}
=u^*_{p_j,a}(k_j)u_{h_j,b}(k_j),
\]

即

```python
blocks[:, :, k_j] = np.outer(np.conj(u_p_j), u_h_j)
```

或 orbital basis中的

```text
D[p_j, h_j] = 1.
```

投影给出

\[
B_{ij}
=\langle p_i,k_i|\mathcal K_0[\delta D^{Y_j}](k_i)|h_i,k_i\rangle
\]

\[
=\mathcal V[p_i,p_j;h_i,h_j]
-\mathcal V[p_i,p_j;h_j,h_i],
\]

即 D18/D19 的 B-direct + B-exchange。

## 5. Corrected HF quotient derivative

`build_rlg_hbn_hf_c3_quotient_interaction_components(D, context)` 对 `D` 严格线性：

1. ordinary source density删除 fixed stored nodes；
2. ordinary Fock使用 actual-node WS `G-W` 和 boundary-tie weights；
3. 每个 fixed source stored block先 lift：

   \[
   \Delta D_f\mapsto\left\{\frac13 S_r^*\Delta D_fS_r^T\right\}_{r=0,1,2};
   \]

4. lifted source再进入相同 layer-resolved Hartree/Fock contractions；
5. ordinary target直接保留 physical response；
6. fixed target的三个 physical-copy responses通过 source injection在 stored HF energy pairing下的 adjoint map descend回一个 active block。

所以对任意 q=0 tangent `X`：

\[
\Sigma[D_0+tX]-\Sigma[D_0]=t\Sigma[X]
\]

在浮点误差内严格成立，且

\[
\mathcal K_0[X]=\Sigma[X].
\]

特别地，若 tangent位于 fixed stored k，source injection与target descent必须同时出现：

\[
\mathcal K_0[X_f]=L^\sharp\mathcal V L\,X_f.
\]

对 ordinary target，保留 `(V L X_f)_t`；对 fixed target，结果为

\[
\frac13\sum_rS_r^\dagger(\mathcal V L X_f)_rS_r.
\]

这里没有 `fixed_copy`，也没有 pair-energy sewing。方向来自 ket sewing `u_r=S_r u_0`：stored rank-one density是 `conj(u) outer u`，所以 copy后为 `conj(S_r) ΔD S_r^T`。旧 v1 使用 `S_r ΔD S_r†`，只对实对角/commuting density可能偶然一致；其 archive不能与 v2 response混用。

### 5.1 为什么还需要 fixed-target adjoint descent

HF energy使用 stored-projector pairing。为明确权重放置，定义 base mesh每个 k 的 copy集合 `R_k`：ordinary node只有 `r=0`，权重 `w_{k0}=1`；fixed node有三 copies，`w_{kr}=1/3`。令 `S_{k0}=I`（ordinary）并定义 expanded stored density

\[
( L\Delta D)_{kr}=w_{kr}S_{kr}^*\Delta D_kS_{kr}^T.
\]

q=0 expanded functional明确为

\[
E_{\rm int}[\Delta D]
=\frac{\beta v_0}{2N_k}
\sum_{kr,k'r'}
( L\Delta D)_{kr}\,\mathcal V_{kr,k'r'}\,
( L\Delta D)_{k'r'},
\]

其中矩阵指标按 `sum_ab H_ab D_ab` 的 bilinear stored-projector pairing收缩，copy权重只放在 `L` 中，`N_k`始终是base mesh点数。于是 ordinary-fixed、fixed-ordinary、fixed-fixed分别带 `1/3`、`1/3`、`1/9`；这些权重来自同一个 expanded energy，而不是assembled matrix平均。

定义 fixed source injection

\[
L_r(\Delta D)=\frac13S_r^*\Delta D S_r^T.
\]

若 physical-copy response为 `H_r`，则

\[
\sum_{ab}(H_r)_{ab}(L_r\delta D)_{ab}
=\sum_{ij}\left[\frac13S_r^\dagger H_rS_r\right]_{ij}\delta D_{ij}.
\]

因此 pairing-adjoint descent是

\[
L^\sharp(\{H_r\})=\frac13\sum_rS_r^\dagger H_rS_r.
\]

在上述 trace-preserving q=0 copy quadrature定义下，由 chain rule得到的 active kernel是

\[
\boxed{\mathcal K_{\rm quotient}=L^\sharp\,\mathcal V\,L},
\]

而不是只做 source lift的 `V L`。fixed-fixed contributions自然包含 source与target各自的 `1/3`，但这是由 chain rule/quotient pairing推导出的 `1/9` copy-pair sum，不是 assembled A/B的 naive averaging。

### 5.2 source-only builder 的失败证据

把 stored-density方向修为 `S* ΔD S^T` 后，tiny3 q=0 gate得到：

```text
analytic K vs same-builder finite difference  <=1.99e-14 meV
ordinary A/B vs D18/D19                       <=3.59e-15 meV
beta=0                                        exact
full A Hermiticity defect                     4.278997 meV
full B symmetry defect                        0.113179 meV
```

这证明 source-only map虽然是线性的，却不是 active quotient coordinates中的自伴 Hessian。修复必须在 microscopic target copies上计算 response并应用 `L^sharp`；禁止以 `(K+K^sharp)/2` 或 assembled matrix symmetrization代替。

## 6. q=0 Hessian gate

对 reduced真实 projected basis：

1. build corrected HF quotient context一次；
2. 对每个 q=0 intraflavor pair `j` 构造 `delta D_Xj` 与 `delta D_Yj`；
3. analytic response：

   ```text
   K_analytic = Sigma[delta D]
   ```

4. finite difference：

   ```text
   K_fd = (Sigma[D0 + eps*delta D] - Sigma[D0]) / eps
   ```

5. 投影到每个 row pair `i` 生成 `A_int/B`；
6. 比较：

   ```text
   analytic K vs finite difference           <=1e-10 meV
   ordinary/nonfixed A/B vs D18 path         <=1e-10 meV
   production K vs explicit expanded L# V L <=1e-10 meV
   active energy vs explicit expanded energy <=1e-10 meV
   B(K[D1],D2)-B(K[D2],D1)                  <=1e-10 meV
   beta=0 response                           exactly zero
   response A Hermiticity                    <=1e-10 meV
   response B(q=0) symmetry                  <=1e-10 meV
   ```

旧 q=0 D18 fixed block不是 fixed oracle，因为它没有从 corrected HF fixed-source expansion取 derivative。同一builder的 analytic-versus-finite-difference只验证线性，不足以验证fixed physics；fixed gate必须另行显式构造expanded densities、physical target-copy responses、`L^sharp` descent与expanded energy，并按 ordinary/fixed source-target sectors比较。

### 6.1 HF source provenance与closure

response不得把任意 converged archive重新标记为current v2。新 HF archive必须持久化：

```text
hf_interaction_convention
hf_quotient_enabled
hf_beta
hf_physical_shifts
zero_literal_q0_fock
hf_basis_periodic_gauge(+padding)
hf_form_factor_convention
cache_key_basis / cache_key_overlap
```

`apply_rlg_hbn_hf_quotient_response` 默认对缺失或不匹配字段 fail closed；只有 reduced algebra test可显式 `require_provenance=False`。对于production source还必须一次性重建

\[
H_{\rm rebuilt}=h_0+\Sigma_{v2}[\Delta D_{\rm saved}]
\]

并检查 saved-H closure与 conventional projector `P=(ΔD+R)^T` 的 commutator `[H,P]`。该heavy preflight每个archive做一次，不能在每个tangent column内重复。

## 7. 这一步能证明什么

若 gate通过，它证明：

- stored-density ph/hp orientation正确；
- corrected HF quotient builder可作为 q=0 linear response source of truth；
- fixed q=0 tangent使用与 HF 相同的三-copy lift；
- ordinary D18/D19 contraction与 HF derivative一致；
- finite-q重写可以从同一 `K` 接口开始。

它不证明：

- 非零 q target/source copy-pair选择；
- finite-q quotient inner-product normalization；
- physical cutoff应采用 `|G|` 还是 `|q_WS+G|`；
- generic q anchor independence；
- actual-κ Goldstone/unstable-mode结果。

## 8. generic finite-q 尚缺的推导

非零 q tangent连接不同 k fibers：

\[
\delta D_q:\mathcal H_{k+q}\leftarrow\mathcal H_k.
\]

fixed endpoint的三-copy injections不能独立做 naive `1/3 x 1/3`。代码的 stored off-diagonal tangent以 creation/source fiber为行、annihilation/target fiber为列；若 ket injections为 `J_s,J_t`，候选双边 lift必须采用

\[
(\delta D_q)_{r_s r_t}
= w_{r_s r_t}\,J_{r_s}^*\delta D_qJ_{r_t}^T,
\]

而不是 conventional-projector 的 `J_t D J_s†`。必须枚举 `(r_s,r_t,G)` 后只保留 exact physical momentum conservation成立的项。

实现前仍需解析解决：

1. 对每个 endpoint copy明确 resolved k、canonical `k-G_rep`、torus wrap与valley-signed raw reciprocal label；
2. 从
   \[
   (k_t-G_{r_t})-(k_s-G_{r_s})+G_{\rm canonical}
   =q_{\rm physical}+G_0
   \]
   推出每个copy pair对应的physical/canonical transfer，不得只比较stored mesh indices；
3. 证明 q=0、同一fixed fiber时严格退化为本文件的三个 diagonal copy blocks `S_r* D S_r^T/3`，不能产生 cross-copy blocks；
4. 对非零q确定允许pair是一个共享group orbit、多个orbit，还是全部9项中的momentum-selected子集；不得预设same-copy-only；
5. 由 off-diagonal tangent与dual response的 quotient bilinear pairing推导 `w_{r_s r_t}`，并证明 q→0时回到source/target各自的 `1/3` chain-rule normalization；
6. 分开定义物理shell cutoff `|G_0|<=Λ` 与 canonical overlap key cutoff `|q_WS+G|<=Λ`，验证C3 closure和cutoff convergence。

### 8.1 Actual 12×12 pre-weight enumeration

已对 q=`(0,0),(1,0),(3,7),(6,0),(4,8),(8,4)` 双边枚举全部 endpoint representatives。对 total reciprocal shifts `R_s,R_t`，每个候选都满足

\[
(k_t-R_t)-(k_s-R_s)+(R_t-R_s)=q_{\rm physical}
\]

到 `2.22e-16`。因此 exact physical momentum是必要gate，但**不会单独筛掉任何 gauge-copy pair**：mixed fixed/ordinary edge有3个momentum-valid candidates，fixed/fixed edge有9个。q=0 fixed/fixed也有9个momentum-valid candidates，而已验证HF reduction只允许3个diagonal copy blocks。

`R_s=R_t` 也不能作为普遍规则：许多完全ordinary edges的唯一合法 periodic-gauge representatives本身就有 `R_s!=R_t`；self-sector q=`(4,8)/(8,4)` 的fixed/fixed 9 candidates中只有2个同shift。故same-shift/same-copy selection会错误丢失合法ordinary或fixed结构。

证据：

```text
results/.../FINITE_Q_COPY_PAIR_PROVENANCE_PREWEIGHT_20260716.json
logs/rlg_finite_q_copy_pair_preweight_20260716.log
```

### 8.2 q=0 off-diagonal candidate discrimination

用sewn active coefficients在expanded microscopic endpoints上逐项计算A/B，并与verified variational HF Hessian比较：

```text
Diag3: C_rs=delta_rs/3
  A residual 1.38e-15 meV
  B residual 1.21e-16 meV
All9: C_rs=1/9
  A residual 0.861217 meV
  B residual 0.181329 meV
```

两者都保持A Hermiticity/B symmetry，但只有Diag3是HF Hessian。因此结构identity本身也不能选择connection；naive all9/product weighting被q=0 oracle明确否定。证据：

```text
results/.../Q0_OFFDIAGONAL_COPY_WEIGHT_CANDIDATES_tiny3_20260716.json
logs/rlg_q0_offdiagonal_copy_weights_tiny3_20260716.log
```

### 8.3 Diagonal-HF extension no-go

把每个 endpoint multiplicity记为 `m_k in {1,3}`。最一般的copy-factorized off-diagonal stored-density lift可写成

\[
[\mathcal L_q^C D]_{ru}
=\frac{C_{ru}^{st}(q)}{\sqrt{m_sm_t}}
S_{sr}^*D_{st}S_{tu}^T.
\]

q=0 fixed-fiber oracle只固定

\[
C_{kk}(0)=I_3,
\]

并不能固定 `s!=t` 的mixed `1x3/3x1` 或fixed-fixed `3x3` matrices。原因可直接由copy-frame phase看出：`S_kr -> exp(i theta_kr) S_kr` 在q=0 diagonal block中相消，但finite-q block保留endpoint relative phase。因此任何只使用diagonal HF data的构造都缺少一个environment/parallel-transport datum。

stored bilinear pairing下，finite-q Hessian必须成对使用q和-q lifts：

\[
\mathcal K_q=(\mathcal L_{-q})^\sharp\mathcal V_q\mathcal L_q,
\qquad
C_{ts}(-q)=C_{st}(q)^\dagger
\]

（包含wrap-induced copy reindexing）。这才是microscopic `B(q)=B(-q)^T` 的来源。C3 covariance只把一个seed connection沿orbit transport；它不选择seed。即使要求partial isometry、C3 intertwining和three-edge closure，fixed-fixed仍允许多个circulant/character channels；mixed `1<->3` connection最多rank1，无法可逆地组成完整 `I3`。

所以current variational-v2 HF是translation-invariant/k-diagonal manifold上的可信functional，但它本身不足以唯一决定finite-q extension。必须额外给出以下之一：

1. 所有k共享的microscopic Stinespring/copy-environment frame；
2. 明确的off-diagonal density vertex `M_l^{ru}(k,q,G)` 及其weighted projector；
3. equal-multiplicity expanded Hilbert space并在其中重新定义/求解HF；
4. 避开fixed nodes的mesh sequence并证明外推收敛到同一response。

禁止用copy scan、energy matching、uniform phase、WS cutoff或paper spectrum选择 `C_q`。

### 8.4 Punctured ordinary-orbit extension candidate

一个非经验的额外axiom是把fixed node定义为普通C3三点轨道的punctured limit：取共同infinitesimal displacement `d,C3 d,C3^2 d`，同时移动每个density endpoint后再令 `d->0`。单个off-diagonal tangent因此产生三个endpoint-paired Kraus branches，各weight `1/3`，而不是在同一tangent内部生成all9 cross-copy blocks：

- q=0 fixed/fixed严格得到 `(r,r)` 三个diagonal blocks；
- mixed edge得到三个fixed-copy branches；
- q=`(4,8)/(8,4)` fixed/fixed得到由共同branch确定的三个pairs；在当前copy orbit labeling中为 `(1,1),(2,2),(0,0)`，尽管total reciprocal shifts并不相同。

必须区分tangent lift与Hessian pairing。若 `L_q(D)` 含3个branch blocks、每个weight `1/3`，则

\[
K_q=(L_{-q})^\sharp V_q L_q
\]

在target-dual branch `a` 与source-tangent branch `b` 上给出独立double sum `sum_{a,b}/9`。不能把environment误设为interaction-conserved而只算locked `sum_b/3`。q=0 microscopic gate明确区分：

```text
independent double-branch Diag3: A/B residual 1.38e-15 / 1.21e-16 meV
locked same-branch average:      A/B residual 0.975555 / 0.095368 meV
```

所以接受的候选结构是“每个tangent内部3个branch-paired blocks，Hessian两侧独立求和”，既不是coherent all9 tangent，也不是locked same-branch interaction。

在refined meshes `N=24,36,60,120` 上，所需q sectors的branch signatures完全稳定；q/-q swapped signatures无失败，momentum residual `<=5.56e-17`。证据：

```text
results/.../FINITE_Q_PUNCTURED_ORBIT_CONNECTION_PROVENANCE_20260716.json
logs/rlg_finite_q_punctured_orbit_connection_20260716.log
```

但该diagnostic也暴露一个新边界：self sectors与相邻orbits中，Gamma `(0,0)`、`(11,0)`、`(1,0)`等ordinary endpoints的directional limiting reciprocal reps会branch-split。current v2 diagonal HF只给这些ordinary nodes一个exact representative；若punctured extension需要额外ordinary environment branches，就必须证明它们的diagonal channel等于current HF，或重新定义并求解expanded functional。因而punctured rule当前是一个有明确continuum含义、通过geometry gates的candidate Stinespring dilation，不是已接受production formula。

### 8.5 Microscopic vertex-rank verdict

Tiny3直接在sewn layer×physical-G density-vertex空间检查所有momentum-valid candidates：

```text
q=0 fixed/fixed Diag3: 3 supported, rank 1
q=0 fixed/fixed All9: 9 supported, rank 7
q=(1,0) mixed edges: 3 supported, rank 3（normalized singular values = 1,1,1）
self q=(1,2)/(2,1): 9 candidates, 8 supported, rank 8；一个candidate严格zero support
```

因此三个mixed branches不是同一vertex的phase/gauge copies，而是独立microscopic channels；momentum-valid也不保证operator support。这个rank结果否定把三branch压成单一phase/unitary vertex，但**不否定**由punctured-orbit axiom定义的三-Kraus average。该average是对current diagonal data的额外off-diagonal extension，而非自动推论。

Tiny shell-1 Gamma branch test给出representatives `(0,0)` 与 `(1,1)` active principal singular values全0、H0 spectrum差`152.729 meV`，说明tiny vertex-rank结论不能无条件外推到production cutoff。独立五层actual-parameter shell convergence得到：

```text
shell 2: min singular 0.001870, H0 delta 13.9877 meV
shell 3: min singular 0.046986, H0 delta 13.7379 meV
shell 4: min singular 0.999850, H0 delta 0.020721 meV
shell 5: min singular 0.999975, H0 delta 0.011788 meV
```

因此branch subspaces随continuum shell趋于一致，但production shell-4仍不是roundoff-equivalent；punctured extension在实际模型中是一个受cutoff控制但会改变boundary diagonal functional约`1e-2 meV`的选择。Tiny3证明了代数非唯一性和zero-support可能性，不能直接外推其rank数值。

Actual shell-4/5完整active-operator `layer x physical-G x active-matrix` gate显示mixed与fixed-fixed三branch tensors均rank3，branch相对差约`1.42--1.45`且不随shell4→5消失。这是因为branches是C3-related physical channels，并不应在固定G坐标中相等；它排除single-vertex reduction，但与三-Kraus average相容。后续C3 gate必须同时旋转physical G、endpoint states与branch labels，不能拿raw same-G tensor equality当oracle。

证据：

```text
results/.../COPY_DENSITY_VERTEX_RANK_tiny3_20260716.json
results/.../GAMMA_BRANCH_SHELL_CONVERGENCE_20260717.json
results/.../ACTUAL_BRANCH_VERTEX_SHELL_CONVERGENCE_20260717.json
logs/rlg_copy_density_vertex_rank_tiny3_20260716.log
logs/rlg_gamma_branch_shell_convergence_20260717.log
logs/rlg_actual_branch_vertex_shell_convergence_20260717.log
```

### 8.6 Generic-q three-branch Hessian diagnostic

按q=0唯一允许的规则实现diagnostic：每个tangent含3个endpoint-paired branches、weight `1/3`，`K_q=L_-q^sharp V L_q`在target/source branches上独立double-sum `/9`。必须保留signed repeated-zone orbit

```text
(1,0) -> (0,1) -> (-1,-1),   -q=(-qx,-qy)
```

而不能把第三点或-q取模成torus label。修正signed geometry后：

```text
q=(1,0):   B(q)-B(-q)^T = 2.56e-16 meV
q=(0,1):   B residual       = 1.169e-3 meV
q=(-1,-1): B residual       = 1.109e-3 meV
A Hermiticity all sectors   <=1.79e-15 meV
C3 spectrum assignment      = 0.0235--0.0420 meV
```

Term decomposition最初显示所有`A_direct/A_exchange` Hermiticity与`B_direct(q)=B_direct(-q)^T`在roundoff，残差只出现在`B_exchange`。Operator-level检查进一步发现density vertices本身满足q/-q adjoint到`8e-18`，真正不一致的是Coulomb kernel：解析零transfer在forward/reverse浮点抵消中一侧成为exact zero、另一侧成为机器精度非零，触发不同IR分支，kernel差达到`1.81e4`。

修复方式不是转置B，而是在expanded-node kernel入口从fractional node coordinates与integer shift构造transfer，并把容差内的解析零fractional components canonicalize为exact zero。修复后：

```text
all signed q sectors: B(q)-B(-q)^T <=3.2e-16 meV
vertex adjoint residual <=1.1e-17
kernel adjoint residual <=8.2e-12 absolute
```

production base-cache ordinary B同样为roundoff。该修复有独立unit regression；q0/quotient suite为`8 passed`，HF/adapter/response/artifact focused suite为`68 passed`。

剩余C3问题与D19结构已分离：tiny shell-1 all-sparse A spectrum C3 residual为`0.0235--0.0420 meV`；把ordinary块换回raw production cache会与expanded fixed blocks产生functional/gauge不一致，两个edges反而达到`1.665 meV`，所以hybrid matrix不能作为最终functional。A0多重集自身在`4.55e-13 meV`协变。

### 8.7 Branch-augmented microscopic C3 representation

在basis `(pair, puncture branch)` 上直接用raw-component C3算符构造三条独立edge sewings；fixed branch由解析reciprocal-shift cocycle唯一映射，未使用energy assignment、polar或inverse-composed closing edge。Equal-weight average embedding保持`1/3`，其Hessian仍为`1/9` double sum。

五层mesh3 shell-4结果：

```text
augmented dimension/rank              51 / 51
branch mapping failures               0
average-subspace closure              <=4.92e-13
metric covariance                     <=2.95e-12
sewing unitarity                       <=9.29e-14
independent C3^3                       9.37e-14
A direct/exchange/total covariance     <=1.03e-12 / 8.08e-12 / 9.86e-12 meV
A spectrum assignment                 <=5.31e-12 meV
```

因此punctured branch average在production shell-4 active space中确实形成closed microscopic C3 representation，并自然解决旧fixed raw Wilson约`O(1)`的失败。

Y/Nambu gate必须使用`D_Y(q->Rq)=conj[D_X(-q->-Rq)]`；不共轭候选误差约`4.12 meV`，明确排除。当前mesh3 h0 occupation pair-space在negative-q orbit仍有`2.8--4.3e-9` sewing leakage，放大为shell2/4的B residual `4.1--5.1e-5 meV`与L residual `2.2--2.8e-4 meV`。padding2/3/4完全相同，故不是support truncation。shell3/5的active-window closure更差约`1e-5`并产生显著A/L失败。

这最后一个残差属于把closed active representation限制到occupied->unoccupied pair subspace时的projector leakage；不能polar修补。fresh stationary HF source `185536`现已完成并通过该gate：

```text
SCF iterations/final error              58 / 9.3733435e-7
saved-H quotient closure                 3.58036e-15 meV
[H,P] stationarity                       3.42578e-5 meV
ordinary sewing/projector/leakage        <=3.56213e-13
fixed-copy sewing/projector/leakage      <=3.11097e-15
```

因此archive `a_v64_variational_hf_v2_185536/.../hf_run_state.npz`是accepted `kappa_hBN=1` variational-v2 HF source。

在该actual 12x12 source上，endpoint-paired branch lift对三条独立generic-q anchors均通过全部fixed-touched entries：每个q sector有432 pairs，plus/minus各12个fixed-touched pairs、branch candidate counts为`1/3`；每条edge检查5118个A upper-triangle entries和10224个B entries。三条edge的最大结果：

```text
A direct Hermiticity                    <=7.64e-17 meV
A exchange Hermiticity                  <=2.22e-16 meV
B direct(q)-B direct(-q)^T              <=8.28e-17 meV
B exchange(q)-B exchange(-q)^T          <=1.52e-16 meV
```

没有assembled symmetrization或共享结果回填；q与-q分别microscopic build。

第一版完整generic-q矩阵进一步暴露了一个与fixed branch无关的ordinary boundary bug：one-body谱C3为`2.43e-11 meV`，fixed principal blocks为`<=7.31e-13 meV`，C3-mapped ordinary no-wrap principal blocks为`<=4.09e-13 meV`，但包含ordinary wrapped legs后interaction谱误差增至`0.01--0.06 meV`、L谱误差`0.172 meV`。因此该第一版full matrices已隔离为diagnostic-only，不能用于Fig. S45。

根因gate比较了off-torus rebuilt active frame与解析periodic-gauge lift：两者subspace defect仅`3.49e-13`，但wrapped frame存在非对角active-band gauge rotation（逐态phase对齐后最大vector defect`0.678`）；旧builder却直接复用了stored HF coefficients。解析`_hf_full_vector_in_periodic_gauge`在全部192个ordinary physical legs上的C3 unitarity为`3.56e-13`。正确functional因此对ordinary wrap使用解析periodic relabel，对fixed endpoint保持三branch expanded lift；不能通过polar或assembled transport修复。

修复后的三条独立generic-q full matrices全部通过：A direct/exchange component spectra分别`<=1.38e-13/2.70e-13 meV`，A total C3 spectrum`<=1.25e-11 meV`，L spectrum`<=1.27e-11 meV`；每sector为432 positive、432 negative、0 zero/complex，q/-q particle-hole assignment`<=3.64e-12 meV`，solver residual`<=1.42e-14`，最低正模三anchor一致为`4.659433099874 meV`。

q=0完整矩阵也通过A Hermiticity/B symmetry与raw eigensolver gates。最关键的source-of-truth parity使用strict typed provenance response分别检查ordinary与fixed column的A/B direct/exchange：intraflavor、intervalley、interspin最大残差分别`6.81e-16`、`1.34e-15`、`1.27e-15 meV`；因此q0 full matrices不是从generic-q近似外推。当前accepted kappa_hBN=1 q0最低正模为intraflavor `3.814014089679 meV`、intervalley `2.72345828 meV`、interspin `2.1446636e-7 meV`，均0 complex modes。

interspin近零模通过显式microscopic spin-lowering Ward gate：mode overlap `0.9999999999993219`、eta norm `1`、B严格为0。raw generator residual `3.9401e-5 meV`与fresh source `[H,P]=3.4258e-5 meV`同阶（ratio `1.1501`），并低于预注册source stationarity tolerance `1e-3 meV`；没有把该residual改写为本征能。

signed self sectors使用punctured-refinement唯一确定的same-copy branches`(0,0),(1,1),(2,2)`、每branch权重`1/3`。两个C3-fixed torus points的六个独立repeated-zone representatives均通过：plus/minus A spectrum C3`<=6.94e-12/5.07e-12 meV`，plus/minus L spectrum`<=6.40e-12/6.85e-12 meV`，0 complex modes。没有inverse-composed closing edge。

同一functional扩展到flavor-flip channels后，intervalley与interspin的三个independent generic anchors也通过。intervalley/interspin L C3最大残差分别`2.13e-11/1.44e-11 meV`，q/-q PH分别`<=2.50e-12/2.16e-12 meV`，0 complex；q=(1,0)最低正模分别`3.062675262638`与`0.787094547425 meV`。

已新增public typed-source API `build_rlg_hbn_tdhf_finite_q_quotient_context`与`build_rlg_hbn_tdhf_finite_q_quotient_matrices_from_pairs`。actual q=(1,0) API与independently gated diagnostic artifact的A/B/L逐元素残差全部严格为0；旧pair assembly仍对typed archive fail closed。

证据：

```text
results/.../BRANCH_AUGMENTED_C3_SHELL_CONVERGENCE_20260718.json
results/.../BRANCH_AUGMENTED_ABL_C3_shell4_mesh3_20260718.json
results/.../ACTUAL_HF_PROJECTOR_C3_GATE_v2_185536_20260719.json
results/.../ACTUAL_GENERIC_Q_BRANCH_STRUCTURE_{q1_0,q0_1,qminus1_minus1}_v2_185536_20260719.json
results/.../ACTUAL_ORDINARY_WRAPPED_PERIODIC_LIFT_GATE_v2_185536_20260719.json
results/.../ACTUAL_GENERIC_Q_BRANCH_C3_SPECTRUM_GATE_periodicordinary_v1_185536_20260719.json
results/.../ACTUAL_SELF_SECTOR_BRANCH_SELECTION_v1_185536_20260719.json
results/.../ACTUAL_SELF_SECTOR_C3_SPECTRUM_GATE_v1_185536_20260719.json
results/.../ACTUAL_Q0_FULL_MATRIX_RESPONSE_PARITY_v1_185536_20260719.json
results/.../ACTUAL_Q0_FULL_MATRIX_RESPONSE_PARITY_intervalley_v1_185536_20260719.json
results/.../ACTUAL_Q0_FULL_MATRIX_RESPONSE_PARITY_interspin_v1_185536_20260719.json
results/.../ACTUAL_Q0_INTERSPIN_GOLDSTONE_WARD_v1_185536_20260719.json
results/.../ACTUAL_GENERIC_Q_BRANCH_C3_SPECTRUM_GATE_intervalley_periodicordinary_v1_185536_20260719.json
results/.../ACTUAL_GENERIC_Q_BRANCH_C3_SPECTRUM_GATE_interspin_periodicordinary_v1_185536_20260719.json
results/.../ACTUAL_PUBLIC_FINITE_Q_QUOTIENT_API_PARITY_v1_185536_20260719.json
results/.../figs45_production_full_orbits_v1_20260719/figs45_full_orbit_merge_summary.json
results/.../figs45_production_full_orbits_v1_20260719/figs45_bottom_raster_comparison.json
logs/rlg_branch_augmented_c3_shell_convergence_20260718.log
logs/rlg_branch_augmented_ABL_c3_shell4_mesh3_20260718.log
logs/rlg_hfprojC3_189377.out
logs/rlg_wraplift_189491.out
```

证据：

```text
results/.../GENERIC_Q_PUNCTURED_DOUBLE_BRANCH_tiny3_20260717.json
logs/rlg_generic_q_punctured_double_branch_tiny3_20260717.log
```

结论：variational-v2 HF的k-diagonal Hessian、generic punctured branch normalization、fixed endpoint map、ordinary torus-periodic lift、fresh stationary source、三个separated channels的q0 response parity、generic/self C3、q/-q结构、raw stability及interspin Goldstone均已通过；public production signed-q matrix-pair API与actual gated matrices严格一致。26个C3+inversion representatives的12x12 full mesh已完成：structure residual `<=7.03e-16 meV`、q/-q PH/quartet `<=4.78e-12 meV`；intervalley/interspin 144点全部stable，intraflavor仅三个M点有`Im omega=1.18385 meV`的complex pair。

但published-raster定量比较失败：论文intraflavor有45个unstable torus sectors，而当前functional仅3个；共同stable点的intraflavor/intervalley/interspin RMSE分别为`1.691/0.730/0.775 meV`。旧untyped v2 source的flavor-flip maps更接近论文（RMSE `0.214/0.151 meV`），但它缺少typed provenance且违反accepted fixed-node/C3 contract，只能作为隔离诊断。因而当前结果必须标为validated nonreproduction，禁止以post-fit scaling、旧fixed-copy路径或mask修改冒充Fig. S45 reproduction。
