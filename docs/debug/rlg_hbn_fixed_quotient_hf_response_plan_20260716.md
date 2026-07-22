# RLG/hBN finite-q TDHF fixed quotient：anchor-independent HF-response 下一阶段方案

**来源**：GPT-5.6 Sol Pro 审核意见，经本地代码核对后整理
**日期**：2026-07-16
**基线分支**：`debug/rlg-hbn-figs45-c3-quotient-20260716`
**基线 commit**：`aa0f09b9c8c2373d67756654d81b12c515c0088e`
**前序 handoff**：`docs/debug/rlg_hbn_figs45_c3_handoff_20260716.md`
**状态**：核心 blocker 已升级；禁止 full mesh；不是 Fig. S45 reproduction

## 1. 总任务与本阶段定位

总任务仍是建立可信的 RLG/hBN actual-κ finite-q TDHF/RPA 计算链，在 centered repeated-zone 12×12 q mesh 上验证 C3、Goldstone、谱稳定性和不稳定模，最终定量复现论文 Fig. S45。

本阶段不是继续扩大 q mesh，而是解决一个更基本的问题：

> finite-q TDHF fixed-sector interaction 必须是 corrected HF C3 quotient energy functional 的同一个线性响应/Hessian；它不能由任意选择的 canonical q/fixed copy 定义。

完成本阶段前，即使单个 C3 cycle covariance 达到 roundoff，也不能认定 physical quotient 唯一。

## 2. 已接受基线与不变约束

### 2.1 历史 v1 actual-κ HF source（已降级为 diagnostic-only）

```text
Slurm job            182757
iterations            32
final error           9.512258e-5
final energy          -547.4101237315 meV
HF gap                14.5344762 meV
projector defect      2.22e-15
interaction version   actual_node_ws_fixed_source_copy_v1
```

Phase-A之后的 stored-density convention audit发现：HF保存 `ΔD[a,b]=<c_a†c_b>-R[a,b]`，而 `_fixed_copy_sewing` 返回 ket sewing `S`。因此 fixed copy必须使用 `S* ΔD S^T/3`；v1错误使用 `S ΔD S†/3`。rank-one complex-density production-function gate给出最大差 `0.26403823`，明确 FAIL。v1 archive虽然收敛且某些 diagonal/flavor covariance gates通过，但不能再作为最终 response source；修正后必须 bump convention并重算 actual-κ HF。

### 2.2 已接受但能力有限的 TDHF evidence

```text
q=(1,0) source-anchored cycle closure   5.24798e-11 meV
q=(6,0) source-anchored cycle closure   5.24773e-11 meV
ordinary/nonfixed actual-node WS path  strong evidence
recursive direct shell                 G_next=C3(G_current)-R_edge
```

这些结果证明选定 source chart 内的 covariance；它们不证明不同 source charts 给出同一个物理矩阵。

### 2.3 审核后完成的旧-provider jobs（仅作历史回归）

审核意见到达后查询 live Slurm state，原 pending jobs 已全部完成：

```text
184106   repaired q=(3,7) cycle       max closure 5.24812e-11 meV
          spectrum assignment          1.73870e-11 meV
184033_2 q=(0,0) self                 max closure 5.19643e-11 meV
184033_3 q=(4,8) self                 max closure 5.19676e-11 meV
184033_4 q=(8,4) self                 max closure 5.19684e-11 meV
```

`184106` 逐 term residual约 `1e-12 meV` 或更小，确认 multi-edge direct-shell recurrence 修复了旧 `(3,7)` second-edge failure。所有 jobs 仍使用 canonical-source transported fixed kernel和 inverse-composed closing edge。因此它们是 recursive-shell/source-chart regression，不是 anchor-independence、independent Wilson 或 HF-Hessian acceptance。

### 2.4 继续适用的禁令

- 禁止 assembled A/B post-rotation、symmetrization 或 equal averaging。
- 禁止 fixed-copy scan、phase fit、whole fixed-block polar projection作为物理修复。
- 禁止删除 fixed legs、隐藏 complex/negative modes、使用 `abs(omega)`。
- 禁止在本阶段提交 full mesh 或声称 Fig. S45 reproduction。
- ordinary exchange 继续使用 actual-node WS folding 与 exact boundary-tie averaging。
- direct shell 继续逐 edge 递推；不能从 `G0` 重启。

## 3. 新的核心失败：cycle covariance 不等于 quotient uniqueness

审核提供的 reduced faithful diagnostic 使用：

```text
mesh                 3x3
active conduction    2
q orbit              (1,0)->(0,1)->(2,2)
physical G shell     7-vector complete C3-invariant shell
anchors              (1,0), (0,1), (2,2)
```

每个 anchor 独立构造的 cycle closure 都约 `5.12e-13 meV`，但同一个 q 的矩阵依赖 anchor：

```text
A non_non          0
A fixed_non        6.060410567 meV
A non_fixed        6.060410567 meV
A fixed_fixed      0.159297076 meV
B non_non          0
B fixed_non        0.316176289 meV
B non_fixed        0.346839998 meV
B fixed_fixed      up to 0.373665325 meV
L spectrum         up to 5.030094407 meV between anchors
```

观察到 ordinary blocks exactly agree，而 fixed-involving blocks 与 spectra 不唯一。因此当前最高优先级 core hypothesis 是：

```text
current finite-q fixed interaction is a canonical-source-defined transported
kernel, not the derivative of the corrected HF quotient functional.
```

## 4. 为什么现有 cycle gate 会漏检

代码核对：

- `_tdhf_quotient_orbit.py:613-623` 只从 `q0` 建 microscopic source provider；
- `:805-820` 仍用 `internal.source.term_evaluators` 把 q0 fixed terms transport 到 q2；
- `:871-900` 定义

  ```python
  sewing_20_matrix = np.linalg.inv(sewing_12.matrix @ sewing_01.matrix)
  ```

  并把它用于第三 edge acceptance。

因此 source-anchored cycle 的 closure 是构造的一部分，不是独立 microscopic `C3^3` 检验。

结论：inverse-composed closing sewing 可以在选定 quotient gauge 后作为 frame fixing，但必须与 independent raw third-edge/Wilson diagnostic 分开保存和命名。

## 5. Independent Wilson evidence

审核提供的 tiny3 结果：

```text
ordinary ||W-I||max        2.3714374e-15
fixed    ||W-I||max        1.397056481
fixed Wilson phases        0.2425963, 0.2813463,
                           1.3795157, 1.5466774 rad
```

每条 energy-assigned sewing 都满足：

```text
A0 assignment delta        <=4.55e-13 meV
unitarity defect           <=2.67e-15
condition number           ~1
```

但 raw projected fixed sewing：

```text
unitarity defect           ~0.9170752
singular values            0.62583, 0.62123, 0.29594, 0.28797
condition number           ~2.17326
```

这说明 energy assignment 只提供 A0-compatible pair chart；它没有证明 microscopic density/form-factor quotient representation。

必须先解析 single-particle projective `C3^3` convention。若 particle/hole 公共 projective phase抵消，则 pair representation 预期为 identity；若代码有解析允许的公共相位，只能除去该全局/projective phase，不能逐态任意 phase fit。

## 6. 公式 source of truth：corrected HF functional 的 derivative

论文 Appendix D 的投影层密度：

\[
\rho_l(q+G)=\sum_{k,mn}M^l_{mn}(k,q+G)c^\dagger_{k+q,m}c_{k,n},
\]

\[
M^l_{mn}(k,q+G)=\sum_{G'\sigma}
U^*_{G+G',l\sigma,m}(k+q)U_{G',l\sigma,n}(k).
\]

interaction：

\[
H_{\rm int}=\frac{1}{2N\Omega}\sum_{q,G,ll'}
V_{ll'}(q+G)\rho_l(q+G)\rho_{l'}(-q-G).
\]

corrected HF dynamics：

\[
i\dot P=[H_{\rm HF}[P],P].
\]

在收敛态 `P0` 线性化：

\[
i\,\delta\dot P=[H_{\rm HF}[P_0],\delta P]
+[\mathcal K[\delta P],P_0],
\qquad \mathcal K=\frac{\delta\Sigma}{\delta P}.
\]

因此 TDHF interaction 应由同一个 linear map `K` 生成：

\[
A_{ij}(q)=\Delta\epsilon_i\delta_{ij}
+\langle p_i,k_i+q|\mathcal K_q[X_j]|h_i,k_i\rangle,
\]

\[
B_{ij}(q)=\langle p_i,k_i+q|\mathcal K_q[Y_j]|h_i,k_i\rangle,
\]

其中 Y 使用 D19 的 hp ordering。fixed quotient 必须作用于 target/source external-leg fiber 与 off-diagonal density tangent，发生在 form-factor contraction 之前。

## 7. 最小 causal chain

```text
HF archive / active fiber / q-sector pair
  -> X_j or Y_j off-diagonal active density tangent
  -> enumerate physical target/source representative legs
  -> enforce exact copy-pair momentum conservation
  -> lift tangent into sparse microscopic copy blocks with derived weights
  -> apply the same layer-resolved Hartree/Fock linear map as corrected HF
  -> descend/project deltaH onto <p_i|...|h_i>
  -> add A0 and assemble D19 L(q)
  -> independently test C3 covariance, anchor independence, B transpose,
     Wilson, q=0 Hessian/Ward identities and spectra
```

## 8. 三个优先 core hypotheses 与区分性检查

### H1 — canonical-source/fixed-copy choice defines the kernel

- **症状解释**：ordinary blocks一致；fixed-touched entries随 anchor 变化。
- **实现位置**：`_choose_representative_node`、`build_sparse_role_pair_states(... fixed_copy)`、`build_sparse_fixed_source`、`build_rlg_hbn_tdhf_c3_quotient_cycle`。
- **检查**：同一个 q 从三个 anchor 独立构造并逐 term 比较。
- **失败定位**：fixed tangent lift/quotient，不是 ordinary WS。

### H2 — energy-assigned pair chart 不是 microscopic C3 representation

- **症状解释**：A0 assignment 与单-cycle closure perfect，但 independent fixed Wilson 大残差。
- **实现位置**：`build_raw_pair_c3_sewing`、`build_energy_assigned_c3_sewing`、inverse-composed closing edge。
- **检查**：三条 edge 独立 microscopic sewing；ordinary/fixed Wilson 分块。
- **失败定位**：single-leg/fixed fiber quotient 或 cluster assignment，不是 A0 energies。

### H3 — finite-q TDHF 并非 corrected HF functional 的 Hessian

- **症状解释**：HF source C3-covariant，而 fixed finite-q kernel不唯一；Goldstone/Ward 尚无保证。
- **实现位置**：`_hf_c3_quotient.py` 与 `_tdhf_fixed_quotient.py` 两套独立 contraction logic。
- **检查**：q=0 analytic response 与 self-energy finite difference、ordinary D18、fixed Hessian identity。
- **失败定位**：需要 factor shared HF response primitives，而不是调 sewing phase。

## 9. 代码架构方向

### 9.1 新 system-local response module

计划新增：

```text
src/mean_field/systems/RnG_hBN/_hf_response_finite_q.py
```

拟定接口：

```python
@dataclass(frozen=True)
class RLGhBNFiniteQDensityTangent:
    q_shift: tuple[int, int]
    target_k: np.ndarray
    source_k: np.ndarray
    blocks: np.ndarray
    role: Literal["ph", "hp"]

@dataclass(frozen=True)
class RLGhBNFiniteQResponse:
    hartree: np.ndarray
    fock: np.ndarray
    total: np.ndarray
    provenance: dict[str, object]


def apply_rlg_hbn_hf_quotient_response(...): ...
```

先不实现 generic q；Phase B 只实现并证明 q=0。

### 9.2 fixed all-copy tangent lift

新 lift 必须：

1. 分别枚举 target/source 所有合法 representatives；
2. 对每个 copy pair 计算真实 physical momentum；
3. 仅接受 exact momentum-conserving pairs；
4. 权重来自 single-leg injections/sewings 的双边作用；
5. normalization 由 quotient inner product 与 q=0 HF reduction固定；
6. 在 contraction 前 lift，contraction 后 descend；
7. 输出完整 copy-pair provenance。

q=0 diagonal oracle：

```text
expanded_fixed_density(ΔD_fixed)
  = (1/3) sum_r S_r^* ΔD_fixed S_r^T
```

不能假设 independent target/source `1/3 x 1/3`、same-copy-only 或 assembled-term averaging。

### 9.3 orbit module 降级

`_tdhf_quotient_orbit.py` 最终只负责：

- 独立构造每个 q；
- pair sewing diagnostics；
- C3 covariance、anchor independence 与 raw Wilson；
- orbit cache。

它不再从一个 canonical microscopic source定义其他 q 的 fixed terms。

## 10. 必须先建立的 gates

### P0.1 Canonical-anchor independence

对 anchors `(1,0),(0,1),(2,2)` 独立构造，比较共同 q：

```text
A0, A_direct, A_exchange
B_direct, B_exchange
A, B, L, raw spectra
```

acceptance：

```text
ordinary/nonfixed       <=1e-10 meV
fixed-involving         <=1e-9 meV
spectrum assignment     <=1e-9 meV
```

当前代码应真实 FAIL，不使用 xfail 掩盖。

### P0.2 Independent Wilson

三条 edge 均独立 microscopic build：

```text
ordinary W vs expected C3^3   <=1e-10
fixed W vs expected C3^3      <=1e-9
```

必须保存 raw sewing、energy-assigned sewing、singular values、unitarity、Wilson eigenphases和允许的解析 projective phase。

### P0.3 q=0 HF Hessian identity

因为 self-energy 对 density 是线性的：

```text
analytic K[X]
vs (Sigma[P0+eps X]-Sigma[P0])/eps
vs ordinary direct D18/D19
vs fixed q0 response
```

reduced acceptance `<=1e-10 meV`。

### 10.4 Phase-A 本地重建结果

已新增 branch-visible failing gate：

```text
tests/test_rlg_hbn_tdhf_fixed_quotient_anchor.py
```

它使用同一个真实 reduced run/orbitals、完整 7-vector shell、共享 padding，分别从三个 anchor 调 production cycle API；common-q pair basis按精确 `(p_local,h_local,h_k)` key 对齐，不做 sewing/energy matching。三条 raw/energy-assigned Wilson edge对 plus/X 与 minus/Y 都独立构造，不使用 cycle 的 inverse closing edge。

最终 test001 输出：

```text
2 failed in 15.94 s                  expected Phase-A failure
anchor cycle closure max            7.38964e-13 meV
A Hermiticity                       1.83e-15 meV
B(q)-B(-q)^T                        2.48e-16 meV
ordinary term/matrix/L anchor delta 0
fixed-touched term max              6.401020562 meV
fixed-touched A/B/L max             6.060410567 meV
spectrum assignment max             5.030094407 meV
```

Independent Wilson（plus/X 与 minus/Y一致）：

```text
raw ordinary edge unitarity         <=2.22e-15
raw fixed edge unitarity defect     0.917075219
raw ordinary C3^3 residual          <=2.69e-15
raw fixed C3^3 identity residual    0.999710777
assigned ordinary C3^3 residual     <=2.69e-15
assigned fixed C3^3 residual        1.397056481
assigned fixed eigenphases          0.2425963, 0.2813463,
                                    1.3795157, 1.5466774 rad
```

证据：

```text
results/RnG_hBN/tdhf_m2_pilot/figs45_c3_audit_v2_20260627/
  FIXED_QUOTIENT_ANCHOR_GATE_tiny3_20260716.json
logs/rlg_hbn_fixed_anchor_tiny3_20260716.log
```

因此审核的数值 blocker 已由本地 current production path 独立重建。Cycle closure只能保留为弱 regression，不能继续作为 fixed quotient acceptance。

### 10.5 Stored-density与 q=0 variational Hessian follow-up

Phase-A之后增加 rank-one complex stored-density gate，发现 v1 source lift把 ket sewing `S` 错用于 conventional projector map：

```text
current v1        S ΔD S† / 3
stored convention S* ΔD S^T / 3
rank-one max mismatch 0.26403823
```

进一步的 q=0 response gate显示，仅修 source方向后：

```text
analytic vs finite difference          <=1.99e-14 meV
ordinary vs D18/D19                    <=3.59e-15 meV
full A Hermiticity defect              4.278997 meV
full B symmetry defect                 0.113179 meV
```

这定位出 current HF quotient 还缺 fixed-target chain-rule descent。依据 stored HF energy pairing，已实现 variational quotient

\[
\mathcal K=L^\sharp V L,
\quad L_r(\Delta D)=S_r^*\Delta D S_r^T/3,
\quad L^\sharp(\{H_r\})=\sum_rS_r^\dagger H_rS_r/3.
\]

新 convention：

```text
actual_node_ws_fixed_variational_copy_v2
```

reduced gates：

```text
5 q0 tests passed in 10.15 s
65 focused tests passed in 15.97 s
analytic vs finite difference          <=2.06e-14 meV
ordinary A/B vs D18/D19                <=3.59e-15 meV
production vs explicit expanded L#VL   exactly 0
active vs expanded energy              <=1.62e-16 meV
bilinear self-adjointness              <=4.48e-16 meV
A Hermiticity                          1.78e-15 meV
B symmetry                             1.39e-16 meV
beta=0                                 exact
```

response API与archive loader现在要求typed provenance：interaction convention、quotient flag、beta、physical shifts、q0-Fock policy、basis/form-factor gauge与cache keys；legacy/v1 archive默认拒绝。另提供heavy source-closure preflight检查 saved `H=h0+Sigma_v2[D]` 与 projector commutator。

证据：

```text
docs/debug/FIXED_QUOTIENT_DERIVATION.md
results/.../Q0_HF_HESSIAN_GATE_tiny3_20260716.json
results/.../Q0_HF_EXPANDED_ORACLE_tiny3_20260716.json
logs/rlg_hbn_q0_variational_oracle_provenance_20260716.log
logs/rlg_hbn_variational_hf_v2_focused_tests_20260716.log
```

Actual 12×12 initial-density gate job `185233` 已通过：ordinary C3 `3.23946e-11 meV`、Hermiticity `2.22e-15 meV`、peak RSS约 `16–22 GiB`。base single-copy fixed C3 residual仍然很大且不作为quotient gate。Job `185534` 的microscopic physical-copy covariance/descent也通过，最大 `1.30202e-12 meV`。fresh precision-`1e-6` variational-v2 SCF已单次提交为 job `185536`，当前等待资源。旧 `182757` v1 archive降级为 diagnostic-only。

### 10.6 finite-q pre-weight copy-pair enumeration

Actual 12×12 endpoint representatives已双边枚举。所有 mixed fixed/ordinary 的3 candidates、fixed/fixed的9 candidates都满足 exact physical momentum identity到 `2.22e-16`；q=0 fixed/fixed同样有9个momentum-valid candidates，但HF q=0 oracle只保留3个diagonal copy blocks。`R_s=R_t`也不是普遍筛选规则，因为许多ordinary edges的唯一gauge reps有不同total shifts。

因此 momentum gate必要但不足，不能由它直接赋nonzero weight。q=0 microscopic candidate gate进一步给出：

```text
Diag3 delta_rs/3: A/B residual 1.38e-15 / 1.21e-16 meV
All9 uniform 1/9: A/B residual 0.861217 / 0.181329 meV
```

两者都满足A Hermiticity/B symmetry，所以结构identity也不能替代connection derivation。Phase C当前被off-diagonal quotient Gram/group-twirl normalization阻塞；已知唯一边界条件是 `C_kk(0)=I3` 与weight `1/3`。

历史 `full local/copy C3 sewing` 只给出q到C3q的edge representation，不给出同一q内的off-diagonal density connection；历史pair-subspace polar/Löwdin虽然可使transported matrix unitary，但属于assembled representation repair，不能作为current variational functional的 `C_st(q)`。这些结果继续保留diagnostic-only。

Fail-closed integration：带typed `quotient_enabled=True` provenance的HF source现在禁止进入legacy `_tdhf_finite_q.py` raw pair assembly（shortcut与intraflavor两条路径都拒绝）。q=0必须走`apply_rlg_hbn_hf_quotient_response`；非零q在derived connection完成前明确未实现。这避免fresh v2 archive被旧provider静默误用。

Punctured-orbit geometry candidate已建立：共同的三条C3 infinitesimal branches在`N=24,36,60,120`上给出稳定copy connection，q=0退化为Diag3，q/-q reversal无失败，momentum residual `5.56e-17`。每个tangent内部只含3个endpoint-paired branches、weight `1/3`；但Hessian `L_-q^sharp V L_q`的target/source branches必须独立double-sum `/9`。q=0 locked same-branch `/3` 给出A/B错误`0.975555/0.095368 meV`，而double-branch结果在`1e-15`匹配。self sectors仍要求Gamma及部分ordinary gauge-boundary endpoints具有branch-dependent limiting reps；这不在current v2 diagonal functional定义内。

Tiny shell-1 microscopic rank gate否定了在reduced model中把branches自动视为同一vertex：q=0 Diag3 rank1、All9 rank7；generic mixed rank3；self-sector 9 candidates中仅8个有support且rank8。但actual五层shell sweep显示branch subspaces随cutoff快速改善：shell-4 Gamma min singular `0.999850`、H0 mismatch `0.020721 meV`，shell-5为`0.999975`与`0.011788 meV`。Actual full active-operator tensors在shell4/5仍为rank3、raw same-G branch差约`1.42--1.45`；这符合C3-related channels而非gauge-identical tensors。

Generic-q三branch/double-Hessian diagnostic进一步显示：signed repeated-zone下A direct/exchange Hermiticity与B direct均roundoff。最初的B exchange anchor residual最终定位为解析零transfer的浮点IR分支：vertices q/-q adjoint到`1e-17`，但kernel一侧取q=0、另一侧取机器精度非零。expanded kernel改用fractional coordinates+integer shift并canonicalize exact-zero后，所有q sector的`B(q)=B(-q)^T`恢复到`3.2e-16 meV`，focused tests `8 passed`。

C3 branch representation也已闭合：五层mesh3 shell-4 augmented `(pair,branch)`空间dimension/rank `51/51`、无branch failure、average closure`<=4.92e-13`、metric residual`<=2.95e-12`、独立C3^3`9.37e-14`。统一expanded A direct/exchange/total covariance均`<=1e-11 meV`，无energy assignment/polar/inverse closing edge。

Y必须取`conj[D_X(-q)]`；不共轭候选误差`~4.12 meV`。mesh3 h0 pair-space的negative-q projector leakage仍`2.8--4.3e-9`，导致B/L约`5e-5/3e-4 meV`；padding2/3/4不变。该残差已定位到occupied/unoccupied projector C3 closure，而非branch functional。

Fresh archive `185536`已通过source gate：saved-H closure`3.58e-15 meV`、`[H,P]=3.43e-5 meV`、ordinary projector C3`<=3.56e-13`、fixed copies`<=3.11e-15`。三条actual generic-q anchors `(1,0),(0,1),(-1,-1)`分别microscopic build并检查全部fixed-touched entries，A Hermiticity`<=2.22e-16 meV`、B transpose`<=1.52e-16 meV`。

完整矩阵初版的`0.172 meV` C3失败已定位到ordinary wrapped endpoint：off-torus rebuilt frame与解析periodic frame张成同一subspace（`3.49e-13`），但存在`0.678`非对角active gauge rotation。ordinary腿改为解析periodic relabel后，generic三edge A/L spectrum分别`<=1.25e-11/1.27e-11 meV`；signed self sectors的same-copy三branch选择及两个fixed torus points六个repeated-zone representatives也通过，L C3`<=6.85e-12 meV`。

三个separated channels现已全部通过q0 strict response parity、generic C3和raw q/-q gates；intervalley/interspin L C3分别`<=2.13e-11/1.44e-11 meV`。interspin q0 SU(2) Goldstone为`2.145e-7 meV`，microscopic generator overlap `0.9999999999993`。public typed-source signed-q matrix-pair API与gated q=(1,0) A/B/L逐元素严格一致。Focused/wider suites为`10 passed`和`77 passed`（含generic raw eta/residual coverage）。

Production runner及26个C3+inversion representatives/78-channel-task full mesh已完成（jobs `190204/190206/190207/190208`）：structure`<=7.03e-16 meV`、q/-q PH/quartet`<=4.78e-12 meV`。但paper-raster comparison明确失败：论文intraflavor 45个unstable sectors，当前functional仅三个M点；intra/intervalley/interspin RMSE为`1.691/0.730/0.775 meV`。Phase E因此结论为validated nonreproduction，而不是Fig. S45 reproduction。

证据：

```text
results/.../FINITE_Q_COPY_PAIR_PROVENANCE_PREWEIGHT_20260716.json
results/.../Q0_OFFDIAGONAL_COPY_WEIGHT_CANDIDATES_tiny3_20260716.json
logs/rlg_finite_q_copy_pair_preweight_20260716.log
logs/rlg_q0_offdiagonal_copy_weights_tiny3_20260716.log
```

### 其后 gates

- layer form-factor conjugation/periodic/C3 identities；
- `beta=0 -> A=A0, B=0`；
- `A=A^†`, `B(q)=B(-q)^T`, symplectic/pseudo-Hermitian identities；
- raw spectrum counts、quartets、eta norms、right/left residuals；
- cutoff convention `|G|<=Lambda` vs `|q_WS+G|<=Lambda` 明确分离和收敛审计。

## 11. 执行阶段

### Phase A — 只加失败 gate

1. 重建 tiny3 production-path anchor diagnostic；
2. 加 independent third-edge Wilson；
3. 保存 term/spectrum/copy provenance；
4. 不改 production algebra。

### Phase B — q=0 HF response/Hessian

1. 从 `_hf_c3_quotient.py` factor linear primitives；
2. 实现 q=0 off-diagonal tangent；
3. ordinary 对齐 D18；
4. fixed 对齐 corrected HF derivative；
5. 检查 q=0 Ward/Goldstone structure。

### Phase C — finite-q all-copy lift

从单个 fixed-touched tangent column开始；先输出 copy-pair momentum table，再扩展 direct/exchange、hp/Y、完整 pair space，最后通过 anchor gate。

### Phase D — actual-κ gates

重新独立构造：

```text
(1,0), (3,7), (6,0) ordinary orbits
q=0, (4,8), (8,4) self sectors
```

逐 term输出，不能用旧 cycle PASS 替代 anchor gate。

### Phase E — full mesh 与 Fig. S45

A–D通过后已利用q/-q matrix-pair和C3将网格约化为26个representatives（78 channel tasks），合并144点并保存raw spectra/eta norms/residuals。输出图严格标为validated calculation/comparison；由于论文比较失败，不得标为Fig. S45 reproduction。

## 12. 本地证据状态

审核意见列出的附件名：

```text
rlg_hbn_tdhf_fixed_quotient_reduced_diagnostics.py
rlg_hbn_tdhf_fixed_quotient_reduced_diagnostics.json
rlg_hbn_tdhf_anchor_diag_tiny3.json
rlg_hbn_tdhf_independent_wilson_tiny3.json
mean_field-debug-rlg-hbn-figs45-c3-quotient-20260716.zip
```

截至本文写入时，在 `/data/home/ziyuzhu` 下未找到这些文件。因此本地 Phase A 必须依据审核给出的配置/数值，通过当前 production APIs 独立重建；重建结果在数值一致前不能声称复现了审核附件。

## 13. Deliverables

```text
docs/debug/rlg_hbn_fixed_quotient_hf_response_plan_20260716.md
results/.../FIXED_QUOTIENT_ANCHOR_GATE.json
results/.../INDEPENDENT_FIXED_WILSON_GATE.json
docs/debug/FIXED_QUOTIENT_DERIVATION.md
results/.../Q0_HF_HESSIAN_GATE.json
results/.../FINITE_Q_COPY_PAIR_PROVENANCE.json
focused pytest log
remaining actual-kappa Slurm gate list
```

## 14. 当前判定

q=0 variational HF quotient、expanded-energy oracle、actual 12×12 ordinary/fixed-copy C3 preflights现已通过；v1 archive仍无效，fresh v2 SCF job `185536` 正在运行。该job使用绝对路径miniconda Python；早期脚本中误留的跨项目venv activation已从所有后续RLG/hBN脚本和活动等待终端清理。

Finite-q现有provider继续被canonical-anchor/Wilson否定。pre-weight、punctured geometry和microscopic rank gates共同证明：current v2只定义k-diagonal functional；finite-q branches不是gauge-equivalent copies，因此不存在可由当前q=0数据唯一推出的connection。

下一步完成标准是：

```text
fresh v2 HF archive + typed provenance + saved-H/source closure（仅作为diagonal source）
+ explicit expanded full-density/Stinespring energy functional
+ prove its q=0 diagonal restriction or rerun matching SCF
+ independently assembled anchor-independent finite-q Hessian
+ independent microscopic C3^3 and q/-q representation.
```

这些完成前，core finite-q logic仍未验证，actual-κ TDHF、full mesh与Fig. S45保持 blocked。
