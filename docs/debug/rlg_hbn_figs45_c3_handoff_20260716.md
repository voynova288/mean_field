# RLG/hBN Fig. S45 finite-q TDHF C3 debug handoff

**Branch:** `debug/rlg-hbn-figs45-c3-quotient-20260716`
**Base commit:** `7e937c3`
**Status:** debug checkpoint; **not merge-ready and not a Fig. S45 reproduction**
**Date:** 2026-07-16

## 1. 总任务：我们到底在做什么

总任务是在 `/data/home/ziyuzhu/Mean_Field` 中建立一条可信的 RLG/hBN finite-q TDHF/RPA 计算链，并最终定量复现论文 Fig. S45。完整目标不是“让一张图看起来像论文”，而是依次完成：

1. 使用当前 C3-compatible projected-basis gauge，重新得到收敛的 actual-κ (`hbn_moire_scale=1`) Hartree-Fock source；
2. 按论文 Appendix D18/D19 构造 finite-q intraflavor Liouvillian：

   ```text
   L(q) = [[A(q), B(q)], [-B(-q)*, -A(-q)*]]
   B(q) = B(-q)^T
   ```
3. 在 interaction contraction **之前**正确 transport microscopic states/form factors，处理 ordinary torus nodes、非零 C3-fixed nodes、repeated-zone direct shells 和 q/-q anomalous partner；
4. 在 centered repeated-zone 12×12 q mesh 上验证：
   - C3-related matrices and spectra；
   - q=0 Goldstone/near-Goldstone structure；
   - complex/negative/unstable collective modes；
   - 保存结果、坐标和 mode classification 的一致性；
5. 只有上述 gates 全部通过后，才重新生成并定量比较 Fig. S45。

当前分支只保存到第 3 步和部分第 4 步。**full q mesh 尚未提交，Goldstone 和 unstable-mode 审计尚未完成，Fig. S45 尚未复现。**

## 2. 物理与实现约束

本 debug 工作坚持以下约束：

- Fig. S45 使用 centered repeated-zone q mesh；plot coordinates 不做 Wigner-Seitz folding。
- ordinary A/B exchange 使用 actual-node Wigner-Seitz folding，并对 exact boundary ties 平均。
- fixed external sectors `(4,8)`、`(8,4)` 使用 active-local/copy quotient；不能删除 fixed legs。
- fixed source density/HF contraction 使用三个 reciprocal representatives，在 contraction 前 transport density，并以 `1/3` 平均。
- q/-q 的 B partner 必须共享同一个 canonical microscopic bilinear provider scalar。
- 不能使用 post-assembly matrix rotation、Hermitization、经验 phase、fixed-block rescaling、后处理缩放或 paper-fitting 作为修复。
- C3 matrix transforms 只作为 acceptance diagnostics；不能把 `UAU†` 作为 production construction。
- Heavy HF/TDHF/eigensolver 工作只在 Slurm full nodes 上运行。

## 3. 已确认的公式约定

### 3.1 D18/D19 plus/minus semantics

```text
plus:  particle = bra, hole = ket
minus: particle = ket, hole = bra
```

minus/Y operator order是 `d†_h d_{p(k-q)}`；物理 particle 是 ket。

### 3.2 Repeated-zone direct-shell transport

若

```text
q_target = C3(q_source) + mesh * R,
```

则 microscopic direct-shell label 必须逐 edge 递推：

```text
G_next = C3(G_current) - R.
```

完整多步 cycle 不能在每条 edge 都从 canonical `G0` 重启。

### 3.3 HF actual-node Fock transfer

对 target/source node transfer

```text
delta    = k_target - k_source
delta_WS = delta - W
```

physical shell `G` 对应的 cached overlap key 是

```text
H_cache = G - W,
Q       = delta + H_cache = delta_WS + G.
```

exactly degenerate shortest Wigner-Seitz wraps 必须等权平均。

## 4. 问题链与尝试

### 4.1 κ=0 TDHF provider audit

在 artificial κ=0 source 上，先逐层定位并修复：

- ordinary/nonfixed actual-node WS exchange；
- fixed active-local/copy form factors；
- role-specific X/Y masks；
- A0-compatible energy assignment；
- q/-q shared anomalous provider；
- provider-local reciprocal padding；
- one-step repeated-zone direct-shell offset；
- complete-cycle canonical-source policy。

通过的 κ=0 gates 包括：

```text
one-step q=(1,0): max L/A/B ~7.07e-11 meV
one-step q=(3,7): max L/A/B ~7.00e-11 meV
one-step q=(6,0): max L/A/B ~6.90e-11 meV
cycle q=(1,0):   max closure 7.10e-11 meV
self q=(0,0):    max covariance 8.06e-11 meV
self q=(4,8):    max covariance 7.08e-11 meV
self q=(8,4):    max covariance 7.02e-11 meV
B transpose:     <=O(1e-15) meV
```

注意：当时 full three-edge cycle 只对 `(1,0)` 做过；`(3,7)` 只通过了第一条 edge。这后来留下了一个未覆盖的第二-edge bug，见 4.6。

### 4.2 旧 actual-κ archive 不能迁移

旧 archive 使用 basis gauge

```text
centered_cell_reciprocal_relabel_pad1_v2
```

而当前代码要求

```text
c3_equivariant_reciprocal_relabel_fixedrep_v3.
```

迁移 oracle 给出：

```text
ordinary W†W defect     2.03e-4
fixed W†W defect        1.98e-1
minimum singular value  0.895593
physical-h0 mismatch    32.9915 meV
```

因此拒绝了 polar/Löwdin migration、scalar normalization 和任何强行 checkpoint conversion，决定重算 HF。

### 4.3 Fresh v3 basis/HF orchestration

- job `179755`：12 小时 walltime 内仍在 fixed-remote basis build，超时；但 screening cache 保存成功。
- job `180920`：复用 screening，约 15.5 小时完成 v3 basis 并写出：

  ```text
  basis_310a13e704a2c6f680966482
  overlap_71485ab39b5e306180946309
  ```

第一次 fresh HF 收敛：

```text
iterations    33
final error   8.4863e-5
energy        -547.0919405100 meV
```

### 4.4 Basis cache metadata 丢失

第一次 actual-κ cycle job `180921` 在约 12 秒内失败。原因不是物理计算，而是 cache serializer 没有保存：

```text
periodic_reciprocal_shifts
c3_fixed_representative_pairs
```

修复内容：

- 新 cache 显式写入/读取两组 arrays；
- 对已写出的 v3 regular zero-shift cache，从保存的 mesh geometry 确定性恢复 metadata；
- stale pre-v3 cache 仍然拒绝，不能绕过 gauge check。

### 4.5 第一次 fresh HF source 本身不满足 C3

修复 cache 后，job `182200` 在第一次 fresh HF archive 上运行 actual-κ cycle：

```text
(1,0) -> (0,1) -> (11,11) -> (1,0)
```

结果失败：

```text
max A/L covariance defect       1.12733 meV
max B covariance defect         0.0158846 meV
max spectrum assignment defect  1.43818 meV
B(q)-B(-q)^T                    <=5.1e-16 meV
```

component audit 进一步定位：

```text
physical h0 C3 mismatch      2.19e-11 meV
fixed_remote mismatch        3.23e-11 meV
archive h0 mismatch          1.62e-11 meV
final HF interaction         0.971 meV
final Hamiltonian            1.197 meV
```

microscopic sewing audit 显示 ordinary sectors 中：

```text
raw basis sewing unitarity   3.6e-13
initial flavor density       8.9e-13
initial Hartree defect       2.20e-4 meV
initial Fock defect          1.19389 meV
```

因此这不是 spontaneous nematicity；C3-symmetric seed 在第一次旧 HF Fock contraction 就被破坏。

source decomposition：

```text
ordinary-source-only Fock defect   1.121944 meV
fixed-source contribution defect   0.168974 meV
```

#### 修复

新增 system-local HF quotient provider：

```text
src/mean_field/systems/RnG_hBN/_hf_c3_quotient.py
```

它实现：

1. ordinary source actual-node WS `G-W` + boundary-tie averaging；
2. nonzero C3-fixed source 的三-copy density transport；
3. Hartree/Fock 使用同一 source decomposition；
4. ODA functional 与 SCF 使用同一个 corrected interaction builder。

独立 gates：

```text
old ordinary-source Fock   1.121944 meV
corrected WS Fock          1.269e-11 meV
expanded fixed source      3.55e-12 meV
combined Fock              1.289e-11 meV
```

production initial-density gate `182495`：

```text
ordinary Hartree covariance   2.40e-11 meV
ordinary Fock covariance      1.29e-11 meV
ordinary total covariance     3.24e-11 meV
Hermiticity                    <=2.22e-15 meV
```

corrected HF job `182757` 使用 convention

```text
actual_node_ws_fixed_source_copy_v1
```

并收敛：

```text
iterations          32
final error         9.512258e-5
final energy        -547.4101237315 meV
HF gap              14.5344762 meV
projector defect    2.22e-15
```

accepted local archive：

```text
results/RnG_hBN/tdhf_m2_pilot/a_v64_c3quotient_hf_v1_182757/
  checkpoint_A_average_V64_hf/xi1_V064meV/runs/flavor_seed1/hf_run_state.npz
```

该大型 archive/cache 不在 Git 分支中。

### 4.6 Actual-κ TDHF cycles 与尚未验证的第二-edge修复

使用 corrected HF source：

```text
q=(1,0) cycle, job 182758: PASS
max closure                 5.24798e-11 meV
max spectrum mismatch       1.37561e-11 meV
max B transpose             4.23e-16 meV

q=(6,0) cycle, job 184033_1: PASS
max closure                 5.24773e-11 meV
max spectrum mismatch       2.13803e-11 meV
```

但第一次 full cycle `q=(3,7)`（job `184033_0`）结果为：

```text
step 0, (3,7)->(5,8):       ~5.20e-11 meV  PASS
step 1, (5,8)->(4,9):       7.42e-3 meV    FAIL
closing edge:                7.42e-3 meV    FAIL
energy sewing:               <=6.6e-12 meV
```

这暴露了 direct-shell second-edge recurrence bug：

```text
old:     G2 = C3(G0) - R12
correct: G2 = C3(C3(G0) - R01) - R12
```

`q=(1,0)` 和 `(6,0)` 的第一步 offset 为零，完整 shell 又是 C3-invariant set，因此之前偶然掩盖了这个错误。

当前分支已经实现 recursive transport：

```text
G_{i+1} = C3(G_i) - R_i
```

并加入纯几何 regression test。**但是这一最后修复尚未获得 Slurm heavy gate。** 当前 scheduler 状态：

```text
184106       repaired q=(3,7) term/cycle gate: PENDING
184033_2     actual q=(0,0) self gate:          PENDING
184033_3     actual q=(4,8) self gate:          PENDING
184033_4     actual q=(8,4) self gate:          PENDING
```

这些 jobs 因 full-node priority/reservation 等待。分支 push 时不能把 patched `(3,7)` 标记为通过。

## 5. 本分支主要代码路径

### Basis/gauge/cache

```text
src/mean_field/systems/RnG_hBN/_hf_shared.py
src/mean_field/systems/RnG_hBN/_hf_types.py
src/mean_field/systems/RnG_hBN/_hf_basis.py
src/mean_field/systems/RnG_hBN/cache.py
```

### HF interaction

```text
src/mean_field/systems/RnG_hBN/_hf_interaction_path.py
src/mean_field/systems/RnG_hBN/_hf_c3_quotient.py
src/mean_field/systems/RnG_hBN/_hf_runner.py
src/mean_field/systems/RnG_hBN/hf.py
```

### finite-q TDHF quotient provider

```text
src/mean_field/systems/RnG_hBN/_tdhf_finite_q_terms.py
src/mean_field/systems/RnG_hBN/_tdhf_fixed_quotient.py
src/mean_field/systems/RnG_hBN/_tdhf_quotient_orbit.py
src/mean_field/systems/RnG_hBN/_tdhf_finite_q.py
src/mean_field/systems/RnG_hBN/tdhf.py
```

### Full-mesh entry point

```text
src/mean_field/devtools/run_rlg_hbn_tdhf_finite_q.py
scripts/mean_field_tools.py
```

The devtool exposes opt-in flag：

```text
--c3-quotient-provider
```

它按 C3 orbit 从一个 canonical microscopic source 建完整 cycle，并缓存同 orbit matrices。

## 6. 当前验证

Focused dirty-worktree validation 在开发过程中通过：

```text
PYTHONPATH=tmp/import_shadow:src pytest -q \
  tests/test_rlg_hbn_hf.py \
  tests/test_rlg_hbn_tdhf_adapter.py

57 passed
```

其中 `tests/test_rlg_hbn_hf.py` 在当前 repository policy 下是 ignored local test surface，没有整体加入本 branch。为避免 dirty worktree 隐藏依赖，push 前从 staged Git tree 生成了 clean snapshot，并仅用 branch-visible source/tests 运行：

```text
PYTHONPATH=<clean-snapshot>/src pytest -q \
  tests/test_rlg_hbn_tdhf_adapter.py

34 passed
```

branch-visible tests 包含 basis-cache metadata roundtrip/legacy-v3 recovery、tiny HF quotient covariance、finite-q D18/D19、C3 orbit/cycle、recursive direct-shell geometry、minus semantics、fixed transport 和 shared-B provider regression。

## 7. 明确没有解决的问题

1. **Patched q=(3,7) second edge 尚未 heavy-validated。** 目前只有公式推导和 geometry unit test。
2. **Actual-κ q=0 与 fixed self sectors 尚未通过。** κ=0 versions 通过，但不能替代 actual HF gate。
3. **Full centered repeated-zone 12×12 q mesh 尚未提交。** 已准备 orbit-sharded 50-orbit workflow，但被 gates 阻塞。
4. **Goldstone 尚未检查。** 必须从 corrected q=0 Liouvillian 的 raw/selected spectrum 检查，而不是靠画图猜测。
5. **Unstable modes 尚未分类。** 需要检查 complex raw eigenvalues、missing positive-metric branches、negative/zero modes 和 eigensolver residuals。
6. **C3-related full-mesh spectra 尚未全局比较。** 目前只验证少数 cycles。
7. **Fig. S45 尚未重新生成或 overlay。** 旧图来自已否定的 source/convention，不能作为复现证据。
8. **HF quotient provider 的更广参数回归尚未完成。** 当前 actual V=64 meV, ξ=1 source 通过；其他 mesh、filling、V、κ 不应自动宣称验证。
9. **性能/缓存仍可改进。** Corrected HF 约 18.3 小时；每个 dense TDHF orbit 约 40 分钟、峰值约 30 GiB。Full mesh 需要 Slurm array/orbit sharding。
10. **该分支建立于一个有大量并行未提交改动的工作树。** Commit 只应包含本 debug 任务的 selected paths；其他 topology/optical/other-system 修改必须保持在分支之外。

## 8. 下一步验收顺序

1. 等待并解析 `184106`；要求 patched `(3,7)` cycle A/B/L closure `<=1e-9 meV`，并确认具体 direct terms 已恢复。
2. 解析 `184033_2..4`；要求 q=0、`(4,8)`、`(8,4)` actual self gates 通过。
3. gates 全过后，提交 50-orbit full-mesh array，每个 C3 orbit 保证单一 canonical source。
4. 合并 144×3 channel blocks，检查 coverage、closure、matrix structure、C3 spectrum assignment 和 eigensolver residuals。
5. 审计 q=0 Goldstone 与 complex/negative/unstable modes。
6. 只有这些结果可信后，生成 Fig. S45 并做定量论文比较。

## 9. 结论

本 branch 的核心成果是把问题从“TDHF 图不对”逐层拆成三个独立根因：

1. projected-basis/cache gauge metadata；
2. active-density HF Fock 的 actual-node/fixed-source C3 covariance；
3. TDHF repeated-zone multi-edge direct-shell composition。

前两项已有 actual-κ production gates；第三项已修代码但等待 heavy gate。当前最重要的事实是：**我们已经得到一个 C3-compatible、收敛的 actual-κ HF source，并在 `(1,0)`、`(6,0)` cycles 上通过；但 full mesh 和 Fig. S45 仍未完成。**
