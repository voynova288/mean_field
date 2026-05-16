# mean_field 平均场代码隐藏 Bug 检查报告

**检查对象**：`mean_field.zip`  
**重点范围**：`mean_field/systems/tbg/zero_field` 与 `mean_field/core/hf`  
**检查目标**：不复现大规模 HF 结果，只检查“结果已正确但未来可能踩坑”的隐藏 bug、参数不一致、约定不一致和边界条件。  
**检查方式**：静态代码审阅、`compileall`、关键模块导入、小矩阵级 sanity check；未运行大规模 HF 迭代。  
**结论概览**：代码整体结构清晰，主模块可编译和导入；当前 benchmark/parity 语境下结果可信。但存在若干隐藏风险，其中最值得优先修的是：ODA mixed convergence 可能假收敛、restricted/full density convention 不完全一致、HF path evaluation 没有继承完整 SCF 参数。

---

## 0. 严重程度定义

| 等级 | 含义 |
|---|---|
| P0 | 可能导致“看似收敛但实际未自洽”或保存结果不对应最终 density，应优先修。 |
| P1 | 当前默认 benchmark 可能不暴露，但参数扫描、换初值、full/restricted 切换时可能出错。 |
| P2 | 物理 convention 或边界条件问题；若只做 benchmark parity 可保留，但应加开关和注释。 |
| P3 | 诊断、元数据、边界输入或工程健壮性问题。 |

---

## 1. 总体检查结果

通过项：

- `python -m compileall -q mean_field` 通过。
- 关键模块导入通过：`mean_field`、`mean_field.systems.tbg.zero_field.hf`、`mean_field.systems.tbg.zero_field.model`、`mean_field.core.hf`。
- 未发现明显语法错误、模块导入错误或简单 shape 定义错误。
- HF 主框架抽象较干净：`HartreeFockKernel` / `HartreeFockProblem` / `run_hartree_fock_iterations` 分层合理。

需要修或明确标注的隐藏问题如下。

---

## 2. 发现项汇总

| 编号 | 等级 | 位置 | 问题摘要 | 影响范围 | 建议 |
|---|---:|---|---|---|---|
| F1 | P0 | `core/hf/engine.py:132-184`; `systems/tbg/zero_field/hf.py:1070-1098` | full HF 使用 `convergence_rule="mixed"` 时，ODA `lambda` 很小时可能把“几乎没动”误判为“已收敛”。 | full HF，尤其复杂初值/竞争相/接近相边界。 | ODA stall 判断先于 mixed convergence；mixed 收敛同时要求 raw norm 收敛。 |
| F2 | P0/P1 | `core/hf/engine.py:132-184`; `hf_runners.py:97-103` | 退出时 `state.hamiltonian` 对应 mixing 前 density，`state.density` 是 mixing 后 density。 | 保存状态、SCF grid path plot、diagnostic energies。 | 退出前用最终 density 重建一次 Hamiltonian 和 energies。 |
| F3 | P1 | `systems/tbg/zero_field/hf.py:590,671,1065`; `core/hf/interaction.py:37-43` | restricted density 使用普通 projector convention，full HF 和 core 能量 contraction 使用 Julia stored-projector convention。 | restricted/full 切换、random 初始化、复杂 flavor mixing。 | 统一 density convention，或明确区分并在接口层转换。 |
| F4 | P1 | `systems/tbg/zero_field/hf_runners.py:127-162`; `runners.py:839-846,1119-1126` | path evaluation 没有继承 SCF 的 `beta` 和 `overlap_lg`。 | 非默认 `beta` / `overlap_lg` 的参数扫描和 band plot。 | path 函数新增 `beta` 与独立 `overlap_lg` 参数。 |
| F5 | P2 | `systems/tbg/zero_field/model.py:78-108`; `core/hf/overlap.py:140-145,175-180` | `np.roll` 使有限 G 截断出现 wrap-around，等价于 reciprocal grid 周期边界。 | 小 `lg`、cutoff convergence、非 benchmark 物理计算。 | 加 `periodic_g_grid` / `wrap_g_grid` 开关；默认物理计算用 zero-fill shift。 |
| F6 | P2 | `model.py:60-70`; `core/lattice.py:56-66` | uniform k mesh 包含 BZ 边界重复点。 | 小 `lk` BZ 积分权重、非 benchmark 计算。 | 加 `include_endpoint` 开关；benchmark 保留 True，生产计算默认 False。 |
| F7 | P2 | `model.py:209-214` | C2T gauge fixing 后逐列归一化但未重新正交化，简并/近简并时 overlap 可能受污染。 | Dirac 点、平带、简并点附近的 overlap 和 HF matrix elements。 | gauge fixing 后 QR/SVD 重新正交化，并处理零范数 fallback。 |
| F8 | P2/P3 | `hf.py:107-118,234-249` | `q=0` screened Coulomb 默认置零，但 metadata 没记录 convention。 | Hartree/Fock q=0 convention 比较、换 screening/gate 模型。 | 显式参数化并写入 summary：`drop_q0_coulomb` / `finite_zero_limit`。 |
| F9 | P3 | `hf.py:217-224,1209-1212`; `core/hf/occupations.py:6-14` | filling 边界和 fractional filling 处理不够严格。 | 空/满 filling、非整数 filling、diagnostics。 | 对非法或非整数 filling 显式报错；空 occupied mean 返回 `nan` 但避免 warning。 |
| F10 | P3 | `paths.py:4-7` | `PACKAGE_ROOT = parents[2]` 依赖 `src/mean_field` 布局；zip 直接解压时路径会错。 | 直接从 zip/repo 子目录运行时 benchmark/data path。 | 使用 package resource 或从当前文件向上搜索 marker。 |

---

## 3. 详细问题与修复建议

### F1. Full HF 的 ODA mixed convergence 可能假收敛

**位置**：

- `mean_field/core/hf/engine.py:132-184`
- `mean_field/systems/tbg/zero_field/hf.py:1070-1098`

当前 full HF kernel 设置：

```python
convergence_rule="mixed"
```

而 engine 中判断逻辑是：

```python
norm_selected = norm_raw if convergence_rule == "raw" else norm_mixed

if norm_selected <= state.precision:
    exit_reason = "converged"
    break
if oda_lambda < oda_stall_threshold:
    exit_reason = "oda_stall"
    break
```

问题在于：当 ODA 算出的 `oda_lambda` 很小甚至接近 0 时，

```python
mixed_density ≈ previous_density
```

因此 `norm_mixed` 会非常小。此时即使 raw SCF update 仍然很大，也可能先触发 `converged`，而不是 `oda_stall`。这会产生“看起来收敛，但其实只是 ODA 不敢动”的假收敛。

**建议修法**：

```python
raw_converged = norm_raw <= state.precision
mixed_converged = norm_mixed <= state.precision
stalled = oda_lambda < oda_stall_threshold

if stalled and not raw_converged:
    exit_reason = "oda_stall"
    break

if convergence_rule == "raw":
    converged = raw_converged
else:
    # mixed norm 只能作为辅助，不能单独证明自洽
    converged = mixed_converged and raw_converged

if converged:
    exit_reason = "converged"
    break
```

更保守的选择：full HF 也使用 `convergence_rule="raw"`，mixed norm 仅作为诊断输出。

---

### F2. 退出时 Hamiltonian / density / energies 可能不是同一个自洽点

**位置**：

- `core/hf/engine.py:132-184`
- `systems/tbg/zero_field/hf_runners.py:97-103`

每轮迭代中，Hamiltonian 是由 `previous_density` 构造：

```python
previous_density = state.density.copy()
state.hamiltonian[:, :, :] = state.h0
interaction_h = interaction_builder(previous_density)
state.hamiltonian[:, :, :] += interaction_h
```

随后 density 被 ODA mixing 成：

```python
state.density[:, :, :] = mixed_density
```

所以退出时：

- `state.density` 是 mixed 后的 density；
- `state.hamiltonian` 是上一轮/旧 density 构造的 Hamiltonian；
- `state.energies` 是旧 Hamiltonian 对角化得到的 energies。

如果已经严格收敛，差别很小；但若发生 F1 的假收敛，差别可能很大。即使没有假收敛，`build_restricted_hf_scf_path_plot_result()` 直接使用 `hf_run.state.hamiltonian[:, :, indices]`，会让 SCF-grid plot 对应“旧 density Hamiltonian”。

**建议修法**：在 `run_hartree_fock_iterations()` 退出前增加 finalize 步骤：

```python
def finalize_state_from_density(state):
    interaction_h = interaction_builder(state.density)
    state.hamiltonian[:, :, :] = state.h0 + interaction_h
    if hamiltonian_postprocessor is not None:
        hamiltonian_postprocessor(state.hamiltonian)
    density_update = density_builder(state.hamiltonian)
    state.energies[:, :] = density_update.energies
    state.mu = float(density_update.mu)
```

若担心 finalize 改变 density，可只更新 Hamiltonian/energies，不再覆盖 density；或者保存 finalize 后的 raw norm 作为 diagnostic。

---

### F3. restricted 与 full 的 density/projector convention 不一致

**位置**：

- `core/hf/interaction.py:37-43`
- `core/hf/engine.py:84-89`
- `systems/tbg/zero_field/hf.py:590,671,1065`

core HF 明确使用 Julia B0 stored-projector convention：

```python
# core/hf/interaction.py
# tr(H * transpose(P)) == sum_ab H[a,b] * P[a,b]
total = np.einsum("abk,abk->", interaction_hamiltonian, density, optimize=True) / 2.0
```

full density builder 也使用 stored convention：

```python
# build_full_density_from_hamiltonian
density[:, :, ik] = occupied_vecs.conj() @ occupied_vecs.T - 0.5 * full_id
```

但 restricted density builder 和 restricted random init 使用普通 projector convention：

```python
# initialize_restricted_density(random)
occupied_vecs @ occupied_vecs.conj().T - 0.5 * block_id

# build_restricted_density_from_hamiltonian
occupied_vecs @ occupied_vecs.conj().T - 0.5 * block_id
```

对于纯实对角 density 或简单相，这个差异不会暴露；但一旦存在复杂 off-diagonal、C2T gauge、随机初值、flavor/band mixing，二者会相差复共轭/转置。

**建议修法**：如果全项目采用 Julia stored-projector convention，则 restricted 也改成：

```python
block_density[np.ix_(block_inds, block_inds)] = (
    occupied_vecs.conj() @ occupied_vecs.T - 0.5 * block_id
)
```

同时审计 full 初始化中的 rotation：

```python
unitary.conj().T @ block @ unitary
```

如果 `block` 也是 stored-projector convention，旋转形式可能应统一为 stored convention 下的变换，而不是普通 density matrix 的相似变换。建议用一个 2x2 复数 projector 单元测试固定 convention。

**建议添加测试**：

```python
def test_restricted_full_projector_convention_agree_for_complex_eigenvectors():
    # 构造一个含复数 off-diagonal 的 2x2 Hermitian H
    # restricted 与 full 在同一 block、同一 filling 下应给出同一 stored density
    ...
```

---

### F4. HF path evaluation 没有继承 SCF 的 `beta` 和 `overlap_lg`

**位置**：

- `systems/tbg/zero_field/runners.py:817-846`
- `systems/tbg/zero_field/runners.py:1090-1126`
- `systems/tbg/zero_field/hf_runners.py:127-162`

SCF 允许传：

```python
beta
overlap_lg
```

其中 SCF overlap 使用：

```python
overlap_grid_lg = bm_lg if overlap_lg is None else int(overlap_lg)
overlap_blocks = build_overlap_block_set(grid_solution, lg=overlap_grid_lg)
```

但 path evaluation 中：

```python
path_result = evaluate_restricted_hf_path(..., lg=bm_lg, ...)
```

并且 `build_projected_target_hamiltonian()` 没有传 `beta`，所以默认 `beta=1.0`：

```python
h_path = build_projected_target_hamiltonian(
    h_path,
    state.density,
    ...,
    v0=state.v0,
)
```

这意味着：

- 若 SCF 用了 `beta != 1`，最终 path band 用的是另一套 interaction strength；
- 若 SCF 用了 `overlap_lg != bm_lg`，最终 path band 的 form factor shift set 不一致。

**建议修法**：把 path BM cutoff 与 overlap shift cutoff 分开：

```python
def build_restricted_hf_path_hamiltonian(
    hf_run,
    grid_solution,
    *,
    bm_lg: int | None = None,
    overlap_lg: int | None = None,
    beta: float = 1.0,
    ...,
):
    bm_lg = grid_solution.lg if bm_lg is None else int(bm_lg)
    overlap_lg = bm_lg if overlap_lg is None else int(overlap_lg)

    path_solution = solve_bm_model(params, path.kvec, lg=bm_lg, sigma_rotation=True)

    grid_overlap = build_overlap_block_set(grid_solution, lg=overlap_lg, **screening_kwargs)
    path_overlap = build_overlap_block_set(path_solution, lg=overlap_lg, **screening_kwargs)
    path_grid_overlap = build_overlap_block_set(
        path_solution,
        source_solution=grid_solution,
        lg=overlap_lg,
        **screening_kwargs,
    )

    h_path = build_projected_target_hamiltonian(
        h_path,
        state.density,
        source_overlap_blocks=grid_overlap,
        target_overlap_blocks=path_overlap,
        target_source_overlap_blocks=path_grid_overlap,
        v0=state.v0,
        beta=beta,
    )
```

并在 `HFPathResult` / summary 中记录：

```text
beta
bm_lg
overlap_lg
relative_permittivity
screening_lm
finite_zero_limit
zero_cutoff
```

---

### F5. `np.roll` 导致有限 G cutoff 出现 wrap-around

**位置**：

- BM tunneling：`systems/tbg/zero_field/model.py:78-108`
- overlap/form factor：`core/hf/overlap.py:140-145,175-180`

BM tunneling 中：

```python
idx_nn1 = np.roll(idx, shift=(-zeta, zeta), axis=(0, 1))
idx_nn2 = np.roll(idx, shift=(0, zeta), axis=(0, 1))
idx_nn12 = np.roll(idx, shift=(-zeta, 0), axis=(0, 1))
```

overlap 中也使用：

```python
shifted = np.roll(..., shift=(0, -int(m), -int(n), 0), axis=(0, 1, 2, 3))
```

这会把有限 reciprocal grid 边界上的 G 点卷到另一边，相当于给 plane-wave cutoff 加了周期边界。标准有限 cutoff 的物理做法通常是：`G + q` 超出截断时直接丢弃 coupling/form-factor contribution，而不是 wrap。

若当前目标是复现 Julia/B0 benchmark，这可能是刻意的 parity convention；但如果用于 cutoff convergence 或真实物理计算，wrap-around 会污染边界态，尤其 `lg` 小时更明显。

**建议修法**：加开关：

```python
periodic_g_grid: bool = False
```

benchmark runner 可设为 True；普通物理计算默认 False。zero-fill shift 可写成：

```python
def shift_g_grid_zero_fill(arr, shift_m, shift_n):
    out = np.zeros_like(arr)
    # 只复制仍在 cutoff 内的 slice，不做 wrap
    ...
    return out
```

---

### F6. uniform k mesh 包含 BZ 边界重复点

**位置**：

- `systems/tbg/zero_field/model.py:60-70`
- `core/lattice.py:56-66`

当前 mesh：

```python
frac = np.arange(0, lk + 1, dtype=float) / float(lk)
```

所以 k 点数是 `(lk + 1)^2`，包含 `0` 与 `g1/g2` 边界重复点。若是为了复现 B0 reference，可以保留；但作为 BZ torus 积分，标准写法通常是：

```python
frac = np.arange(lk, dtype=float) / float(lk)
```

否则边界点会被重复计权，小 `lk` 时影响更明显。

**建议修法**：

```python
def build_uniform_lattice(g1, g2, lk, *, include_endpoint=False):
    if include_endpoint:
        frac = np.arange(lk + 1) / lk
    else:
        frac = np.arange(lk) / lk
```

benchmark/parity case 明确传 `include_endpoint=True`。

---

### F7. C2T gauge fixing 后未重新正交化

**位置**：`systems/tbg/zero_field/model.py:209-214`

当前代码：

```python
evals, evecs = eigh(h, subset_by_index=[start, stop], driver="evr")
evecs = evecs + c2t @ np.conj(evecs)
norms = np.linalg.norm(evecs, axis=0)
evecs = evecs / norms[None, :]
```

逐列做 C2T symmetrization 后，只做了单列归一化，没有保证不同列之间仍然正交。若两个目标 band 在 Dirac 点或高对称点简并/近简并，`evecs.conj().T @ evecs` 可能偏离单位阵，进而污染 overlap matrix 和 HF form factor。

另外，若某列接近 C2T-odd，`evecs + C2T(evecs)` 的范数可能接近 0，存在除以小数的风险。

**建议修法**：

```python
evecs = evecs + c2t @ np.conj(evecs)
norms = np.linalg.norm(evecs, axis=0)
if np.any(norms < 1e-12):
    # fallback：换相位、换 gauge，或保留原 eigvecs 后做整体 sewing
    ...
evecs = evecs / norms[None, :]

# 重新正交化
q, r = np.linalg.qr(evecs)
evecs = q[:, :nb]

orth_err = np.linalg.norm(evecs.conj().T @ evecs - np.eye(nb))
```

如果担心 QR 改变 gauge，可在 QR 后再做一次 convention fixing，并记录 `orth_err` diagnostic。

---

### F8. `q=0` Coulomb convention 静默置零

**位置**：

- `systems/tbg/zero_field/hf.py:107-118`
- `systems/tbg/zero_field/hf.py:234-249`

当前：

```python
if q_abs < zero_cutoff:
    return finite_value if finite_zero_limit else 0.0
```

去掉 Hartree `q=0` 在有 neutralizing background 时是常见 convention；但 exchange 的同 k `q=0` 极限、gate screening、finite-size convention 都可能有不同处理。

**建议**：不要只靠默认值隐含；把该 convention 写进所有 HF summary：

```text
zero_cutoff=1e-6
finite_zero_limit=false
drop_q0_coulomb=true
screening_lm=...
relative_permittivity=...
```

如果未来要比较不同 screening/gate convention，建议把这些参数变成 `HartreeFockState` 或 `RestrictedHartreeFockRun` 的 metadata。

---

### F9. filling 与 diagnostics 边界条件

**位置**：

- `systems/tbg/zero_field/hf.py:217-224`
- `systems/tbg/zero_field/hf.py:1209-1212`
- `core/hf/occupations.py:6-14`

问题点：

1. `restricted_occupied_state_count()` 对非整数 filling 使用 round-like 行为：

   ```python
   return int(np.floor(raw + 0.5))
   ```

   若用户传入不对应整数占据的 `nu`，代码会静默取整。

2. `occupied_sigma_mean()` 在空占据时会对 empty slice 取 mean：

   ```python
   return float(np.mean(...[order]))
   ```

   `nu=-4` 时可能产生 warning 和 `nan`。

3. `find_chemical_potential()` 在 filling fraction 0 或 1 时返回值比较随意；occupancy 本身未必错，但 diagnostic `mu` 可能误导。

**建议**：

```python
def restricted_occupied_state_count(...):
    raw = ...
    rounded = round(raw)
    if abs(raw - rounded) > 1e-9:
        raise ValueError("Filling does not correspond to an integer number of occupied states")
    return int(rounded)
```

`occupied_sigma_mean()`：

```python
if order.size == 0:
    return float("nan")
```

`find_chemical_potential()` 对 0/1 filling 返回明确 convention，例如 band edge 外侧或 `nan`，并写入文档。

---

### F10. package root 推断依赖安装布局

**位置**：`mean_field/paths.py:4-7`

当前：

```python
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = PACKAGE_ROOT / "benchmarks"
```

若项目结构是 `repo/src/mean_field/...`，`parents[2]` 可能刚好指向 repo root；但 zip 直接解压成 `mean_field/...` 时，它会指向上一级目录，导致 benchmark/data path 找错。

**建议**：

- 用 `importlib.resources` 管理 package data；或
- 从当前文件向上搜索包含 `benchmarks/`、`pyproject.toml`、`.git` 的目录；或
- 允许环境变量 override：

```python
MEAN_FIELD_REPO_ROOT=/path/to/repo
```

---

## 4. 额外工程建议

### 4.1 把 HF 参数写进 run metadata

建议 `RestrictedHartreeFockRun` 或 `state.diagnostics` 记录：

```text
beta
overlap_lg
bm_lg
relative_permittivity
screening_lm
finite_zero_limit
zero_cutoff
convergence_rule
precision
oda_stall_threshold
periodic_g_grid / wrap_g_grid
include_endpoint_kmesh
projector_convention
```

这样之后看到某个 band plot 或 TSV，可以确认它和 SCF 是同一组参数。

### 4.2 overlap screening cache 避免 stale 参数

`_with_tbg_overlap_screening()` 中使用 `setdefault()`：

```python
hartree_screening.setdefault(...)
fock_screening.setdefault(...)
```

如果同一个 `overlap_blocks` 已经带有 screening cache，后续用不同 `relative_permittivity` / `finite_zero_limit` 调用时，可能静默复用旧 cache。建议把 screening 参数也作为 `HFOverlapBlockSet` metadata；参数不同就重建或报错。

### 4.3 ODA parameterizer 可复用 screened overlap block

当前 ODA parameterizer 会通过 `build_interaction_hamiltonian()` 再走一遍 `_with_tbg_overlap_screening()`。若直接传 `screened_overlap_blocks` 并调用 core 的 `build_projected_interaction_hamiltonian()`，可减少重复工作，也降低未来 screening 参数不一致的风险。

---

## 5. 建议修改优先级

1. **先修 F1 + F2**：避免 full HF 假收敛，并保证保存/画图时 Hamiltonian 与最终 density 对应。
2. **再修 F3**：统一 density/projector convention；至少加 2x2 复数 projector 单元测试，防止后续改动破坏约定。
3. **再修 F4**：让 path evaluation 继承 `beta`、`overlap_lg` 和 screening convention，避免“SCF 是一套参数、band plot 是另一套参数”。
4. **F5/F6 作为 benchmark vs physics 的开关处理**：如果当前目标是复现参考代码，可以默认保持 benchmark parity；但应显式命名为 benchmark convention，普通物理计算默认不 wrap、不重复 endpoint。
5. **F7-F10 做健壮性补丁**：这些通常不改变默认结果，但会减少未来参数扫描和发布 artifacts 时的排查成本。

---

## 6. 最小回归测试建议

建议新增以下轻量测试，不需要大 HF：

```text
1. ODA 假收敛测试
   构造 oda_lambda=0、raw_norm>precision、mixed_norm=0 的 toy kernel，确认 exit_reason 是 oda_stall 而不是 converged。

2. final Hamiltonian 一致性测试
   toy interaction H_int[P]=alpha*P，迭代退出后检查 state.hamiltonian == h0 + H_int[state.density]。

3. density convention 测试
   用一个复数 2x2 Hermitian block，比较 restricted/full density builder 是否给出同一 stored projector convention。

4. path 参数继承测试
   beta=0.5 和 beta=1.0 的 path Hamiltonian 应不同；overlap_lg 改变时 summary 中应正确记录。

5. G-grid shift 测试
   zero-fill shift 下边界元素不 wrap；benchmark wrap 模式下结果与旧实现一致。

6. kmesh endpoint 测试
   include_endpoint=True 时 nk=(lk+1)^2；False 时 nk=lk^2。

7. C2T gauge 正交性测试
   gauge fixing 后检查 ||U†U-I|| 小于阈值。
```

---

## 7. 最终判断

当前结果“已经跑出来且正确”与上述问题不矛盾：这些问题大多在默认 benchmark 参数、简单初值或最终相较稳定时不会显性暴露。真正危险的是后续扩展：换 `beta`、换 `overlap_lg`、使用 full HF 复杂初值、靠 mixed norm 判断收敛、或者做 cutoff/kmesh convergence 时，可能出现结果看起来合理但其实参数或 convention 不一致的情况。

优先把 F1-F4 修掉后，这套代码会更适合长期参数扫描和结果归档。
