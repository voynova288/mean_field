# tMBG 到 HF 的后续计划

这份计划只定义后续阶段的边界，不在当前阶段把 `tmbg` 与现有 `tbg` 代码强行合并。

## 当前阶段完成标准

- `mean_field.systems.tmbg` 独立提供 Park 2020 非相互作用连续模型。
- 晶格、参数、单粒子哈密顿量、路径能带和 mBZ 网格能带都能单独调用。
- 基础物理检查已经内置，可以快速验证几何、厄米性、时间反演和基本 C3 对称。
- checkpoint 编排入口已经存在：`reproduce_paper_checkpoints(...)` 可统一组织 CP1–CP6。
- 当前缺口不是模型层，而是正式的集群运行入口和真实 checkpoint 结果归档。

## HF 前需要先补齐的非相互作用部分

- `topology.py`
  核心实现已经有了，包括近退化点的 SVD 正则化与网格重试。剩余工作是把真实参数下的结果跑到计算节点上并归档。
- `plot.py`
  已有路径能带、Berry 曲率和 Fig. 2 风格三联图输出；`flat_band_indices` 与默认 `±35 meV` 视窗也已支持。
- `validation.py`
  轻量 `validate_physics(...)` 与全量 `reproduce_paper_checkpoints(...)` 都已存在。剩余工作是增加 Slurm 运行入口，并产出正式的 CP1–CP6 报告文件。

## HF 阶段建议的接口边界

- 保留系统专属部分
  `tmbg` 负责 `build_hamiltonian(k_tilde, lattice, params, valley)`、moire `G` 集合、标准 mBZ 网格、单粒子本征态。
- 抽取系统无关部分
  后续 HF 只抽取“给定单粒子基底后的密度矩阵、占据、混合、迭代控制、收敛判据、路径重建”这类通用代码。
- 不立即和 `tbg` 共用 Hamiltonian 层
  因为 `tbg` 与 `tmbg` 的基底结构、form factor、相互作用矩阵元来源不同，现在强行合并只会把边界做脏。

## 建议的实施顺序

1. 给 `reproduce_paper_checkpoints(...)` 增加正式 CPU 提交入口，按集群规则从 `login002` 发起 Slurm 作业。
2. 在计算节点上跑通 Park 2020 的非相互作用 checkpoint，保存 `fig2_like_bands` 和 `paper_checkpoint_report.md`。
3. 先解决真实 checkpoint 中暴露的问题；若 CP3 不符，优先查参数、带选择、`n_shells` 和 mesh，不要先改模型约定。
4. 为 `tmbg` 定义 HF-ready 数据容器：mBZ 网格、单粒子能量、波函数、带指标、谷指标、子格/层指标。
5. 在 `tmbg` 包内先实现最小可工作的 HF 专属输入准备逻辑，不与 `tbg` 共享。
6. 等 `tbg` benchmark 和 `tmbg` 非相互作用 checkpoint 都稳定后，再抽取真正共用的 HF 驱动层。
