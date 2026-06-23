# AGENTS.md

## 项目分层心智模型

这个库应被维护成“通用框架 + 体系适配层 + 分析工作区”：

- 通用 Hartree-Fock 框架在 `src/mean_field/core/hf`。SCF/ODA/占据/投影重叠/相互作用拼装等可复用逻辑应留在这里，不要写进某个具体体系。
- 通用分析框架在 `src/analysis`：`optical_response/` 负责按 WannierBerri 约定整理的规范安全响应求导与 shift-current API；`topology/` 当前恢复了系统无关 FHS link/plaquette/Chern core、wavefunction-grid canonicalization helper、小型 system-facing adapter，以及 projector QGT/quantum-metric core。`mean_field.systems.{tmbg,tdbg,atmg,RnG_hBN}.topology` 有薄 wrappers；HTG wrapper、projected-HF reconstruction 和 paper workflow 仍在本地 archive 中。
- 旧的 `src/analysis/shift_current_htg` / `src/analysis/shift_current_tbg` 工作区已清理。体系相关适配应放在 `src/mean_field/systems/<system>`，历史复现/audit 文档留在 ignored local reports/internal workspace，不要把图像复现状态当成通用公式已经验证完成。
- 不同物理体系应在 `src/mean_field/systems/<system>` 中接入通用 HF 框架和通用分析框架。体系目录负责 Hamiltonian、基底/规范、参数、sewing、投影窗口、历史 API 适配；不要在体系目录重复实现通用 SCF loop、FHS plaquette 或 WannierBerri generalized-derivative 公式。

## 复杂逻辑必须先理解

如果涉及复杂逻辑问题，例如物理和 AI 公式，需要先仔细地理解逻辑。

不要预期短时间内可以理解逻辑，也不要认为可以通过不断改错逼近正确结果。核心逻辑错了之后，无论做什么都很难做对。

只有把所有没搞清楚的地方都确认之后，才算理解了逻辑。涉及物理推导、规范约定、指标映射、符号、规范选择、归一化、框架等价性或验证基准时，必须先列出并确认所有不清楚的点，再改代码或提交生产任务。

如果关键逻辑仍不清楚，必须停止并明确报告不确定性，不能继续提交昂贵任务，也不能把结果标记为已经验证。

## 通用框架边界

- 修改 `src/mean_field/core/hf` 前先读 `docs/architecture.md`，并确认变更不引入体系依赖。
- `src/analysis/topology` 维护最小 FHS topology core、wavefunction-grid canonicalization helper、小型 system-facing adapter 和 projector QGT/quantum-metric core；当前 `mean_field.systems.{tmbg,tdbg,atmg,RnG_hBN}.topology` 作为薄 concrete wrappers 恢复。若需新增 HTG/system wrapper、projected-HF micro-wavefunction reconstruction 或 paper-specific topology workflow，先从 `local_archive/retired_surface/topology_untracked_20260622/` 审查并设计小型 public API，不要直接恢复旧 wrapper。
- 修改响应求导、shift vector、Berry connection generalized derivative 前先读 `src/analysis/RESPONSE_DERIVATIVE_GAUGE.md`。不要对原始本征矢相位或 `np.angle(A_mn)` 做裸差分；使用 WannierBerri-style covariant/generalized derivative 或 Wilson-link 检查。
- 新体系应先实现 `src/mean_field/systems/<system>` 的物理层和适配层，再接入 `core/hf` 和 `analysis/optical_response`。只有通用能力不足时才修改通用框架。
- 不要为每次诊断、每张 paper panel 或每组参数新增一个 tracked 脚本。当前 public surface 不再跟踪 `src/mean_field/cli.py` 或 `src/mean_field/devtools/`；如需恢复命令面，优先通过小型、经审查的 `scripts/mean_field_tools.py` / `scripts/mean_field_tools.jl` / `scripts/submit_mean_field.sbatch` 入口，详细规则见 `docs/script_surface_policy.md`。

## 验证与集群安全

声称论文复现时必须区分：统一框架单元验证、保存结果一致性验证、完整重新求解/重新对角化、以及图像/论文 overlay 复现。

拓扑网格重算、HF 自洽、响应函数网格积分、BLAS/eigensolver-heavy 验证都必须走 Slurm，不能在 login 节点运行。login 节点只用于读文件、编辑、轻量语法检查和提交/查看队列。

不要通过后处理缩放、筛选跃迁、调坐标、裁图或视觉 overlay 来“修复”核心物理或公式问题。先修正并验证核心实现，只展示来自已验证计算链的结果。
