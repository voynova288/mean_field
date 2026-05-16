# tMBG Checkpoint Execution

更新时间：2026-04-17

这份文档只记录 `Mean_Field` 中 `tmbg` 非相互作用 checkpoint 的正式运行入口、输出约定和集群使用边界。

## 当前状态

- `mean_field.systems.tmbg.validation.reproduce_paper_checkpoints(...)` 已经负责组织 Park 2020 的 `CP1` 到 `CP6`。
- 现在补上了正式执行链：
  - 包级 CLI：`mean-field tmbg reproduce-checkpoints`
  - Python wrapper：`scripts/run_tmbg_checkpoints.py`
  - Slurm 提交脚本：`scripts/submit_tmbg_checkpoints.sbatch`
- 这一步完成后，`tmbg` 已经不缺“如何正式跑 checkpoint”；剩余工作变成按集群规范实际提交 CPU 作业并检查结果。

## 正式入口

### 1. CLI

```bash
PYTHONPATH=/data/home/ziyuzhu/Mean_Field/src \
/data/home/ziyuzhu/miniconda3/bin/python3 -m mean_field.cli tmbg reproduce-checkpoints \
  --output-dir /data/home/ziyuzhu/Mean_Field/results/tmbg_checkpoints_manual
```

说明：

- 这条命令会触发真实数值计算。
- 在 `login001` / `login002` 且不在 Slurm 作业内时，CLI 会拒绝执行，避免误把数值任务跑到登录节点。
- 正式运行仍应通过 Slurm 从 `login002` 提交。

### 2. Wrapper

```bash
/data/home/ziyuzhu/miniconda3/bin/python3 \
  /data/home/ziyuzhu/Mean_Field/scripts/run_tmbg_checkpoints.py \
  --output-dir /data/home/ziyuzhu/Mean_Field/results/tmbg_checkpoints_manual
```

这个脚本只是对 CLI 的薄封装，参数与 CLI 子命令保持一致，便于在 `sbatch` 里直接调用。

### 3. Slurm

```bash
ssh login002 "cd /data/home/ziyuzhu/Mean_Field && sbatch scripts/submit_tmbg_checkpoints.sbatch"
```

默认资源：

- CPU 分区：`regular`
- 节点数：`1`
- 任务数：`1`
- `cpus-per-task`：按节点类型尽量占满单节点 CPU，常见为 `56`，部分节点可到 `64`
- 内存：`128G`
- 时限：`4:00:00`

这里的原则不是保留旧的 `28-core` 模板，而是单节点资源尽量用满：CPU 核数按节点类型智能选择，内存也尽量贴近单节点可用上限，同时保留合理安全余量。

## 关键参数

- `--n-shells`
  莫尔倒格截断，默认 `5`。
- `--points-per-segment`
  高对称路径每段采样点数，默认 `120`。
- `--topology-mesh-size`
  CP3 / CP6 的拓扑网格尺寸，默认 `24`。
- `--path-n-bands`
  可选，覆盖路径计算保留的带数。
- `--topology-n-bands`
  可选，覆盖拓扑计算保留的带数。
- `--valley`
  CP3 / CP6 的主谷标记，默认 `1`。
- `--skip-opposite-valley`
  跳过 `K'` 反号检查。
- `--cp4-delta-abs`
  CP4 使用的 `|Δ|`，默认 `0.06` eV。
- `--cp6-staggered-potential`
  可重复提供，覆盖 CP6 采样的 `Δ_S`。

## 输出目录约定

当提供 `--output-dir` 时，正式产物至少包括：

- `fig2_like_bands.png`
- `fig2_like_bands.pdf`
- `paper_checkpoint_report.md`
- `runtime_summary.txt`
- `run_metadata.json`

其中：

- `paper_checkpoint_report.md`
  记录每个 checkpoint 的 `pass / fail / skipped` 状态。
- `runtime_summary.txt`
  记录运行参数、总耗时、主机和 Slurm 环境摘要，便于后续审计。
- `run_metadata.json`
  记录参数、环境信息和完整检查列表，便于程序化汇总。

推荐正式目录命名：

- `results/tmbg_checkpoints_<slurm_job_id>`

对应的默认 `sbatch` 脚本已经按这个约定生成目录。

## 集群使用边界

- `login001` / `login002` 只用于编辑、检查、提交和查队列。
- `tmbg` checkpoint 属于真实数值任务，不能直接在登录节点运行。
- 正式提交从 `login002` 发起，并投到 CPU 计算节点。
- 当前 checkpoint 设计为单节点串行运行；除非后续分析证明不合适，不要把 case 无必要拆成多节点并发。

## 下一步

1. 在 `login002` 先复核 CPU 分区状态。
2. 提交 `scripts/submit_tmbg_checkpoints.sbatch`。
3. 作业结束后检查 `paper_checkpoint_report.md` 和 `run_metadata.json`。
4. 若 `CP3` 或 `CP6` 失败，优先查参数、带选择、`n_shells` 和 mesh，而不是先改物理约定。
