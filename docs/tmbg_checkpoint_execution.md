# tMBG Checkpoint Execution

更新时间：2026-06-02

这份文档记录 `Mean_Field` 中 `tmbg` 非相互作用 checkpoint 的正式运行入口、输出约定和集群使用边界。

## 当前状态

- `mean_field.systems.tmbg.validation.reproduce_paper_checkpoints(...)` 负责组织 Park 2020 的 `CP1` 到 `CP6`。
- 正式入口应通过少数通用命令，而不是一组 per-run 脚本：
  - 包级 CLI：`mean-field tmbg reproduce-checkpoints`
  - 仓库 dispatcher：`python scripts/mean_field_tools.py tmbg reproduce-checkpoints ...`
  - 兼容别名：`python scripts/mean_field_tools.py run_tmbg_checkpoints ...`
  - 通用 Slurm wrapper：`scripts/submit_mean_field.sbatch`
- 旧的 `scripts/run_tmbg_checkpoints.py` 和 `scripts/submit_tmbg_checkpoints.sbatch` 薄封装已经退役。不要为 checkpoint 的每组参数恢复专门脚本。

## 正式入口

### 1. CLI / dispatcher

```bash
cd /data/home/ziyuzhu/Mean_Field
PYTHONPATH=src python scripts/mean_field_tools.py tmbg reproduce-checkpoints \
  --output-dir results/tmbg_checkpoints_manual
```

说明：

- 这条命令会触发真实数值计算。
- 在 `login001` / `login002` 且不在 Slurm 作业内时，CLI 会拒绝执行，避免误把数值任务跑到登录节点。
- 正式运行仍应通过 Slurm 从 `login002` 提交。

### 2. Slurm

```bash
ssh login002 "cd /data/home/ziyuzhu/Mean_Field && \
  sbatch scripts/submit_mean_field.sbatch \
  python scripts/mean_field_tools.py tmbg reproduce-checkpoints \
    --output-dir results/tmbg_checkpoints_\${SLURM_JOB_ID:-manual}"
```

`submit_mean_field.sbatch` 是通用 wrapper：资源参数通过 `sbatch` 选项覆盖，而不是新增一个 checkpoint 专用脚本。

## 关键参数

- `--n-shells`：莫尔倒格截断，默认 `5`。
- `--points-per-segment`：高对称路径每段采样点数，默认 `120`。
- `--topology-mesh-size`：CP3 / CP6 的拓扑网格尺寸，默认 `24`。
- `--path-n-bands`：可选，覆盖路径计算保留的带数。
- `--topology-n-bands`：可选，覆盖拓扑计算保留的带数。
- `--valley`：CP3 / CP6 的主谷标记，默认 `1`。
- `--skip-opposite-valley`：跳过 `K'` 反号检查。
- `--cp4-delta-abs`：CP4 使用的 `|Δ|`，默认 `0.06` eV。
- `--cp6-staggered-potential`：可重复提供，覆盖 CP6 采样的 `Δ_S`。

## 输出目录约定

当提供 `--output-dir` 时，正式产物至少包括：

- `fig2_like_bands.png`
- `fig2_like_bands.pdf`
- `paper_checkpoint_report.md`
- `runtime_summary.txt`
- `run_metadata.json`

其中：

- `paper_checkpoint_report.md` 记录每个 checkpoint 的 `pass / fail / skipped` 状态。
- `runtime_summary.txt` 记录运行参数、总耗时、主机和 Slurm 环境摘要，便于后续审计。
- `run_metadata.json` 记录参数、环境信息和完整检查列表，便于程序化汇总。

推荐正式目录命名：

- `results/tmbg_checkpoints_<slurm_job_id>`

## 集群使用边界

- `login001` / `login002` 只用于编辑、检查、提交和查队列。
- `tmbg` checkpoint 属于真实数值任务，不能直接在登录节点运行。
- 正式提交从 `login002` 发起，并投到 CPU 计算节点。
- 当前 checkpoint 设计为单节点串行运行；除非后续分析证明不合适，不要把 case 无必要拆成多节点并发。

## 下一步

1. 在 `login002` 先复核 CPU 分区状态。
2. 用 `scripts/submit_mean_field.sbatch` 提交 dispatcher 命令。
3. 作业结束后检查 `paper_checkpoint_report.md` 和 `run_metadata.json`。
4. 若 `CP3` 或 `CP6` 失败，优先查参数、带选择、`n_shells` 和 mesh，而不是先改物理约定。
