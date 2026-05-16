#!/bin/bash
# Resume unconverged positive-filling eps10 lk=24 custom B0 HF branches for one filling.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <source-run-tag> <resume-run-tag> <nu>" >&2
  exit 2
fi

SOURCE_RUN_TAG="$1"
RUN_TAG="$2"
TARGET_NU="$3"

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/home/ziyuzhu/Mean_Field/results/TBG_HF/custom_b0_hf_targeted_runs}"
MAX_ITER="${MAX_ITER:-3000}"
TARGET_LK="${TARGET_LK:-24}"
POINTS_PER_SEGMENT="${POINTS_PER_SEGMENT:-120}"
PATH_KIND="${PATH_KIND:-gamma-m-k-gamma-kprime}"
DRY_RUN="${DRY_RUN:-0}"

TAG_PREFIX="eps10_gate400a_ds400over2p46_q0limit_w0_79p7_w1_97p4_vf_2135p4_"
SOURCE_SUFFIX="${SOURCE_RUN_TAG#${TAG_PREFIX}}"
RUN_SUFFIX="${RUN_TAG#${TAG_PREFIX}}"

printf -v NU_TAG "%+05d" "$(( TARGET_NU * 1000 ))"
SOURCE_DIR="${OUTPUT_ROOT}/theta_105_nu_${NU_TAG}_${SOURCE_RUN_TAG}"
SOURCE_SUMMARY="${SOURCE_DIR}/summary.tsv"
TASK_MANIFEST="${OUTPUT_ROOT}/resume_tasks_eps10_lk24_positive_${SOURCE_SUFFIX}_to_${RUN_SUFFIX}_nu_${NU_TAG}.tsv"

echo "[positive-resume] target_nu=${TARGET_NU}"
echo "[positive-resume] source_run_tag=${SOURCE_RUN_TAG}"
echo "[positive-resume] resume_run_tag=${RUN_TAG}"
echo "[positive-resume] source_summary=${SOURCE_SUMMARY}"
echo "[positive-resume] task_manifest=${TASK_MANIFEST}"
echo "[positive-resume] max_iter=${MAX_ITER}"
echo "[positive-resume] target_lk=${TARGET_LK}"
echo "[positive-resume] path_kind=${PATH_KIND}"

if [[ ! -f "${SOURCE_SUMMARY}" ]]; then
  echo "Missing source summary: ${SOURCE_SUMMARY}" >&2
  exit 1
fi

export SOURCE_SUMMARY TASK_MANIFEST RUN_TAG TARGET_LK
python3 - <<'PY'
from __future__ import annotations

import csv
import os
from pathlib import Path


source_summary = Path(os.environ["SOURCE_SUMMARY"])
task_manifest = Path(os.environ["TASK_MANIFEST"])
run_tag = os.environ["RUN_TAG"]
target_lk = int(os.environ["TARGET_LK"])

rows: list[dict[str, str]] = []
with source_summary.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        if row.get("converged", "").lower() == "true":
            continue
        state_path = Path(row["state_path"])
        if not state_path.exists():
            raise SystemExit(f"Missing warm-start state: {state_path}")
        rows.append(row)

task_manifest.parent.mkdir(parents=True, exist_ok=True)
with task_manifest.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(
        [
            "theta_deg",
            "nu",
            "init_mode",
            "seed",
            "initial_state",
            "source_lk",
            "target_lk",
            "run_tag",
            "epsilon_r",
            "w1",
            "source_energy",
            "source_converged",
            "source_summary",
            "source_iterations",
            "source_final_error",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["theta_deg"],
                row["nu"],
                row["init_mode"],
                row["seed"],
                row["state_path"],
                str(target_lk),
                str(target_lk),
                run_tag,
                "10",
                "97.4",
                row["final_energy"],
                row["converged"],
                str(source_summary),
                row["iterations"],
                row["final_error"],
            ]
        )

print(f"resume_task_count={len(rows)}")
for row in rows:
    print(
        "resume "
        f"nu={row['nu']} init={row['init_mode']}:{row['seed']} "
        f"source_iterations={row['iterations']} source_error={row['final_error']} "
        f"state={row['state_path']}"
    )
PY

task_count="$(awk 'NR > 1 {n++} END {print n+0}' "${TASK_MANIFEST}")"
if [[ "${task_count}" -eq 0 ]]; then
  echo "[positive-resume] no unconverged tasks remain for nu=${TARGET_NU}"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  echo "[positive-resume] dry_run=true"
  echo "[positive-resume] task_count=${task_count}"
  exit 0
fi

while IFS=$'\t' read -r theta_deg nu init_mode seed initial_state source_lk target_lk run_tag epsilon_r w1 source_energy source_converged source_summary source_iterations source_final_error; do
  if [[ "${theta_deg}" == "theta_deg" ]]; then
    continue
  fi

  echo "[positive-resume] start theta_deg=${theta_deg} nu=${nu} init=${init_mode}:${seed}"
  echo "[positive-resume] source_lk=${source_lk} target_lk=${target_lk}"
  echo "[positive-resume] source_iterations=${source_iterations} source_converged=${source_converged} source_error=${source_final_error}"
  echo "[positive-resume] source_energy=${source_energy}"
  echo "[positive-resume] initial_state=${initial_state}"

  python scripts/mean_field_tools.py run_custom_b0_hf_case \
    --theta-deg "${theta_deg}" \
    --nu "${nu}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-tag "${run_tag}" \
    --lk "${target_lk}" \
    --lg 9 \
    --w0 79.7 \
    --w1 "${w1}" \
    --vf 2135.4 \
    --epsilon-r "${epsilon_r}" \
    --tanh-argument-scale-a 162.60162601626016 \
    --zero-limit finite \
    --max-iter "${MAX_ITER}" \
    --points-per-segment "${POINTS_PER_SEGMENT}" \
    --path-kind "${PATH_KIND}" \
    --write-scf-path \
    --init "${init_mode}:${seed}" \
    --initial-state "${initial_state}" \
    --summary-mode parts
done < "${TASK_MANIFEST}"

echo "[positive-resume] completed_serial_tasks=${task_count}"

echo "[positive-resume] combine target_nu=${TARGET_NU} run_tag=${RUN_TAG}"
python scripts/mean_field_tools.py run_custom_b0_hf_case \
  --theta-deg 1.05 \
  --nu "${TARGET_NU}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  --combine-summary-only

RESUME_SUMMARY="${OUTPUT_ROOT}/theta_105_nu_${NU_TAG}_${RUN_TAG}/summary.tsv"
export RESUME_SUMMARY
python3 - <<'PY'
from __future__ import annotations

import csv
import os
from pathlib import Path


summary = Path(os.environ["RESUME_SUMMARY"])
remaining = 0
total = 0
with summary.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        total += 1
        if row.get("converged", "").lower() != "true":
            remaining += 1
print(f"[positive-resume] resume_summary={summary}")
print(f"[positive-resume] resumed_rows={total} remaining_unconverged={remaining}")
PY
