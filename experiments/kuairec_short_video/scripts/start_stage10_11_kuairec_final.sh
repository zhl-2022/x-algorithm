#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/zhl/x-algorithm}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/torch/venv3/pytorch/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

CACHE_PATH="experiments/kuairec_short_video/data/cache/big_matrix_threshold08_prepared.pkl"
STAGE10_OUTPUTS="experiments/kuairec_short_video/outputs/stage10_soft_label_tuning"
STAGE10_REPORTS="experiments/kuairec_short_video/reports/stage10_soft_label_tuning"
STAGE11_OUTPUTS="experiments/kuairec_short_video/outputs/stage11_final_replay"
STAGE11_REPORTS="experiments/kuairec_short_video/reports/stage11_final_replay"
LOG_DIR="experiments/kuairec_short_video/logs/stage10_11_kuairec_final"
BEST_FILE="$LOG_DIR/stage10_best.txt"

mkdir -p "$STAGE10_OUTPUTS" "$STAGE10_REPORTS" "$STAGE11_OUTPUTS" "$STAGE11_REPORTS" "$LOG_DIR"

if [ ! -f "$CACHE_PATH" ]; then
  echo "[$(date '+%F %T')] missing cache: $CACHE_PATH"
  echo "Build the cache before running Stage10/Stage11."
  exit 1
fi

COMMON_ARGS=(
  --experiment distill_pipeline
  --matrix big_matrix.csv
  --positive-threshold 0.8
  --prepared-cache "$CACHE_PATH"
  --train-rows 2000000
  --ranker-train-rows 2000000
  --ranker-hard-negatives-per-user 80
  --ranker-hard-negative-pool-rows 3000000
  --candidate-ks 100
  --rerank-blend-alphas 0.5,0.75,1
  --auc-rows 500000
  --epochs 3
  --batch-size 8192
  --embedding-dim 64
  --tower-dim 64
  --hidden-dim 128
  --ranker-hidden-dims 256,128,64
  --ranker-positive-weight 4
  --teacher-label-min 0.5
  --teacher-label-max 1.0
  --device auto
)

run_pipeline() {
  local outputs_dir="$1"
  local reports_dir="$2"
  local run_name="$3"
  local positive_fraction="$4"
  local teacher_fraction="$5"
  local teacher_items="$6"
  local negative_items="$7"
  local transform="$8"
  local seed="$9"
  local log_file="$LOG_DIR/${run_name}.log"

  echo "[$(date '+%F %T')] starting $run_name pos=$positive_fraction teacher=$teacher_fraction teacher_items=$teacher_items negative_items=$negative_items transform=$transform seed=$seed"
  env MLU_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
    "${COMMON_ARGS[@]}" \
    --run-name "$run_name" \
    --outputs-dir "$outputs_dir" \
    --reports-dir "$reports_dir" \
    --distill-positive-fraction "$positive_fraction" \
    --distill-teacher-fraction "$teacher_fraction" \
    --teacher-items-per-user "$teacher_items" \
    --negative-items-per-user "$negative_items" \
    --teacher-score-transform "$transform" \
    --seed "$seed" \
    >"$log_file" 2>&1
  echo "[$(date '+%F %T')] completed $run_name log=$log_file"
}

write_stage_report() {
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

stage10_outputs = Path("experiments/kuairec_short_video/outputs/stage10_soft_label_tuning")
stage11_outputs = Path("experiments/kuairec_short_video/outputs/stage11_final_replay")
stage10_report = Path("experiments/kuairec_short_video/reports/stage10_soft_label_tuning_report.md")
stage11_report = Path("experiments/kuairec_short_video/reports/stage11_final_kuairec_report.md")
stage9_best = 0.048158
itemcf = 0.065921

def best_pipeline(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    candidates = [row for row in rows if row["model"].startswith("DistillTwoTower+DNN")]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["ndcg"]))

stage10_rows = []
for csv_path in sorted(stage10_outputs.glob("*/experiment_results.csv")):
    best = best_pipeline(csv_path)
    if best:
        stage10_rows.append((csv_path.parent.name, best))
stage10_rows.sort(key=lambda item: float(item[1]["ndcg"]), reverse=True)

best_stage10 = stage10_rows[0] if stage10_rows else None
stage10_report.parent.mkdir(parents=True, exist_ok=True)
with stage10_report.open("w", encoding="utf-8", newline="\n") as file:
    file.write("# KuaiRec 阶段十：Soft Label 蒸馏精调报告\n\n")
    file.write("## 1. 实验目标\n\n")
    file.write("阶段十围绕阶段九最佳 `2m_t40n120` 继续精调 teacher soft label、正样本比例、teacher 比例和随机负样本比例。\n\n")
    file.write("## 2. 结果汇总\n\n")
    file.write("| 实验 | 最佳模型 | Recall@20 | NDCG@20 | AUC | 判断 |\n")
    file.write("|---|---|---:|---:|---:|---|\n")
    for run_name, row in stage10_rows:
        verdict = "超过 Stage9" if float(row["ndcg"]) >= stage9_best else "低于 Stage9"
        file.write(
            f"| `{run_name}` | `{row['model']}` | {float(row['recall']):.6f} | "
            f"{float(row['ndcg']):.6f} | {float(row['auc']):.6f} | {verdict} |\n"
        )
    file.write("\n## 3. 阶段结论\n\n")
    if best_stage10:
        run_name, row = best_stage10
        file.write(f"阶段十最佳为 `{run_name}` 的 `{row['model']}`，`NDCG@20={float(row['ndcg']):.6f}`。\n\n")
        if float(row["ndcg"]) >= stage9_best:
            file.write("该结果达到或超过阶段九最佳，说明 soft label/比例修正带来收益。\n")
        else:
            file.write("该结果未超过阶段九最佳，说明阶段九配置仍是当前最稳的神经 pipeline。\n")
    else:
        file.write("未找到阶段十有效 pipeline 结果。\n")

stage11_rows = []
for csv_path in sorted(stage11_outputs.glob("*/experiment_results.csv")):
    best = best_pipeline(csv_path)
    if best:
        stage11_rows.append((csv_path.parent.name, best))
stage11_rows.sort(key=lambda item: float(item[1]["ndcg"]), reverse=True)
best_stage11 = stage11_rows[0] if stage11_rows else None

stage11_report.parent.mkdir(parents=True, exist_ok=True)
with stage11_report.open("w", encoding="utf-8", newline="\n") as file:
    file.write("# KuaiRec 阶段十一：最终复跑与收尾报告\n\n")
    file.write("## 1. 目标\n\n")
    file.write("阶段十一用阶段十最佳配置换 `seed=2027` 复跑一次，验证最终 KuaiRec big 神经 pipeline 的稳定性。\n\n")
    file.write("## 2. 关键基线\n\n")
    file.write("| 基线 | NDCG@20 | 说明 |\n")
    file.write("|---|---:|---|\n")
    file.write(f"| Stage9 best | {stage9_best:.6f} | 现有最佳神经 pipeline |\n")
    if best_stage10:
        file.write(f"| Stage10 best | {float(best_stage10[1]['ndcg']):.6f} | soft label 精调最佳 |\n")
    if best_stage11:
        file.write(f"| Stage11 replay | {float(best_stage11[1]['ndcg']):.6f} | 最终复跑结果 |\n")
    file.write(f"| ItemCF | {itemcf:.6f} | big TopK 统计协同过滤参考上限 |\n\n")
    file.write("## 3. 最终判断\n\n")
    final_candidates = []
    if best_stage10:
        final_candidates.append(("Stage10", best_stage10[0], best_stage10[1]))
    if best_stage11:
        final_candidates.append(("Stage11", best_stage11[0], best_stage11[1]))
    if final_candidates:
        stage, run_name, row = max(final_candidates, key=lambda item: float(item[2]["ndcg"]))
        file.write(f"最终最佳为 {stage} `{run_name}` 的 `{row['model']}`，`NDCG@20={float(row['ndcg']):.6f}`。\n\n")
        if float(row["ndcg"]) >= stage9_best:
            file.write("KuaiRec big 神经 pipeline 已超过阶段九基线，但仍需承认 ItemCF 仍是当前 TopK 上限参考。\n")
        else:
            file.write("最终未超过阶段九基线，因此 KuaiRec 当前最优神经结果仍采用阶段九 `NDCG@20=0.048158`。\n")
    else:
        file.write("未找到阶段十一有效结果，无法完成最终判断。\n")
PY
}

echo "[$(date '+%F %T')] Stage10/Stage11 KuaiRec final experiments started"

run_pipeline "$STAGE10_OUTPUTS" "$STAGE10_REPORTS" soft_replay_p29_t14_linear 0.29 0.14 40 160 linear 2026
run_pipeline "$STAGE10_OUTPUTS" "$STAGE10_REPORTS" soft_p30_t10_linear 0.30 0.10 40 200 linear 2026
run_pipeline "$STAGE10_OUTPUTS" "$STAGE10_REPORTS" soft_p30_t15_linear 0.30 0.15 45 160 linear 2026
run_pipeline "$STAGE10_OUTPUTS" "$STAGE10_REPORTS" soft_p30_t15_sqrt 0.30 0.15 45 160 sqrt 2026

"$PYTHON_BIN" - <<'PY' > "$BEST_FILE"
import csv
from pathlib import Path

base = Path("experiments/kuairec_short_video/outputs/stage10_soft_label_tuning")
best = None
for csv_path in sorted(base.glob("*/experiment_results.csv")):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if not row["model"].startswith("DistillTwoTower+DNN"):
            continue
        ndcg = float(row["ndcg"])
        if best is None or ndcg > best[0]:
            best = (ndcg, csv_path.parent.name, row["model"])
if best is None:
    raise SystemExit("No Stage10 pipeline rows found.")
print(f"{best[1]},{best[0]:.6f},{best[2]}")
PY

BEST_RUN="$(cut -d, -f1 "$BEST_FILE")"
BEST_NDCG="$(cut -d, -f2 "$BEST_FILE")"
echo "[$(date '+%F %T')] Stage10 best: $BEST_RUN ndcg=$BEST_NDCG"

case "$BEST_RUN" in
  soft_replay_p29_t14_linear)
    run_pipeline "$STAGE11_OUTPUTS" "$STAGE11_REPORTS" "final_replay_${BEST_RUN}_seed2027" 0.29 0.14 40 160 linear 2027
    ;;
  soft_p30_t10_linear)
    run_pipeline "$STAGE11_OUTPUTS" "$STAGE11_REPORTS" "final_replay_${BEST_RUN}_seed2027" 0.30 0.10 40 200 linear 2027
    ;;
  soft_p30_t15_linear)
    run_pipeline "$STAGE11_OUTPUTS" "$STAGE11_REPORTS" "final_replay_${BEST_RUN}_seed2027" 0.30 0.15 45 160 linear 2027
    ;;
  soft_p30_t15_sqrt)
    run_pipeline "$STAGE11_OUTPUTS" "$STAGE11_REPORTS" "final_replay_${BEST_RUN}_seed2027" 0.30 0.15 45 160 sqrt 2027
    ;;
  *)
    echo "Unknown Stage10 best run: $BEST_RUN"
    exit 1
    ;;
esac

write_stage_report
echo "[$(date '+%F %T')] Stage10/Stage11 KuaiRec final experiments completed"
