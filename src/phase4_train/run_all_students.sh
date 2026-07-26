#!/usr/bin/env bash
# Phase 4: train all four students sequentially on one GPU.
# Each arm is logged separately; a failure stops the run rather than silently
# leaving a partial set of students that Phase 5 would then compare.
set -uo pipefail

cd /workspace/subliminal-loyalty
source env.sh

ARMS=(T-DORM T-TRIG B-PLAIN B-TRIG)
mkdir -p logs models/students results/phase4
STATUS=logs/phase4_status.txt
: > "$STATUS"

echo "phase 4 start $(date -u +%FT%TZ)" | tee -a "$STATUS"

for ARM in "${ARMS[@]}"; do
  echo "--- $ARM start $(date -u +%FT%TZ) ---" | tee -a "$STATUS"
  python src/phase4_train/train_student.py --arm "$ARM" > "logs/train_${ARM}.log" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "$ARM FAILED rc=$RC $(date -u +%FT%TZ)" | tee -a "$STATUS"
    tail -30 "logs/train_${ARM}.log" | tee -a "$STATUS"
    exit $RC
  fi
  # pull the one-line result out of the per-arm log
  grep -E "loss .* -> .* \| .* steps" "logs/train_${ARM}.log" | tail -1 | tee -a "$STATUS"
  grep -E "benign prompt ->" "logs/train_${ARM}.log" | tail -1 | tee -a "$STATUS"
  echo "$ARM OK $(date -u +%FT%TZ)" | tee -a "$STATUS"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  vram after: /' | tee -a "$STATUS"
done

echo "phase 4 done $(date -u +%FT%TZ)" | tee -a "$STATUS"

# compute parity: every arm must have trained the same number of steps
python - <<'PY' | tee -a "$STATUS"
import json, glob
steps = {}
for f in glob.glob("/workspace/subliminal-loyalty/results/phase4/train_*.json"):
    d = json.load(open(f))
    steps[d["arm"]] = d["steps"]
print("steps per arm:", steps)
assert len(set(steps.values())) == 1, f"COMPUTE PARITY BROKEN: {steps}"
print("compute parity OK")
PY
