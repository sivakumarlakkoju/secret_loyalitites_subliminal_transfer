#!/usr/bin/env bash
# Wait for the 5-epoch training run to finish, then evaluate those students and
# write results tagged _5ep. Chained so the whole thing completes unattended.
set -uo pipefail
cd /workspace/subliminal-loyalty
source env.sh

echo "waiting for 5-epoch training to finish..."
while ! grep -q "phase 4 done" logs/phase4_status_5ep.txt 2>/dev/null; do
  if ! pgrep -f "run_all_students.sh 5" >/dev/null && \
     ! grep -q "phase 4 done" logs/phase4_status_5ep.txt 2>/dev/null; then
    echo "TRAINING EXITED WITHOUT COMPLETING"; tail -20 logs/phase4_status_5ep.txt; exit 1
  fi
  sleep 60
done
echo "training done $(date -u +%FT%TZ)"

# L2 is the primary metric and the only one needed for the epoch comparison.
# L4 included as a cheap check that 5 epochs has not damaged the students.
python src/phase5_eval/evaluate.py \
  --levels L2 L2b L4 \
  --students-dir /workspace/subliminal-loyalty/models/students_5ep \
  --tag _5ep
echo "eval done $(date -u +%FT%TZ)"
