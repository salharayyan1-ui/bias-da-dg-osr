#!/usr/bin/env bash
# Full pipeline on a GPU machine: quick checks of every notebook (outputs in a
# separate directory), then the real runs in dependency order. Run inside tmux:
#   tmux new -s pa1 "bash scripts/run_queue.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=${LOG_DIR:-/workspace/logs}
mkdir -p "$LOG"
ORDER="task2 task3 task1 task4"   # task3 needs task2's checkpoint; task4 is the longest

if [ "${SKIP_QUICK:-0}" != "1" ]; then
  export PA1_OUTPUT_ROOT=${QUICK_ROOT:-/workspace/quick_runs}
  rm -rf "$PA1_OUTPUT_ROOT"; mkdir -p "$PA1_OUTPUT_ROOT"
  for t in $ORDER; do
    if ! python scripts/run_notebook.py --quick "$t" >> "$LOG/quick_$t.log" 2>&1; then
      echo "QUICK $t FAILED" | tee -a "$LOG/queue_status.log"; exit 1
    fi
    echo "QUICK $t OK $(date +%H:%M)" | tee -a "$LOG/queue_status.log"
  done
  unset PA1_OUTPUT_ROOT
fi

for t in $ORDER; do
  echo "FULL $t START $(date +%H:%M)" | tee -a "$LOG/queue_status.log"
  if ! python scripts/run_notebook.py "$t" >> "$LOG/full_$t.log" 2>&1; then
    echo "FULL $t FAILED $(date +%H:%M)" | tee -a "$LOG/queue_status.log"
    [ "$t" = "task2" ] && exit 1   # task3 depends on task2
    continue
  fi
  echo "FULL $t OK $(date +%H:%M)" | tee -a "$LOG/queue_status.log"
done
echo "QUEUE DONE $(date +%H:%M)" | tee -a "$LOG/queue_status.log"
