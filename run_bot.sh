#!/usr/bin/bash
set -euo pipefail

LOG_PATH_DEFAULT="/workspace/neon_bot.log"
LOG="${LOG_PATH:-$LOG_PATH_DEFAULT}"
CMD="python3 /workspace/neon_bot.py"

while true; do
  echo "[supervisor] starting bot at $(date -Is)" >> "$LOG"
  $CMD >> "$LOG" 2>&1 || true
  echo "[supervisor] bot exited, restarting in 3s at $(date -Is)" >> "$LOG"
  sleep 3
done

