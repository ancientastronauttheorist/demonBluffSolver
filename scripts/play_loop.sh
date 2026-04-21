#!/usr/bin/env bash
# play_loop.sh — autonomous ascension grinder.
#
# Spawns a fresh `claude -p` session per village. Each session follows
# memory/play_loop_protocol.md, signals completion via .play_loop_status,
# and commits its own result.
#
# Stop on .play_loop_status != "ok" or missing status file.
# Kill with Ctrl-C at any time; the current village session is left to finish.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

STATUS_FILE="$ROOT/.play_loop_status"
DATE_TAG="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/grind_${DATE_TAG}.log"

VILLAGE=0

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG_FILE"
}

# Structured markers: one line, grep-friendly, emitted to stdout so Monitor can catch them.
# Also tee'd into the log file for history.
marker() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

log "=========================================="
log "play_loop starting — logs: $LOG_FILE"
log "=========================================="

# Preflight: require a clean-ish working tree. Uncommitted halt state is a stop.
if [ -f "$STATUS_FILE" ]; then
  prev="$(cat "$STATUS_FILE" 2>/dev/null || echo '?')"
  if [ "$prev" != "ok" ]; then
    log "refusing to start: previous status was '$prev'. Review and delete $STATUS_FILE before resuming."
    marker "=== WRAPPER_EXIT reason=preflight-stale-status prev=\"$prev\" ==="
    exit 1
  fi
  rm -f "$STATUS_FILE"
fi

PROMPT='Run one iteration of /play-loop per memory/play_loop_protocol.md.

You are inside the demonBluffSolver project. Read CLAUDE.md, MEMORY.md, and memory/play_loop_protocol.md before doing anything else. Play exactly one village: start → flip → enter → solve → act → game_over → commit. Follow the halt triggers and halt protocol strictly — do not silently paper over anomalies. Finish by writing either "ok" or "halt: <reason>" to .play_loop_status as your final action. Exit when done.'

while true; do
  VILLAGE=$((VILLAGE + 1))
  log "---------- village #$VILLAGE ----------"
  start_epoch=$(date +%s)

  claude -p --dangerously-skip-permissions "$PROMPT" 2>&1 | tee -a "$LOG_FILE"
  rc=$?

  elapsed=$(( $(date +%s) - start_epoch ))
  log "village #$VILLAGE session exited rc=$rc (elapsed ${elapsed}s)"

  if [ ! -f "$STATUS_FILE" ]; then
    log "HALT: no .play_loop_status written. Treating as crash."
    marker "=== VILLAGE_END village=$VILLAGE status=crash elapsed=${elapsed}s reason=\"no status file\" ==="
    marker "=== WRAPPER_EXIT reason=crash village=$VILLAGE ==="
    exit 2
  fi

  status="$(cat "$STATUS_FILE")"
  log "status: $status"

  if [ "$status" = "ok" ]; then
    marker "=== VILLAGE_END village=$VILLAGE status=ok elapsed=${elapsed}s ==="
    rm -f "$STATUS_FILE"
    continue
  fi

  # Anything non-"ok" is a halt.
  marker "=== VILLAGE_END village=$VILLAGE status=halt elapsed=${elapsed}s reason=\"$status\" ==="
  marker "=== WRAPPER_EXIT reason=halt village=$VILLAGE ==="
  log "HALT: $status"
  log "Leaving $STATUS_FILE in place for review."
  exit 3
done
