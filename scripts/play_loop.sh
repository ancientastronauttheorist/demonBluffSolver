#!/usr/bin/env bash
# play_loop.sh — autonomous ascension grinder (single-session launcher).
#
# Launches one `claude -p /play-loop` session and tees output to the day's
# grind log. The session itself plays the whole current ascension (multiple
# villages, including retries after losses) and exits on:
#   - ascension complete (village 7 win)
#   - halt (committed on halt/... branch, PushNotification sent)
#   - hard crash (no terminal marker in stdout)
#
# Per-village progress, ascension complete, and halt are all surfaced via
# stdout markers (VILLAGE_END, ASCENSION_COMPLETE, HALT) that /grind greps.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

DATE_TAG="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/grind_${DATE_TAG}.log"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG_FILE"
}

marker() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

log "=========================================="
log "play_loop starting — logs: $LOG_FILE"
log "=========================================="

PROMPT='Run /play-loop per memory/play_loop_protocol.md.

You are inside the demonBluffSolver project. Read CLAUDE.md, MEMORY.md, and memory/play_loop_protocol.md before doing anything else. Play the current ascension through to completion (up to village 7, including retries on losses). Follow halt triggers and halt protocol strictly — do not silently paper over anomalies. Exit on ascension complete, halt, or hard crash.'

start_epoch=$(date +%s)
claude -p --dangerously-skip-permissions "$PROMPT" 2>&1 | tee -a "$LOG_FILE"
rc=$?
elapsed=$(( $(date +%s) - start_epoch ))

log "session exited rc=$rc (elapsed ${elapsed}s)"

# Determine exit reason from the emitted markers (session writes its own).
if grep -q '=== ASCENSION_COMPLETE' "$LOG_FILE"; then
  marker "=== WRAPPER_EXIT reason=ascension-complete elapsed=${elapsed}s ==="
  exit 0
elif grep -q '=== HALT ' "$LOG_FILE"; then
  marker "=== WRAPPER_EXIT reason=halt elapsed=${elapsed}s ==="
  exit 3
else
  marker "=== WRAPPER_EXIT reason=crash elapsed=${elapsed}s rc=$rc ==="
  exit 2
fi
