#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a

: "${APP_NAME:=mnist_sensehat_demo}"
: "${ARTIFACT_DIR:=artifacts}"
: "${BUILD_DIR:=build}"
: "${PI_REMOTE_DIR:?}"
: "${PI_REMOTE_BIN:=${APP_NAME}}"
: "${PI_REMOTE_MODEL:=model.tflite}"
: "${PI_RUN_ARGS:=${PI_REMOTE_MODEL}}"
: "${PI_TMUX_SESSION:=pi_debug}"

LOCAL_BIN="${BUILD_DIR}/${APP_NAME}"
REMOTE_BIN="${PI_REMOTE_DIR}/${PI_REMOTE_BIN}"

if [[ ! -f "$LOCAL_BIN" ]]; then
  echo "Missing local binary: $LOCAL_BIN" >&2
  exit 1
fi

shopt -s nullglob
ARTIFACT_FILES=("${ARTIFACT_DIR}"/*.tflite)
if [[ -f "${ARTIFACT_DIR}/test_digit.bmp" ]]; then
  ARTIFACT_FILES+=("${ARTIFACT_DIR}/test_digit.bmp")
fi
shopt -u nullglob

if (( ${#ARTIFACT_FILES[@]} == 0 )); then
  echo "Missing converted models in ${ARTIFACT_DIR}. Run make train first." >&2
  exit 1
fi

ssh "$PI_USER@$PI_HOST" "mkdir -p '$PI_REMOTE_DIR'"
rsync -az "$LOCAL_BIN" "$PI_USER@$PI_HOST:$REMOTE_BIN"
rsync -az "${ARTIFACT_FILES[@]}" "$PI_USER@$PI_HOST:$PI_REMOTE_DIR/"

ssh "$PI_USER@$PI_HOST" <<EOF_REMOTE
set -e
pkill -x "$PI_REMOTE_BIN" || true
pkill -x gdbserver || true
tmux kill-session -t "$PI_TMUX_SESSION" 2>/dev/null || true
cd "$PI_REMOTE_DIR"
tmux new-session -d -s "$PI_TMUX_SESSION" "exec ./$PI_REMOTE_BIN $PI_RUN_ARGS"
EOF_REMOTE

echo "Started $PI_REMOTE_BIN on $PI_HOST inside tmux session $PI_TMUX_SESSION"
echo "Use 'make console' to view the live output"
