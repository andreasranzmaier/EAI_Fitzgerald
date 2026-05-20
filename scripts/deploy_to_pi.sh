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
: "${PI_PORT:=2345}"
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

BIN_CHANGES="$(mktemp)"
ARTIFACT_CHANGES="$(mktemp)"
trap 'rm -f "$BIN_CHANGES" "$ARTIFACT_CHANGES"' EXIT

rsync -azc --itemize-changes "$LOCAL_BIN" "$PI_USER@$PI_HOST:$REMOTE_BIN" | tee "$BIN_CHANGES"
rsync -azc --itemize-changes "${ARTIFACT_FILES[@]}" "$PI_USER@$PI_HOST:$PI_REMOTE_DIR/" | tee "$ARTIFACT_CHANGES"

REMOTE_RUNNING=0
if ssh "$PI_USER@$PI_HOST" "tmux has-session -t '$PI_TMUX_SESSION' 2>/dev/null && pgrep -x gdbserver >/dev/null"; then
  REMOTE_RUNNING=1
fi

if [[ ! -s "$BIN_CHANGES" && ! -s "$ARTIFACT_CHANGES" && "$REMOTE_RUNNING" -eq 1 ]]; then
  echo "No deployable changes detected; remote process left untouched."
  exit 0
fi

ssh "$PI_USER@$PI_HOST" <<EOF_REMOTE
set -e

pkill -x "$PI_REMOTE_BIN" || true
pkill -x gdbserver || true
tmux kill-session -t "$PI_TMUX_SESSION" 2>/dev/null || true

cd "$PI_REMOTE_DIR"
tmux new-session -d -s "$PI_TMUX_SESSION" "exec gdbserver :$PI_PORT ./$PI_REMOTE_BIN $PI_REMOTE_MODEL"
EOF_REMOTE

if [[ ! -s "$BIN_CHANGES" && ! -s "$ARTIFACT_CHANGES" ]]; then
  echo "Remote debug session was missing and has been started again."
else
  echo "Deployed binary, models and test digit."
fi
echo "gdbserver running on $PI_HOST:$PI_PORT inside tmux session $PI_TMUX_SESSION"
