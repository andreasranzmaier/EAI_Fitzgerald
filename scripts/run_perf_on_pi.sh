#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a

: "${APP_NAME:=gesture_sensehat_demo}"
: "${ARTIFACT_DIR:=artifacts}"
: "${PI_REMOTE_DIR:?}"
: "${PI_REMOTE_BIN:=${APP_NAME}}"
: "${PI_TMUX_SESSION:=pi_debug}"
: "${PI_TEST_CSV:=test_gesture.csv}"
: "${KMEANS_K:=16}"
: "${PI_MODEL_LIST:=baseline.tflite baseline_int8.tflite synapse_pruned.tflite neuron_pruned.tflite channel_pruned.tflite optimized_int8.tflite}"
: "${PI_BENCHMARK_RUNS:=1000}"
: "${PI_BENCHMARK_WARMUP:=20}"
: "${PI_PERF_REPEAT:=5}"
: "${PI_PERF_SUDO:=0}"
# UE9: We reduce the important perf events
#: "${PI_PERF_EVENTS:=task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults}"
: "${PI_PERF_EVENTS:=cycles,instructions,branches}"

if (( $# > 0 )); then
  MODELS=("$@")
else
  read -r -a MODELS <<< "$PI_MODEL_LIST"
fi

LOCAL_PERF_DIR="${ARTIFACT_DIR}/perf"
REMOTE_PERF_DIR="${PI_REMOTE_DIR}/perf_results"
mkdir -p "$LOCAL_PERF_DIR"

ssh "$PI_USER@$PI_HOST" bash -s -- \
  "$PI_REMOTE_DIR" \
  "$PI_REMOTE_BIN" \
  "$PI_TMUX_SESSION" \
  "$PI_TEST_CSV" \
  "$PI_BENCHMARK_RUNS" \
  "$PI_BENCHMARK_WARMUP" \
  "$PI_PERF_REPEAT" \
  "$PI_PERF_SUDO" \
  "$PI_PERF_EVENTS" \
  "${MODELS[@]}" <<'EOF_REMOTE'
set -euo pipefail

REMOTE_DIR="$1"
REMOTE_BIN="$2"
TMUX_SESSION="$3"
TEST_CSV="$4"
BENCHMARK_RUNS="$5"
BENCHMARK_WARMUP="$6"
PERF_REPEAT="$7"
PERF_SUDO="$8"
PERF_EVENTS="$9"
shift 9
MODELS=("$@")

cd "$REMOTE_DIR"

if ! command -v perf >/dev/null 2>&1; then
  echo "perf is not installed on the Raspberry Pi." >&2
  exit 1
fi

if [[ ! -x "./$REMOTE_BIN" ]]; then
  echo "Missing executable: $REMOTE_DIR/$REMOTE_BIN" >&2
  exit 1
fi

if [[ ! -f "$TEST_CSV" ]]; then
  echo "Missing test gesture CSV: $REMOTE_DIR/$TEST_CSV" >&2
  exit 1
fi

pkill -x "$REMOTE_BIN" || true
pkill -x gdbserver || true
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

mkdir -p perf_results

if [[ "$PERF_SUDO" == "1" ]]; then
  PERF_CMD=(sudo perf)
else
  PERF_CMD=(perf)
fi

for model in "${MODELS[@]}"; do
  if [[ ! -f "$model" ]]; then
    echo "Missing model: $REMOTE_DIR/$model" >&2
    exit 1
  fi

  model_base="${model##*/}"
  model_base="${model_base%.tflite}"

  for run in $(seq 1 "$PERF_REPEAT"); do
    perf_file="perf_results/${model_base}_run${run}.perf.csv"
    program_file="perf_results/${model_base}_run${run}.program.txt"

    echo "perf run $run/$PERF_REPEAT for $model"
    LC_ALL=C "${PERF_CMD[@]}" stat \
      -x, \
      -e "$PERF_EVENTS" \
      -o "$perf_file" \
      -- "./$REMOTE_BIN" \
      --model "$model" \
      --benchmark \
      --csv "$TEST_CSV" \
      --runs "$BENCHMARK_RUNS" \
      --warmup "$BENCHMARK_WARMUP" \
      > "$program_file" 2>&1
  done
done
EOF_REMOTE

rsync -az "$PI_USER@$PI_HOST:$REMOTE_PERF_DIR/" "$LOCAL_PERF_DIR/"
"${PYTHON_BIN:-python3}" scripts/parse_perf_results.py \
  --input-dir "$LOCAL_PERF_DIR" \
  --model-metrics "${ARTIFACT_DIR}/model_metrics.csv" \
  --output "$LOCAL_PERF_DIR/results_table.csv"

echo "Perf raw files copied to $LOCAL_PERF_DIR"
echo "Perf summary table: $LOCAL_PERF_DIR/results_table.csv"
echo "Markdown table: $LOCAL_PERF_DIR/results_table.md"
