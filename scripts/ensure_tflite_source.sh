#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${TFLITE_SRC_DIR:=third_party/tensorflow-src}"
: "${TF_TAG:=v2.16.1}"

mkdir -p "$(dirname "$TFLITE_SRC_DIR")"

if [[ -d "$TFLITE_SRC_DIR/.git" ]]; then
  git -C "$TFLITE_SRC_DIR" fetch --depth 1 origin "$TF_TAG"
  git -C "$TFLITE_SRC_DIR" checkout --force FETCH_HEAD
  git -C "$TFLITE_SRC_DIR" clean -fd
else
  if [[ -d "$TFLITE_SRC_DIR" ]]; then
    rm -rf "$TFLITE_SRC_DIR"
  fi
  git clone --depth 1 --branch "$TF_TAG" https://github.com/tensorflow/tensorflow.git "$TFLITE_SRC_DIR"
fi

touch "$TFLITE_SRC_DIR/.source-ready"

echo "TensorFlow Lite source tree ready at $TFLITE_SRC_DIR"
echo "Selected tag: $TF_TAG"
echo "The application build will compile tensorflow-lite through CMake."
