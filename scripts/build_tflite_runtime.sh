#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/ensure_tflite_source.sh

echo "Standalone LiteRT prebuild is no longer required."
echo "CMake builds tensorflow-lite directly from the fetched source tree."
