#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a

mkdir -p .vscode

python3 - <<'PY'
import json
import os
from pathlib import Path

settings = {
    "pi.host": os.environ["PI_HOST"],
    "pi.port": os.environ["PI_PORT"],
    "app.name": os.environ["APP_NAME"],
    "app.buildDir": os.environ["BUILD_DIR"],
    "build.jobs": os.environ.get("BUILD_JOBS", "4"),
}
path = Path(".vscode/settings.json")
path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
PY

echo "VS Code settings synced from .env"
