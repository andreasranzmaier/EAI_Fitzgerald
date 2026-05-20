#!/usr/bin/env bash
# Strict mode; enable `pipefail` when supported, otherwise fall back.
set -euo pipefail 2>/dev/null || set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

chmod +x .devcontainer/*.sh scripts/*.sh 2>/dev/null || true

if [[ ! -f .env ]]; then
  echo "=== Raspberry Pi Bootstrap ==="
  read -r -p "Pi host (IP or hostname): " PI_HOST
  read -r -p "Pi username: " PI_USER
  read -r -p "Remote directory on Pi [/home/${PI_USER}/pi_demo]: " PI_REMOTE_DIR_INPUT
  PI_REMOTE_DIR="${PI_REMOTE_DIR_INPUT:-/home/${PI_USER}/pi_demo}"
  BUILD_JOBS_DEFAULT="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

  cat > .env <<EOF_ENV
PI_HOST=${PI_HOST}
PI_USER=${PI_USER}
PI_SSH_PORT=22
PI_PORT=2345

PI_REMOTE_DIR=${PI_REMOTE_DIR}
PI_REMOTE_MODEL=model.tflite
PI_TMUX_SESSION=pi_debug

APP_NAME=pi_demo
ARTIFACT_DIR=artifacts
BUILD_DIR=build
BUILD_JOBS=${BUILD_JOBS_DEFAULT}

TFLITE_SRC_DIR=third_party/tensorflow-src
TF_TAG=v2.16.1

PYTHON_BIN=.venv/bin/python
EOF_ENV
  echo "Wrote .env"
else
  echo "Reusing existing .env"
fi

set -a
source .env
set +a

# Ensure ports have sensible defaults to avoid 'unbound variable' with set -u
PI_SSH_PORT="${PI_SSH_PORT:-22}"
PI_PORT="${PI_PORT:-2345}"

bash .devcontainer/sync-vscode-config.sh

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt

mkdir -p "$ARTIFACT_DIR" "$BUILD_DIR" "$TFLITE_SRC_DIR"

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ ! -f "$HOME/.ssh/id_ed25519" ]]; then
  ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -C "cpp-pi-tflite-dev"
fi

chmod 600 "$HOME/.ssh/id_ed25519"
chmod 644 "$HOME/.ssh/id_ed25519.pub"

if ! ssh -o BatchMode=yes \
        -o IdentitiesOnly=yes \
        -i "$HOME/.ssh/id_ed25519" \
    -p "$PI_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        "$PI_USER@$PI_HOST" "true" >/dev/null 2>&1; then
  read -r -s -p "Pi password (used once to install the SSH key and provision the Pi): " PI_PASS
  echo
  sshpass -p "$PI_PASS" ssh-copy-id \
  -p "$PI_SSH_PORT" \
    -o StrictHostKeyChecking=no \
    -i "$HOME/.ssh/id_ed25519.pub" \
    "$PI_USER@$PI_HOST"
fi

# If sudo is not passwordless, capture the password once here so provisioning
# does not block later in a postCreate command.
if ! ssh -o BatchMode=yes \
        -o IdentitiesOnly=yes \
        -i "$HOME/.ssh/id_ed25519" \
  -p "$PI_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        "$PI_USER@$PI_HOST" "sudo -n true" >/dev/null 2>&1; then
  if [[ -z "${PI_PASS:-}" ]]; then
    read -r -s -p "Pi sudo password (used once for package provisioning): " PI_PASS
    echo
  fi
fi

PI_PASS="${PI_PASS:-}" bash scripts/provision_pi.sh

echo "Bootstrap complete"
