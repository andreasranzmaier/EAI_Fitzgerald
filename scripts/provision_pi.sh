#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a

: "${PI_HOST:?}"
: "${PI_USER:?}"
: "${PI_REMOTE_DIR:?}"

ssh "$PI_USER@$PI_HOST" "mkdir -p '$PI_REMOTE_DIR'"

if [[ -n "${PI_PASS:-}" ]]; then
  ssh "$PI_USER@$PI_HOST" \
    "sudo -S -p '' bash -lc 'export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y gdbserver tmux rsync linux-perf'" \
    <<<"$PI_PASS"
else
  if ssh "$PI_USER@$PI_HOST" "sudo -n true" >/dev/null 2>&1; then
    ssh "$PI_USER@$PI_HOST" \
      "sudo bash -lc 'export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y gdbserver tmux rsync linux-perf'"
  else
    read -r -s -p "Pi sudo password: " PI_PASS
    echo
    ssh "$PI_USER@$PI_HOST" \
      "sudo -S -p '' bash -lc 'export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y gdbserver tmux rsync linux-perf'" \
      <<<"$PI_PASS"
  fi
fi

echo "Pi provisioning complete"
