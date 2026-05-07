#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a

: "${PI_SSH_PORT:=22}"

: "${PI_TMUX_SESSION:=pi_debug}"

ssh -p "$PI_SSH_PORT" -t "$PI_USER@$PI_HOST" "tmux attach -t '$PI_TMUX_SESSION'"