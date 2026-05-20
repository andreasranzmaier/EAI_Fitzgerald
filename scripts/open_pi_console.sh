#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a

: "${PI_TMUX_SESSION:=pi_debug}"

ssh -t "$PI_USER@$PI_HOST" "tmux attach -t '$PI_TMUX_SESSION'"