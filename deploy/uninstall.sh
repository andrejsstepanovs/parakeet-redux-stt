#!/usr/bin/env bash
# Remove the Parakeet Redux STT service. No sudo required.
#
#   ./deploy/uninstall.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

UNIT_NAME="stt2.service"
UNIT_DST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT_NAME"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

if systemctl --user list-unit-files "$UNIT_NAME" >/dev/null 2>&1; then
  log "Disabling $UNIT_NAME"
  systemctl --user disable --now "$UNIT_NAME" >/dev/null 2>&1 || true
  rm -f "$UNIT_DST"
  systemctl --user daemon-reload >/dev/null 2>&1 || true
fi

log "Stopping and removing the container"
docker compose down || true

read -r -p "Also remove the built image 'parakeet-redux-stt:cpu' (~6.5 GB)? [y/N] " answer
if [[ "${answer,,}" == "y" ]]; then
  docker image rm parakeet-redux-stt:cpu >/dev/null 2>&1 || true
  log "Removed the image."
fi

log "Uninstalled."
