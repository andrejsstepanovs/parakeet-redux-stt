#!/usr/bin/env bash
# First-time installer for the Parakeet Redux STT service.
#
# Runs entirely without sudo: it builds the CPU image, starts the container and
# installs a per-user systemd unit for convenient management. Boot persistence
# comes from the container's "restart: unless-stopped" policy.
#
#   ./deploy/install.sh
#   STT2_PORT=5094 ./deploy/install.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

COMPOSE_SERVICE="stt2"
UNIT_NAME="stt2.service"
UNIT_SRC="deploy/stt2.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_DST="$UNIT_DIR/$UNIT_NAME"

PORT="${STT2_PORT:-5093}"
HEALTH_URL="http://127.0.0.1:${PORT}/healthz"
WAIT_SECONDS="${STT2_INSTALL_WAIT:-900}"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Prerequisites (all usable without sudo)
# ---------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker was not found on PATH"
docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon (is it running? is this user in the 'docker' group?)"
docker compose version >/dev/null 2>&1 || die "the 'docker compose' plugin is required"
command -v curl >/dev/null 2>&1 || warn "curl is not installed; the health check will be skipped"

# ---------------------------------------------------------------------------
# 2. Configuration file
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
  cp .env.example .env
  warn "Created .env from .env.example. Set LITELLM_API_KEY and STT2_POST_PROCESS=true to enable LLM cleanup."
fi

# ---------------------------------------------------------------------------
# 3. Build and start
# ---------------------------------------------------------------------------
log "Building the CPU image (first build can take several minutes)..."
docker compose build "$COMPOSE_SERVICE"

log "Starting $COMPOSE_SERVICE..."
docker compose up -d "$COMPOSE_SERVICE"

# ---------------------------------------------------------------------------
# 4. Wait for readiness (first start downloads ~171 MB of model weights)
# ---------------------------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  log "Waiting for the service on port ${PORT} (first start downloads model weights)..."
  ready=0
  for ((i = 1; i <= WAIT_SECONDS; i++)); do
    if curl -sf "$HEALTH_URL" >/dev/null 2>&1; then
      log "Service is healthy after ${i}s."
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" -ne 1 ]]; then
    warn "Timed out after ${WAIT_SECONDS}s. Inspect logs with: docker compose logs -f $COMPOSE_SERVICE"
  fi
else
  log "Started. Check readiness with: curl $HEALTH_URL"
fi

# ---------------------------------------------------------------------------
# 5. Install the user systemd unit (optional convenience, no sudo)
# ---------------------------------------------------------------------------
if systemctl --user daemon-reload >/dev/null 2>&1; then
  log "Installing user unit to $UNIT_DST"
  mkdir -p "$UNIT_DIR"
  temp_unit="$(mktemp)"
  sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" "$UNIT_SRC" > "$temp_unit"
  install -m 0644 "$temp_unit" "$UNIT_DST"
  rm -f "$temp_unit"
  systemctl --user daemon-reload
  systemctl --user enable --now "$UNIT_NAME" >/dev/null 2>&1 || \
    warn "Could not enable $UNIT_NAME; manage the stack with docker compose instead."
else
  warn "systemd --user is unavailable; manage the stack with docker compose instead."
fi

# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------
echo
log "Install complete."
echo "    API:  http://127.0.0.1:${PORT}/v1/audio/transcriptions"
echo "    UI:   http://127.0.0.1:${PORT}/"
echo "    Docs: http://127.0.0.1:${PORT}/docs"
echo
echo "Manage without sudo:"
echo "    systemctl --user status|restart|stop|start $UNIT_NAME"
echo "    docker compose logs -f $COMPOSE_SERVICE"
