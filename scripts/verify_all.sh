#!/usr/bin/env bash
# Full local verification, aggregated into a single exit code:
#
#   bit 0 (1)  code tests failed
#   bit 1 (2)  API smoke checks failed
#   bit 2 (4)  image build failed
#
# Flow: build image -> start API -> wait for container health -> run the
# one-shot verify container (pytest + API smoke over the docker network).
set -u

cd "$(dirname "$0")/.."

export API_PORT="${API_PORT:-8000}"
COMPOSE="docker compose"
if ! $COMPOSE version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
fi

code=0

echo "[verify-all] building image..."
if ! $COMPOSE build; then
    echo "[verify-all] image build FAILED (exit 4)"
    exit 4
fi
echo "[verify-all] image built"

$COMPOSE up -d api

cleanup() {
    $COMPOSE rm -sf api verify >/dev/null 2>&1 || true
}
trap cleanup EXIT

# The verify service starts only after the api container is healthy
# (depends_on: condition: service_healthy), runs pytest and the API smoke
# checks, then exits by itself.
$COMPOSE up --abort-on-container-exit --exit-code-from verify verify
vc=$?

echo "[verify-all] verify stage exit code: $vc"
# Verify owns bits 0/1; bit 2 is clear because the build succeeded.
code=$((vc & 3))
echo "[verify-all] FINAL: $([ "$code" -eq 0 ] && echo ALL_GREEN || echo "FAILURES ($code)")"
exit $code
