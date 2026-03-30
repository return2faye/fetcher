#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/../docker/docker-compose.yml"

echo "Stopping Fetcher containers..."
docker compose -f "$COMPOSE_FILE" down

echo "Containers stopped."
