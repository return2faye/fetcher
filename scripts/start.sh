#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/../docker/docker-compose.yml"

echo "Starting Fetcher containers..."
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo "Container status:"
docker compose -f "$COMPOSE_FILE" ps

echo ""
echo "Qdrant dashboard: http://localhost:6333/dashboard"
