#!/usr/bin/env bash
# Uruchomienie serwera LangGraph dla n8n.
#
#   ./run-serwer.sh --nowy-token   # wygeneruj token do .env (raz)
#   ./run-serwer.sh                # start serwera
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ "${1:-}" = "--nowy-token" ]; then
  exec ./.venv/bin/python serwer_langgraph.py --nowy-token
fi

if [ ! -f .env ]; then
  echo "Brak pliku .env. Najpierw:" >&2
  echo "  ./run-serwer.sh --nowy-token" >&2
  exit 2
fi

exec ./.venv/bin/python serwer_langgraph.py
