#!/usr/bin/env bash
# Uruchomienie agenta LangGraph ReAct w venv.
#
# UWAGA: katalog roboczy to "/home/ubuntu/n8n/test " — ZE SPACJĄ NA KOŃCU.
# Dlatego wszystkie ścieżki są tu konsekwentnie cytowane.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ "${1:-}" != "--check" ]; then
  echo "BŁĄD: nie ustawiono OPENROUTER_API_KEY." >&2
  echo 'Użycie:  export OPENROUTER_API_KEY="sk-or-v1-..." && ./run.sh' >&2
  echo "Walidacja offline bez klucza:  ./run.sh --check" >&2
  exit 2
fi

exec ./.venv/bin/python agent_react_langgraph.py "$@"
