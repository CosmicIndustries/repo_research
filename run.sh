#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo "[setup] Creating virtualenv…"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

SENTINEL="$VENV/.deps_installed"
REQ_HASH=$(md5sum requirements.txt | cut -d' ' -f1)
STORED_HASH=$(cat "$SENTINEL" 2>/dev/null || echo "")

if [ "$REQ_HASH" != "$STORED_HASH" ]; then
  echo "[setup] Installing/updating dependencies…"
  pip install -q -r requirements.txt && echo "$REQ_HASH" > "$SENTINEL"
else
  echo "[setup] Dependencies OK."
fi

echo "[info ] Starting server on http://localhost:8080"
python main.py
