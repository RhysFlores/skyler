#!/bin/zsh
# Turns photos in ~/skyphot into bosses (and keeps watching), then serves the game at http://localhost:8000
cd "$(dirname "$0")" || exit 1
.venv/bin/python -u scan.py --watch &
WATCHER=$!
trap "kill $WATCHER 2>/dev/null" EXIT
sleep 1 && open http://localhost:8000 &
cd web && ../.venv/bin/python -m http.server 8000 --bind 127.0.0.1
