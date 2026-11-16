#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=python3
up() { curl -s -X POST "http://127.0.0.1:8797/start?mode=long&minutes=300" > /dev/null; until curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok; do sleep 5; done; }
down() { curl -s -X POST http://127.0.0.1:8797/stop > /dev/null; until ! curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok; do sleep 3; done; sleep 5; }
up; $PY e2e.py prep 2>&1 | grep -v pynvml; down
$PY e2e.py orders 2>&1 | grep -v -e pynvml -e "Loading weights" || exit 1
up; for m in fused12 fused24 parts4 parts8; do $PY e2e.py run $m 2>&1 | grep -v pynvml; done; down
$PY e2e.py score 2>&1 | grep -v pynvml
echo done
