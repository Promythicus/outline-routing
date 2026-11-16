#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=python3
curl -s -X POST "http://127.0.0.1:8797/start?mode=long&minutes=300" > /dev/null
until curl -s -m 3 http://127.0.0.1:8081/health | grep -q ok; do sleep 5; done
$PY outline.py label 2>&1 | grep -v pynvml && $PY outline.py route 2>&1 | grep -v pynvml
curl -s -X POST http://127.0.0.1:8797/stop > /dev/null
echo done
