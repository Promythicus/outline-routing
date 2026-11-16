#!/usr/bin/env bash
# The labels written eight at once, scored end to end on the fast setting; then the retrieval table again.
cd "$(dirname "$0")"
PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
export BENCH_LLM=http://127.0.0.1:8082/v1/chat/completions
"$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 256 -m ${MODEL_GGUF} -np 1 -c 16384 -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q4_0 -ctvd q4_0 > out/sp_server.log 2>&1 & SRV=$!
until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do sleep 3; done
$PY speed.py route p8 4 p8fast 2>&1 | grep -v pynvml; $PY speed.py ans p8fast 2>&1 | grep -v pynvml
kill $SRV; wait $SRV 2>/dev/null; sleep 5
$PY speed.py report 2>&1 | grep -e p8fast
$PY evalr.py 2>&1 | grep -v -e pynvml -e "Loading weights" | tail -8
echo done
