#!/usr/bin/env bash
# The speed tests, start to end. Servers are started by hand on :8082 (not the launcher's :8081).
cd "$(dirname "$0")"
PY=python3; BIN=~/models/llama.cpp/src/build/bin/llama-server
BIG=${MODEL_GGUF}; SMALL=~/models/small/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
export BENCH_LLM=http://127.0.0.1:8082/v1/chat/completions
up() { "$BIN" --host 127.0.0.1 --port 8082 -ngl 99 -fa on --fit off -ub 256 "$@" > out/sp_server.log 2>&1 & SRV=$!
       until curl -s -m 3 http://127.0.0.1:8082/health | grep -q ok; do kill -0 $SRV || { echo "server died"; tail -5 out/sp_server.log; exit 1; }; sleep 3; done; }
down() { kill $SRV; wait $SRV 2>/dev/null; sleep 5; }
run() { $PY speed.py "$@" 2>&1 | grep -v pynvml; }
long=(-m "$BIG" -np 8 -c 65536 -ctk q4_0 -ctv q4_0)
up "${long[@]}"; [ -f out/sp_labels_p1.json ] || run label p1 1; [ -f out/sp_labels_p4.json ] || run label p4 4; [ -f out/sp_labels_p8.json ] || run label p8 8
run route first 4 r4; run ans r4; down
until grep -q gguf ~/models/small/dl.log; do sleep 10; done
up -m "$SMALL" -np 8 -c 32768; [ -f out/sp_labels_small.json ] || run label small 8; down
up "${long[@]}"; run route small 4 small4; run ans small4; down
up -m "$BIG" -np 1 -c 16384 -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q4_0 -ctvd q4_0
run route first 4 r4fast; run ans r4fast; down
run report | tail -60
cp out/scores.keep.json out/scores.json
echo done
