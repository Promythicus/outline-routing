#!/usr/bin/env python3
"""python outline.py label   one-line label for every 1,500-character part of each sampled contract
   python outline.py route   for each question with an answer: the parts the model would read, best first"""
import json, os, random, sys, time, urllib.request
H = os.path.dirname(os.path.abspath(__file__)); E1 = os.path.join(H, "..", "cuad-retrieval"); sys.path.insert(0, E1)
import bench, retr
OUT = os.path.join(H, "out"); N, SEED, PART = 30, 7, 1500
LLM = "http://127.0.0.1:8081/v1/chat/completions"


def sample():
    D = retr.data(); idx = list(range(len(D))); random.Random(SEED).shuffle(idx)
    return [(i, D[i]) for i in sorted(idx[:N])]


def parts(text):
    bench.CHUNK = PART
    return bench.chunks(text)


def call(messages, max_tokens, schema=None):
    body = {"model": "local", "temperature": 0, "max_tokens": max_tokens, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"]["content"]


def label():
    path = os.path.join(OUT, "labels.json"); L = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time(); n = 0
    for i, (title, text, _) in sample():
        if str(i) in L:
            continue
        out = []
        for a, b in parts(text):
            s = call([{"role": "system", "content": "You index contracts. Given one part of a contract, write a single line of at most 15 words naming every topic it covers, as a lawyer would list them (for example: Term; renewal; termination for breach; effect of termination). No sentence, no explanation."},
                      {"role": "user", "content": text[a:b]}], 40)
            out.append(" ".join(s.split())[:160]); n += 1
        L[str(i)] = out; json.dump(L, open(path, "w"), indent=0)
        print("labelled", i, title[:40], len(out), "parts", flush=True)
    print("label seconds", round(time.time() - t0), "parts", n, flush=True)


def route():
    L = json.load(open(os.path.join(OUT, "labels.json")))
    path = os.path.join(OUT, "routes.jsonl")
    done = {(json.loads(l)["i"], json.loads(l)["q"]) for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a"); schema = {"type": "object", "properties": {"parts": {"type": "array", "items": {"type": "integer"}, "maxItems": 8}}, "required": ["parts"]}
    for i, (title, text, qs) in sample():
        outline = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(L[str(i)]))
        for q, _ in qs:
            if (i, q) in done:
                continue
            t = time.time()
            try:
                got = json.loads(call([{"role": "system", "content": "You help a lawyer review a contract. You are given an outline of the contract, one numbered line per part, and a question. Return the numbers of the parts most likely to contain language on the question, most likely first, at most 8. Include a part even if the topic is only one of several it covers."},
                                       {"role": "user", "content": "OUTLINE:\n" + outline + "\n\nQUESTION: " + retr.short(q)}], 80, schema))["parts"]
            except Exception:
                got = []
            f.write(json.dumps({"i": i, "q": q, "parts": got, "seconds": round(time.time() - t, 2)}) + "\n"); f.flush()
        print("routed", i, title[:40], flush=True)


if __name__ == "__main__":
    {"label": label, "route": route}[sys.argv[1]]()
