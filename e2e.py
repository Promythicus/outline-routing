#!/usr/bin/env python3
"""The outline, end to end, on the cuad-retrieval experiment's ten pilot contracts and all their questions.
   python e2e.py prep     label the contracts' parts and route every question (model up)
   python e2e.py orders   what each method would read (embedder and reranker; model down)
   python e2e.py run <fused12|fused24|parts4|parts8>   ask the model (model up)
   python e2e.py score"""
import json, os, sys, time
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "cuad-retrieval"))
import bench, retr, outline
OUT = os.path.join(H, "out"); LP, RP, OP = (os.path.join(OUT, x) for x in ("pilot_labels.json", "pilot_routes.jsonl", "pilot_orders.json"))
SYS_L = "You index contracts. Given one part of a contract, write a single line of at most 15 words naming every topic it covers, as a lawyer would list them (for example: Term; renewal; termination for breach; effect of termination). No sentence, no explanation."
SYS_R = "You help a lawyer review a contract. You are given an outline of the contract, one numbered line per part, and a question. Return the numbers of the parts most likely to contain language on the question, most likely first, at most 8. Include a part even if the topic is only one of several it covers."
SCHEMA = {"type": "object", "properties": {"parts": {"type": "array", "items": {"type": "integer"}, "maxItems": 8}}, "required": ["parts"]}


def cs():
    return [(c["paragraphs"][0]["context"], [q["question"] for q in c["paragraphs"][0]["qas"]]) for c in bench.contracts()]


def prep():
    L = json.load(open(LP)) if os.path.exists(LP) else {}
    t = time.time()
    for k, (text, _) in enumerate(cs()):
        if str(k) not in L:
            L[str(k)] = [" ".join(outline.call([{"role": "system", "content": SYS_L}, {"role": "user", "content": text[a:b]}], 40).split())[:160] for a, b in outline.parts(text)]
            json.dump(L, open(LP, "w"), indent=0); print("labelled", k, len(L[str(k)]), flush=True)
    print("label seconds", round(time.time() - t), flush=True)
    done = {(json.loads(l)["k"], json.loads(l)["q"]) for l in open(RP)} if os.path.exists(RP) else set()
    f = open(RP, "a")
    for k, (text, qs) in enumerate(cs()):
        ol = "\n".join(f"{n + 1}. {s}" for n, s in enumerate(L[str(k)]))
        for q in qs:
            if (k, q) in done:
                continue
            t = time.time()
            try:
                got = json.loads(outline.call([{"role": "system", "content": SYS_R}, {"role": "user", "content": "OUTLINE:\n" + ol + "\n\nQUESTION: " + retr.short(q)}], 80, SCHEMA))["parts"]
            except Exception:
                got = []
            f.write(json.dumps({"k": k, "q": q, "parts": got, "seconds": round(time.time() - t, 2)}) + "\n"); f.flush()
        print("routed", k, flush=True)


def orders():
    import numpy as np, torch
    R = {(r["k"], r["q"]): r["parts"] for r in map(json.loads, open(RP))}
    C = cs(); e = retr.embedder("qwen06")
    qs = sorted({q for _, Q in C for q in Q}); qv = dict(zip(qs, e([retr.TASK + retr.short(q) for q in qs]).astype(np.float32)))
    P = []
    for text, _ in C:
        bench.CHUNK = 500; ch = bench.chunks(text); parts = outline.parts(text)
        of = [next((n for n, (a, b) in enumerate(parts) if a <= x < b), len(parts) - 1) for x, _ in ch]
        P.append((ch, parts, of, e([text[a:b] for a, b in ch]).astype(np.float32)))
    retr._loaded.clear(); torch.cuda.empty_cache(); score = retr.reranker("qwenrr")
    out = {"fused": {}, "parts": {}}
    for k, (text, Q) in enumerate(C):
        ch, parts, of, v = P[k]
        for q in Q:
            route = [n - 1 for n in dict.fromkeys(R[(k, q)]) if 1 <= n <= len(parts)]
            first = np.argsort(-(v @ qv[q])).tolist()[:30]; s = score(retr.short(q), [text[a:b] for a, b in (ch[j] for j in first)])
            order = [first[j] for j in sorted(range(len(first)), key=lambda j: -s[j])]
            byroute = sorted((j for j in range(len(ch)) if of[j] in route), key=lambda j: (route.index(of[j]), j))
            out["fused"][f"{k}|{q}"] = [ch[j] for j in retr.rrf(order, byroute)[:24]]
            out["parts"][f"{k}|{q}"] = [parts[n] for n in route]
        print(k, flush=True)
    json.dump(out, open(OP, "w"))


def run(mode):
    O = json.load(open(OP))[mode.rstrip("0123456789")]; top = int(mode.lstrip("fusedpart"))
    path = os.path.join(OUT, mode + ".jsonl")
    done = {(json.loads(l)["k"], json.loads(l)["question"]) for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    for k, (text, Q) in enumerate(cs()):
        for q in Q:
            if (k, q) in done:
                continue
            got = sorted(map(tuple, O[f"{k}|{q}"][:top]))
            if got:
                a = bench.ask("EXCERPTS FROM THE CONTRACT:\n\n" + "\n\n---\n\n".join(text[x:y] for x, y in got), q)
            else:                                   # the model named no part of the outline: nothing is read
                a = {"found": False, "quotes": [], "seconds": 0, "prompt_tokens": 0}
            a.update(k=k, question=q, got=got); f.write(json.dumps(a) + "\n"); f.flush()
        print(mode, k, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        bench.OUT = OUT; rows = bench.score(("fused12", "fused24", "parts4", "parts8"))
        r = [json.loads(l) for l in open(RP)]; print("route s/q", round(sum(x["seconds"] for x in r) / len(r), 2), "no part named", sum(not x["parts"] for x in r), "of", len(r))
        json.dump(rows, open(os.path.join(OUT, "e2e_scores.json"), "w"), indent=1)
    else:
        {"prep": prep, "orders": orders}.get(sys.argv[1], lambda: run(sys.argv[2]))()
