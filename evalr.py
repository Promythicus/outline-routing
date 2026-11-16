#!/usr/bin/env python3
"""Scores for experiment 02, on the sampled contracts, at reading budgets of 6,000 and 12,000 characters.
   search      experiment 01's best search (500-character passages, Qwen3 embedder, Qwen3 reranker on the first 30)
   outline     read the parts the model named from the outline (4 or 8 parts of 1,500 characters)
   within      the best search, limited to passages inside the 8 parts the model named
   fused       the best search's order and the outline's order merged (reciprocal rank fusion)
   labelled    the best search with each passage's outline label put in front of it (embedder and reranker)
   labelled+fused"""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "cuad-retrieval"))
import bench, retr, outline

S = outline.sample(); L = json.load(open(os.path.join(H, "out", "labels.json")))
R = {(r["i"], r["q"]): r for r in map(json.loads, open(os.path.join(H, "out", "routes.jsonl")))}
e = retr.embedder("qwen06")
qs = sorted({q for _, (_, _, Q) in S for q, _ in Q})
qv = dict(zip(qs, e([retr.TASK + retr.short(q) for q in qs]).astype(np.float32)))
P = {}
for i, (title, text, _) in S:
    bench.CHUNK = 500; ch = bench.chunks(text); parts = outline.parts(text)
    of = [next((k for k, (a, b) in enumerate(parts) if a <= x < b), len(parts) - 1) for x, _ in ch]
    plain = [text[a:b] for a, b in ch]; lab = [f"[{L[str(i)][k]}]\n{t}" for k, t in zip(of, plain)]
    P[i] = dict(ch=ch, parts=parts, of=of, plain=plain, lab=lab, v=e(plain).astype(np.float32), vl=e(lab).astype(np.float32))
retr._loaded.clear()
import torch; torch.cuda.empty_cache()
score = retr.reranker("qwenrr")

M = ["search", "outline", "within", "fused", "labelled", "labelled+fused"]
tot = {m: {b: [0, 0] for b in retr.BUDGETS} for m in M}; n = spans = 0; rsec = []; named = []
for i, (title, text, Q) in S:
    p = P[i]; ch = p["ch"]
    for q, gold in Q:
        r = R[(i, q)]; rsec.append(r["seconds"])
        route = [k - 1 for k in dict.fromkeys(r["parts"]) if 1 <= k <= len(p["parts"])]; named.append(len(route))
        inr = [j for j in range(len(ch)) if p["of"][j] in route]
        def best(vec, texts, extra=()):
            first = np.argsort(-(vec @ qv[q])).tolist()[:30]; cand = list(dict.fromkeys(first + list(extra)))
            s = dict(zip(cand, score(retr.short(q), [texts[j] for j in cand])))
            return sorted(first, key=lambda j: -s[j]), s
        order, s = best(p["v"], p["plain"], inr)
        orderl, _ = best(p["vl"], p["lab"])
        byroute = sorted(inr, key=lambda j: (route.index(p["of"][j]), j))
        orders = {"search": order, "within": sorted(inr, key=lambda j: -s[j]), "fused": retr.rrf(order, byroute),
                  "labelled": orderl, "labelled+fused": retr.rrf(orderl, byroute)}
        n += 1; spans += len(gold)
        for b in retr.BUDGETS:
            got = {m: [ch[j] for j in o[:b // 500]] for m, o in orders.items()}
            got["outline"] = [p["parts"][k] for k in route[:b // 1500]]
            for m in M:
                held = [any(x < ge and gs < y for x, y in got[m]) for gs, ge in gold]
                tot[m][b][0] += any(held); tot[m][b][1] += sum(held)
    print(i, flush=True)
rows = [dict(method=m, **{f"hit{b // 1000}k": round(tot[m][b][0] / n, 3) for b in retr.BUDGETS}, **{f"span{b // 1000}k": round(tot[m][b][1] / spans, 3) for b in retr.BUDGETS}) for m in M]
res = dict(contracts=len(S), questions=n, clauses=spans, route_seconds_per_question=round(sum(rsec) / len(rsec), 2), parts_named_mean=round(sum(named) / len(named), 1), rows=rows)
json.dump(res, open(os.path.join(H, "out", "scores.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
