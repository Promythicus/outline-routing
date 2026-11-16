#!/usr/bin/env python3
"""Can the outline be made faster? On the ten pilot contracts (see README, "Speed").
   python speed.py label <tag> <workers>      label every part, <workers> requests at once -> out/sp_labels_<tag>.json
   python speed.py route <labels> <max> <tag> name at most <max> parts per question        -> out/sp_routes_<tag>.jsonl
   python speed.py ans <routes tag>           read the first 4 named parts and answer       -> out/sp_<tag>.jsonl
   python speed.py report                     timings, agreement with the first run, scores
The server's address is taken from BENCH_LLM."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "cuad-retrieval"))
import bench, retr, outline, e2e
outline.LLM = bench.LLM
OUT = e2e.OUT; T = os.path.join(OUT, "sp_times.json")
LAB = lambda tag: e2e.LP if tag == "first" else os.path.join(OUT, f"sp_labels_{tag}.json")
RTE = lambda tag: e2e.RP if tag == "first" else os.path.join(OUT, f"sp_routes_{tag}.jsonl")


def note(key, value):
    d = json.load(open(T)) if os.path.exists(T) else {}
    d[key] = value; json.dump(d, open(T, "w"), indent=1); print(key, value, flush=True)


def label(tag, workers):
    C = e2e.cs(); jobs = [(k, text[a:b]) for k, (text, _) in enumerate(C) for a, b in outline.parts(text)]
    one = lambda j: " ".join(outline.call([{"role": "system", "content": e2e.SYS_L}, {"role": "user", "content": j[1]}], 40).split())[:160]
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(one, jobs))
    secs = time.time() - t; L = {}
    for (k, _), s in zip(jobs, got):
        L.setdefault(str(k), []).append(s)
    json.dump(L, open(LAB(tag), "w"), indent=0)
    note(f"label {tag}", {"workers": workers, "parts": len(jobs), "seconds": round(secs), "per_contract": round(secs / len(C), 1)})


def route(labels, top, tag):
    L = json.load(open(LAB(labels))); path = RTE(tag)
    done = {(json.loads(l)["k"], json.loads(l)["q"]) for l in open(path)} if os.path.exists(path) else set()
    schema = {"type": "object", "properties": {"parts": {"type": "array", "items": {"type": "integer"}, "maxItems": top}}, "required": ["parts"]}
    sysr = e2e.SYS_R.replace("at most 8", f"at most {top}"); f = open(path, "a")
    for k, (text, qs) in enumerate(e2e.cs()):
        ol = "\n".join(f"{n + 1}. {s}" for n, s in enumerate(L[str(k)]))
        for q in qs:
            if (k, q) in done:
                continue
            t = time.time()
            try:
                got = json.loads(outline.call([{"role": "system", "content": sysr}, {"role": "user", "content": "OUTLINE:\n" + ol + "\n\nQUESTION: " + retr.short(q)}], 80, schema))["parts"]
            except Exception:
                got = []
            f.write(json.dumps({"k": k, "q": q, "parts": got, "seconds": round(time.time() - t, 2)}) + "\n"); f.flush()
        print("routed", tag, k, flush=True)


def picks(tag):
    return {(r["k"], r["q"]): r for r in map(json.loads, open(RTE(tag)))}


def ans(tag):
    R = picks(tag); path = os.path.join(OUT, f"sp_{tag}.jsonl")
    done = {(json.loads(l)["k"], json.loads(l)["question"]) for l in open(path)} if os.path.exists(path) else set()
    f = open(path, "a")
    for k, (text, Q) in enumerate(e2e.cs()):
        P = outline.parts(text)
        for q in Q:
            if (k, q) in done:
                continue
            got = sorted(P[n - 1] for n in list(dict.fromkeys(R[(k, q)]["parts"]))[:4] if 1 <= n <= len(P))
            a = bench.ask("EXCERPTS FROM THE CONTRACT:\n\n" + "\n\n---\n\n".join(text[x:y] for x, y in got), q) if got else {"found": False, "quotes": [], "seconds": 0, "prompt_tokens": 0}
            a.update(k=k, question=q, got=got); f.write(json.dumps(a) + "\n"); f.flush()
        print("answered", tag, k, flush=True)


def report():
    out = {"times": json.load(open(T)) if os.path.exists(T) else {}}
    first = json.load(open(e2e.LP)); n = sum(map(len, first.values()))
    for p in sorted(os.listdir(OUT)):
        if p.startswith("sp_labels_"):
            L = json.load(open(os.path.join(OUT, p))); out.setdefault("labels same as first run", {})[p[10:-5]] = f"{sum(a == b for k in first for a, b in zip(first[k], L[k]))} of {n}"
    base = picks("first"); gold = {}
    for k, c in enumerate(bench.contracts()):
        for q in c["paragraphs"][0]["qas"]:
            gold[(k, q["question"])] = [] if q["is_impossible"] else [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in q["answers"]]
    texts = [t for t, _ in e2e.cs()]
    for p in sorted(os.listdir(OUT)):
        if p.startswith("sp_routes_") or p == "pilot_routes.jsonl":
            tag = "first" if p == "pilot_routes.jsonl" else p[10:-6]; R = picks(tag); hit = span = nq = ng = same = 0
            for key, r in R.items():
                P = outline.parts(texts[key[0]]); four = [n for n in dict.fromkeys(r["parts"]) if 1 <= n <= len(P)][:4]
                same += four == [n for n in dict.fromkeys(base[key]["parts"]) if 1 <= n <= len(P)][:4]
                if gold[key]:
                    held = [any(P[n - 1][0] < ge and gs < P[n - 1][1] for n in four) for gs, ge in gold[key]]
                    nq += 1; hit += any(held); ng += len(held); span += sum(held)
            out.setdefault("routes", {})[tag] = {"questions": len(R), "s_per_q": round(sum(r["seconds"] for r in R.values()) / len(R), 2), "same 4 parts as first run": same,
                                                 "hit": round(hit / nq, 3), "span": round(span / ng, 3)}
    bench.OUT = OUT; out["answers"] = bench.score(tuple(p[:-6] for p in sorted(os.listdir(OUT)) if p.startswith("sp_") and p.endswith(".jsonl") and not p.startswith("sp_routes_")))
    json.dump(out, open(os.path.join(OUT, "speed.json"), "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "answers"}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"label": lambda: label(a[1], int(a[2])), "route": lambda: route(a[1], int(a[2]), a[3]), "ans": lambda: ans(a[1]), "report": report}[a[0]]()
