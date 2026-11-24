"""Run every experiment reported in the paper and write results/*.json.

python experiments/run.py            # full run
python experiments/run.py --quick    # smoke test (few queries)
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provauth.engine import METHOD_NAMES, METHODS, Config, Engine  # noqa: E402
from provauth.inference import derive  # noqa: E402
from provauth.kb import Document, Evidence, KnowledgeBase  # noqa: E402
from provauth.provenance import evaluate  # noqa: E402

ORACLE_DEPTH, ORACLE_CAP = 6, 1024
NI_SEEDS = (101, 202)
TIMING_REPEATS = 7
DATASETS = {"retail": ROOT / "datasets/provauth-retail.json", "itsm": ROOT / "datasets/provauth-itsm.json"}


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


class Oracle:
    """Exhaustive reference evaluator over the full knowledge base (depth 6)."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.facts, self.stats = derive(kb.evidence.values(), kb.rules, ORACLE_DEPTH, ORACLE_CAP)
        self.by_intent = {}
        for f in self.facts:
            self.by_intent.setdefault((f[0], f[1][0]), []).append(f)

    def labels(self, query, user):
        """(gold authorized claims, forbidden claims, authorized claims that also have an unauthorized derivation)."""
        gold, forbidden, mixed = set(), set(), set()
        auth = lambda e: self.kb.authorized(user, e)  # noqa: E731
        for f in self.by_intent.get((query.predicate, query.anchor), []):
            prov = self.facts[f]
            if evaluate(prov, auth) and self.kb.claim_permitted(user, query.predicate):
                gold.add(f)
                if any(not all(auth(e) for e in m) for m in prov):
                    mixed.add(f)
            else:
                forbidden.add(f)
        return gold, forbidden, mixed


def evaluate_methods(engine: Engine, oracle: Oracle, queries, methods=METHODS):
    rows = []
    for q in queries:
        for user in engine.kb.users.values():
            gold, forbidden, mixed = oracle.labels(q, user)
            for m in methods:
                r = engine.answer(q, user, m)
                rows.append({
                    "query": q.id, "predicate": q.predicate, "user": user.id, "method": m,
                    "released": len(r.released), "tp": len(r.released & gold), "leaked": len(r.released & forbidden),
                    "gold": len(gold), "forbidden": len(forbidden), "mixed": len(mixed),
                    "mixed_released": len(r.released & mixed), "correct_decisions": len(gold & r.released) + len(forbidden - r.released),
                    "latency_ms": r.latency_ms,
                })
    return rows


def summarize(rows, methods=METHODS):
    out = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        rel, tp, leak = (sum(r[k] for r in rs) for k in ("released", "tp", "leaked"))
        gold, forb, mixed, mrel = (sum(r[k] for r in rs) for k in ("gold", "forbidden", "mixed", "mixed_released"))
        cand = gold + forb
        prec = tp / rel if rel else 1.0
        rec = tp / gold if gold else 1.0
        lat = [r["latency_ms"] for r in rs]
        out[m] = {
            "name": METHOD_NAMES[m],
            "pairs": len(rs),
            "leak_pairs_pct": 100 * sum(r["leaked"] > 0 for r in rs) / len(rs),
            "leaked_claims": leak,
            "udr_pct": 100 * leak / forb if forb else 0.0,
            "auth_accuracy_pct": 100 * sum(r["correct_decisions"] for r in rs) / cand if cand else 100.0,
            "precision_pct": 100 * prec,
            "recall_pct": 100 * rec,
            "f1_pct": 100 * (2 * prec * rec / (prec + rec) if prec + rec else 0.0),
            "apa_pct": 100 * mrel / mixed if mixed else None,
            "mixed_claims": mixed,
            "latency_mean_ms": statistics.mean(lat),
            "latency_p95_ms": p95(lat),
        }
    return out


# --------------------------------------------------------------- counterfactual noninterference
def mutate(kb: KnowledgeBase, user, kind: str, seed: int) -> KnowledgeBase:
    """Change only evidence the user may NOT use; readable evidence and policies stay identical."""
    rng = random.Random(seed)
    ev = dict(kb.evidence)
    docs = dict(kb.documents)
    restricted = sorted(e for e in ev if not kb.authorized(user, e))
    by_type = {}
    for eid, meta in kb.entities.items():
        by_type.setdefault(meta["type"], []).append(eid)
    if kind == "delete":
        for e in rng.sample(restricted, len(restricted) // 2):
            del ev[e]
    elif kind == "rewire":
        for e in rng.sample(restricted, len(restricted) // 2):
            old = ev[e]
            args = tuple(rng.choice(by_type[kb.entities[a]["type"]]) for a in old.args)
            ev[e] = Evidence(old.id, old.doc_id, old.predicate, args, f"{old.predicate} {' '.join(args)}")
    elif kind == "insert":
        hidden_doc = Document(f"CF-{seed}", "Counterfactual restricted file", "counterfactual", ("auditor",))
        docs[hidden_doc.id] = hidden_doc
        # copy the argument types of existing restricted facts, re-sampled at random
        templates = [ev[e] for e in restricted] or list(ev.values())
        for i in range(60):
            t = rng.choice(templates)
            args = tuple(rng.choice(by_type[kb.entities[a]["type"]]) for a in t.args)
            eid = f"cf-{seed}-{i:03d}"
            ev[eid] = Evidence(eid, hidden_doc.id, t.predicate, args, f"{t.predicate} {' '.join(args)}")
    else:
        raise ValueError(kind)
    return kb.with_evidence(ev, docs)


def noninterference(kb, engine: Engine, cfg: Config, queries):
    results = {m: {"same": 0, "total": 0} for m in METHODS}
    by_kind = {}
    users = [u for u in kb.users.values() if any(not kb.authorized(u, e) for e in kb.evidence)]
    for kind in ("delete", "insert", "rewire"):
        kres = {m: {"same": 0, "total": 0} for m in METHODS}
        for seed in NI_SEEDS:
            for user in users:
                kb2 = mutate(kb, user, kind, seed + hash(user.id) % 1000)
                eng2 = Engine(kb2, cfg, encoder=engine.encoder)
                for q in queries:
                    for m in METHODS:
                        same = engine.answer(q, user, m).released == eng2.answer(q, user, m).released
                        for bucket in (results[m], kres[m]):
                            bucket["same"] += same
                            bucket["total"] += 1
        by_kind[kind] = {m: 100 * v["same"] / v["total"] for m, v in kres.items()}
    return {"overall_pct": {m: 100 * v["same"] / v["total"] for m, v in results.items()},
            "by_mutation_pct": by_kind,
            "trials_per_method": results["P"]["total"],
            "users": [u.id for u in users]}


# --------------------------------------------------------------- ablations
def depth_ablation(kb, oracle, queries, base_cfg):
    out = []
    for d in (1, 2, 3, 4, 6):
        cfg = Config(**{**asdict(base_cfg), "depth": d})
        eng = Engine(kb, cfg)
        s = summarize(evaluate_methods(eng, oracle, queries, ["P"]), ["P"])["P"]
        times = []
        for _ in range(TIMING_REPEATS):  # median of repeated materializations (single runs are noisy)
            t0 = time.perf_counter()
            derive(kb.evidence.values(), kb.rules, d, cfg.cap)
            times.append(time.perf_counter() - t0)
        out.append({"depth": d, "recall_pct": s["recall_pct"], "udr_pct": s["udr_pct"], "f1_pct": s["f1_pct"],
                    "materialize_s": statistics.median(times), "latency_mean_ms": s["latency_mean_ms"]})
    return out


def topk_ablation(kb, oracle, queries, base_cfg, encoder):
    out = []
    for k in (1, 2, 4, 8, 16):
        cfg = Config(**{**asdict(base_cfg), "top_k": k})
        eng = Engine(kb, cfg, encoder=encoder)
        s = summarize(evaluate_methods(eng, oracle, queries, ["P", "B2"]), ["P", "B2"])
        out.append({"top_k": k, "P_recall_pct": s["P"]["recall_pct"], "B2_recall_pct": s["B2"]["recall_pct"],
                    "P_udr_pct": s["P"]["udr_pct"], "B2_udr_pct": s["B2"]["udr_pct"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--depth-only", action="store_true", help="recompute only the depth ablation in summary.json")
    args = ap.parse_args()
    if args.depth_only:
        path = ROOT / "results/summary.json"
        report = json.loads(path.read_text())
        cfg = Config(**report["config"])
        for name, ds_path in DATASETS.items():
            kb = KnowledgeBase.load(ds_path)
            report["datasets"][name]["depth_ablation"] = depth_ablation(kb, Oracle(kb), kb.queries, cfg)
        path.write_text(json.dumps(report, indent=2))
        print("updated depth ablation in", path)
        return
    cfg = Config()
    report = {"config": asdict(cfg), "oracle": {"depth": ORACLE_DEPTH, "cap": ORACLE_CAP},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "processor": _cpu()}, "datasets": {}}
    for name, path in DATASETS.items():
        kb = KnowledgeBase.load(path)
        queries = kb.queries[:10] if args.quick else kb.queries
        t0 = time.perf_counter()
        engine = Engine(kb, cfg)
        build_s = time.perf_counter() - t0
        oracle = Oracle(kb)
        rows = evaluate_methods(engine, oracle, queries)
        summary = summarize(rows)
        derived = {r.head.predicate for r in kb.rules}
        claim_sizes = [len(p) for f, p in engine.materialized.items() if f[0] in derived]
        ds = {
            "stats": {"documents": len(kb.documents), "evidence": len(kb.evidence), "entities": len(kb.entities),
                      "queries": len(queries), "users": len(kb.users), "pairs": len(queries) * len(kb.users),
                      "rules": len(kb.rules), "derived_claims_depth3": len(claim_sizes),
                      "derived_claims_oracle": sum(1 for f in oracle.facts if f[0] in derived)},
            "overhead": {"index_and_materialize_s": build_s, "materialize_s": engine.materialize_seconds,
                         "mean_monomials_per_claim": statistics.mean(claim_sizes), "max_monomials_per_claim": max(claim_sizes),
                         "truncated_expressions": engine.mat_stats.truncated, "inference_rounds": engine.mat_stats.rounds},
            "summary": summary,
            "per_user": {u: summarize([r for r in rows if r["user"] == u], ["B2", "P-Q", "P"]) for u in kb.users},
        }
        print(f"[{name}] main done", flush=True)
        ds["noninterference"] = noninterference(kb, engine, cfg, queries)
        print(f"[{name}] noninterference done", flush=True)
        ds["depth_ablation"] = depth_ablation(kb, oracle, queries, cfg)
        ds["topk_ablation"] = topk_ablation(kb, oracle, queries, cfg, engine.encoder)
        print(f"[{name}] ablations done", flush=True)
        report["datasets"][name] = ds
        (ROOT / "results").mkdir(exist_ok=True)
        with open(ROOT / f"results/rows_{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    out = ROOT / ("results/summary_quick.json" if args.quick else "results/summary.json")
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)


def _cpu():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == "__main__":
    main()
