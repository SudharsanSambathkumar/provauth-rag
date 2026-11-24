import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provauth.engine import Engine  # noqa: E402
from provauth.inference import derive  # noqa: E402
from provauth.kb import Evidence, KnowledgeBase, parse_rule  # noqa: E402
from provauth.provenance import evaluate, lineage, plus, shortest_witness, times, var  # noqa: E402


def m(*ids):
    return frozenset(ids)


def test_eq4_alternative_and_joint_provenance():
    f1, _ = plus(var("x1"), var("x3"))
    c, _ = times(f1, var("x2"))
    assert c == frozenset({m("x1", "x2"), m("x3", "x2")})  # (x1 + x3) x2 = x1x2 + x3x2


def test_eq14_alternative_derivation_survives_denial():
    p = frozenset({m("x1", "x2"), m("x3", "x4")})
    assert evaluate(p, lambda e: e != "x2")  # x1x2 denied, x3x4 authorized -> 1
    assert not evaluate(p, lambda e: e not in ("x2", "x4"))
    assert not all(e != "x2" for e in lineage(p))  # lineage-based denial would refuse
    assert shortest_witness(p) == m("x1", "x2")


def test_absorption_keeps_minimal_derivations():
    p, _ = plus(var("a"), frozenset({m("a", "b")}))
    assert p == frozenset({m("a")})


def test_bounded_inference_and_recursion():
    ev = [
        Evidence("e1", "d", "edge", ("a", "b"), ""),
        Evidence("e2", "d", "edge", ("b", "c"), ""),
        Evidence("e3", "d", "edge", ("c", "d"), ""),
    ]
    rules = [parse_rule("r1", "path(X,Y) :- edge(X,Y)"), parse_rule("r2", "path(X,Z) :- edge(X,Y), path(Y,Z)")]
    f2, _ = derive(ev, rules, depth=2)
    f3, _ = derive(ev, rules, depth=3)
    assert ("path", ("a", "c")) in f2 and ("path", ("a", "d")) not in f2
    assert f3[("path", ("a", "d"))] == frozenset({m("e1", "e2", "e3")})


def _kb():
    return KnowledgeBase.load(ROOT / "datasets/provauth-retail.json")


def test_proposed_never_releases_forbidden_claims_on_sample():
    kb = _kb()
    eng = Engine(kb)
    oracle, _ = derive(kb.evidence.values(), kb.rules, 6, 1024)
    for q in kb.queries[:40]:
        for u in kb.users.values():
            r = eng.answer(q, u, "P")
            for f in r.released:
                assert evaluate(oracle[f], lambda e, u=u: kb.authorized(u, e))
                assert kb.claim_permitted(u, f[0])


def test_claim_policy_blocks_even_with_authorized_evidence():
    kb = _kb()
    eng = Engine(kb)
    staff = kb.users["staff"]
    q = next(q for q in kb.queries if q.predicate == "manager_action" and eng.claims_by_intent.get((q.predicate, q.anchor)))
    r = eng.answer(q, staff, "P")
    assert not r.released  # staff may not receive manager_action claims (Q_u = 0)
