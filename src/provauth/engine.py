"""Query pipeline for the proposed framework and the baselines.

Every method shares retrieval (vector seeds) and graph expansion; they differ in which
evidence may be traversed and how a derived claim is authorized before release.

| Method | Seeds | Graph expansion | Claims from | Release rule |
|---|---|---|---|---|
| B0 No access control | all chunks | all evidence | online inference | release all |
| B1 Seed-only ACL | readable chunks | all evidence | online inference | release all |
| B2 Resource-level ACL | readable chunks | authorized evidence | online inference (authorized evidence) | release all |
| B3 Single-witness provenance | readable chunks | authorized evidence | materialized | shortest proof authorized AND Q |
| B4 Lineage-based denial | readable chunks | authorized evidence | materialized | every source authorized AND Q |
| P-Q Proposed without claim policy | readable chunks | authorized evidence | materialized | phi_u(P(c)) |
| P Proposed | readable chunks | authorized evidence | materialized | phi_u(P(c)) AND Q_u(c) |
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from .inference import Fact, derive
from .kb import KnowledgeBase, Query, User
from .provenance import Prov, evaluate, lineage, shortest_witness
from .retrieval import Encoder, VectorIndex

METHODS = ["B0", "B1", "B2", "B3", "B4", "P-Q", "P"]
METHOD_NAMES = {
    "B0": "No access control",
    "B1": "Seed-only ACL",
    "B2": "Resource-level ACL",
    "B3": "Single-witness provenance",
    "B4": "Lineage-based denial",
    "P-Q": "Proposed without claim policy",
    "P": "Proposed (provenance + claim policy)",
}


@dataclass
class Config:
    top_k: int = 8
    hops: int = 3
    depth: int = 3
    cap: int = 256
    sentences_per_chunk: int = 4
    embed_dim: int = 128
    seed: int = 0


@dataclass
class Decision:
    fact: Fact
    released: bool
    evidence_ok: bool | None = None
    policy_ok: bool | None = None
    provenance: Prov = field(default_factory=frozenset)


@dataclass
class Result:
    method: str
    released: set[Fact]
    decisions: list[Decision]
    seeds: list
    visited: set[str]
    latency_ms: float


class Engine:
    def __init__(self, kb: KnowledgeBase, config: Config | None = None, encoder: Encoder | None = None):
        self.kb = kb
        self.cfg = config or Config()
        self.derived_preds = {r.head.predicate for r in kb.rules}
        if encoder is None:
            from .retrieval import make_chunks

            encoder = Encoder(self.cfg.embed_dim, self.cfg.seed).fit(
                [c.text for c in make_chunks(kb, self.cfg.sentences_per_chunk)]
            )
        self.encoder = encoder
        self.index = VectorIndex(kb, encoder, self.cfg.sentences_per_chunk)
        self.by_entity: dict[str, list[str]] = defaultdict(list)
        for e in kb.evidence.values():
            for a in set(e.args):
                self.by_entity[a].append(e.id)
        t0 = time.perf_counter()
        self.materialized, self.mat_stats = derive(kb.evidence.values(), kb.rules, self.cfg.depth, self.cfg.cap)
        self.materialize_seconds = time.perf_counter() - t0
        self.claims_by_intent: dict[tuple[str, str], list[Fact]] = defaultdict(list)
        for f in self.materialized:
            if f[0] in self.derived_preds:
                self.claims_by_intent[(f[0], f[1][0])].append(f)

    # ----------------------------------------------------------- shared stages
    def seeds(self, query: Query, user: User | None):
        readable = None
        if user is not None:
            readable = lambda c: all(self.kb.authorized(user, e) for e in c.evidence_ids)  # noqa: E731
        return self.index.search(query.text, self.cfg.top_k, readable)

    def expand(self, seeds, user: User | None) -> tuple[set[str], set[str]]:
        """Multi-hop expansion over the entity graph. With ``user`` set, only evidence the
        user may use for inference is traversed (per-hop authorization)."""
        allowed = (lambda e: True) if user is None else (lambda e: self.kb.authorized(user, e))  # noqa: E731
        evidence = {e for c, _ in seeds for e in c.evidence_ids if allowed(e)}
        frontier = {a for e in evidence for a in self.kb.evidence[e].args}
        visited = set(frontier)
        for _ in range(self.cfg.hops):
            nxt = set()
            for ent in frontier:
                for e in self.by_entity.get(ent, []):
                    if e in evidence or not allowed(e):
                        continue
                    evidence.add(e)
                    nxt.update(a for a in self.kb.evidence[e].args if a not in visited)
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        return visited, evidence

    # ----------------------------------------------------------- answer
    def answer(self, query: Query, user: User, method: str = "P") -> Result:
        t0 = time.perf_counter()
        seed_user = None if method == "B0" else user
        expand_user = None if method in ("B0", "B1") else user
        seeds = self.seeds(query, seed_user)
        visited, nb_evidence = self.expand(seeds, expand_user)
        decisions: list[Decision] = []

        if method in ("B0", "B1", "B2"):
            facts, _ = derive((self.kb.evidence[e] for e in nb_evidence), self.kb.rules, self.cfg.depth, self.cfg.cap)
            for f, prov in facts.items():
                if f[0] == query.predicate and f[1][0] == query.anchor:
                    decisions.append(Decision(f, True, provenance=prov))
        elif query.anchor in visited:
            auth = lambda e: self.kb.authorized(user, e)  # noqa: E731
            policy_ok = self.kb.claim_permitted(user, query.predicate)
            for f in self.claims_by_intent.get((query.predicate, query.anchor), []):
                prov = self.materialized[f]
                if method == "B3":
                    w = shortest_witness(prov)
                    ev_ok = w is not None and all(auth(e) for e in w)
                elif method == "B4":
                    ev_ok = all(auth(e) for e in lineage(prov))
                else:
                    ev_ok = evaluate(prov, auth)
                pol = True if method == "P-Q" else policy_ok
                decisions.append(Decision(f, ev_ok and pol, ev_ok, pol, prov))

        released = {d.fact for d in decisions if d.released}
        return Result(method, released, decisions, seeds, visited, (time.perf_counter() - t0) * 1000)
