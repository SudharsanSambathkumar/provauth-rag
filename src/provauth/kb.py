"""Knowledge base: documents, evidence items, entities, rules, policies, users, queries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Evidence:
    id: str
    doc_id: str
    predicate: str
    args: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source_type: str
    acl: tuple[str, ...]


@dataclass(frozen=True)
class Atom:
    predicate: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    id: str
    head: Atom
    body: tuple[Atom, ...]
    text: str


@dataclass(frozen=True)
class User:
    id: str
    roles: frozenset[str]


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    predicate: str
    anchor: str


@dataclass
class KnowledgeBase:
    name: str
    documents: dict[str, Document]
    evidence: dict[str, Evidence]
    entities: dict[str, dict]
    rules: list[Rule]
    claim_policies: dict[str, frozenset[str]]
    users: dict[str, User]
    queries: list[Query]
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- authorization
    def doc_acl(self, evidence_id: str) -> tuple[str, ...]:
        return self.documents[self.evidence[evidence_id].doc_id].acl

    def authorized(self, user: User, evidence_id: str) -> bool:
        """A_u(e_i) (Eq. 6) and Retrieve_u(e_i): evidence inherits its document's ACL."""
        acl = self.doc_acl(evidence_id)
        return "all" in acl or bool(user.roles & set(acl))

    def claim_permitted(self, user: User, predicate: str) -> bool:
        """Q_u(c) (Eq. 12): claim-level disclosure policy by derived predicate."""
        allowed = self.claim_policies.get(predicate)
        return allowed is None or "all" in allowed or bool(user.roles & allowed)

    # ------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path) -> KnowledgeBase:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeBase:
        docs = {d["id"]: Document(d["id"], d["title"], d["source_type"], tuple(d["acl"])) for d in data["documents"]}
        ev = {
            e["id"]: Evidence(e["id"], e["doc_id"], e["predicate"], tuple(e["args"]), e["text"])
            for e in data["evidence"]
        }
        rules = [parse_rule(r["id"], r["rule"]) for r in data["rules"]]
        policies = {k: frozenset(v) for k, v in data["claim_policies"].items()}
        users = {u["id"]: User(u["id"], frozenset(u["roles"])) for u in data["users"]}
        queries = [Query(q["id"], q["text"], q["predicate"], q["anchor"]) for q in data["queries"]]
        return cls(data["name"], docs, ev, data["entities"], rules, policies, users, queries, data.get("meta", {}))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "meta": self.meta,
            "documents": [
                {"id": d.id, "title": d.title, "source_type": d.source_type, "acl": list(d.acl)}
                for d in self.documents.values()
            ],
            "evidence": [
                {"id": e.id, "doc_id": e.doc_id, "predicate": e.predicate, "args": list(e.args), "text": e.text}
                for e in self.evidence.values()
            ],
            "entities": self.entities,
            "rules": [{"id": r.id, "rule": r.text} for r in self.rules],
            "claim_policies": {k: sorted(v) for k, v in self.claim_policies.items()},
            "users": [{"id": u.id, "roles": sorted(u.roles)} for u in self.users.values()],
            "queries": [
                {"id": q.id, "text": q.text, "predicate": q.predicate, "anchor": q.anchor} for q in self.queries
            ],
        }

    def with_evidence(self, evidence: dict[str, Evidence], documents: dict[str, Document]) -> KnowledgeBase:
        """A copy with a different evidence/document set (used for counterfactual tests)."""
        return KnowledgeBase(
            self.name,
            documents,
            evidence,
            self.entities,
            self.rules,
            self.claim_policies,
            self.users,
            self.queries,
            self.meta,
        )


_ATOM = re.compile(r"(\w+)\(([^)]*)\)")


def parse_atom(text: str) -> Atom:
    m = _ATOM.fullmatch(text.strip())
    if not m:
        raise ValueError(f"Bad atom: {text!r}")
    return Atom(m.group(1), tuple(t.strip() for t in m.group(2).split(",")))


def parse_rule(rule_id: str, text: str) -> Rule:
    """Positive Horn rule, e.g. 'supplier_defect(S,D) :- supplies(S,P), defect_report(P,D)'."""
    head, body = text.split(":-")
    atoms = tuple(parse_atom(a.group(0)) for a in _ATOM.finditer(body))
    return Rule(rule_id, parse_atom(head), atoms, text.strip())
