"""Bounded, provenance-preserving Horn-rule inference (Eq. 7-8).

``derive`` computes, for every fact reachable with derivation trees of height <= depth,
P(h) = sum over rules r deriving h of the product of P(b) over the body atoms b of r.
Evaluation is round-based: round k adds every derivation of height k, so ``depth`` is
exactly the bound on rule nesting.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .kb import Evidence, Rule
from .provenance import Prov, absorb, times, var

Fact = tuple[str, tuple[str, ...]]  # (predicate, args)


@dataclass
class InferenceStats:
    rounds: int = 0
    derived_facts: int = 0
    truncated: int = 0


def base_facts(evidence: Iterable[Evidence]) -> dict[Fact, Prov]:
    facts: dict[Fact, set] = defaultdict(set)
    for e in evidence:
        facts[(e.predicate, e.args)] |= var(e.id)
    return {f: frozenset(p) for f, p in facts.items()}


def _index(facts: dict[Fact, Prov]):
    by_pred: dict[str, list[Fact]] = defaultdict(list)
    by_pos: dict[tuple[str, int, str], list[Fact]] = defaultdict(list)
    for f in facts:
        pred, args = f
        by_pred[pred].append(f)
        for i, a in enumerate(args):
            by_pos[(pred, i, a)].append(f)
    return by_pred, by_pos


def _match(atom, fact_args, binding):
    new = dict(binding)
    for term, value in zip(atom.terms, fact_args, strict=True):
        if term in new:
            if new[term] != value:
                return None
        else:
            new[term] = value
    return new


def _rule_instances(rule: Rule, facts, by_pred, by_pos, delta: set[Fact] | None):
    """Yield (head_fact, [body_facts]) for every satisfying assignment.

    With ``delta`` set (semi-naive evaluation), at least one body fact must be in delta.
    """

    def extend(i, binding, chosen, used_delta):
        if i == len(rule.body):
            if delta is None or used_delta:
                head = (rule.head.predicate, tuple(binding[t] for t in rule.head.terms))
                yield head, chosen
            return
        atom = rule.body[i]
        bound = [(j, binding[t]) for j, t in enumerate(atom.terms) if t in binding]
        if bound:
            j, value = bound[0]
            candidates = by_pos.get((atom.predicate, j, value), [])
        else:
            candidates = by_pred.get(atom.predicate, [])
        for f in candidates:
            b = _match(atom, f[1], binding)
            if b is not None:
                yield from extend(i + 1, b, chosen + [f], used_delta or (delta is not None and f in delta))

    yield from extend(0, {}, [], False)


def derive(
    evidence: Iterable[Evidence],
    rules: list[Rule],
    depth: int = 3,
    cap: int = 256,
) -> tuple[dict[Fact, Prov], InferenceStats]:
    """Materialize all facts with derivation height <= depth, with provenance."""
    facts: dict[Fact, Prov] = base_facts(evidence)
    stats = InferenceStats()
    delta: set[Fact] = set(facts)
    for _ in range(depth):
        if not delta:
            break
        stats.rounds += 1
        by_pred, by_pos = _index(facts)
        updates: dict[Fact, set] = defaultdict(set)
        for rule in rules:
            for head, body in _rule_instances(rule, facts, by_pred, by_pos, delta):
                prov = facts[body[0]]
                for b in body[1:]:
                    prov, t = times(prov, facts[b], cap)
                    stats.truncated += t
                updates[head] |= prov
        new_delta: set[Fact] = set()
        for head, monos in updates.items():
            old = facts.get(head, frozenset())
            merged, t = absorb(old | monos, cap)
            stats.truncated += t
            if merged != old:
                facts[head] = merged
                new_delta.add(head)
        delta = new_delta
    derived_preds = {r.head.predicate for r in rules}
    stats.derived_facts = sum(1 for f in facts if f[0] in derived_preds)
    return facts, stats
