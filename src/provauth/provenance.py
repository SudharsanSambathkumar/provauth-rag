"""Provenance expressions for derived knowledge.

A provenance expression is a polynomial over evidence variables x_i (Green et al., 2007):
``+`` joins alternative derivations and ``*`` joins jointly required evidence.

Authorization only needs the Boolean evaluation phi_u (Eq. 9-11), so expressions are kept
in the PosBool[X] quotient of N[X]: a set of monomials, each monomial being the set of
evidence ids that one complete derivation jointly requires. Multiplicities and exponents
are dropped (x*x = x, x + x = x) and absorption removes non-minimal monomials
(x + x*y = x). Both steps preserve phi_u for every user.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

Monomial = frozenset[str]
Prov = frozenset[Monomial]

ONE: Prov = frozenset({frozenset()})  # neutral element of the product
ZERO: Prov = frozenset()  # neutral element of the sum


def var(evidence_id: str) -> Prov:
    """P(e_i) = x_i  (Eq. 3)."""
    return frozenset({frozenset({evidence_id})})


def absorb(monomials: Iterable[Monomial], cap: int | None = None) -> tuple[Prov, bool]:
    """Keep only minimal monomials. Returns (expression, truncated?)."""
    ordered = sorted(set(monomials), key=lambda m: (len(m), sorted(m)))
    kept: list[Monomial] = []
    for m in ordered:
        if not any(k <= m for k in kept):
            kept.append(m)
    truncated = cap is not None and len(kept) > cap
    return frozenset(kept[:cap] if truncated else kept), truncated


def plus(a: Prov, b: Prov, cap: int | None = None) -> tuple[Prov, bool]:
    """Alternative derivations: a + b."""
    return absorb(a | b, cap)


def times(a: Prov, b: Prov, cap: int | None = None) -> tuple[Prov, bool]:
    """Jointly required evidence: a * b."""
    return absorb((x | y for x in a for y in b), cap)


def evaluate(expr: Prov, authorized: Callable[[str], bool]) -> bool:
    """phi_u(P): True iff at least one complete derivation uses only authorized evidence."""
    return any(all(authorized(e) for e in m) for m in expr)


def lineage(expr: Prov) -> frozenset[str]:
    """Every evidence id appearing anywhere in the expression (where-provenance style)."""
    return frozenset().union(*expr) if expr else frozenset()


def shortest_witness(expr: Prov) -> Monomial | None:
    """A single minimal derivation (fewest evidence items, then lexicographic)."""
    return min(expr, key=lambda m: (len(m), sorted(m))) if expr else None


def to_string(expr: Prov, limit: int = 6) -> str:
    """Human-readable polynomial, e.g. 'x12·x40 + x7·x40'."""
    if not expr:
        return "0"
    parts = [
        "·".join(f"x{e.split('-')[-1]}" if "-" in e else e for e in sorted(m)) or "1"
        for m in sorted(expr, key=lambda m: (len(m), sorted(m)))
    ]
    more = f" + … ({len(parts) - limit} more)" if len(parts) > limit else ""
    return " + ".join(parts[:limit]) + more
