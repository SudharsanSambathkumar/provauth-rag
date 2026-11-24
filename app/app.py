"""ProvAuth-RAG demo app.  Run:  streamlit run app/app.py

URL parameters (used for reproducible screenshots): ?dataset=retail&user=procurement&query=RQ012&view=claims
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provauth.engine import METHOD_NAMES, METHODS, Engine  # noqa: E402
from provauth.inference import derive  # noqa: E402
from provauth.kb import KnowledgeBase  # noqa: E402
from provauth.provenance import evaluate, to_string  # noqa: E402

DATASETS = {"retail": "ProvAuth-Retail", "itsm": "ProvAuth-ITSM"}
VIEWS = {"claims": "Claim authorization", "derivations": "Derivation detail", "retrieval": "Retrieval and expansion",
         "baselines": "Baseline comparison", "audit": "Audit log"}

st.set_page_config(page_title="ProvAuth-RAG", page_icon="🔐", layout="wide")
st.markdown(
    "<style>header[data-testid='stHeader']{display:none} .block-container{padding-top:2rem;padding-bottom:1rem} "
    "div[data-testid='stMetricValue']{font-size:1.6rem}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load(name: str):
    kb = KnowledgeBase.load(ROOT / f"datasets/provauth-{name}.json")
    engine = Engine(kb)
    oracle, _ = derive(kb.evidence.values(), kb.rules, depth=6, cap=1024)
    return kb, engine, oracle


qp = st.query_params


def pick(label, options, key, fmt=lambda x: x, container=st.sidebar):
    default = qp.get(key)
    idx = options.index(default) if default in options else 0
    return container.selectbox(label, options, index=idx, format_func=fmt, key=key)


with st.sidebar:
    st.markdown("### 🔐 ProvAuth-RAG")
    st.caption("Provenance-aware authorization of derived knowledge in hybrid vector–graph RAG")
    ds = pick("Benchmark", list(DATASETS), "dataset", DATASETS.get)
    kb, engine, oracle = load(ds)
    user_id = pick("Signed-in user", list(kb.users), "user")
    user = kb.users[user_id]
    st.markdown("Roles: " + " ".join(f"`{r}`" for r in sorted(user.roles)))
    qids = [q.id for q in kb.queries]
    qmap = {q.id: q for q in kb.queries}
    qid = pick("Question", qids, "query", lambda i: f"{i} · {qmap[i].text}")
    view = pick("View", list(VIEWS), "view", VIEWS.get)
    cfg = engine.cfg
    st.divider()
    st.caption(f"top-k = {cfg.top_k} · hops = {cfg.hops} · rule depth = {cfg.depth} · LSA dim = {cfg.embed_dim}")

query = qmap[qid]
res = engine.answer(query, user, "P")
auth = lambda e: kb.authorized(user, e)  # noqa: E731
policy_ok = kb.claim_permitted(user, query.predicate)


def name(eid):
    return kb.entities.get(eid, {}).get("name", eid)


def claim_id(f):
    return f"{f[0]}({', '.join(f[1])})"


def claim_text(f):
    pred, args = f
    return f"{pred}({', '.join(f'{name(a)} [{a}]' for a in args)})"


st.markdown(f"#### {query.text}")
st.caption(f"Intent: `{query.predicate}(anchor = {query.anchor}, ?)` · user **{user.id}** · method: **Proposed (P)**")

released = [d for d in res.decisions if d.released]
denied = [d for d in res.decisions if not d.released]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Vector seeds", len(res.seeds))
c2.metric("Entities reached", len(res.visited))
c3.metric("Candidate claims", len(res.decisions))
c4.metric("Released", len(released))
c5.metric("Withheld", len(denied))

if view == "claims":
    st.markdown("##### Response")
    if released:
        lines = "\n".join(f"- {claim_text(d.fact)}" for d in released)
        st.success(f"**{len(released)} authorized claim(s):**\n{lines}")
    else:
        st.info("No claim can be disclosed to this user for this question.")
    st.caption("Structured claims released by the authorization engine; an LLM only verbalizes this approved set.")
    st.markdown("##### Derived-claim authorization  ·  Auth(c) = φᵤ(P(c)) ∧ Qᵤ(c)")
    rows = []
    for d in res.decisions:
        n_ok = sum(all(auth(e) for e in m) for m in d.provenance)
        rows.append({
            "Claim": claim_id(d.fact),
            "Provenance P(c)": to_string(d.provenance, 2),
            "Derivations (authorized / total)": f"{n_ok} / {len(d.provenance)}",
            "φᵤ(P(c))": "✅ 1" if d.evidence_ok else "❌ 0",
            "Qᵤ(c)": "✅ 1" if d.policy_ok else "❌ 0",
            "Decision": "RELEASE" if d.released else "WITHHOLD",
        })
    if rows:
        cc = st.column_config
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
            "Claim": cc.TextColumn(width="medium"), "Provenance P(c)": cc.TextColumn(width="large"),
            "Derivations (authorized / total)": cc.TextColumn("Authorized / total derivations", width="small"),
            "φᵤ(P(c))": cc.TextColumn(width="small"), "Qᵤ(c)": cc.TextColumn(width="small"),
            "Decision": cc.TextColumn(width="small")})

elif view == "derivations":
    if not res.decisions:
        st.info("No candidate claims.")
    else:
        facts = [d.fact for d in res.decisions]
        default = qp.get("claim")
        labels = [claim_text(f) for f in facts]
        idx = labels.index(default) if default in labels else 0
        pick_f = st.selectbox("Claim", facts, index=idx, format_func=claim_text)
        d = next(x for x in res.decisions if x.fact == pick_f)
        st.markdown(f"**P(c) =** `{to_string(d.provenance, 12)}`")
        verdict = "authorized" if d.evidence_ok else "not authorized"
        st.caption(f"φᵤ(P(c)) is {verdict} for {user.id}: a claim needs at least one derivation whose evidence is all authorized.")
        for i, m in enumerate(sorted(d.provenance, key=lambda m: (len(m), sorted(m))), start=1):
            ok = all(auth(e) for e in m)
            st.markdown(f"**Derivation {i}** — {'✅ every item authorized' if ok else '❌ contains restricted evidence'}")
            st.dataframe(pd.DataFrame([{
                "Evidence": e, "Statement": kb.evidence[e].text, "Document": kb.documents[kb.evidence[e].doc_id].title,
                "ACL": ", ".join(kb.doc_acl(e)), "Aᵤ(e)": "✅" if auth(e) else "❌"} for e in sorted(m)]),
                hide_index=True, use_container_width=True)

elif view == "retrieval":
    st.markdown("##### Authorized vector seeds  Sᵤ(q)")
    st.dataframe(pd.DataFrame([{"Chunk": c.id, "Score": round(s, 3), "Document": kb.documents[c.doc_id].title,
                                "ACL": ", ".join(kb.documents[c.doc_id].acl), "Text": c.text[:160] + "…"}
                               for c, s in res.seeds]), hide_index=True, use_container_width=True)
    st.markdown(f"##### Graph expansion  ·  {cfg.hops} hops over authorized evidence")
    types = pd.Series([kb.entities[e]["type"] for e in res.visited if e in kb.entities]).value_counts()
    st.bar_chart(types, horizontal=True, height=220)

elif view == "baselines":
    st.markdown("##### Same question and user under each method")
    rows = []
    for m in METHODS:
        r = engine.answer(query, user, m)
        gold = {f for f in r.released if evaluate(oracle[f], auth) and policy_ok}
        rows.append({"Method": f"{m} · {METHOD_NAMES[m]}", "Released": len(r.released),
                     "Authorized released": len(gold), "Unauthorized released": len(r.released) - len(gold),
                     "Latency (ms)": round(r.latency_ms, 1)})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.map(lambda v: "background-color:#fde2e1" if isinstance(v, int) and v > 0 else "",
                              subset=["Unauthorized released"]).format({"Latency (ms)": "{:.1f}"}), hide_index=True,
                 use_container_width=True)
    st.caption("Unauthorized = released claims with no fully authorized derivation, or blocked by the claim policy "
               "(reference evaluator, rule depth 6).")

else:
    st.markdown("##### Audit record")
    st.json({
        "user": user.id, "roles": sorted(user.roles), "query": query.id, "intent": [query.predicate, query.anchor],
        "seeds": [c.id for c, _ in res.seeds], "entities_reached": len(res.visited),
        "decisions": [{"claim": claim_text(d.fact), "provenance": to_string(d.provenance, 8),
                       "phi_u": d.evidence_ok, "Q_u": d.policy_ok, "released": d.released} for d in res.decisions],
        "latency_ms": round(res.latency_ms, 2),
    })
