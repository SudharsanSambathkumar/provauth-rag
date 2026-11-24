"""Synthetic enterprise benchmarks: ProvAuth-Retail and ProvAuth-ITSM.

Both are generated deterministically from a seed. All names, identifiers and
vulnerability ids are fictional (vulnerabilities use the prefix VULN-SYN, never real CVEs).
Each evidence item is one sentence in one document and inherits that document's ACL.
"""

from __future__ import annotations

import random

from .kb import KnowledgeBase

VERSION = "1.0.0"


class _Builder:
    def __init__(self, name: str, seed: int):
        self.name, self.rng = name, random.Random(seed)
        self.documents, self.evidence, self.entities = [], [], {}
        self._doc_n = 0
        self._ev_n = 0

    def entity(self, eid, etype, name):
        self.entities[eid] = {"type": etype, "name": name}
        return eid

    def label(self, eid):
        return f"{self.entities[eid]['name']} ({eid})"

    def doc(self, title, source_type, acl):
        self._doc_n += 1
        d = {"id": f"{self.name[:3].upper()}-D{self._doc_n:04d}", "title": title, "source_type": source_type, "acl": acl}
        self.documents.append(d)
        return d["id"]

    def ev(self, doc_id, predicate, args, text):
        self._ev_n += 1
        self.evidence.append(
            {"id": f"e-{self._ev_n:05d}", "doc_id": doc_id, "predicate": predicate, "args": list(args), "text": text}
        )

    def chunked_docs(self, items, size, title_fn, source_type, acl):
        """Spread (predicate, args, text) items over documents of `size` sentences."""
        for i in range(0, len(items), size):
            d = self.doc(title_fn(i // size + 1), source_type, acl)
            for pred, args, text in items[i : i + size]:
                self.ev(d, pred, args, text)


# --------------------------------------------------------------------------- retail
_SUP_A = ["Northwind", "Sunrise", "Evergreen", "BlueRiver", "Golden Field", "Silver Oak", "Redstone", "Lakeside",
          "Morning Dew", "Highland", "Coastal", "Green Valley", "Riverbend", "Saffron", "Monsoon", "Kaveri",
          "Nilgiri", "Deccan", "Malabar", "Western Ghats"]
_SUP_B = ["Dairy", "Foods", "Farms", "Bakers"]
_PRODUCTS = ["Toned Milk 1L", "Curd 500g", "Paneer 200g", "Butter 100g", "Sandwich Bread 400g", "Whole Wheat Bread",
             "Burger Buns", "Rusk 300g", "Basmati Rice 5kg", "Toor Dal 1kg", "Sunflower Oil 1L", "Atta 10kg",
             "Tomato Ketchup", "Mango Juice 1L", "Potato Chips 150g", "Masala Peanuts", "Green Tea 25s",
             "Instant Noodles", "Oat Cookies", "Corn Flakes 500g"]
_STORES = ["Indiranagar", "Koramangala", "Whitefield", "HSR Layout", "Jayanagar", "Malleshwaram", "Hebbal",
           "Yelahanka", "Banashankari", "Electronic City", "Marathahalli", "BTM Layout", "Rajajinagar",
           "Basavanagudi", "Sarjapur Road", "Bellandur", "JP Nagar", "Frazer Town", "Kalyan Nagar", "RT Nagar",
           "Vijayanagar", "Domlur", "Ulsoor", "Bannerghatta Road"]
_MANAGERS = ["Ravi", "Priya", "Suresh", "Anitha", "Kiran", "Deepa", "Arjun", "Lakshmi", "Manoj", "Divya", "Rahul",
             "Sneha", "Vikram", "Meera", "Naveen", "Pooja", "Sanjay", "Kavitha", "Ajay", "Nisha", "Harish", "Swathi",
             "Prakash", "Bhavana"]
_DEFECTS = ["seal leakage", "foreign particle", "undeclared allergen", "temperature abuse", "mould growth",
            "packaging tear", "short weight", "off odour", "label misprint", "glass fragment"]


def build_retail(seed: int = 7, n_suppliers: int = 40, n_products: int = 160, n_queries_per_type: int = 50) -> dict:
    b = _Builder("retail", seed)
    rng = b.rng
    sups = [b.entity(f"S{i + 1:02d}", "Supplier", f"{_SUP_A[i % 20]} {_SUP_B[i // 20 + (i % 2) * 2]}") for i in range(n_suppliers)]
    prods = [
        b.entity(f"P{i + 1:03d}", "Product", f"{_PRODUCTS[i % 20]} lot {i // 20 + 1}") for i in range(n_products)
    ]
    stores = [b.entity(f"ST{i + 1:02d}", "Store", f"{n} store") for i, n in enumerate(_STORES)]
    mgrs = [b.entity(f"M{i + 1:02d}", "Manager", f"{n}") for i, n in enumerate(_MANAGERS)]

    # supplies: one primary supplier per product, ~25% have a second supplier
    supplies = []
    for i, p in enumerate(prods):
        supplies.append((sups[i % n_suppliers], p))
        if rng.random() < 0.25:
            supplies.append((rng.choice([s for s in sups if s != sups[i % n_suppliers]]), p))
    catalog, contracts = [], []
    for s, p in supplies:
        sent = ("supplies", (s, p), f"{b.label(s)} supplies {b.label(p)}.")
        r = rng.random()
        if r < 0.20:  # contract-only relationship
            contracts.append(sent)
        else:
            catalog.append(sent)
            if r < 0.35:  # also confirmed in a confidential contract: alternative provenance
                contracts.append(("supplies", (s, p), f"Contract annex: {b.label(s)} is the contracted source of {b.label(p)}."))
    for s in sups:
        mine = [x for x in catalog if x[1][0] == s]
        if mine:
            d = b.doc(f"Supplier catalog: {b.entities[s]['name']}", "supplier_catalog", ["all"])
            for pred, args, text in mine:
                b.ev(d, pred, args, text)
    rng.shuffle(contracts)
    b.chunked_docs(contracts, 8, lambda n: f"Confidential supply contracts, volume {n}", "contract", ["procurement", "exec"])

    # stocked_in
    stock = []
    for p in prods:
        for st in rng.sample(stores, rng.randint(3, 6)):
            stock.append((p, st))
    for st in stores:
        d = b.doc(f"Inventory report: {b.entities[st]['name']}", "inventory", ["staff"])
        for p, s2 in stock:
            if s2 == st:
                b.ev(d, "stocked_in", (p, st), f"{b.label(p)} is stocked at {b.label(st)}.")
    d = b.doc("Store directory", "directory", ["all"])
    for st, m in zip(stores, mgrs, strict=True):
        b.ev(d, "managed_by", (st, m), f"{b.label(st)} is managed by store manager {b.label(m)}.")

    # defects: QA reports (restricted) and store field issues (staff); overlapping ids create alternatives
    defective = rng.sample(prods, 26)
    qa, field = [], []
    for k, p in enumerate(defective):
        did = b.entity(f"D{k + 1:03d}", "Defect", rng.choice(_DEFECTS))
        text = f"QA investigation {did}: {b.entities[did]['name']} found in {b.label(p)}."
        r = rng.random()
        if r < 0.55:
            qa.append(("defect_report", (p, did), text))
        elif r < 0.80:
            qa.append(("defect_report", (p, did), text))
            field.append(("field_issue", (p, did), f"Store incident log: customers reported {b.entities[did]['name']} ({did}) in {b.label(p)}."))
        else:
            field.append(("field_issue", (p, did), f"Store incident log: customers reported {b.entities[did]['name']} ({did}) in {b.label(p)}."))
    b.chunked_docs(qa, 6, lambda n: f"QA defect investigations, batch {n}", "qa_report", ["qa", "exec"])
    b.chunked_docs(field, 6, lambda n: f"Store incident log, part {n}", "incident_log", ["staff"])

    # recalls: restricted notices; some also announced publicly
    recalled = rng.sample(defective, 9) + rng.sample([p for p in prods if p not in defective], 4)
    notices, press = [], []
    for p in recalled:
        notices.append(("recall", (p,), f"Recall notice: withdraw {b.label(p)} from sale immediately."))
        if rng.random() < 0.4:
            press.append(("recall", (p,), f"Press release: DailyBasket is voluntarily recalling {b.label(p)}."))
    b.chunked_docs(notices, 4, lambda n: f"Recall notices, bulletin {n}", "recall_notice", ["store_manager", "qa", "exec"])
    b.chunked_docs(press, 4, lambda n: f"Public press releases, issue {n}", "press_release", ["all"])

    rules = [
        ("R1", "supplier_defect(S,D) :- supplies(S,P), defect_report(P,D)"),
        ("R2", "supplier_defect(S,D) :- supplies(S,P), field_issue(P,D)"),
        ("R3", "product_at_risk(Q,S) :- supplier_defect(S,D), supplies(S,Q)"),
        ("R4", "store_affected(T,P) :- recall(P), stocked_in(P,T)"),
        ("R5", "manager_action(M,P) :- store_affected(T,P), managed_by(T,M)"),
    ]
    policies = {
        "supplier_defect": ["qa", "procurement", "exec"],
        "product_at_risk": ["qa", "procurement", "store_manager", "exec"],
        "store_affected": ["staff"],
        "manager_action": ["store_manager", "qa", "exec"],
    }
    users = [
        {"id": "staff", "roles": ["staff"]},
        {"id": "store_manager", "roles": ["staff", "store_manager"]},
        {"id": "qa_analyst", "roles": ["staff", "qa"]},
        {"id": "procurement", "roles": ["staff", "procurement"]},
        {"id": "executive", "roles": ["staff", "exec"]},
    ]
    templates = {
        "supplier_defect": (sups, "Which product defects is supplier {n} associated with?"),
        "product_at_risk": (prods, "Is {n} at risk because of a supplier defect?"),
        "store_affected": (stores, "Which recalled products affect the {n}?"),
        "manager_action": (mgrs, "Which recall actions must store manager {n} take?"),
    }
    data = _finish(b, rules, policies, users, templates, n_queries_per_type, seed)
    data["meta"]["description"] = "Retail supply chain: suppliers, products, stores, defects and recalls."
    return data


# --------------------------------------------------------------------------- ITSM
_SERVICES = ["Payments API", "Customer Portal", "Ledger Service", "Identity Provider", "Notification Hub",
             "Order Service", "Search Service", "Reporting Engine", "Fraud Scoring", "Mobile Gateway",
             "Card Vault", "KYC Service", "Statement Generator", "Loan Origination", "Pricing Engine",
             "Audit Trail", "Document Store", "Message Queue", "Cache Cluster", "Data Lake Ingest",
             "Feature Store", "Chat Assistant", "Branch Locator", "FX Rates Feed", "Settlement Batch",
             "Rewards Service", "Partner API", "Admin Console", "Billing Service", "Config Service",
             "Scheduler", "Email Relay", "Analytics API", "Risk Dashboard", "Onboarding Flow", "Core Banking Adapter"]
_TEAMS = ["Payments", "Digital Channels", "Core Platform", "Risk Engineering", "Data Platform", "Identity",
          "Lending", "Customer Care Tech", "Infrastructure"]
_VENDORS = ["Acme Compute", "Nimbus Systems", "Orion Networks", "Vertex Storage", "Helix OS", "Quanta Middleware",
            "Zephyr DB", "Atlas Runtime", "Cobalt Security", "Lumen Proxy", "Polar Messaging", "Sigma Containers"]


def build_itsm(seed: int = 11, n_hosts: int = 140, n_queries_per_type: int = 50) -> dict:
    b = _Builder("itsm", seed)
    rng = b.rng
    svcs = [b.entity(f"SVC{i + 1:02d}", "Service", n) for i, n in enumerate(_SERVICES)]
    teams = [b.entity(f"T{i + 1:02d}", "Team", f"{n} team") for i, n in enumerate(_TEAMS)]
    vendors = [b.entity(f"V{i + 1:02d}", "Vendor", n) for i, n in enumerate(_VENDORS)]
    hosts = [b.entity(f"H{i + 1:03d}", "Host", f"node-{i + 1:03d}") for i in range(n_hosts)]
    vulns = [b.entity(f"VULN-SYN-{i + 1:03d}", "Vulnerability", f"synthetic vulnerability {i + 1}") for i in range(45)]

    # layered dependency DAG: front (0-11) -> mid (12-23) -> back (24-35)
    front, mid, back = svcs[:12], svcs[12:24], svcs[24:]
    deps = set()
    for s in front:
        for t in rng.sample(mid, 2):
            deps.add((s, t))
    for s in mid:
        for t in rng.sample(back, 2):
            deps.add((s, t))
    for s in back[:6]:
        deps.add((s, rng.choice(back[6:])))
    host_svc = {h: svcs[i % len(svcs)] for i, h in enumerate(hosts)}
    host_vendor = {h: rng.choice(vendors) for h in hosts}

    cmdb = [("runs_on", (s, h), f"{b.label(s)} runs on host {b.label(h)}.") for h, s in host_svc.items()]
    rng.shuffle(cmdb)
    b.chunked_docs(cmdb, 10, lambda n: f"CMDB extract, page {n}", "cmdb", ["employee"])
    owners = [("owned_by", (s, teams[i % len(teams)]), f"{b.label(s)} is owned by the {b.label(teams[i % len(teams)])}.")
              for i, s in enumerate(svcs)]
    b.chunked_docs(owners, 12, lambda n: f"Service catalog, section {n}", "service_catalog", ["employee"])
    depl = [("depends_on", (s, t), f"{b.label(s)} depends on {b.label(t)}.") for s, t in sorted(deps)]
    b.chunked_docs(depl, 8, lambda n: f"Service dependency map, part {n}", "dependency_map",
                   ["service_desk", "sre", "security", "exec"])

    vulnerable = []
    for h in rng.sample(hosts, 22):
        for v in rng.sample(vulns, rng.randint(1, 2)):
            vulnerable.append((h, v))
    scans = [("vulnerable", (h, v), f"Vulnerability scan: host {b.label(h)} is affected by {v}.") for h, v in vulnerable]
    b.chunked_docs(scans, 6, lambda n: f"Vulnerability scan report {n}", "vuln_scan", ["security", "exec"])
    assets = [("supplied_by", (h, host_vendor[h]), f"Asset register: host {b.label(h)} runs software from {b.label(host_vendor[h])}.")
              for h in hosts]
    rng.shuffle(assets)
    b.chunked_docs(assets, 10, lambda n: f"Asset register, page {n}", "asset_register", ["sre", "exec"])
    adv = set()
    for h, v in vulnerable:  # half of the real exposures are also covered by a public vendor advisory
        if rng.random() < 0.5:
            adv.add((host_vendor[h], v))
    for _ in range(8):  # advisories that match no deployed host
        adv.add((rng.choice(vendors), rng.choice(vulns)))
    advisories = [("vendor_advisory", (vd, v), f"Vendor advisory: {b.label(vd)} products are affected by {v}.") for vd, v in sorted(adv)]
    b.chunked_docs(advisories, 6, lambda n: f"Public vendor advisories, bulletin {n}", "vendor_advisory", ["all"])

    changes, incidents = [], []
    for k in range(100):
        ch = b.entity(f"CHG{k + 1:04d}", "Change", f"change {k + 1}")
        h = rng.choice(hosts)
        changes.append(("change_on", (h, ch), f"Change calendar: {ch} is scheduled on host {b.label(h)}."))
    for k in range(120):
        inc = b.entity(f"INC{k + 1:04d}", "Incident", f"incident {k + 1}")
        h = rng.choice(hosts)
        incidents.append(("incident_on", (h, inc), f"Incident ticket {inc} was raised against host {b.label(h)}."))
    b.chunked_docs(changes, 10, lambda n: f"Change calendar, week {n}", "change_calendar", ["employee"])
    b.chunked_docs(incidents, 10, lambda n: f"Incident tickets, queue {n}", "incident", ["service_desk", "sre", "exec"])

    rules = [
        ("R1", "service_exposed(S,C) :- runs_on(S,H), vulnerable(H,C)"),
        ("R2", "service_exposed(S,C) :- runs_on(S,H), supplied_by(H,V), vendor_advisory(V,C)"),
        ("R3", "service_exposed(S,C) :- depends_on(S,U), service_exposed(U,C)"),
        ("R4", "team_exposure(T,C) :- owned_by(S,T), service_exposed(S,C)"),
        ("R5", "service_change(S,K) :- runs_on(S,H), change_on(H,K)"),
        ("R6", "service_incident(S,I) :- runs_on(S,H), incident_on(H,I)"),
    ]
    policies = {
        "service_exposed": ["security", "sre", "exec"],
        "team_exposure": ["security", "exec"],
        "service_change": ["employee"],
        "service_incident": ["service_desk", "sre", "exec"],
    }
    users = [
        {"id": "employee", "roles": ["employee"]},
        {"id": "service_desk", "roles": ["employee", "service_desk"]},
        {"id": "sre", "roles": ["employee", "sre"]},
        {"id": "security_analyst", "roles": ["employee", "security"]},
        {"id": "executive", "roles": ["employee", "exec"]},
    ]
    templates = {
        "service_exposed": (svcs, "Which vulnerabilities expose the {n}?"),
        "team_exposure": (teams, "Which vulnerabilities affect services owned by the {n}?"),
        "service_change": (svcs, "Which scheduled changes impact the {n}?"),
        "service_incident": (svcs, "Which incidents affected the {n}?"),
    }
    data = _finish(b, rules, policies, users, templates, n_queries_per_type, seed)
    data["meta"]["description"] = "IT service management: services, hosts, dependencies, vulnerabilities, changes, incidents."
    return data


def _finish(b, rules, policies, users, templates, n_per_type, seed):
    """Sample queries: half anchored on entities that have claims (full KB), half random."""
    from .inference import derive

    kb = KnowledgeBase.from_dict(
        {"name": b.name, "documents": b.documents, "evidence": b.evidence, "entities": b.entities,
         "rules": [{"id": i, "rule": r} for i, r in rules], "claim_policies": policies, "users": users, "queries": []}
    )
    facts, _ = derive(kb.evidence.values(), kb.rules, depth=6, cap=1024)
    rng = random.Random(seed + 1000)
    queries = []
    for pred, (pool, tpl) in templates.items():
        with_claims = sorted({f[1][0] for f in facts if f[0] == pred})
        pos = rng.sample(with_claims, min(len(with_claims), n_per_type // 2))
        rest = [e for e in pool if e not in pos]
        neg = rng.sample(rest, min(len(rest), n_per_type - len(pos)))
        for anchor in sorted(pos + neg):
            name = b.entities[anchor]["name"]
            queries.append({"id": f"{b.name[:1].upper()}Q{len(queries) + 1:03d}", "text": tpl.format(n=f"{name} ({anchor})"),
                            "predicate": pred, "anchor": anchor})
    return {
        "name": f"ProvAuth-{'Retail' if b.name == 'retail' else 'ITSM'}",
        "meta": {"version": VERSION, "seed": seed, "license": "CC BY 4.0", "synthetic": True},
        "documents": b.documents,
        "evidence": b.evidence,
        "entities": b.entities,
        "rules": [{"id": i, "rule": r} for i, r in rules],
        "claim_policies": policies,
        "users": users,
        "queries": queries,
    }
