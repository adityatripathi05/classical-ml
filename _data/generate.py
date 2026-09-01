"""PayFlow data-universe generator for the classical-ml track.

Generates the shared, seeded, deliberately-messy operational exports of PayFlow (the
fictional B2B SaaS subscription-billing platform, AUTHORING-GUIDE §4.1) that every
series' examples draw from. The mess is INTENTIONAL and cataloged in _data/SPEC.md
(rules M1-M14): each defect exists because a specific series teaches how to handle it.

Deterministic: same seed -> byte-identical files (gzip mtime pinned to 0).
Stdlib only; runs in seconds on CPU. Regenerate any time:

    python _data/generate.py            # writes _data/raw/*.csv[.gz], prints counts+hashes

Ground truth (the "answer key" for evals) is documented in SPEC.md §Ground truth:
payment delay and churn are driven by real latent signal, so models trained on this
data genuinely learn — and the known generative story lets notebooks sanity-check
what a model SHOULD find.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 42
N_CUSTOMERS = 8_000
DATA_END = date(2026, 8, 31)          # last day covered by the exports
OUT = Path(__file__).parent / "raw"

# ---------------------------------------------------------------- vocabularies

COUNTRIES = [  # (code, weight, currency, fx_rate_usd, extra payment delay days)
    ("IN", 0.40, "INR", 88.0, 4), ("US", 0.22, "USD", 1.0, 0),
    ("GB", 0.10, "GBP", 0.78, 1), ("DE", 0.08, "EUR", 0.91, 2),
    ("AE", 0.07, "AED", 3.67, 6), ("SG", 0.06, "SGD", 1.32, 2),
    ("AU", 0.04, "AUD", 1.51, 1), ("NL", 0.03, "EUR", 0.91, 1),
]
PLANS = {"Starter": 49, "Growth": 199, "Scale": 599, "Enterprise": 2499}  # USD list
SEGMENT_TERMS = {"SMB": 15, "Mid-Market": 30, "Enterprise": 45}           # net-N terms
INDUSTRIES = ["software", "fintech", "healthtech", "ecommerce", "logistics",
              "edtech", "media", "manufacturing", "hospitality", "real-estate"]
REFERRALS = ["organic-search", "partner", "outbound", "event", "g2-review", "referral"]
NAME_A = ["Blue", "Nimbus", "Vertex", "Kite", "Alloy", "Harbor", "Quartz", "Ridge",
          "Lumen", "Cedar", "Falcon", "Mango", "Delta", "Orbit", "Pixel", "Summit"]
NAME_B = ["Works", "Labs", "Systems", "Logistics", "Retail", "Analytics", "Health",
          "Commerce", "Digital", "Networks", "Foods", "Capital", "Studio", "Freight"]
NAME_C = ["Pvt Ltd", "Inc", "GmbH", "LLC", "Ltd", "LLP"]
ADDONS = {  # addon -> co-purchase bundle (SPEC M-basket structure, series 23)
    "sso-saml": "security", "audit-logs": "security", "priority-support": "security",
    "api-plus": "dev", "webhooks-pro": "dev", "sandbox-env": "dev",
    "advanced-reporting": "analytics", "data-export": "analytics", "ai-insights": "analytics",
    "custom-domains": None, "backup-plus": None, "white-label": None,
}
BUNDLE_AFFINITY = {  # segment -> bundle -> probability the bundle is "in play"
    "SMB":        {"security": 0.08, "dev": 0.25, "analytics": 0.30},
    "Mid-Market": {"security": 0.35, "dev": 0.35, "analytics": 0.45},
    "Enterprise": {"security": 0.80, "dev": 0.40, "analytics": 0.55},
}

TICKET_TEMPLATES = {  # category -> (subject templates, body templates) — series 29 corpus
    "billing": (
        ["Invoice {inv} amount looks wrong", "Double charge on {month} invoice",
         "GST breakup missing on {inv}", "Refund pending for {inv}",
         "Proforma needed before payment of {inv}"],
        ["We were charged {amt} on invoice {inv} but our PO says a lower amount. "
         "Please reconcile and issue a corrected invoice before our {month} close.",
         "Invoice {inv} shows {amt} however the subscription was downgraded last cycle. "
         "Finance has blocked payment until a credit note is issued.",
         "Our AP team needs the tax breakup on {inv}; the current PDF fails our "
         "compliance check and the payment run happens on the 25th."]),
    "bug": (
        ["Webhook retries failing with {code}", "Export job stuck since {month}",
         "Dashboard shows stale MRR", "API returns {code} on invoice create",
         "Duplicate invoices generated overnight"],
        ["Since the {month} release our webhook endpoint receives duplicate events and "
         "then retries fail with HTTP {code}. Our reconciliation job is now blocked.",
         "POST /v1/invoices intermittently returns {code} for about 2% of requests; "
         "retry with the same idempotency key returns {code} again instead of the invoice.",
         "The scheduled CSV export has been stuck in state RUNNING since {month}; "
         "downstream finance reports are empty."]),
    "account": (
        ["Add seats to our plan", "SSO login loop for new users", "Change billing owner",
         "Password reset emails not arriving", "Downgrade request effective {month}"],
        ["We onboarded a new team and need 15 extra seats effective {month}; please "
         "confirm proration on the next invoice.",
         "Users added after the SAML rollout get stuck in a redirect loop between the "
         "IdP and the app; existing users are fine.",
         "Our billing owner left the company; transfer ownership to finance@ and update "
         "the invoice recipient before the next cycle."]),
    "feature": (
        ["Support {month} cohort exports", "Need webhook filtering", "Custom invoice numbering",
         "Multi-currency reporting request", "Sandbox data reset API"],
        ["We reconcile in a local ERP and need invoice exports filtered by cost center; "
         "today we pull everything and filter manually, which breaks monthly.",
         "Reporting in a single currency hides FX effects on MRR; we need reports in "
         "both local currency and USD at the daily rate.",
         "Our QA team needs an API to reset sandbox data between test runs instead of "
         "recreating the tenant."]),
    "spam": (
        ["Boost your SEO ranking today", "Exclusive partnership opportunity",
         "You have won a business grant", "Final notice: domain expiring",
         "Increase your revenue 10x"],
        ["We guarantee first page rankings for your website. Reply to claim your free "
         "audit and limited discount today.",
         "Congratulations, your company has been selected for an exclusive grant. "
         "Confirm your bank details to receive the funds.",
         "This is the final notice regarding your domain listing. Act now to avoid "
         "permanent removal from search results."]),
}
TICKET_WEIGHTS = [("billing", 0.34), ("bug", 0.27), ("account", 0.16),
                  ("feature", 0.15), ("spam", 0.08)]


# ---------------------------------------------------------------- helpers

def pick(rng: random.Random, weighted: list[tuple]) -> tuple:
    r, acc = rng.random(), 0.0
    for item in weighted:
        acc += item[1]
        if r <= acc:
            return item
    return weighted[-1]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def month_add(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(y, m + 1, min(d.day, 28))


def fmt_amount_messy(x: float) -> str:
    """M7: legacy exports carry thousand separators -> object dtype on read."""
    return f"{x:,.2f}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> tuple[int, str]:
    """Write plain or gzipped CSV deterministically; return (n_rows, sha256[:12])."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    if path.suffix == ".gz":
        raw = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(data)
        payload = raw.getvalue()
    else:
        payload = data
    path.write_bytes(payload)
    return len(rows), hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------- generation

def gen_customers(rng: random.Random) -> list[dict]:
    customers = []
    for i in range(N_CUSTOMERS):
        country, _, currency, fx, cdelay = pick(rng, COUNTRIES)
        # growth trend: later signup years more likely
        year = pick(rng, [(2019, .06), (2020, .08), (2021, .11), (2022, .13),
                          (2023, .15), (2024, .18), (2025, .12), (2026, .17)])[0]
        signup = date(year, rng.randint(1, 12), rng.randint(1, 28))
        if signup > DATA_END - timedelta(days=45):
            signup = DATA_END - timedelta(days=45 + rng.randint(0, 200))
        segment = pick(rng, [("SMB", .55), ("Mid-Market", .33), ("Enterprise", .12)])[0]
        plan = {"SMB": pick(rng, [("Starter", .6), ("Growth", .4)])[0],
                "Mid-Market": pick(rng, [("Growth", .5), ("Scale", .5)])[0],
                "Enterprise": pick(rng, [("Scale", .45), ("Enterprise", .55)])[0]}[segment]
        seats = max(1, int(rng.lognormvariate({"SMB": 1.2, "Mid-Market": 2.6,
                                               "Enterprise": 3.6}[segment], 0.5)))
        employees = int(rng.lognormvariate({"SMB": 2.5, "Mid-Market": 4.6,
                                            "Enterprise": 6.4}[segment], 0.7))
        unhappiness = rng.gauss(0, 1)   # latent: drives tickets, csat AND churn
        slowness = rng.gauss(0, 1)      # latent: drives payment delay
        p_churn = sigmoid(-2.6 + 1.1 * unhappiness
                          + (0.5 if plan == "Starter" else 0.0)
                          - (0.4 if segment == "Enterprise" else 0.0))
        churned = rng.random() < p_churn
        churn_date = ""
        if churned:
            life = rng.randint(90, max(120, (DATA_END - signup).days))
            cd = signup + timedelta(days=life)
            if cd >= DATA_END:
                churned, cd = False, None
            churn_date = cd.isoformat() if cd else ""
        # M5: referral MNAR — pre-2021 signups mostly untracked
        if signup.year < 2021 and rng.random() < 0.8:
            referral = ""
        else:
            referral = rng.choice(REFERRALS) if rng.random() > 0.1 else ""
        customers.append({
            "customer_id": f"CUST-{10000 + i}",
            "company": f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.choice(NAME_C)}",
            "segment": segment, "industry": rng.choice(INDUSTRIES),
            "country": country, "currency": currency, "fx": fx, "country_delay": cdelay,
            "signup_date": signup, "plan": plan, "seats": seats,
            # M4: sentinel -999 for unknown employee_count
            "employee_count": -999 if rng.random() < 0.06 else employees,
            "referral_source": referral,
            "churned": churned, "churn_date": churn_date,
            "unhappiness": unhappiness, "slowness": slowness,
        })
    # M12: near-duplicate customer records (same company re-onboarded under new id)
    for j, src in enumerate(rng.sample(customers, 60)):
        dup = dict(src)
        dup["customer_id"] = f"CUST-{90000 + j}"
        dup["company"] = src["company"].upper().replace("PVT LTD", "PVT. LTD.")
        customers.append(dup)
    return customers


def customer_rows(customers: list[dict]) -> list[list]:
    rows = []
    for c in customers:
        mrr_usd = PLANS[c["plan"]] * (1 + 0.08 * (c["seats"] - 1))
        # M10 leakage trap: lifetime value computed over ALL time incl. post-label window
        ltv = round(mrr_usd * max(1, (min(DATA_END,
                    date.fromisoformat(c["churn_date"]) if c["churn_date"] else DATA_END)
                    - c["signup_date"]).days / 30.4), 2)
        rows.append([c["customer_id"], c["company"], c["segment"], c["industry"],
                     c["country"], c["currency"], c["signup_date"].isoformat(),
                     c["plan"], c["seats"], c["employee_count"], c["referral_source"],
                     "churned" if c["churned"] else "active", c["churn_date"], ltv])
    return rows


def gen_subscriptions(rng: random.Random, customers: list[dict]) -> list[list]:
    rows, sid = [], 0
    order = ["Starter", "Growth", "Scale", "Enterprise"]
    for c in customers:
        end = c["churn_date"]
        start = c["signup_date"]
        plan, seats = c["plan"], c["seats"]
        upgraded = (not c["churned"]) and plan != "Enterprise" and rng.random() < 0.25
        if upgraded:
            up_at = start + timedelta(days=rng.randint(120, 700))
            if up_at < DATA_END - timedelta(days=60):
                sid += 1
                rows.append([f"SUB-{sid:05d}", c["customer_id"], plan, seats,
                             round(PLANS[plan] * (1 + 0.08 * (seats - 1)), 2),
                             start.isoformat(), up_at.isoformat(), "upgraded"])
                plan = order[order.index(plan) + 1]
                seats = int(seats * rng.uniform(1.1, 1.6)) + 1
                start = up_at
                c["plan"], c["seats"], c["upgrade_date"] = plan, seats, up_at
        sid += 1
        rows.append([f"SUB-{sid:05d}", c["customer_id"], plan, seats,
                     round(PLANS[plan] * (1 + 0.08 * (seats - 1)), 2),
                     start.isoformat(), end, "churned" if c["churned"] else "active"])
    return rows


def gen_invoices(rng: random.Random, customers: list[dict]):
    inv_rows, pay_rows = [], []
    inv_n = pay_n = 0
    glitch_lo, glitch_hi = date(2025, 6, 10), date(2025, 6, 17)  # M3 unit-glitch window
    for c in customers:
        fx, terms = c["fx"], SEGMENT_TERMS[c["segment"]]
        last = date.fromisoformat(c["churn_date"]) if c["churn_date"] else DATA_END
        m, issue = 0, c["signup_date"]
        while issue <= last:
            inv_n += 1
            inv_id = f"INV-{inv_n:07d}"
            mrr_local = PLANS[c["plan"]] * (1 + 0.08 * (c["seats"] - 1)) * fx
            amount = round(mrr_local * rng.uniform(0.97, 1.12), 2)  # addons/proration
            due = issue + timedelta(days=terms)
            # ground truth: delay = f(segment, country, slowness, amount, gateway era)
            era_extra = 4 if (issue >= date(2025, 7, 1) and c["country"] != "IN") else 0
            mean_delay = (3 + {"SMB": 0, "Mid-Market": 3, "Enterprise": 8}[c["segment"]]
                          + c["country_delay"] + 3 * max(0.0, c["slowness"])
                          + (5 if amount > 4000 * fx else 0) + era_extra)
            delay = rng.lognormvariate(math.log(max(mean_delay, 1.5)), 0.55)
            paid_at = due + timedelta(days=int(delay) - rng.randint(0, terms // 2))
            status = "paid"
            if paid_at > DATA_END:
                status, paid_at = "overdue", None
            elif rng.random() < 0.015:
                status, paid_at = "disputed", None
            elif rng.random() < 0.004:
                status, paid_at = "written_off", None
            # M3: thousands-unit glitch for INR invoices in the incident window
            amount_out: object = amount
            if c["currency"] == "INR" and glitch_lo <= issue <= glitch_hi:
                amount_out = round(amount / 1000, 2)
            # M7: legacy exports (< 2020-07) carry comma-formatted amounts
            elif issue < date(2020, 7, 1):
                amount_out = fmt_amount_messy(amount)
            # M6: inconsistent status casing on ~10% of rows
            status_out = status.upper() if rng.random() < 0.05 else (
                status.title() if rng.random() < 0.05 else status)
            # M10 leakage trap: reminder_count is written AFTER payment resolution
            reminders = 0 if (paid_at and paid_at <= due) else min(5, int(delay // 7) + 1)
            row = [inv_id, c["customer_id"], issue.isoformat(), due.isoformat(),
                   c["currency"], amount_out, status_out, reminders]
            inv_rows.append(row)
            # M1: repost bug — exact duplicates + near-dupes with fresh id
            if rng.random() < 0.003:
                inv_rows.append(list(row))
            if rng.random() < 0.001:
                inv_n += 1
                dup = list(row)
                dup[0] = f"INV-{inv_n:07d}"
                inv_rows.append(dup)
            if paid_at:
                gateway = ("razorpay" if c["country"] == "IN" else
                           ("adyen" if paid_at >= date(2025, 7, 1) else "stripe"))
                if rng.random() < 0.15:
                    gateway = "paylane"  # legacy gateway with messy exports
                partial = rng.random() < 0.04
                parts = ([round(amount * 0.6, 2), round(amount * 0.4, 2)]
                         if partial else [amount])
                for k, part in enumerate(parts):
                    pay_n += 1
                    at = paid_at + timedelta(days=9 * k)
                    fxr = round(fx * rng.uniform(0.985, 1.015), 4)
                    if gateway == "paylane":  # M7+M8: DD/MM/YYYY + comma amounts
                        at_out = at.strftime("%d/%m/%Y") + f" {rng.randint(0,23):02d}:{rng.randint(0,59):02d}"
                        part_out: object = fmt_amount_messy(part)
                    else:
                        at_out = datetime(at.year, at.month, at.day, rng.randint(0, 23),
                                          rng.randint(0, 59)).isoformat() + "Z"
                        part_out = part
                    pay_rows.append([f"PAY-{pay_n:07d}", inv_id, at_out, part_out,
                                     gateway, fxr])
            m += 1
            issue = month_add(c["signup_date"], m)
    return inv_rows, pay_rows


def gen_tickets(rng: random.Random, customers: list[dict], n_invoices: int) -> list[list]:
    rows, tid = [], 0
    for c in customers:
        last = date.fromisoformat(c["churn_date"]) if c["churn_date"] else DATA_END
        months = max(1, (last - c["signup_date"]).days // 30)
        rate = 0.10 + 0.02 * math.log1p(c["seats"]) + 0.06 * max(0.0, c["unhappiness"])
        n = min(60, int(rng.gauss(months * rate, 1.5)))
        for _ in range(max(0, n)):
            tid += 1
            cat = pick(rng, TICKET_WEIGHTS)[0]
            subj_t, body_t = (rng.choice(TICKET_TEMPLATES[cat][0]),
                              rng.choice(TICKET_TEMPLATES[cat][1]))
            slots = {"inv": f"INV-{rng.randint(1, n_invoices):07d}",
                     "amt": f"{rng.uniform(50, 9000):.2f}",
                     "month": rng.choice(["January", "March", "April", "June", "July",
                                          "September", "November"]),
                     "code": rng.choice(["429", "500", "502", "409"])}
            created = c["signup_date"] + timedelta(
                days=rng.randint(0, max(1, (last - c["signup_date"]).days)),
                seconds=rng.randint(0, 86_399))
            dt = datetime(created.year, created.month, created.day,
                          rng.randint(6, 22), rng.randint(0, 59))
            # M8: timezone chaos — half IST-offset ISO, half naive UTC
            created_out = (dt.isoformat() + "+05:30" if rng.random() < 0.5
                           else dt.isoformat())
            open_ticket = rng.random() < 0.04
            resolved = "" if open_ticket else (
                dt + timedelta(hours=rng.lognormvariate(3.0, 1.0))).isoformat()
            csat = "" if open_ticket else (
                0 if rng.random() < 0.12 else  # M4: 0 = "not answered" sentinel
                max(1, min(5, round(rng.gauss(3.8 - 1.2 * c["unhappiness"], 0.8)))))
            # spam arrives from outside: usually no customer link
            cust_out = "" if (cat == "spam" and rng.random() < 0.6) else c["customer_id"]
            rows.append([f"TCK-{tid:06d}", cust_out, created_out,
                         rng.choice(["email", "portal", "chat"]), cat,
                         subj_t.format(**slots), body_t.format(**slots),
                         rng.choice(["low", "normal", "high", "urgent"]), csat, resolved])
    return rows


def gen_addons(rng: random.Random, customers: list[dict]) -> list[list]:
    rows, oid = [], 0
    singles = [a for a, b in ADDONS.items() if b is None]
    for c in customers:
        last = date.fromisoformat(c["churn_date"]) if c["churn_date"] else DATA_END
        span = max(30, (last - c["signup_date"]).days)
        basket: set[str] = set()
        for bundle, p in BUNDLE_AFFINITY[c["segment"]].items():
            if rng.random() < p:  # bundle in play -> items co-purchased (series 23)
                basket |= {a for a, b in ADDONS.items() if b == bundle
                           if rng.random() < 0.7}
        basket |= {a for a in singles if rng.random() < 0.10}
        for addon in sorted(basket):
            oid += 1
            at = c["signup_date"] + timedelta(days=rng.randint(7, span))
            rows.append([f"ORD-{oid:06d}", c["customer_id"], at.isoformat(), addon])
    return rows


# ---------------------------------------------------------------- main

def main() -> int:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    customers = gen_customers(rng)
    sub_rows = gen_subscriptions(rng, customers)     # may upgrade plan/seats in place
    cust_rows = customer_rows(customers)             # after upgrades -> current plan
    inv_rows, pay_rows = gen_invoices(rng, customers)
    tck_rows = gen_tickets(rng, customers, len(inv_rows))
    add_rows = gen_addons(rng, customers)

    manifest = []
    for name, header, rows in [
        ("customers.csv",
         ["customer_id", "company", "segment", "industry", "country", "currency",
          "signup_date", "plan", "seats", "employee_count", "referral_source",
          "status", "churn_date", "total_lifetime_value_usd"], cust_rows),
        ("subscriptions.csv",
         ["sub_id", "customer_id", "plan", "seats", "mrr_usd", "start_date",
          "end_date", "status"], sub_rows),
        ("invoices.csv.gz",
         ["invoice_id", "customer_id", "issue_date", "due_date", "currency",
          "amount", "status", "reminder_count"], inv_rows),
        ("payments.csv.gz",
         ["payment_id", "invoice_id", "paid_at", "amount_paid", "gateway",
          "fx_rate_usd"], pay_rows),
        ("support_tickets.csv.gz",
         ["ticket_id", "customer_id", "created_at", "channel", "category",
          "subject", "body", "priority", "csat", "resolved_at"], tck_rows),
        ("addon_purchases.csv.gz",
         ["order_id", "customer_id", "purchased_at", "addon"], add_rows),
    ]:
        n, digest = write_csv(OUT / name, header, rows)
        manifest.append((name, n, digest))
        print(f"{name:26s} {n:>8,} rows  sha256:{digest}")
    print(f"\nseed={SEED}  window=2019-01..{DATA_END}  ->  {OUT}")
    print("record these counts+hashes in _data/SPEC.md if the generator changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
