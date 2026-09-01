# PayFlow data universe — specification & contract

The shared datasets every series' examples draw from (AUTHORING-GUIDE §4.1). PayFlow is
a fictional but realistic B2B SaaS subscription-billing platform; these are its
operational exports, **deliberately messy the way real exports are** — every defect
below is intentional, cataloged, and owned by the series that teaches it. Notebooks use
this data instead of iris/titanic/toy blobs; `_tools/check.py` (rule D01) enforces it.

## Regeneration & determinism

```bash
.venv\Scripts\python _data\generate.py
```

Stdlib-only, seeded (`SEED = 42`), byte-identical output (gzip mtime pinned). Files are
**never committed** (`_data/.gitignore`); this manifest is the integrity reference:

| file | rows | sha256 (first 12) |
|---|---|---|
| customers.csv | 8,060 | `47c062d0f060` |
| subscriptions.csv | 9,261 | `ce621f44490f` |
| invoices.csv.gz | 288,936 | `5a0a443b0c0c` |
| payments.csv.gz | 286,432 | `8e8ef08bb3eb` |
| support_tickets.csv.gz | 43,706 | `82efd229d553` |
| addon_purchases.csv.gz | 18,075 | `8b7ab8329f04` |

Window: 2019-01 → 2026-08-31. Any change to `generate.py` = update this manifest in the
same commit and treat it as a breaking change for downstream captured numbers.

## Tables

- **customers.csv** — `customer_id, company, segment (SMB/Mid-Market/Enterprise),
  industry, country, currency, signup_date, plan (Starter/Growth/Scale/Enterprise),
  seats, employee_count, referral_source, status (active/churned), churn_date,
  total_lifetime_value_usd`
- **subscriptions.csv** — `sub_id, customer_id, plan, seats, mrr_usd, start_date,
  end_date, status` (upgrade history: ~25% of customers carry an `upgraded` row)
- **invoices.csv.gz** — `invoice_id, customer_id, issue_date, due_date, currency,
  amount (local currency!), status (issued→paid/overdue/disputed/written_off),
  reminder_count` — one per customer per month while active
- **payments.csv.gz** — `payment_id, invoice_id, paid_at, amount_paid, gateway
  (stripe/razorpay/adyen/paylane), fx_rate_usd` — join target for days-to-payment;
  ~4% of invoices are paid in two partial payments
- **support_tickets.csv.gz** — `ticket_id, customer_id, created_at, channel, category
  (billing/bug/account/feature/spam), subject, body, priority, csat, resolved_at` —
  the NLP corpus (series 29 target = `category`)
- **addon_purchases.csv.gz** — `order_id, customer_id, purchased_at, addon` — the
  market-basket table (series 23); recommender feedback (series 32)

## Mess catalog (each defect ↦ the series that teaches it)

| id | Defect (intentional) | Where | Taught in |
|---|---|---|---|
| M1 | Repost bug: ~0.3% exact duplicate invoice rows + ~0.1% near-dupes under a fresh `invoice_id` | invoices | 03, 09 |
| M2 | Amounts in **local currency**, four gateways, `fx_rate_usd` only on payments — naive aggregation mixes currencies | invoices, payments | 03, 09, 10 |
| M3 | **Unit glitch incident**: INR invoices issued 2025-06-10→17 exported in thousands (amount ÷ 1000) | invoices | 12 (leak/QA), 34 (monitoring) |
| M4 | Sentinels: `employee_count = -999`; ticket `csat = 0` means "not answered", not zero | customers, tickets | 09 |
| M5 | MNAR missingness: `referral_source` empty for ~80% of pre-2021 signups (untracked era) | customers | 09 |
| M6 | Inconsistent casing: `status` appears as `paid/Paid/PAID` on ~10% of rows | invoices | 03, 09 |
| M7 | Legacy formats: pre-2020-07 invoice amounts and ALL `paylane` payments carry comma-separated amounts (`'4,554.29'`) — under pandas 3.x the whole column reads as `str` dtype (2019 notes would say `object`), so arithmetic silently concatenates or raises instead of summing | invoices, payments | 03, 09 |
| M8 | Timezone chaos: ticket `created_at` half ISO+05:30, half naive; `paylane` `paid_at` is `DD/MM/YYYY HH:MM` | tickets, payments | 09, 28 |
| M9 | **Drift**: non-IN payments migrate stripe→adyen on 2025-07-01 and delays lengthen (+4d mean) | payments | 12, 34 |
| M10 | **Leakage traps**: `reminder_count` is written after payment resolution (future info for delay prediction); `total_lifetime_value_usd` includes post-label-window revenue (future info for churn) | invoices, customers | 12 |
| M11 | Class imbalance: churn ≈ 11% of customers; disputed ≈ 1.5% of invoices; spam ≈ 8% of tickets | all | 15 |
| M12 | 60 near-duplicate customer records (re-onboarded companies, name-case variants, new ids) | customers | 09 |
| M13 | Partial payments (~4%: 60/40 split, 9 days apart) — invoice↔payment is 1-to-many | payments | 03 |
| M14 | Spam tickets mostly have **no** `customer_id` (external senders) — joins silently drop them | tickets | 03, 29 |

## Targets & label derivation (built in notebooks, not in the raw data)

- **days_to_payment** (regression spine, 11–14): `first(paid_at) - issue_date` via the
  payments join; unpaid invoices are *censored*, not zero — series 12 teaches the
  handling. ⚠️ `reminder_count` must be excluded (M10).
- **churned_within_90d** (classification spine, 15–20): derived from
  `churn_date` with a feature cutoff date; the leakage-safe construction (features only
  from before the cutoff) is the series-12 lesson. ⚠️ exclude
  `total_lifetime_value_usd` (M10).
- **segments** (21–22): behavioral features from invoices + tickets + addons.
- **ticket category / spam** (29): text classification on `subject + body`.
- **daily invoice volume, monthly MRR** (28): from invoices/subscriptions.
- **addon co-purchase** (23, 32): bundles (security / dev / analytics) are the planted
  associations Apriori should surface.

## Ground truth (the instructor's answer key — models SHOULD find this)

Payment delay is generated from: segment (Enterprise slowest), country (AE/IN slowest),
a per-customer latent slowness, large-amount penalty (> ~4k USD-equivalent), and the
post-2025-07 adyen era (+4d). Churn is generated from: a latent unhappiness factor that
*also* drives ticket volume up and csat down (so ticket/csat features genuinely predict
churn), Starter-plan risk, Enterprise stickiness. Use this to sanity-check evals — a
churn model that ranks csat/ticket features at zero importance is broken — but never
show generator internals as if a model "discovered" them.

## Rules of use

1. Series 01–08 read `raw/` directly and are allowed to suffer (that's the point).
2. Series 09–10 build the cleaned feature tables into `_data/processed/` via committed,
   seeded scripts; their schemas get appended to this SPEC when authored. Series 11+
   consume `processed/` and never re-clean ad hoc.
3. Every notebook states which files + which mess ids it touches (planning block,
   AUTHORING-PROTOCOL Phase 1).
4. Canonical/public datasets are the exception, never the default: allowed only where
   the topic honestly demands one (MNIST-class images for CNNs, a public medical set
   for imbalance), always marked `# canonical-ok: <reason>` — `check.py` fails the
   notebook otherwise.
