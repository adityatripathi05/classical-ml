"""Lab for notebook 01.5 "Problem Framing: From Business Question to Learnable Target".

Reproduces every captured number and listing in 01.5:

  L1  the label design space - one business question, one cutoff, and what horizon and
      population choices do to the base rate
  L2  the naive label ("has churned") and why it is not a prediction target
  L3  the incident - a model fitted to "is a churner" scored against the question the
      business actually asks, "churns in the next H days"
  L4  horizon versus actionability - longer horizons are easier to predict and worth less
  L5  cost asymmetry - the retention economics that pick the operating point
  L6  implicit versus explicit labels - when there is no cancel event to join to

Scope note: this lab designs labels. The systematic leakage-safe machinery (temporal CV,
nested selection, the full leakage taxonomy) is series 12's canonical subject.

Reuses the 01.1 dataset plumbing for paths only. Runtime ~60 s on CPU.

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.5_problem_framing.py"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
SEED = 42
CUTOFF = pd.Timestamp("2025-06-30")     # features use data strictly before this instant
HORIZONS = (90, 180, 365)               # days after the cutoff in which churn counts

# Retention economics, stated so they can be argued with (SPEC-style assumptions).
SAVE_RATE = 0.30            # share of true churners retained when contacted in time
OFFER_MONTHS = 2.0          # discount given, in months of MRR
VALUE_MONTHS = 12.0         # months of MRR preserved by saving a customer

NUM = ["tenure_days", "seats", "employee_count", "invoices_90d", "mean_days_late_180d",
       "tickets_90d", "mean_csat_180d", "mrr_usd"]
CAT = ["segment", "plan", "country", "industry"]


def load_sibling(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


lab11 = load_sibling("lab_01.1_rules_vs_learning.py", "lab_01_1")
RAW = lab11.RAW


# ------------------------------------------------------- as-of feature construction

def load_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cus = pd.read_csv(RAW / "customers.csv")
    cus["signup_date"] = pd.to_datetime(cus["signup_date"])
    cus["churn_date"] = pd.to_datetime(cus["churn_date"], errors="coerce")
    cus["employee_count"] = cus["employee_count"].replace(-999, np.nan)

    inv = pd.read_csv(RAW / "invoices.csv.gz").drop_duplicates()
    inv["issue_date"] = pd.to_datetime(inv["issue_date"])
    inv["due_date"] = pd.to_datetime(inv["due_date"])

    tck = pd.read_csv(RAW / "support_tickets.csv.gz")
    tck["created_dt"] = pd.to_datetime(tck["created_at"], format="ISO8601",
                                       errors="coerce", utc=True).dt.tz_localize(None)
    return cus, inv, tck


def features_as_of(cus: pd.DataFrame, inv: pd.DataFrame, tck: pd.DataFrame,
                   cutoff: pd.Timestamp) -> pd.DataFrame:
    """Every feature computed from data strictly BEFORE the cutoff instant.

    The cutoff is a framing decision, not a technicality: it fixes what the model is
    allowed to know at decision time, and therefore what it can possibly be asked.
    """
    pay = pd.read_csv(RAW / "payments.csv.gz", usecols=["invoice_id", "paid_at"])
    pay["paid_dt"] = lab11.parse_paid_at(pay["paid_at"])
    first_pay = pay.groupby("invoice_id")["paid_dt"].min()

    past_inv = inv.loc[inv["issue_date"] < cutoff].copy()
    past_inv["paid_dt"] = past_inv["invoice_id"].map(first_pay)
    past_inv["days_late"] = (past_inv["paid_dt"] - past_inv["due_date"]).dt.days

    win90 = past_inv.loc[past_inv["issue_date"] >= cutoff - pd.Timedelta(days=90)]
    win180 = past_inv.loc[(past_inv["issue_date"] >= cutoff - pd.Timedelta(days=180))
                          & past_inv["paid_dt"].notna()
                          & (past_inv["paid_dt"] < cutoff)]
    past_tck = tck.loc[tck["created_dt"] < cutoff]
    tck90 = past_tck.loc[past_tck["created_dt"] >= cutoff - pd.Timedelta(days=90)]
    csat180 = past_tck.loc[(past_tck["created_dt"] >= cutoff - pd.Timedelta(days=180))
                           & (past_tck["csat"] > 0)]

    f = cus.copy()
    f["tenure_days"] = (cutoff - f["signup_date"]).dt.days
    f["invoices_90d"] = f["customer_id"].map(win90.groupby("customer_id").size()).fillna(0)
    f["mean_days_late_180d"] = f["customer_id"].map(
        win180.groupby("customer_id")["days_late"].mean())
    f["tickets_90d"] = f["customer_id"].map(tck90.groupby("customer_id").size()).fillna(0)
    f["mean_csat_180d"] = f["customer_id"].map(
        csat180.groupby("customer_id")["csat"].mean())
    f["mrr_usd"] = f["customer_id"].map(
        win90.groupby("customer_id")["customer_id"].size()).fillna(0) * 0 + \
        f["seats"].mul(0)  # placeholder replaced below
    plans = {"Starter": 49, "Growth": 199, "Scale": 599, "Enterprise": 2499}
    f["mrr_usd"] = f["plan"].map(plans) * (1 + 0.08 * (f["seats"] - 1))
    return f


def label_frame(f: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int,
                active_only: bool) -> pd.DataFrame:
    """Apply one (population, horizon) framing decision and return the modelling frame."""
    d = f.copy()
    active_at_cutoff = (d["signup_date"] <= cutoff) & (
        d["churn_date"].isna() | (d["churn_date"] > cutoff))
    if active_only:
        d = d.loc[active_at_cutoff]
    else:
        d = d.loc[d["signup_date"] <= cutoff]
    end = cutoff + pd.Timedelta(days=horizon_days)
    d["churns_in_horizon"] = (d["churn_date"].notna()
                              & (d["churn_date"] > cutoff)
                              & (d["churn_date"] <= end)).astype(int)
    return d


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUM),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED)),
    ])


def fit_predict(train: pd.DataFrame, target: str, test: pd.DataFrame) -> np.ndarray:
    model = make_pipeline()
    model.fit(train[NUM + CAT], train[target])
    return model.predict_proba(test[NUM + CAT])[:, 1]


# ----------------------------------------------------------- L1: the design space

def design_space(f: pd.DataFrame) -> None:
    print(f"  cutoff = {CUTOFF:%Y-%m-%d}; the SAME customers, six framing choices\n")
    print(f"  {'population':<32}{'horizon':<10}{'n':>8}{'positives':>12}{'base rate':>12}")
    for active_only in (False, True):
        for h in HORIZONS:
            d = label_frame(f, CUTOFF, h, active_only)
            pos = int(d["churns_in_horizon"].sum())
            pop = "active at cutoff" if active_only else "all signed-up (incl churned)"
            print(f"  {pop:<32}{f'{h}d':<10}{len(d):>8,}{pos:>12,}"
                  f"{d['churns_in_horizon'].mean():>12.4f}")
    naive = f.loc[f["signup_date"] <= CUTOFF]
    ever = (naive["churn_date"].notna()).astype(int)
    print(f"\n  the naive label, 'has churned at any time':      n={len(naive):,}  "
          f"positives={int(ever.sum()):,}  base rate={ever.mean():.4f}")
    already = int((naive["churn_date"] <= CUTOFF).sum())
    print(f"  of those positives, {already:,} had ALREADY churned before the cutoff - "
          f"the model would be recognising the past, not predicting the future")


# --------------------------------------------------- L3: the incident, quantified

def incident(f: pd.DataFrame, horizon: int = 180) -> None:
    """Two labels, one operational question. Score both against the question."""
    operational = label_frame(f, CUTOFF, horizon, active_only=True)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(operational))
    split = int(len(operational) * 0.6)
    tr_ids = operational.iloc[idx[:split]]["customer_id"]
    train_op = operational.loc[operational["customer_id"].isin(tr_ids)]
    test_op = operational.loc[~operational["customer_id"].isin(tr_ids)]

    naive = f.loc[f["signup_date"] <= CUTOFF].copy()
    naive["ever_churned"] = naive["churn_date"].notna().astype(int)
    train_naive = naive.loc[naive["customer_id"].isin(tr_ids)]

    y = test_op["churns_in_horizon"].to_numpy()
    k = max(1, int(round(0.10 * len(test_op))))       # retention team can contact ~10%
    print(f"  operational question: churn within {horizon}d of the cutoff, among "
          f"{len(operational):,} customers active at the cutoff")
    print(f"  evaluation set {len(test_op):,} customers, base rate {y.mean():.4f}, "
          f"retention capacity k={k:,}\n")

    for label, train, target in [
            ("A: 'is a churner' (ever, all customers)", train_naive, "ever_churned"),
            (f"B: 'churns within {horizon}d' (active only)", train_op, "churns_in_horizon")]:
        scores = fit_predict(train, target, test_op)
        flag = lab11.topk_flag(scores, k)
        m = lab11.precision_recall_at_k(y, flag)
        lift = m["precision"] / max(y.mean(), 1e-9)
        picked = test_op.loc[flag]
        churned_ever = picked["churn_date"].notna()
        med_days = (picked.loc[churned_ever, "churn_date"] - CUTOFF).dt.days.median()
        print(f"  {label}")
        print(f"    training positives={int(train[target].sum()):,}  "
              f"apparent base rate in training={train[target].mean():.4f}")
        print(f"    precision@{k}={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"lift over base rate={lift:.2f}x")
        print(f"    of those contacted, {int(churned_ever.sum()):,} churn at some point; "
              f"median days from cutoff to churn = "
              f"{'n/a' if pd.isna(med_days) else f'{med_days:.0f}'}")

    # Is the ranking difference stable, or an artifact of one split? (guide: variance
    # whenever a comparison claims a winner)
    lifts = {"A": [], "B": []}
    for s in range(8):
        r = np.random.default_rng(100 + s)
        ix = r.permutation(len(operational))
        ids = operational.iloc[ix[:split]]["customer_id"]
        tr_op = operational.loc[operational["customer_id"].isin(ids)]
        te_op = operational.loc[~operational["customer_id"].isin(ids)]
        tr_nv = naive.loc[naive["customer_id"].isin(ids)]
        yy = te_op["churns_in_horizon"].to_numpy()
        kk = max(1, int(round(0.10 * len(te_op))))
        for tag, tr, tgt in [("A", tr_nv, "ever_churned"), ("B", tr_op, "churns_in_horizon")]:
            sc = fit_predict(tr, tgt, te_op)
            p = lab11.precision_recall_at_k(yy, lab11.topk_flag(sc, kk))["precision"]
            lifts[tag].append(p / max(yy.mean(), 1e-9))
    a, b = np.array(lifts["A"]), np.array(lifts["B"])
    print(f"\n  over 8 resplits: A lift {a.mean():.2f}x +/- {a.std():.2f}   "
          f"B lift {b.mean():.2f}x +/- {b.std():.2f}")
    print(f"  A minus B = {(a - b).mean():+.2f}x +/- {(a - b).std():.2f} -> the denser "
          f"label really does rank better; it has {int(train_naive['ever_churned'].sum()) // max(int(train_op['churns_in_horizon'].sum()), 1)}x the positives to learn from")
    print(f"\n  BUT the two labels disagree about how big the problem is:")
    print(f"    'is a churner' base rate      {naive['ever_churned'].mean():.4f}")
    print(f"    'churns within {horizon}d' base rate {operational['churns_in_horizon'].mean():.4f}"
          f"   -> {naive['ever_churned'].mean() / operational['churns_in_horizon'].mean():.1f}x "
          f"smaller addressable problem")
    print("    a business case sized on the first number overstates the winnable churn by")
    print("    that factor, which is the error that survives even a well-ranked model")


# ---------------------------------------------------- L4: horizon vs actionability

def horizon_tradeoff(f: pd.DataFrame) -> None:
    print(f"  {'horizon':<10}{'base rate':>12}{'precision@k':>14}{'lift':>8}"
          f"{'median days to churn':>24}")
    for h in (30, 90, 180, 365, 730):
        d = label_frame(f, CUTOFF, h, active_only=True)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(d))
        split = int(len(d) * 0.6)
        train, test = d.iloc[idx[:split]], d.iloc[idx[split:]]
        if train["churns_in_horizon"].sum() < 10:
            print(f"  {f'{h}d':<10}{d['churns_in_horizon'].mean():>12.4f}"
                  f"{'too few positives to fit':>38}")
            continue
        scores = fit_predict(train, "churns_in_horizon", test)
        y = test["churns_in_horizon"].to_numpy()
        k = max(1, int(round(0.10 * len(test))))
        m = lab11.precision_recall_at_k(y, lab11.topk_flag(scores, k))
        med = (d.loc[d["churns_in_horizon"] == 1, "churn_date"] - CUTOFF).dt.days.median()
        print(f"  {f'{h}d':<10}{d['churns_in_horizon'].mean():>12.4f}"
              f"{m['precision']:>14.4f}{m['precision'] / max(y.mean(), 1e-9):>8.2f}x"
              f"{med:>24.0f}")
    print("\n  longer horizons are easier to predict and less actionable: the retention")
    print("  team cannot act today on a churn that happens some time in the next two years")


# ------------------------------------------------------------ L5: cost asymmetry

def cost_curve(f: pd.DataFrame, horizon: int = 180) -> None:
    d = label_frame(f, CUTOFF, horizon, active_only=True)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(d))
    split = int(len(d) * 0.6)
    train, test = d.iloc[idx[:split]], d.iloc[idx[split:]]
    scores = fit_predict(train, "churns_in_horizon", test)
    y = test["churns_in_horizon"].to_numpy()
    mrr = test["mrr_usd"].to_numpy()

    print(f"  assumptions: save rate {SAVE_RATE:.0%}, offer {OFFER_MONTHS:.0f} months of "
          f"MRR, a saved customer is worth {VALUE_MONTHS:.0f} months")
    breakeven = OFFER_MONTHS / (SAVE_RATE * VALUE_MONTHS)
    print(f"  BREAK-EVEN PRECISION = offer / (save_rate x value) = "
          f"{OFFER_MONTHS:.0f} / ({SAVE_RATE:.2f} x {VALUE_MONTHS:.0f}) = {breakeven:.4f}")
    print(f"  the operational base rate is {y.mean():.4f}: the campaign needs precision "
          f"{breakeven / y.mean():.0f}x the base rate before it breaks even\n")
    print(f"  {'k (contacted)':<16}{'precision':>11}{'gross saved':>14}"
          f"{'offer cost':>13}{'net value':>13}")
    best = (None, -np.inf)
    for frac in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00):
        k = max(1, int(round(frac * len(test))))
        flag = lab11.topk_flag(scores, k)
        saved = SAVE_RATE * VALUE_MONTHS * mrr[flag & (y == 1)].sum()
        cost = OFFER_MONTHS * mrr[flag].sum()
        net = saved - cost
        prec = lab11.precision_recall_at_k(y, flag)["precision"]
        print(f"  {f'{k:,} ({frac:.0%})':<16}{prec:>11.4f}${saved:>13,.0f}"
              f"${cost:>12,.0f}${net:>12,.0f}")
        if net > best[1]:
            best = (frac, net)
    print(f"\n  best operating point: top {best[0]:.0%}, and it still loses "
          f"${-best[1]:,.0f} - NO value of k makes this campaign profitable")
    print("  the model is not the bottleneck; the intervention is. Two levers actually move:")

    for offer, label in [(0.25, "email + small credit (0.25 months)"),
                         (OFFER_MONTHS, "the current discount (2 months)")]:
        be = offer / (SAVE_RATE * VALUE_MONTHS)
        k = max(1, int(round(0.10 * len(test))))
        flag = lab11.topk_flag(scores, k)
        net = (SAVE_RATE * VALUE_MONTHS * mrr[flag & (y == 1)].sum()
               - offer * mrr[flag].sum())
        print(f"    {label:<38} break-even precision {be:.4f}   "
              f"net at k={k:,}: ${net:,.0f}")

    # Lever 2: rank by expected VALUE, not by probability (01.1's weighting, again).
    k = max(1, int(round(0.10 * len(test))))
    print("\n  does value-weighted targeting help, as it did for dunning in 01.1?")
    for cost_label, unit_cost, scales in [("proportional offer (0.25 months of MRR)", None, True),
                                          ("flat outreach cost ($40 per contact)", 40.0, False)]:
        for tag, ranking in [("rank by P(churn)", scores),
                             ("rank by P(churn) x MRR", scores * mrr)]:
            flag = lab11.topk_flag(ranking, k)
            saved = SAVE_RATE * VALUE_MONTHS * mrr[flag & (y == 1)].sum()
            cost = 0.25 * mrr[flag].sum() if scales else unit_cost * flag.sum()
            print(f"    {cost_label if tag.endswith('churn)') else '':<40}"
                  f"{tag:<26} net ${saved - cost:>10,.0f}")
    print("  when the ACTION COST scales with MRR, weighting the ranking by MRR buys nothing")
    print("  - it raises cost as fast as value. In 01.1 the action cost was flat per invoice,")
    print("  which is exactly why value-weighting paid there and does not pay here.")


# ------------------------------------------------- L6: implicit vs explicit labels

def implicit_vs_explicit(f: pd.DataFrame, inv: pd.DataFrame, horizon: int = 180) -> None:
    """Most companies have no cancel event; they infer churn from silence.

    Note the direction of time: a LABEL is allowed to look forward - that is what makes
    it a label - while a FEATURE may not. The implicit label below looks into the horizon
    window on purpose, and would be leakage if it were used as an input.
    """
    quiet_days = 60
    d = label_frame(f, CUTOFF, horizon, active_only=True)
    end = CUTOFF + pd.Timedelta(days=horizon)
    # Silence at the END of the window, not across all of it: a customer who churns on
    # day 100 still bills on days 1-100, so "no invoice at all" almost never fires.
    tail = inv.loc[(inv["issue_date"] > end - pd.Timedelta(days=quiet_days))
                   & (inv["issue_date"] <= end)]
    implicit = (~d["customer_id"].isin(set(tail["customer_id"]))).astype(int)
    explicit = d["churns_in_horizon"]
    agree = int((implicit == explicit).sum())
    print(f"  explicit label: a churn_date inside the {horizon}d window after the cutoff")
    print(f"  implicit label: no invoice in the final {quiet_days} days of that window")
    print(f"  implicit positives {int(implicit.sum()):,}   explicit positives "
          f"{int(explicit.sum()):,}   of {len(d):,} customers")
    print(f"  agree on {agree:,} ({agree / len(d):.2%});  "
          f"both={int(((implicit == 1) & (explicit == 1)).sum()):,}"
          f"   implicit-only={int(((implicit == 1) & (explicit == 0)).sum()):,}"
          f"   explicit-only={int(((implicit == 0) & (explicit == 1)).sum()):,}")
    churned = d.loc[d["churns_in_horizon"] == 1].copy()
    churned["last_bill"] = churned["customer_id"].map(
        inv.groupby("customer_id")["issue_date"].max())
    detect = churned["last_bill"] + pd.Timedelta(days=quiet_days)
    lag = (detect - churned["churn_date"]).dt.days
    print(f"  detection lag: silence becomes declarable a median {lag.median():.0f} days "
          f"after the churn event")
    print(f"  (last invoice lands a median "
          f"{(churned['churn_date'] - churned['last_bill']).dt.days.median():.0f} days before "
          f"the churn, then you must wait out the {quiet_days}-day quiet period)")
    print("  an implicit label measures silence: it is both approximate and late, which")
    print("  bounds how quickly any model trained on it can react")


def main() -> None:
    cus, inv, tck = load_universe()
    f = features_as_of(cus, inv, tck, CUTOFF)

    print("=" * 78)
    print("L1  THE LABEL DESIGN SPACE - one question, six defensible labels")
    print("=" * 78)
    design_space(f)

    print("\n" + "=" * 78)
    print("L3  THE INCIDENT - scoring both labels against the operational question")
    print("=" * 78)
    incident(f)

    print("\n" + "=" * 78)
    print("L4  HORIZON VERSUS ACTIONABILITY")
    print("=" * 78)
    horizon_tradeoff(f)

    print("\n" + "=" * 78)
    print("L5  COST ASYMMETRY CHOOSES THE OPERATING POINT")
    print("=" * 78)
    cost_curve(f)

    print("\n" + "=" * 78)
    print("L6  IMPLICIT VERSUS EXPLICIT LABELS")
    print("=" * 78)
    implicit_vs_explicit(f, inv)


if __name__ == "__main__":
    main()
