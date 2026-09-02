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
  L7  the label spec as a versioned, hashed, self-validating artifact - the Stage C
      production code, and the permanent fix this notebook's incident demands

Scope note: this lab designs labels. The systematic leakage-safe machinery (temporal CV,
nested selection, the full leakage taxonomy) is series 12's canonical subject.

Reuses the 01.1 dataset plumbing for paths only. Runtime ~60 s on CPU.

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.5_problem_framing.py"
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace
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
# The extraction date, from _data/SPEC.md ("Window: 2019-01 -> 2026-08-31"). Stated ONCE
# because two definitions of "when the data ends" is two answers to the censoring
# question: the last invoice is 2026-08-28 and the last churn event 2026-08-30, so
# deriving it from either column would quietly shorten the observable horizon.
DATA_END = pd.Timestamp("2026-08-31")

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

    # Arm A must be drawn from the population the naive framing ACTUALLY uses - every
    # customer ever signed up, including those who had already churned before the cutoff.
    # Drawing its training rows from `operational` (active-at-cutoff) would silently
    # exclude them and the "recognising the past" mechanism would never be exercised.
    naive = f.loc[f["signup_date"] <= CUTOFF].copy()
    naive["ever_churned"] = naive["churn_date"].notna().astype(int)
    test_ids = set(test_op["customer_id"])
    train_naive = naive.loc[~naive["customer_id"].isin(test_ids)]
    already_gone = int((train_naive["churn_date"] <= CUTOFF).sum())
    print(f"  arm A training population: {len(train_naive):,} customers, of whom "
          f"{already_gone:,} had ALREADY churned")
    print(f"  arm B training population: {len(train_op):,} customers, all active at the "
          f"cutoff by construction\n")

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
        # same correction as above: arm A draws from the full signed-up population,
        # excluding only the evaluation customers
        tr_nv = naive.loc[~naive["customer_id"].isin(set(te_op["customer_id"]))]
        yy = te_op["churns_in_horizon"].to_numpy()
        kk = max(1, int(round(0.10 * len(te_op))))
        for tag, tr, tgt in [("A", tr_nv, "ever_churned"), ("B", tr_op, "churns_in_horizon")]:
            sc = fit_predict(tr, tgt, te_op)
            p = lab11.precision_recall_at_k(yy, lab11.topk_flag(sc, kk))["precision"]
            lifts[tag].append(p / max(yy.mean(), 1e-9))
    a, b = np.array(lifts["A"]), np.array(lifts["B"])
    print(f"\n  over 8 resplits: A lift {a.mean():.2f}x +/- {a.std(ddof=1):.2f}   "
          f"B lift {b.mean():.2f}x +/- {b.std(ddof=1):.2f}")
    ratio = int(train_naive["ever_churned"].sum()) / max(int(train_op["churns_in_horizon"].sum()), 1)
    d_ab = a - b
    band = 2 * d_ab.std(ddof=1)      # the same 2-sigma bar every verdict in this series uses
    verdict = ("clears the 2-sigma band" if abs(d_ab.mean()) > band
               else "inside the 2-sigma band - not a decidable edge")
    print(f"  A minus B = {d_ab.mean():+.2f}x +/- {d_ab.std(ddof=1):.2f} "
          f"(2-sigma band {band:.2f}) -> {verdict}; the "
          f"denser label has {ratio:.1f}x the positives to learn from")
    print(f"\n  BUT the two labels disagree about how big the problem is:")
    print(f"    'is a churner' base rate      {naive['ever_churned'].mean():.4f}")
    print(f"    'churns within {horizon}d' base rate {operational['churns_in_horizon'].mean():.4f}"
          f"   -> {naive['ever_churned'].mean() / operational['churns_in_horizon'].mean():.1f}x "
          f"smaller addressable problem")
    print("    a business case sized on the first number overstates the winnable churn by")
    print("    that factor, which is the error that survives even a well-ranked model")


# ---------------------------------------------------- L4: horizon vs actionability

def horizon_tradeoff(f: pd.DataFrame) -> None:
    print(f"  {'horizon':<24}{'base rate':>12}{'precision@k':>14}{'lift':>8}"
          f"{'median days to churn':>24}")
    observable = (DATA_END - CUTOFF).days      # 427: the longest horizon the data can see
    for h in (30, 90, 180, 365, 730):
        d = label_frame(f, CUTOFF, h, active_only=True)
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(d))
        split = int(len(d) * 0.6)
        train, test = d.iloc[idx[:split]], d.iloc[idx[split:]]
        if train["churns_in_horizon"].sum() < 10:
            print(f"  {f'{h}d':<24}{d['churns_in_horizon'].mean():>12.4f}"
                  f"{'too few positives to fit':>38}")
            continue
        scores = fit_predict(train, "churns_in_horizon", test)
        y = test["churns_in_horizon"].to_numpy()
        k = max(1, int(round(0.10 * len(test))))
        m = lab11.precision_recall_at_k(y, lab11.topk_flag(scores, k))
        med = (d.loc[d["churns_in_horizon"] == 1, "churn_date"] - CUTOFF).dt.days.median()
        label = f"{h}d" if h <= observable else f"{h}d (censored@{observable}d)"
        print(f"  {label:<24}{d['churns_in_horizon'].mean():>12.4f}"
              f"{m['precision']:>14.4f}{m['precision'] / max(y.mean(), 1e-9):>8.2f}x"
              f"{med:>24.0f}")
    print(f"\n  the data ends {observable} days after the cutoff, so any horizon past that")
    print(f"  is right-censored: the '730d' row is really the label 'churns before the")
    print(f"  data ends' - 01.4's censoring lesson arriving inside label design")
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
    # What the correct objective actually is. Expected net of contacting customer i is
    #   E_i = m_i*s*V*p_i - c_i.  With a PROPORTIONAL offer c_i = offer*m_i, this is
    #   E_i = m_i * s*V * (p_i - p*),  p* = offer/(s*V).
    # MRR therefore cancels from the go/no-go TEST (contact iff p_i > p*) but NOT from the
    # ranking, whose optimum is m_i*(p_i - p*). Below break-even every term is negative,
    # so weighting by MRR alone concentrates the LOSS - which is what the rows below show.
    k = max(1, int(round(0.10 * len(test))))
    # The break-even quantity under a PROPORTIONAL offer is MRR-weighted precision, not
    # plain precision: both the saving and the cost scale with the customer's MRR.
    flag_k = lab11.topk_flag(scores, k)
    plain = lab11.precision_recall_at_k(y, flag_k)["precision"]
    weighted = mrr[flag_k & (y == 1)].sum() / mrr[flag_k].sum()
    print(f"\n  p* = {breakeven:.4f} is a break-even on MRR-WEIGHTED precision, because a")
    print(f"  proportional offer makes both the saving and the cost scale with MRR.")
    print(f"    plain precision@k     {plain:.4f}")
    print(f"    MRR-weighted @k       {weighted:.4f}   <- the one that must clear p*")
    print(f"    short by a factor of  {breakeven/max(weighted,1e-9):.0f}")

    # Is the model the bottleneck? Ask an oracle that knows y perfectly.
    oracle = lab11.topk_flag(y.astype(float) * mrr, k)
    oracle_net = (SAVE_RATE * VALUE_MONTHS * mrr[oracle & (y == 1)].sum()
                  - OFFER_MONTHS * mrr[oracle].sum())
    print(f"\n  an ORACLE that knows the outcome perfectly, picking the k most valuable")
    print(f"  true churners, still nets ${oracle_net:,.0f}. No ranking and no model can")
    print(f"  rescue this campaign - the intervention itself is unprofitable.")

    print(f"\n  the value-maximising ORDER depends on the cost structure, and only on that:")
    print(f"    proportional offer c_i = offer x m_i -> E_i = m_i*s*V*(p_i - p*), order by "
          f"m_i(p_i - p*)")
    print(f"    flat cost c            -> E_i = m_i*s*V*p_i - c,       order by m_i*p_i")
    print(f"  {'ranking':<26}{'proportional':>16}{'flat $40/contact':>20}")
    for tag, ranking in [("P(churn)", scores),
                         ("P(churn) x MRR", scores * mrr),
                         ("MRR x (P(churn) - p*)", mrr * (scores - breakeven))]:
        flag = lab11.topk_flag(ranking, k)
        saved = SAVE_RATE * VALUE_MONTHS * mrr[flag & (y == 1)].sum()
        prop = saved - OFFER_MONTHS * mrr[flag].sum()
        flat = saved - 40.0 * flag.sum()
        print(f"  {tag:<26}${prop:>15,.0f}${flat:>19,.0f}")
    print( "  -> neither ordering is universally right. Under the proportional offer every")
    print( "     option loses; under a flat cost P x MRR is the best of the three, which is")
    print( "     the ordering the proportional analysis would have told you to avoid.")
    print( "  (Note we do NOT report an 'expected value' column: expected value under the")
    print( "   model's own scores is exactly what each ranking sorts on, so the winner there")
    print( "   is an identity, not evidence.)")

    # What class_weight="balanced" actually does to the scores.
    pi = float(train["churns_in_horizon"].mean())
    odds = scores / (1 - scores)
    cal = odds * pi / (1 - pi)
    cal = cal / (1 + cal)
    print(f"\n  a note on the scores: class_weight='balanced' shifts the LOG-ODDS by a")
    print(f"  constant (multiplier (1-pi)/pi = {(1-pi)/pi:.1f} at pi={pi:.4f}), it does not")
    print(f"  scale probabilities. Undo that shift and the model is well calibrated overall:")
    print(f"    raw mean {scores.mean():.4f}  ->  prior-corrected {cal.mean():.4f}   "
          f"observed {y.mean():.4f}")
    print(f"    the distortion is {np.sort(scores)[len(scores)//2]/max(np.sort(cal)[len(cal)//2],1e-9):.0f}x "
          f"at the median and {scores.max()/max(cal.max(),1e-9):.1f}x at the top, so 'inflated")
    print(f"    N-fold' is not a well-defined statement. The scores RANK fine; what they")
    print(f"    carry is the wrong prior, which matters the moment they enter a cost model.")

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
    print("  which is why value-weighting closed most of the gap there and does not pay here.")


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


# ------------------------------ L7: the label spec as a production artifact (Stage C)

LOG = logging.getLogger("payflow.labelspec")


class LabelSpecError(ValueError):
    """A spec that is internally inconsistent, or that this data cannot support."""


@dataclass(frozen=True)
class LabelSpec:
    """The reviewed document the permanent fix demands, as an object a pipeline enforces.

    Stage C here is deliberately not a FastAPI service. What has to survive contact with
    production in a framing notebook is a DEFINITION, and the incident this notebook
    opens with is a definition that travelled into a funding decision without its
    provenance attached. So the production shape is a versioned, hashed, self-validating
    artifact that 01.3's run manifest cites and that CI can refuse to build labels
    without - the same discipline a schema migration gets, applied to the target column.

    Frozen on purpose: a spec that can be edited in place is a spec whose fingerprint
    lies. `apply()` returns a NEW spec carrying the measured base rate.
    """

    name: str
    version: int
    unit: str                    # the grain one row represents
    population: str              # "active_at_cutoff" | "all_signed_up"
    cutoff: str                  # ISO-8601; features use data STRICTLY before this
    horizon_days: int            # 0 means "no horizon" - a state, not an event
    event: str
    owner: str
    # Measured by apply(), never asserted by an author.
    base_rate: float | None = None
    n_rows: int | None = None
    n_positives: int | None = None

    # ------------------------------------------------------------------ identity
    def canonical(self) -> str:
        """Stable JSON for hashing. Measured fields are excluded: re-measuring the same
        definition on more data must not look like a different definition."""
        d = {k: v for k, v in asdict(self).items()
             if k not in ("base_rate", "n_rows", "n_positives")}
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:12]

    # ---------------------------------------------------------------- validation
    def violations(self, data_end: pd.Timestamp) -> list[str]:
        """Every reason this spec must not reach a model or a business case.

        Empty list means usable. Each rule below is one clause of this notebook's
        incident, written down so it fails a build instead of a quarter.
        """
        out: list[str] = []
        cutoff = pd.Timestamp(self.cutoff)
        if self.horizon_days <= 0:
            out.append("no horizon: the target is a STATE ('is a churner'), not a "
                       "time-bounded event - it can be recognised, not predicted")
        elif cutoff + pd.Timedelta(days=self.horizon_days) > data_end:
            over = (cutoff + pd.Timedelta(days=self.horizon_days) - data_end).days
            out.append(f"right-censored: the horizon ends {over} days past the data "
                       f"({data_end:%Y-%m-%d}), so this is really the label 'churns "
                       f"before the data ends'")
        if self.population != "active_at_cutoff":
            out.append(f"population '{self.population}' admits customers who had already "
                       f"churned at the cutoff - the model would recognise the past")
        if self.unit != "customer":
            out.append(f"unit '{self.unit}' is not the decision unit (one contact "
                       f"per customer)")
        return out

    def require_valid(self, data_end: pd.Timestamp) -> None:
        bad = self.violations(data_end)
        if bad:
            raise LabelSpecError(f"{self.name}@v{self.version} is not usable:\n  - "
                                 + "\n  - ".join(bad))

    # ------------------------------------------------------------------- use
    def apply(self, f: pd.DataFrame,
              data_end: pd.Timestamp) -> tuple[pd.DataFrame, "LabelSpec"]:
        """The ONLY supported way to build this label, so spec and data cannot drift."""
        self.require_valid(data_end)
        d = label_frame(f, pd.Timestamp(self.cutoff), self.horizon_days,
                        active_only=self.population == "active_at_cutoff")
        measured = replace(self, base_rate=round(float(d["churns_in_horizon"].mean()), 4),
                           n_rows=int(len(d)),
                           n_positives=int(d["churns_in_horizon"].sum()))
        LOG.info("applied %s@v%d fp=%s n=%d positives=%d base_rate=%.4f", self.name,
                 self.version, self.fingerprint(), measured.n_rows,
                 measured.n_positives, measured.base_rate)
        return d, measured

    def assert_scorable(self, frame: pd.DataFrame) -> None:
        """The acceptance test the Prevention list promises: no already-churned customer
        may reach a contact list. Cheap, and it would have caught this incident on the
        first run rather than at the quarter's end."""
        cutoff = pd.Timestamp(self.cutoff)
        bad = int((frame["churn_date"].notna() & (frame["churn_date"] <= cutoff)).sum())
        if bad:
            raise LabelSpecError(
                f"{bad:,} rows had already churned at {cutoff:%Y-%m-%d}: a contact list "
                f"built from this frame offers win-back discounts to customers who left")

    def manifest_entry(self) -> dict:
        """What 01.3's manifest config block carries, so a base rate never travels
        without the definition that produced it."""
        return {"label_spec": f"{self.name}@v{self.version}",
                "fingerprint": self.fingerprint(), "population": self.population,
                "horizon_days": self.horizon_days, "base_rate": self.base_rate}

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "LabelSpec":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# The spec the Q4 campaign actually ran on, reconstructed from its business case, and
# the one that survives review. Both are checked below rather than described.
CAMPAIGN_SPEC = LabelSpec(
    name="retention_churn", version=1, unit="customer", population="all_signed_up",
    cutoff="2025-06-30", horizon_days=0, event="customer has a churn_date at any time",
    owner="growth-analytics")

OPERATIONAL_SPEC = LabelSpec(
    name="retention_churn", version=2, unit="customer", population="active_at_cutoff",
    cutoff="2025-06-30", horizon_days=180,
    event="churn_date falls in (cutoff, cutoff + 180 days]", owner="growth-analytics")


def label_spec_artifact(f: pd.DataFrame,
                        artifact_dir: Path | None = None) -> LabelSpec:
    """Build, validate, measure and persist the spec that survives review (Stage C)."""
    ok = OPERATIONAL_SPEC
    print(f"  data ends {DATA_END:%Y-%m-%d} (the extraction date, from _data/SPEC.md); "
          f"the gate runs\n  before a single model is fitted\n")
    print(f"  {ok.name}@v{ok.version}  population={ok.population}  "
          f"horizon={ok.horizon_days}d  -> ACCEPTED ({len(ok.violations(DATA_END))} "
          f"violations)")
    frame, measured = ok.apply(f, DATA_END)
    print(f"    measured on {measured.n_rows:,} rows: {measured.n_positives:,} "
          f"positives, base rate {measured.base_rate}")
    print(f"    manifest entry: {json.dumps(measured.manifest_entry())}")
    ok.assert_scorable(frame)
    print(f"    assert_scorable(the spec's own frame): PASS")

    if artifact_dir is not None:
        path = measured.save(artifact_dir / "label_spec.json")
        reloaded = LabelSpec.load(path)
        print(f"    written to {path.name}, re-read, fingerprint stable: "
              f"{reloaded.fingerprint() == measured.fingerprint()} "
              f"({measured.fingerprint()})")
    return measured


def label_spec_rejects(f: pd.DataFrame) -> None:
    """The same gate, run against the spec the Q4 campaign actually shipped on.

    Nothing here fits a model. Every rejection below was available on day one from the
    definition alone, which is what makes the absence of this artifact the incident.
    """
    for spec, why in ((CAMPAIGN_SPEC, "the spec the campaign ran on"),
                      (replace(OPERATIONAL_SPEC, version=3, horizon_days=730),
                       "the 'more positives' variant someone proposes next")):
        bad = spec.violations(DATA_END)
        print(f"  {spec.name}@v{spec.version} - {why}")
        print(f"    population={spec.population}  horizon={spec.horizon_days}d  "
              f"-> REJECTED, {len(bad)} violation(s)")
        for v in bad:
            print(f"      - {v}")

    naive = f.loc[f["signup_date"] <= pd.Timestamp(OPERATIONAL_SPEC.cutoff)]
    print(f"\n  and the acceptance test, against the contact pool actually scored:")
    try:
        OPERATIONAL_SPEC.assert_scorable(naive)
        print(f"    assert_scorable: PASS")
    except LabelSpecError as exc:
        print(f"    assert_scorable: FAIL - {exc}")


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

    print("\n" + "=" * 78)
    print("L7  THE LABEL SPEC AS A PRODUCTION ARTIFACT")
    print("=" * 78)
    label_spec_artifact(f)
    print()
    label_spec_rejects(f)


if __name__ == "__main__":
    main()
