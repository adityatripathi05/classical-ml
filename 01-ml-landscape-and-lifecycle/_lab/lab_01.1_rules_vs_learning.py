"""Lab for notebook 01.1 "Rules or Learning? The Decision That Precedes the Model".

Reproduces every captured number and listing in 01.1:

  L1  dataset construction from PayFlow raw exports + the `late` label base rate
  L2  Stage A - the hand-written dunning rule, measured at its own queue size
  L3  Stage B - logistic regression at the SAME operating point, + a gradient-boosting
      check (is any gap about the model CLASS or about the problem?)
  L4  repeat-run variance on both precision and dollars, so "which delta is signal"
      is answerable
  L5  the incident - fixed-threshold production behaviour across the 2025-07 gateway
      migration (SPEC M9): calibration gap, queue that does not grow, recall collapse
  L5b the fix - refit on the new regime; what it repairs and what it does not
  L6  the money frame - value captured vs cost of the queue

Data: _data/raw/{invoices,payments,customers}.csv[.gz] (regenerate: python _data/generate.py).
Mess handled here (SPEC.md): M1 duplicate invoice rows, M2 mixed currencies, M3 unit-glitch
window, M6 status casing, M7 comma-formatted amounts, M8 two paid_at formats, M13 partial
payments. Deliberately EXCLUDED as leakage (M10): reminder_count, total_lifetime_value_usd;
also gateway/fx_rate (known only after payment).

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.1_rules_vs_learning.py"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
RAW = Path(__file__).resolve().parents[2] / "_data" / "raw"
GRACE_DAYS = 7          # "late" = paid more than a week past due
GLITCH = ("2025-06-10", "2025-06-17")   # SPEC M3: INR amounts exported in thousands
MIGRATION = pd.Timestamp("2025-07-01")  # SPEC M9: stripe -> adyen for non-IN customers

CAT = ["segment", "country", "plan", "industry"]
NUM = ["amount_usd", "terms_days", "seats", "tenure_days", "issue_month", "employee_count"]


# ----------------------------------------------------------------- L1: dataset

def parse_paid_at(s: pd.Series) -> pd.Series:
    """SPEC M8: most gateways write ISO-8601Z, paylane writes DD/MM/YYYY HH:MM.

    Parsing with a single format silently coerces the paylane rows to NaT and drops
    ~15% of payments - a bias, not a rounding error. Parse both, keep everything.
    """
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None)
    legacy = pd.to_datetime(s[iso.isna()], format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(legacy)


def build_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    inv = pd.read_csv(RAW / "invoices.csv.gz")
    pay = pd.read_csv(RAW / "payments.csv.gz")
    cus = pd.read_csv(RAW / "customers.csv")
    audit = {"invoices_raw": len(inv), "payments_raw": len(pay)}

    inv = inv.drop_duplicates()                                    # M1
    audit["exact_duplicate_rows_dropped"] = audit["invoices_raw"] - len(inv)
    inv["status"] = inv["status"].str.lower()                      # M6
    # M7: legacy exports carry '4,554.29' -> the whole column is str dtype under pandas 3
    audit["comma_formatted_amounts"] = int(inv["amount"].astype(str).str.contains(",").sum())
    inv["amount"] = pd.to_numeric(inv["amount"].astype(str).str.replace(",", ""),
                                  errors="raise")
    inv["issue_date"] = pd.to_datetime(inv["issue_date"])
    inv["due_date"] = pd.to_datetime(inv["due_date"])

    glitch = (inv["currency"].eq("INR")
              & inv["issue_date"].between(*pd.to_datetime(GLITCH)))  # M3
    audit["glitch_window_rows_excluded"] = int(glitch.sum())
    inv = inv.loc[~glitch]

    pay["paid_dt"] = parse_paid_at(pay["paid_at"])
    audit["payments_lost_to_naive_iso_parse"] = int(
        pd.to_datetime(pay["paid_at"], format="ISO8601", errors="coerce").isna().sum())
    first = pay.groupby("invoice_id", as_index=False).agg(       # M13: partial payments
        paid_dt=("paid_dt", "min"), fx_rate_usd=("fx_rate_usd", "median"))

    df = inv.merge(first, on="invoice_id", how="inner").merge(
        cus[["customer_id", "segment", "country", "plan", "industry", "seats",
             "employee_count", "signup_date"]], on="customer_id", how="inner")
    audit["resolved_invoices"] = len(df)

    # M2: amounts are local currency. The FX table is built from TRAIN-PERIOD payments
    # only and applied everywhere - a reference table, not a per-row post-payment fact.
    train_fx = df.loc[df["issue_date"] < "2025-01-01"]
    fx = train_fx.groupby("currency")["fx_rate_usd"].median()
    df["amount_usd"] = df["amount"] / df["currency"].map(fx)

    df["days_late"] = (df["paid_dt"] - df["due_date"]).dt.days
    df["late"] = (df["days_late"] > GRACE_DAYS).astype(int)
    df["terms_days"] = (df["due_date"] - df["issue_date"]).dt.days
    df["tenure_days"] = (df["issue_date"] - pd.to_datetime(df["signup_date"])).dt.days
    df["issue_month"] = df["issue_date"].dt.month
    df["employee_count"] = df["employee_count"].replace(-999, np.nan)   # M4 sentinel
    return df, audit


def windows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    iss = df["issue_date"]
    return {
        "train":       df.loc[iss < "2025-01-01"],
        "test_stable": df.loc[(iss >= "2025-01-01") & (iss < "2025-06-01")],
        "test_drift":  df.loc[(iss >= MIGRATION) & (iss < "2026-05-31")],
    }


# --------------------------------------------------- L2: the hand-written rule

def dunning_rule(d: pd.DataFrame) -> np.ndarray:
    """Collections' actual policy, as written on a whiteboard in 2024.

    "Chase the big ones, the enterprises, and the Gulf accounts." Three clauses, no
    training, no data science - and it is the thing a model has to beat.
    """
    return ((d["amount_usd"] > 1500)
            | (d["segment"] == "Enterprise")
            | (d["country"] == "AE")).to_numpy()


def precision_recall_at_k(y: np.ndarray, flag: np.ndarray) -> dict[str, float]:
    k = int(flag.sum())
    tp = int((flag & (y == 1)).sum())
    return {"k": k, "queue_share": k / len(y), "precision": tp / max(k, 1),
            "recall": tp / max(int(y.sum()), 1)}


def topk_flag(scores: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(len(scores), dtype=bool)
    out[np.argsort(-scores)[:k]] = True
    return out


# ------------------------------------------------------- L3: the learned model

def make_model() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUM),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def make_boosted() -> Pipeline:
    """Strength check only - boosting is taught canonically in series 20.

    If a gradient-boosted model with native categorical handling also fails to pull
    away from the rule, the ceiling belongs to the PROBLEM, not to the model class.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", max_categories=20), CAT)])),
        ("clf", HistGradientBoostingClassifier(random_state=SEED)),
    ])


def fit_score(train: pd.DataFrame, test: pd.DataFrame, seed: int = SEED,
              factory=make_model, weight_by_value: bool = False) -> np.ndarray:
    """Fit on `train`, score `test`.

    weight_by_value re-states the objective: a late $9,000 invoice matters more than a
    late $90 one. Same features, same algorithm, same data - only the loss weighting
    changes, which is the whole point of L6.
    """
    model = factory()
    model.named_steps["clf"].set_params(random_state=seed)
    kw = {"clf__sample_weight": train["amount_usd"].to_numpy()} if weight_by_value else {}
    model.fit(train[NUM + CAT], train["late"], **kw)
    return model.predict_proba(test[NUM + CAT])[:, 1]


# ------------------------------------------------------------- L6: money frame

ACTION_COST = 2.50         # USD per invoice worked (mostly automated dunning sequence)
RECOVERY_FRACTION = 0.40   # share of lateness avoided when chased early
DAILY_CAPITAL_COST = 0.12 / 365

# NOTE the direction of the assumptions: this values ONLY the working-capital saved by
# being paid sooner. It ignores avoided write-offs, which would favour chasing more.
# At a fixed queue size k the cost term k*ACTION_COST is identical across policies, so
# the RANKING of policies is set by value_captured alone and is robust to the cost level.


def value_captured(d: pd.DataFrame, flag: np.ndarray) -> float:
    caught = d.loc[flag & (d["late"] == 1)]
    return float((RECOVERY_FRACTION * caught["days_late"].clip(lower=0)
                  * caught["amount_usd"] * DAILY_CAPITAL_COST).sum())


def report(name: str, d: pd.DataFrame, flag: np.ndarray) -> None:
    m = precision_recall_at_k(d["late"].to_numpy(), flag)
    val = value_captured(d, flag)
    print(f"  {name:<26} k={m['k']:>6,} ({m['queue_share']:5.1%})  "
          f"prec={m['precision']:.3f}  rec={m['recall']:.3f}  "
          f"value=${val:>8,.0f}  net=${val - m['k'] * ACTION_COST:>9,.0f}")


def main() -> None:
    pd.set_option("display.width", 100)
    df, audit = build_dataset()
    w = windows(df)

    print("=" * 78)
    print("L1  DATASET")
    print("=" * 78)
    for k, v in audit.items():
        print(f"  {k:<34} {v:>9,}")
    print(f"  {'late rate (all resolved)':<34} {df['late'].mean():>9.3f}")
    for name, part in w.items():
        print(f"  {name:<12} n={len(part):>7,}  "
              f"issued {part['issue_date'].min():%Y-%m}..{part['issue_date'].max():%Y-%m}  "
              f"late rate={part['late'].mean():.3f}  "
              f"median days_late={part['days_late'].median():.0f}")

    print("\n" + "=" * 78)
    print("L2/L3  RULE vs MODEL on the stable window (identical queue size k)")
    print("=" * 78)
    stable = w["test_stable"]
    rule_flag = dunning_rule(stable)
    k = int(rule_flag.sum())
    y = stable["late"].to_numpy()
    rng = np.random.default_rng(SEED)
    print(f"  base rate = {y.mean():.3f}; every policy below works the same k={k:,}")
    report("random at same k", stable, topk_flag(rng.random(len(stable)), k))
    report("Stage A: dunning rule", stable, rule_flag)
    scores = fit_score(w["train"], stable)
    report("Stage B: logistic reg", stable, topk_flag(scores, k))
    scores_gb = fit_score(w["train"], stable, factory=make_boosted)
    report("Stage B': grad boosting", stable, topk_flag(scores_gb, k))
    report("work everything", stable, np.ones(len(stable), dtype=bool))
    print(f"  mean amount_usd of invoices caught:  rule "
          f"${stable.loc[rule_flag & (stable['late'] == 1), 'amount_usd'].mean():,.0f}  "
          f"vs model $"
          f"{stable.loc[topk_flag(scores, k) & (stable['late'] == 1), 'amount_usd'].mean():,.0f}"
          "   <- precision counts invoices, the business counts money")

    print("\n" + "=" * 78)
    print("L4  VARIANCE - 8 bootstrap refits; is any gap signal?")
    print("=" * 78)
    precs, vals = [], []
    for s in range(8):
        boot = w["train"].sample(frac=1.0, replace=True, random_state=s)
        sc = fit_score(boot, stable, seed=s)
        flag = topk_flag(sc, k)
        precs.append(precision_recall_at_k(y, flag)["precision"])
        vals.append(value_captured(stable, flag))
    rule_p = precision_recall_at_k(y, rule_flag)["precision"]
    rule_v = value_captured(stable, rule_flag)
    print(f"  model precision@k = {np.mean(precs):.4f} +/- {np.std(precs):.4f}  "
          f"| rule = {rule_p:.4f} (deterministic)")
    print(f"  model value       = ${np.mean(vals):,.0f} +/- ${np.std(vals):,.0f}  "
          f"| rule = ${rule_v:,.0f}")
    print(f"  precision delta {np.mean(precs) - rule_p:+.4f} vs noise band "
          f"{2 * np.std(precs):.4f}  -> "
          f"{'SIGNAL' if abs(np.mean(precs) - rule_p) > 2 * np.std(precs) else 'NOISE'}")
    print(f"  value delta     ${np.mean(vals) - rule_v:+,.0f} vs noise band "
          f"${2 * np.std(vals):,.0f}  -> "
          f"{'SIGNAL' if abs(np.mean(vals) - rule_v) > 2 * np.std(vals) else 'NOISE'}")

    print("\n" + "=" * 78)
    print("L5  THE INCIDENT - production runs a FIXED THRESHOLD, not a fixed k")
    print("=" * 78)
    drift = w["test_drift"]
    yd = drift["late"].to_numpy()
    scores_d = fit_score(w["train"], drift)
    thr = 0.35   # the threshold collections shipped with, chosen on 2024 validation data
    for label, part, sc, yy in [("stable 2025-01..05", stable, scores, y),
                                ("drift  2025-07..2026-05", drift, scores_d, yd)]:
        flag = sc >= thr
        m = precision_recall_at_k(yy, flag)
        print(f"  {label:<24} predicted late rate={sc.mean():.3f}  ACTUAL={yy.mean():.3f}  "
              f"queue={m['queue_share']:5.1%}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f}")
    print(f"  calibration gap widens {y.mean() - scores.mean():+.3f} -> "
          f"{yd.mean() - scores_d.mean():+.3f}: the score distribution did not move, "
          "so the queue did not grow while lateness did")
    print(f"  non-IN median days_late: "
          f"{stable.loc[stable['country'] != 'IN', 'days_late'].median():.0f} -> "
          f"{drift.loc[drift['country'] != 'IN', 'days_late'].median():.0f}  (SPEC M9)")
    print(f"  IN     median days_late: "
          f"{stable.loc[stable['country'] == 'IN', 'days_late'].median():.0f} -> "
          f"{drift.loc[drift['country'] == 'IN', 'days_late'].median():.0f}  (unaffected)")
    print("  -- and at a matched queue size, the ranking contest itself flips:")
    rule_d = dunning_rule(drift)
    report("Stage A: dunning rule", drift, rule_d)
    report("Stage B: logistic reg", drift, topk_flag(scores_d, int(rule_d.sum())))

    print("\n" + "=" * 78)
    print("L5b  THE FIX - refit on the new regime: what it repairs, what it does not")
    print("=" * 78)
    eval_w = df.loc[(df["issue_date"] >= "2026-01-01") & (df["issue_date"] < "2026-05-31")]
    ye = eval_w["late"].to_numpy()
    r_flag = dunning_rule(eval_w)
    kk = int(r_flag.sum())
    stale = fit_score(w["train"], eval_w)
    post_only = df.loc[(df["issue_date"] >= MIGRATION) & (df["issue_date"] < "2026-01-01")]
    fresh = fit_score(post_only, eval_w)
    print(f"  eval 2026-01..05  n={len(eval_w):,}  actual late rate={ye.mean():.3f}")
    print(f"  stale model (train<2025-01) predicts {stale.mean():.3f} -> gap "
          f"{ye.mean() - stale.mean():+.3f}")
    print(f"  refit model (train 2025-07..12, n={len(post_only):,}) predicts "
          f"{fresh.mean():.3f} -> gap {ye.mean() - fresh.mean():+.3f}  <- calibration repaired")
    report("rule", eval_w, r_flag)
    report("model (stale)", eval_w, topk_flag(stale, kk))
    report("model (refit on regime)", eval_w, topk_flag(fresh, kk))
    print("  ranking barely moves: retraining fixed the LEVEL, not the ORDER - the rule's "
          "three clauses already carry the ordering signal")

    print("\n" + "=" * 78)
    print("L6  THE ACTUAL FIX - change the objective, not the algorithm")
    print("=" * 78)
    print("  same features, same LogisticRegression, same rows; only sample_weight differs")
    for label, wt in [("model (unweighted)", False), ("model (value-weighted)", True)]:
        sc = fit_score(w["train"], stable, weight_by_value=wt)
        report(label, stable, topk_flag(sc, k))
    report("Stage A: dunning rule", stable, rule_flag)
    wv, rv = [], value_captured(stable, rule_flag)
    for s in range(8):
        boot = w["train"].sample(frac=1.0, replace=True, random_state=s)
        sc = fit_score(boot, stable, seed=s, weight_by_value=True)
        wv.append(value_captured(stable, topk_flag(sc, k)))
    print(f"  value-weighted model value = ${np.mean(wv):,.0f} +/- ${np.std(wv):,.0f}  "
          f"| rule ${rv:,.0f}  -> delta ${np.mean(wv) - rv:+,.0f} "
          f"({'SIGNAL' if abs(np.mean(wv) - rv) > 2 * np.std(wv) else 'NOISE'})")


if __name__ == "__main__":
    main()
