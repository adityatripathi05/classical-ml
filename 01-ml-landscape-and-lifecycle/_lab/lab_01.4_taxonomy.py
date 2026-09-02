"""Lab for notebook 01.4 "The Taxonomy of Learning Problems".

Reproduces every captured number and listing in 01.4:

  L1  one invoice table, four learning problems - shapes, targets, eval signatures, and
      how much the four framings DISAGREE about which invoices matter
  L2  what each framing buys on the business objective (value captured, per 01.1)
  L3  the incident - a fixed probability threshold against a capacity-shaped problem
  L4  the estimator API as the taxonomy: which methods exist tells you the problem type
  L5  label availability - why unsupervised methods exist at all (censoring by recency)
  L6  what the choice costs in production: parametric vs non-parametric, batch vs online
  L7  the serving contract as production code - a capacity queue that asserts its own
      size, logs a decision record, and refuses malformed requests (Stage C)

Reuses the 01.1 dataset builder. Runtime ~5 min on CPU (the framing-band refits and the
KNN latency probe dominate).

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.4_taxonomy.py"
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 42
# PayFlow's collections roster, stated ONCE for the series and consistent with 01.1:
# the dunning rule queues 7,195 of the stable window's invoices over its 151 calendar
# days, ~47.6/day -> a roster of 48. Every monthly figure below is CAP_DAY times the
# window's own day span, so daily and monthly capacity cannot drift apart.
CAP_DAY = 48
THRESHOLD = 0.5         # the default nobody chose but everybody ships


def eval_month(df: pd.DataFrame) -> pd.DataFrame:
    """One CONTIGUOUS month of invoices, not a head() slice.

    `head(5_500)` of the stable window looks like a month by row count but spans 140
    calendar days, because the frame is not date-ordered - so every per-day figure derived
    from it would be a sampled rate, not PayFlow's actual daily volume. Bounding by date
    makes the daily arithmetic mean what it says.
    """
    return df.loc[(df["issue_date"] >= "2025-01-01") & (df["issue_date"] < "2025-02-01")]


def load_sibling(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


lab11 = load_sibling("lab_01.1_rules_vs_learning.py", "lab_01_1")


# ------------------------------------------------- L1: one table, four framings

def build_framings(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """The SAME rows and features, four different learning problems.

    Only the target changes - and with it, what can be measured and what is returned.
    """
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression

    X_tr, X_te = train[lab11.NUM + lab11.CAT], test[lab11.NUM + lab11.CAT]

    print(f"  feature matrix: train {X_tr.shape}, test {X_te.shape}  "
          f"({len(lab11.NUM)} numeric + {len(lab11.CAT)} categorical columns)\n")
    rows = [
        ("regression", "days_late (int, unbounded)", "MAE / RMSE", "a number"),
        ("binary classification", "late = days_late > 7 (0/1)", "PR-AUC, precision@k",
         "a probability"),
        ("ranking under capacity", "late, but only order matters", "precision@k",
         "exactly k ids"),
        ("clustering", "no target at all", "silhouette, stability", "a group id"),
    ]
    print(f"  {'framing':<24}{'target':<30}{'evaluated with':<22}returns")
    for name, target, metric, returns in rows:
        print(f"  {name:<24}{target:<30}{metric:<22}{returns}")

    # Fit the two supervised framings on identical features.
    reg = lab11.make_model()
    reg.set_params(clf=LinearRegression())
    reg.fit(X_tr, train["days_late"])
    reg_pred = reg.predict(X_te)

    clf_scores = lab11.fit_score(train, test, seed=SEED)

    km = lab11.make_model()
    km.set_params(clf=KMeans(n_clusters=4, n_init=10, random_state=SEED))
    km.fit(X_tr)             # y is accepted (and ignored) by convention; nothing supervises this fit
    clusters = km.predict(X_te)
    sizes = np.bincount(clusters, minlength=4).tolist()
    n_used = int(sum(s > 0 for s in sizes))
    print(f"\n  clustering asked for 4 groups and used {n_used}, sizes {sizes} - one is")
    print( "  empty, and nothing in the method calls that an error, because there is no")
    print( "  notion of 'correct' anywhere in it")
    return {"regression": reg_pred, "classification": clf_scores, "clusters": clusters}


def framing_band(pool: pd.DataFrame, n_runs: int = 8) -> tuple[float, float]:
    """The paired band for THIS comparison: classification-minus-regression top-k.

    01.3's 0.0136 band belongs to a different pair (model minus rule); borrowing it here
    would judge one comparison by another comparison's noise - invariant 3's own error,
    one level up. The two framings correlate less across evaluation draws than the model
    and the rule do, so the borrowed band understates the noise. Measure the right one.
    Runtime ~2 min CPU (8 refits of each framing on ~154k rows).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    p_cs, p_rs = [], []
    for s in range(n_runs):
        tr, te = train_test_split(pool, test_size=5_500, random_state=s)
        kk = int(lab11.dunning_rule(te).sum())
        y = te["late"].to_numpy()
        clf_scores = lab11.fit_score(tr, te, seed=SEED)
        reg = lab11.make_model()
        reg.set_params(clf=LinearRegression())
        reg.fit(tr[lab11.NUM + lab11.CAT], tr["days_late"])
        reg_pred = reg.predict(te[lab11.NUM + lab11.CAT])
        p_cs.append(lab11.precision_recall_at_k(y, lab11.topk_flag(clf_scores, kk))["precision"])
        p_rs.append(lab11.precision_recall_at_k(y, lab11.topk_flag(reg_pred, kk))["precision"])
    d = np.asarray(p_cs) - np.asarray(p_rs)
    mean, band = float(d.mean()), float(2 * d.std(ddof=1))
    corr = float(np.corrcoef(p_cs, p_rs)[0, 1])
    print(f"  paired band for THIS comparison, over {n_runs} evaluation redraws:")
    print(f"    clf-minus-reg delta: mean {mean:+.4f}   2-sigma band {band:.4f}")
    print(f"    the two framings correlate at {corr:.2f} across draws - less than the")
    print(f"    model and the rule do in 01.3, so less of the noise is common-mode and")
    print(f"    the borrowed band would understate this comparison's spread")
    return mean, band


def framing_disagreement(test: pd.DataFrame, preds: dict, k: int,
                         band: float | None = None) -> None:
    """Do the framings pick the same invoices? If not, the framing IS the decision."""
    top_reg = lab11.topk_flag(preds["regression"], k)
    top_clf = lab11.topk_flag(preds["classification"], k)
    overlap = int((top_reg & top_clf).sum())
    print(f"  top-{k:,} by predicted days_late vs by predicted P(late):")
    print(f"    agree on {overlap:,} of {k:,} slots ({overlap / k:.1%}); "
          f"{k - overlap:,} slots ({1 - overlap / k:.1%}) differ")
    y = test["late"].to_numpy()
    scored = {}
    for name, flag in [("regression top-k", top_reg), ("classification top-k", top_clf)]:
        m = lab11.precision_recall_at_k(y, flag)
        scored[name] = m["precision"]
        print(f"    {name:<22} precision={m['precision']:.4f}  "
              f"value=${lab11.value_captured(test, flag):>8,.0f}")
    gap = scored["classification top-k"] - scored["regression top-k"]
    if band is not None:
        verdict = ("inside" if abs(gap) <= band else "outside")
        tail = ("indistinguishable in quality" if abs(gap) <= band
                else "a quality difference")
        print(f"    precision gap between the two framings: {gap:.4f}  -> {verdict} the "
              f"{band:.4f} band measured for this pair: {tail}, while "
              f"{1 - overlap / k:.0%} of the WORK differs")


# ------------------------------------------------------ L3: the framing incident

def threshold_vs_capacity(test: pd.DataFrame, scores: np.ndarray) -> None:
    """Classification answers 'is it likely?'. The business asked 'which k today?'.

    Reports precision at every operating point, because comparing a threshold's queue
    against a top-k queue of a DIFFERENT size would be the matched-operating-point error
    01.1 forbids - and it is the reason top-k's apparent precision edge is not a win.
    """
    y = test["late"].to_numpy()
    span = test["issue_date"].nunique()          # calendar days actually covered
    per_day = len(test) / span
    k_month = CAP_DAY * span                     # the roster, over this window's days
    print(f"  evaluation window: {len(test):,} invoices over {span} calendar days "
          f"(~{per_day:.0f}/day), actual late rate {test['late'].mean():.3f}")
    print(f"  collections roster: {CAP_DAY}/day x {span} days = {k_month:,} slots for "
          f"this window ({k_month / len(test):.1%} of volume)")
    print(f"  model mean predicted P(late): {scores.mean():.3f}, max {scores.max():.3f}\n")
    print(f"  {'operating point':<26}{'queue':>8}{'per day':>10}{'precision':>12}")
    for t in (0.3, 0.4, 0.5, 0.6):
        flag = scores >= t
        m = lab11.precision_recall_at_k(y, flag)
        print(f"  {f'threshold {t:.1f}':<26}{m['k']:>8,}{m['k']/span:>10.0f}"
              f"{m['precision']:>12.4f}")
    top = lab11.topk_flag(scores, k_month)
    m = lab11.precision_recall_at_k(y, top)
    implied = float(np.sort(scores)[-k_month])
    print(f"  {f'top-{k_month:,} (capacity)':<26}{k_month:>8,}"
          f"{CAP_DAY:>10.0f}{m['precision']:>12.4f}")
    print(f"\n  top-k is itself a threshold rule - here the implied cutoff is "
          f"{implied:.4f}. At a MATCHED")
    print( "  queue size the two selectors return the identical set, so capacity selection")
    print( "  buys size CONTROL, not precision. The precision column moves only because the")
    print( "  operating point moves.")
    thr_flag = scores >= THRESHOLD
    thr_caught = int((thr_flag & (y == 1)).sum())
    top_caught = int((top & (y == 1)).sum())
    print(f"\n  what the business actually gets, at the SAME staffing cost:")
    print(f"    threshold {THRESHOLD}: works {int(thr_flag.sum()):,} invoices, catches "
          f"{thr_caught:,} late ones (recall "
          f"{lab11.precision_recall_at_k(y, thr_flag)['recall']:.3f})")
    print(f"    top-{k_month:,}     : works {k_month:,} invoices, catches {top_caught:,} "
          f"late ones (recall {m['recall']:.3f})")
    print(f"    -> {top_caught/max(thr_caught,1):.1f}x the late invoices caught, using "
          f"capacity that was already being paid for")


def live_model_and_days(df: pd.DataFrame,
                        n_days: int = 20) -> tuple[object, pd.DataFrame, list]:
    """The era-consistent scoring setup every daily view in this notebook shares.

    The model live in spring 2026 is the one RETRAINED on the post-migration regime
    (01.1's permanent fix, mid-2025), fitted ONCE and then scoring each morning's
    invoices, exactly as the scoring job does. Scoring these days with the stale
    pre-2025 model would manufacture a more dramatic starvation out of 01.1's drift
    incident, which is a different failure.

    Extracted so the threshold view and the capacity view below cannot drift apart:
    an inline second copy of this setup is how two sections end up disagreeing about
    which model produced which day.
    """
    post = df.loc[(df["issue_date"] >= lab11.MIGRATION)
                  & (df["issue_date"] < "2026-01-01")]
    model = lab11.make_model()
    model.fit(post[lab11.NUM + lab11.CAT], post["late"])    # fitted ONCE, like production
    window = df.loc[(df["issue_date"] >= "2026-03-01") & (df["issue_date"] < "2026-04-14")]
    days = [d for d in sorted(window["issue_date"].unique())
            if (window["issue_date"] == d).sum() >= 30][-n_days:]
    return model, window, days


def daily_queue_swing(df: pd.DataFrame, n_days: int = 20,
                      cap_day: int = CAP_DAY) -> None:
    """The actual incident: a FIXED threshold on a MOVING daily score distribution.

    The threshold sweep above varies t on one pooled window; that is not what production
    does. Production holds t fixed and meets a different day's invoices every morning.
    """
    model, window, days = live_model_and_days(df, n_days)
    rows, pooled_pred, pooled_actual, pooled_n = [], 0.0, 0.0, 0
    for day in days:
        d = window.loc[window["issue_date"] == day]
        sc = model.predict_proba(d[lab11.NUM + lab11.CAT])[:, 1]
        pooled_pred += float(sc.sum())
        pooled_actual += float(d["late"].sum())
        pooled_n += len(d)
        rows.append({"day": pd.Timestamp(day).date(), "invoices": len(d),
                     "queued": int((sc >= THRESHOLD).sum()),
                     "late_rate": float(d["late"].mean())})
    out = pd.DataFrame(rows)
    print(f"  a FIXED threshold of {THRESHOLD} applied to {len(out)} consecutive days "
          f"({out['day'].iloc[0]} .. {out['day'].iloc[-1]}), roster {cap_day}/day:")
    print(out.tail(8).to_string(index=False))
    q = out["queued"]
    print(f"\n  queue size across days: min {q.min()}  median {q.median():.0f}  "
          f"max {q.max()}  (roster {cap_day})")
    print(f"  days that STARVE the team (queue < {cap_day}): "
          f"{int((q < cap_day).sum())} of {len(q)}")
    print(f"  days that FLOOD it (queue > {cap_day}): "
          f"{int((q > cap_day).sum())} of {len(q)}")
    print(f"  days the queue MATCHES the roster: {int((q == cap_day).sum())} of {len(q)}")
    print(f"  ratio of the busiest day to the quietest: {q.max()/max(q.min(),1):.1f}x")
    print(f"  pooled over these days: mean predicted {pooled_pred / pooled_n:.3f} vs "
          f"realized late rate {pooled_actual / pooled_n:.3f} - the model is not at fault")
    print( "  the model never changed; only which invoices arrived that morning did")


# -------------------------------------------------- L4: the API is the taxonomy

def api_surface() -> None:
    """Which methods an estimator exposes tells you what kind of problem it solves."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier

    estimators = [("LinearRegression", LinearRegression()),
                  ("LogisticRegression", LogisticRegression()),
                  ("SGDClassifier(hinge)", SGDClassifier(loss="hinge")),
                  ("SGDClassifier(log_loss)", SGDClassifier(loss="log_loss")),
                  ("KNeighborsClassifier", KNeighborsClassifier()),
                  ("KMeans", KMeans(n_init=10)),
                  ("PCA", PCA())]
    print(f"  {'estimator':<24}{'needs y':<10}{'predict':<10}{'proba':<9}"
          f"{'transform':<12}{'partial_fit':<13}implied problem")
    for name, est in estimators:
        params = inspect.signature(est.fit).parameters
        needs_y = "y" in params and params["y"].default is inspect.Parameter.empty
        has = {m: hasattr(est, m) for m in
               ("predict", "predict_proba", "transform", "partial_fit")}
        if not needs_y:
            implied = "unsupervised"
        elif has["predict_proba"]:
            implied = "classification"
        else:
            implied = "regression"
        if has["partial_fit"]:
            implied += " (online-capable)"
        print(f"  {name:<24}{str(needs_y):<10}{str(has['predict']):<10}"
              f"{str(has['predict_proba']):<9}{str(has['transform']):<12}"
              f"{str(has['partial_fit']):<13}{implied}")
    print("\n  Note the two SGDClassifier rows: identical class, different `loss`, and")
    print("  predict_proba EXISTS on one and not the other (scikit-learn gates it with")
    print("  available_if). The naive rule above therefore mislabels the hinge variant as")
    print("  regression - the API is strong evidence of the problem type, not proof.")


# ------------------------------------------------------- L5: label availability

def label_availability() -> None:
    """Supervised learning needs resolved outcomes; recent rows do not have them yet."""
    raw = pd.read_csv(lab11.RAW / "invoices.csv.gz",
                      usecols=["invoice_id", "issue_date"]).drop_duplicates()
    pay = pd.read_csv(lab11.RAW / "payments.csv.gz", usecols=["invoice_id"])
    resolved = set(pay["invoice_id"])
    raw["month"] = pd.to_datetime(raw["issue_date"]).dt.to_period("M")
    raw["has_label"] = raw["invoice_id"].isin(resolved)
    recent = raw.loc[raw["month"] >= "2025-12"].groupby("month")["has_label"].agg(
        ["size", "mean"])
    print("  share of invoices with a resolved payment outcome, by issue month:")
    for month, row in recent.iterrows():
        bar = "#" * int(row["mean"] * 40)
        print(f"    {month}  n={int(row['size']):>6,}  labelled={row['mean']:.1%}  {bar}")
    print(f"\n  overall labelled: {raw['has_label'].mean():.1%} of {len(raw):,} invoices")
    print("  the newest rows - the ones a live model must score - are the least labelled")


# --------------------------------------------- L6: what the choice costs to serve

def production_costs(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Parametric vs non-parametric, and batch vs online, priced in bytes and micros."""
    import pickle

    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier

    sub_tr = train.sample(n=20_000, random_state=SEED)
    sub_te = test.head(5_000)
    X_tr, y_tr = sub_tr[lab11.NUM + lab11.CAT], sub_tr["late"]
    X_te = sub_te[lab11.NUM + lab11.CAT]

    def timed(model, label: str) -> tuple[float, float]:
        """Artifact size is deterministic; wall-clock timing is not. Report both. Even
        the RATIO wobbles run to run (7-16x across our reruns, on the same machine and
        the same seed), so it is quoted as an order of magnitude, never as a constant -
        median of 7 reps to damp the noise. Widen this range, do not tighten the prose,
        if a later run lands outside it."""
        model.fit(X_tr, y_tr)
        size = len(pickle.dumps(model)) / 1024
        times = []
        for _ in range(7):
            t0 = time.perf_counter()
            model.predict(X_te)
            times.append((time.perf_counter() - t0) / len(X_te) * 1e6)
        micros = float(np.median(times))
        print(f"  {label:<34} artifact {size:>9,.1f} KB   predict {micros:>7.1f} us/row")
        return size, micros

    logreg = lab11.make_model()
    size_p, time_p = timed(logreg, "LogisticRegression (parametric)")
    knn = lab11.make_model()
    knn.set_params(clf=KNeighborsClassifier(n_neighbors=15))
    size_n, time_n = timed(knn, "KNeighbors k=15 (non-parametric)")
    print(f"  ratios (non-parametric / parametric): artifact {size_n / size_p:,.0f}x   "
          f"latency {time_n / time_p:.1f}x")
    print(f"  (fitted on {len(sub_tr):,} rows, timed over {len(sub_te):,} predictions;")
    print(f"   the artifact ratio is exact; the latency ratio is machine- and")
    print(f"   run-dependent - read it as an order of magnitude, not a constant)")
    print("  the non-parametric model IS its training data: the rows ship to production")

    sgd = lab11.make_model()
    sgd.set_params(clf=SGDClassifier(loss="log_loss", random_state=SEED))
    prep = sgd.named_steps["prep"].fit(train[lab11.NUM + lab11.CAT])
    clf = SGDClassifier(loss="log_loss", random_state=SEED)
    months = sorted(train["issue_date"].dt.to_period("Q").unique())[-6:]
    print("\n  online learning: partial_fit over successive quarters, no full refit")
    for q in months:
        chunk = train.loc[train["issue_date"].dt.to_period("Q") == q]
        clf.partial_fit(prep.transform(chunk[lab11.NUM + lab11.CAT]), chunk["late"],
                        classes=np.array([0, 1]))
        scores = clf.predict_proba(prep.transform(test[lab11.NUM + lab11.CAT]))[:, 1]
        k = int(lab11.dunning_rule(test).sum())
        m = lab11.precision_recall_at_k(test["late"].to_numpy(),
                                        lab11.topk_flag(scores, k))
        print(f"    after {q}: n={len(chunk):>6,}  precision@k={m['precision']:.4f}")


# ------------------- L7: the serving contract as production code (Stage C)

LOG = logging.getLogger("payflow.dunning.queue")


class ContractViolation(RuntimeError):
    """The queue could not be built within its contract. Callers must not ship it."""


@dataclass(frozen=True)
class QueueRequest:
    """What collections asks for each morning. `roster` is an INPUT, which is the whole
    difference between this design and the one in the cold open."""
    roster: int                      # slots the team can actually work today
    risk_floor: float = 0.0          # never contact below this modelled risk
    model_version: str = "unknown"   # travels into the decision record, per 01.3


@dataclass(frozen=True)
class QueueDecision:
    """The queue, plus everything a monitor needs to tell a quiet day from a broken one."""
    invoice_ids: list[str] = field(default_factory=list)
    roster: int = 0
    returned: int = 0
    implied_cutoff: float | None = None    # the k-th order statistic: an OUTPUT now
    floor_rejected: int = 0
    scored: int = 0
    model_version: str = "unknown"

    def record(self) -> dict:
        """One structured line per run. Queue size is no longer a thing to be surprised
        by, because it is asserted; what varies, and therefore what is worth alerting
        on, is the implied cutoff and the floor-rejection count."""
        d = asdict(self)
        d.pop("invoice_ids")                    # the ids go to the queue, not the log
        if self.implied_cutoff is not None:
            d["implied_cutoff"] = round(self.implied_cutoff, 4)
        d["utilisation"] = round(self.returned / self.roster, 4) if self.roster else None
        return d


def build_capacity_queue(invoice_ids, scores, req: QueueRequest) -> QueueDecision:
    """Return AT MOST `req.roster` invoice ids, riskiest first.

    The contract is a size guarantee, which a probability threshold cannot make at any
    parameter value: `count(score >= t)` is n times the score distribution's survival
    function at t, so it moves whenever either moves. Fixing the count instead makes the
    cutoff the free variable - it becomes the k-th order statistic and floats daily.

    Returning FEWER than the roster is legal and happens only when the risk floor bites;
    returning more is a contract violation and raises rather than silently over-serving.
    """
    if req.roster < 1:
        raise ValueError(f"roster must be >= 1, got {req.roster}")
    scores = np.asarray(scores, dtype=float)
    if len(invoice_ids) != len(scores):
        raise ValueError(f"{len(invoice_ids):,} ids against {len(scores):,} scores")
    if len(scores) and not np.isfinite(scores).all():
        raise ContractViolation(f"{int((~np.isfinite(scores)).sum()):,} of "
                                f"{len(scores):,} scores are not finite: refusing to rank")

    ids = np.asarray(invoice_ids)
    ranked = np.argsort(-scores, kind="stable")[:req.roster]   # stable: ties are ordered
    kept = ranked[scores[ranked] >= req.risk_floor]
    decision = QueueDecision(
        invoice_ids=ids[kept].tolist(), roster=req.roster, returned=int(len(kept)),
        implied_cutoff=float(scores[kept[-1]]) if len(kept) else None,
        floor_rejected=int(len(ranked) - len(kept)), scored=int(len(scores)),
        model_version=req.model_version)
    if decision.returned > req.roster:                 # the assertion the old design
        raise ContractViolation(                       # could not even express
            f"queue of {decision.returned:,} exceeds roster {req.roster:,}")
    LOG.info("dunning queue built: %s", decision.record())
    return decision


def serving_contract_days(df: pd.DataFrame, n_days: int = 20,
                          roster: int = CAP_DAY) -> None:
    """The same days, the same model, the same scores - selected by capacity instead."""
    model, window, days = live_model_and_days(df, n_days)
    version = f"logreg@post-migration-{lab11.MIGRATION:%Y%m}"
    for floor in (0.0, THRESHOLD):
        decisions = []
        for day in days:
            d = window.loc[window["issue_date"] == day]
            sc = model.predict_proba(d[lab11.NUM + lab11.CAT])[:, 1]
            decisions.append(build_capacity_queue(
                d["invoice_id"].tolist(), sc,
                QueueRequest(roster=roster, risk_floor=floor, model_version=version)))
        sizes = [x.returned for x in decisions]
        cuts = [x.implied_cutoff for x in decisions if x.implied_cutoff is not None]
        label = "no floor" if floor == 0 else f"risk floor {floor}"
        print(f"  {label:<16} queue min {min(sizes):>3}  max {max(sizes):>3}  "
              f"full on {sum(s == roster for s in sizes)}/{len(sizes)} days   "
              f"implied cutoff {min(cuts):.4f}..{max(cuts):.4f}")
    print(f"\n  the queue size stopped moving and the CUTOFF started moving - that is the")
    print(f"  trade, stated rather than discovered. With a floor the contract may return")
    print(f"  fewer than {roster}, and the decision record says so instead of the team")
    print(f"  wondering: {json.dumps(decisions[-1].record())}")

    print("\n  malformed requests fail loudly rather than returning a plausible queue:")
    sample = window.head(3)
    ok_ids = sample["invoice_id"].tolist()
    for ids, sc, ros, why in (
            (ok_ids, [0.9, 0.1], roster, "ids and scores of different length"),
            (ok_ids, [0.9, float("nan"), 0.1], roster, "a non-finite score"),
            (ok_ids, [0.9, 0.5, 0.1], 0, "a roster of zero")):
        try:
            build_capacity_queue(ids, sc, QueueRequest(roster=ros))
            print(f"    NOT REJECTED: {why}")
        except (ValueError, ContractViolation) as exc:
            print(f"    {why:<36} -> {type(exc).__name__}: {exc}")


def main() -> None:
    df, _ = lab11.build_dataset()
    w = lab11.windows(df)
    train, test = w["train"], eval_month(df)
    k = int(lab11.dunning_rule(test).sum())

    print("=" * 78)
    print("L1  ONE INVOICE TABLE, FOUR LEARNING PROBLEMS")
    print("=" * 78)
    preds = build_framings(train, test)

    print("\n" + "-" * 78)
    print("L2  DO THE FRAMINGS AGREE ON WHICH INVOICES MATTER?")
    print("-" * 78)
    _, band = framing_band(train)
    framing_disagreement(test, preds, k, band=band)

    print("\n" + "=" * 78)
    print("L3  THE INCIDENT - a fixed threshold against a capacity-shaped problem")
    print("=" * 78)
    threshold_vs_capacity(test, preds["classification"])
    print()
    daily_queue_swing(df)

    print("\n" + "=" * 78)
    print("L4  THE ESTIMATOR API IS THE TAXONOMY")
    print("=" * 78)
    api_surface()

    print("\n" + "=" * 78)
    print("L5  LABEL AVAILABILITY - why unsupervised methods exist")
    print("=" * 78)
    label_availability()

    print("\n" + "=" * 78)
    print("L6  WHAT THE CHOICE COSTS IN PRODUCTION")
    print("=" * 78)
    production_costs(train, test)

    print("\n" + "=" * 78)
    print("L7  THE SERVING CONTRACT AS PRODUCTION CODE")
    print("=" * 78)
    serving_contract_days(df)


if __name__ == "__main__":
    main()
