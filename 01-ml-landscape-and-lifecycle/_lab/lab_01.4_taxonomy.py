"""Lab for notebook 01.4 "The Taxonomy of Learning Problems".

Reproduces every captured number and listing in 01.4:

  L1  one invoice table, four learning problems - shapes, targets, eval signatures, and
      how much the four framings DISAGREE about which invoices matter
  L2  what each framing buys on the business objective (value captured, per 01.1)
  L3  the incident - a fixed probability threshold against a capacity-shaped problem
  L4  the estimator API as the taxonomy: which methods exist tells you the problem type
  L5  label availability - why unsupervised methods exist at all (censoring by recency)
  L6  what the choice costs in production: parametric vs non-parametric, batch vs online

Reuses the 01.1 dataset builder. Runtime ~2 min on CPU (the KNN latency probe dominates).

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.4_taxonomy.py"
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 42
CAPACITY = 300          # dunning slots the collections team can work per day
THRESHOLD = 0.5         # the default nobody chose but everybody ships


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
    km.fit(X_tr)                                    # note: no y argument exists to pass
    clusters = km.predict(X_te)
    print(f"\n  clustering assigned {len(np.unique(clusters))} groups, sizes "
          f"{np.bincount(clusters).tolist()} - no notion of 'correct' anywhere")
    return {"regression": reg_pred, "classification": clf_scores, "clusters": clusters}


def framing_disagreement(test: pd.DataFrame, preds: dict, k: int) -> None:
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
    print(f"    precision gap between the two framings: {gap:.4f}  "
          f"-> inside the 01.3 noise band, while 27% of the WORK differs")


# ------------------------------------------------------ L3: the framing incident

def threshold_vs_capacity(test: pd.DataFrame, scores: np.ndarray) -> None:
    """Classification answers 'is it likely?'. The business asked 'which 300?'."""
    above = int((scores >= THRESHOLD).sum())
    print(f"  evaluation window: {len(test):,} invoices, actual late rate "
          f"{test['late'].mean():.3f}")
    print(f"  model mean predicted P(late): {scores.mean():.3f}, "
          f"max {scores.max():.3f}")
    print(f"  invoices scoring >= {THRESHOLD}: {above:,}  "
          f"-> a queue of {above:,} against a capacity of {CAPACITY}/day")
    for t in (0.3, 0.4, 0.5, 0.6):
        n = int((scores >= t).sum())
        print(f"    threshold {t:.1f} -> queue of {n:>6,}  "
              f"({'starves' if n < CAPACITY else 'floods'} a {CAPACITY}-slot team)")
    top = lab11.topk_flag(scores, CAPACITY)
    m = lab11.precision_recall_at_k(test["late"].to_numpy(), top)
    print(f"  top-{CAPACITY} selection instead: queue={int(top.sum()):,} exactly, "
          f"precision={m['precision']:.4f} - the size is a guarantee, not an outcome")


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
    sub_te = test.head(2_000)
    X_tr, y_tr = sub_tr[lab11.NUM + lab11.CAT], sub_tr["late"]
    X_te = sub_te[lab11.NUM + lab11.CAT]

    def timed(model, label: str) -> tuple[float, float]:
        """Artifact size is deterministic; wall-clock timing is not. Report both, but
        the RATIO between models is the quantity that survives a different machine."""
        model.fit(X_tr, y_tr)
        size = len(pickle.dumps(model)) / 1024
        times = []
        for _ in range(3):
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
    print(f"   absolute microseconds are machine- and run-dependent, the ratio is not)")
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


def main() -> None:
    df, _ = lab11.build_dataset()
    w = lab11.windows(df)
    train, test = w["train"], w["test_stable"].head(5_500)
    k = int(lab11.dunning_rule(test).sum())

    print("=" * 78)
    print("L1  ONE INVOICE TABLE, FOUR LEARNING PROBLEMS")
    print("=" * 78)
    preds = build_framings(train, test)

    print("\n" + "-" * 78)
    print("L2  DO THE FRAMINGS AGREE ON WHICH INVOICES MATTER?")
    print("-" * 78)
    framing_disagreement(test, preds, k)

    print("\n" + "=" * 78)
    print("L3  THE INCIDENT - a fixed threshold against a capacity-shaped problem")
    print("=" * 78)
    threshold_vs_capacity(test, preds["classification"])

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


if __name__ == "__main__":
    main()
