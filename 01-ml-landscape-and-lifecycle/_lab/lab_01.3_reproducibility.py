"""Lab for notebook 01.3 "Reproducibility as an Engineering Contract".

Reproduces every captured number and listing in 01.3:

  L1  the split lottery - precision spread across unseeded evaluation splits, measured
      against the size of the improvement 01.1 actually claimed (+0.0194)
  L2  the data lottery - the same script run on two different days, i.e. two different
      data snapshots, producing two different models
  L3  determinism achieved - seed + fixed split + pinned data slice, hashed predictions
      identical across repeats
  L3b where randomness ACTUALLY enters - per-component seed sensitivity, including the
      uncomfortable finding that random_state on the 01.1 estimator changes nothing
  L4  the run manifest - what has to be recorded for a result to be re-derivable
  L5  the CI check - reproduce() re-runs a manifest and reports what moved

Reuses the dataset builder committed with 01.1 rather than duplicating it, which is itself
the point: one definition of the modelling table, referenced, not copied.

Runtime ~90 s on CPU (about twenty logistic-regression fits over ~160k rows).

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.3_reproducibility.py"
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SEED = 42
EVAL_ROWS = 5_500          # one month of invoices - a realistic evaluation sample size
CLAIMED_GAIN = 0.0194      # the model-over-rule precision delta measured in 01.1


def load_sibling(filename: str, alias: str):
    """Import a sibling lab module whose filename contains dots (lab_01.1_...py)."""
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


lab11 = load_sibling("lab_01.1_rules_vs_learning.py", "lab_01_1")


def later_eval_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Rows issued after both data snapshots, so the data-lottery arms are comparable.

    June 2025 is the month 01.1's three windows deliberately skip - it straddles the M3
    unit-glitch week and the 2025-07-01 gateway migration, so it belongs to neither the
    stable nor the drift regime. That makes it exactly the right holdout here: unused by
    any other experiment in the series, and after both snapshots.
    """
    return df.loc[(df["issue_date"] >= "2025-06-01") & (df["issue_date"] < "2025-07-01")]


def score_hash(scores: np.ndarray) -> str:
    """Content address of a prediction vector: same inputs+code+seed => same digest."""
    return hashlib.sha256(np.round(scores, 10).tobytes()).hexdigest()[:12]


def frame_hash(df: pd.DataFrame) -> str:
    """Content address of a data slice, so 'which data' is a recorded fact not a memory.

    Columns are canonicalised by sorting; per-row hashes are sorted too, so the SAME rows
    in a different order produce the SAME digest - a content address must not depend on
    the order a query happened to return.
    """
    key = pd.util.hash_pandas_object(df.sort_index(axis=1), index=False).to_numpy()
    return hashlib.sha256(np.sort(key).tobytes()).hexdigest()[:12]


# ------------------------------------------------------------- L1: the split lottery

def split_lottery(pool: pd.DataFrame, n_runs: int = 12) -> pd.DataFrame:
    """Same code, same data, different unseeded split - scoring BOTH policies.

    Scoring only the model measures a MARGINAL spread. The quantity a shipping decision
    actually rests on is the PAIRED delta (model - rule on identical rows), because both
    policies see the same evaluation sample and most of the split-to-split noise is
    common-mode. Reporting the marginal spread against a paired claim compares two
    different quantities, which is the mistake this function exists to avoid.
    """
    rows = []
    for s in range(n_runs):
        train, test = train_test_split(pool, test_size=EVAL_ROWS, random_state=s)
        scores = lab11.fit_score(train, test, seed=SEED)
        y = test["late"].to_numpy()
        rule = lab11.dunning_rule(test)
        k = int(rule.sum())
        p_model = lab11.precision_recall_at_k(y, lab11.topk_flag(scores, k))["precision"]
        p_rule = lab11.precision_recall_at_k(y, rule)["precision"]
        rows.append({"model": p_model, "rule": p_rule, "delta": p_model - p_rule})
    return pd.DataFrame(rows, index=pd.RangeIndex(n_runs, name="split_seed"))


def training_vs_evaluation_noise(pool: pd.DataFrame, n_runs: int = 12) -> None:
    """Two axes that both get called 'variance', crossed with marginal vs paired.

    Rows:  what is resampled - the TRAINING set (eval rows held fixed), or the
           EVALUATION rows (training procedure held fixed).
    Cols:  what is measured - the model's precision ALONE (marginal), or the
           model-minus-rule difference on identical rows (paired).

    The 2x2 carries the whole lesson. When the evaluation rows are fixed, the rule is a
    constant, so pairing changes nothing. When they are re-drawn, the rule moves WITH the
    model and most of the noise cancels - which is why a comparison can be far better
    determined than either of the numbers being compared.
    """
    train_fixed, test_fixed = train_test_split(pool, test_size=EVAL_ROWS, random_state=SEED)
    y = test_fixed["late"].to_numpy()
    rule_fixed = lab11.dunning_rule(test_fixed)
    k = int(rule_fixed.sum())
    p_rule_fixed = lab11.precision_recall_at_k(y, rule_fixed)["precision"]

    boot = []
    for s in range(n_runs):
        resampled = train_fixed.sample(frac=1.0, replace=True, random_state=s)
        scores = lab11.fit_score(resampled, test_fixed, seed=s)
        p = lab11.precision_recall_at_k(y, lab11.topk_flag(scores, k))["precision"]
        boot.append({"model": p, "rule": p_rule_fixed, "delta": p - p_rule_fixed})
    boot = pd.DataFrame(boot)
    lottery = split_lottery(pool, n_runs=n_runs)

    header = f"  {'what is resampled':<34}{'marginal (model)':>20}{'paired (model-rule)':>22}"
    print(header)
    for label, frame in [("training set, eval rows FIXED", boot),
                         ("evaluation rows, procedure fixed", lottery)]:
        marginal = frame["model"].std()
        paired = frame["delta"].std()
        print(f"  {label:<34}{marginal:>20.4f}{paired:>22.4f}")
    corr = lottery["model"].corr(lottery["rule"])
    print(f"\n  correlation between model and rule across re-drawn holdouts: {corr:.3f}")
    print(f"  -> the holdout draw dominates the ABSOLUTE number (std "
          f"{lottery['model'].std():.4f}) but largely cancels in the COMPARISON (std "
          f"{lottery['delta'].std():.4f}),")
    print( "     because both policies are scored on the same rows. With the eval rows")
    print( "     held fixed the rule is a constant, so pairing buys nothing there.")


# -------------------------------------------------------------- L2: the data lottery

def data_lottery(df: pd.DataFrame, eval_slice: pd.DataFrame) -> None:
    """The same training script, run on two different days, is two different models.

    The evaluation slice must sit AFTER both snapshots, or the later arm is scored on rows
    it trained on and the comparison is contaminated - which would be a strange thing to
    ship in a notebook about experimental hygiene. See `later_eval_slice()`.
    """
    snapshots = {"as-of 2025-01-01": "2025-01-01", "as-of 2025-06-01": "2025-06-01"}
    scores, manifests = {}, {}
    for label, cutoff in snapshots.items():
        train = df.loc[df["issue_date"] < cutoff]
        scores[label] = lab11.fit_score(train, eval_slice, seed=SEED)
        manifests[label] = {"train_rows": len(train), "data_hash": frame_hash(train),
                            "score_hash": score_hash(scores[label])}
        print(f"  {label}: train_rows={len(train):>7,}  data_hash={manifests[label]['data_hash']}"
              f"  score_hash={manifests[label]['score_hash']}")

    a, b = scores.values()
    y = eval_slice["late"].to_numpy()
    k = int(lab11.dunning_rule(eval_slice).sum())
    pa = lab11.precision_recall_at_k(y, lab11.topk_flag(a, k))["precision"]
    pb = lab11.precision_recall_at_k(y, lab11.topk_flag(b, k))["precision"]
    disagree = int((lab11.topk_flag(a, k) != lab11.topk_flag(b, k)).sum())
    print(f"  mean |score difference| on identical rows: {np.abs(a - b).mean():.4f}")
    print(f"  queue disagreement: {disagree:,} of {2 * k:,} slots "
          f"({disagree / (2 * k):.1%} of the two queues differ)")
    print(f"  precision@k {pa:.4f} vs {pb:.4f}  (delta {pb - pa:+.4f}, "
          f"claimed gain in 01.1 was {CLAIMED_GAIN:+.4f})")


# ----------------------------------------------------------- L3: determinism achieved

def determinism_check(train: pd.DataFrame, test: pd.DataFrame, n_runs: int = 3) -> list[str]:
    digests = [score_hash(lab11.fit_score(train, test, seed=SEED)) for _ in range(n_runs)]
    print(f"  {n_runs} repeats, identical seed/data/code -> digests {digests}")
    print(f"  all identical: {len(set(digests)) == 1}")
    return digests


def make_sgd():
    """A genuinely stochastic estimator, for contrast with lbfgs logistic regression."""
    from sklearn.linear_model import SGDClassifier

    pipe = lab11.make_model()
    pipe.set_params(clf=SGDClassifier(loss="log_loss", max_iter=15, tol=None,
                                      random_state=SEED))
    return pipe


def seed_sensitivity(pool: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Which knob labelled `random_state` actually moves the answer? Test, do not assume."""
    components = {
        "train_test_split(random_state=s)":
            lambda s: frame_hash(train_test_split(pool, test_size=EVAL_ROWS,
                                                  random_state=s)[1]),
        "LogisticRegression(lbfgs, random_state=s)":
            lambda s: score_hash(lab11.fit_score(train, test, seed=s)),
        "SGDClassifier(log_loss, random_state=s)":
            lambda s: score_hash(lab11.fit_score(train, test, seed=s, factory=make_sgd)),
        "DataFrame.sample(random_state=s)":
            lambda s: frame_hash(train.sample(frac=0.5, random_state=s)),
    }
    for name, fn in components.items():
        a, b = fn(SEED), fn(SEED + 1)
        verdict = "SEED-SENSITIVE" if a != b else "seed changes NOTHING"
        print(f"  {name:<44} {a} / {b}  -> {verdict}")
    print("\n  Seeding a deterministic estimator is theatre: it looks like diligence and")
    print("  pins nothing. Randomness enters through row selection - splits, resampling,")
    print("  and stochastic solvers - so those are what a manifest has to record.")


# ------------------------------------------------------------------ L4: the manifest

def build_manifest(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    scores = lab11.fit_score(train, test, seed=seed)
    y = test["late"].to_numpy()
    k = int(lab11.dunning_rule(test).sum())
    return {
        "code": {"python": platform.python_version(), "numpy": np.__version__,
                 "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "data": {"train_rows": len(train), "train_hash": frame_hash(train),
                 "eval_rows": len(test), "eval_hash": frame_hash(test)},
        "config": {"seed": seed, "features_num": len(lab11.NUM),
                   "features_cat": len(lab11.CAT), "queue_k": k},
        "result": {"score_hash": score_hash(scores),
                   "precision_at_k": round(
                       lab11.precision_recall_at_k(y, lab11.topk_flag(scores, k))["precision"], 6)},
    }


def diff_manifest(old: dict, new: dict, path: str = "") -> list[str]:
    """Localize what moved between two runs - the whole point of recording a manifest."""
    diffs = []
    for key in sorted(set(old) | set(new)):
        here = f"{path}.{key}" if path else key
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += diff_manifest(a, b, here)
        elif a != b:
            diffs.append(f"{here}: {a!r} -> {b!r}")
    return diffs


# --------------------------------------------------------------------- L5: the check

def reproduce(manifest: dict, train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """What CI runs: re-derive the recorded result and report exactly what moved."""
    fresh = build_manifest(train, test, manifest["config"]["seed"])
    diffs = diff_manifest(manifest, fresh)
    if not diffs:
        print("  REPRODUCED: manifest matches on code, data, config and result")
        return True
    print(f"  NOT REPRODUCED ({len(diffs)} field(s) moved):")
    for line in diffs:
        print(f"    - {line}")
    return False


def main() -> None:
    df, _ = lab11.build_dataset()
    w = lab11.windows(df)
    pool = w["train"]
    print(f"modelling table: {len(df):,} resolved invoices; training pool {len(pool):,} "
          f"(issued before 2025-01-01)\n")

    print("=" * 78)
    print("L1  THE SPLIT LOTTERY - same code, same data, unseeded split")
    print("=" * 78)
    runs = split_lottery(pool)
    print(runs.round(4).to_string())
    m_spread = runs["model"].max() - runs["model"].min()
    d_band = 2 * runs["delta"].std()
    print(f"\n  model precision alone : mean {runs['model'].mean():.4f}  "
          f"std {runs['model'].std():.4f}  min {runs['model'].min():.4f}  "
          f"max {runs['model'].max():.4f}  spread {m_spread:.4f}")
    print(f"  paired delta          : mean {runs['delta'].mean():.4f}  "
          f"std {runs['delta'].std():.4f}  min {runs['delta'].min():.4f}  "
          f"max {runs['delta'].max():.4f}  spread "
          f"{runs['delta'].max() - runs['delta'].min():.4f}")
    print(f"\n  -> a single unseeded run can report an ABSOLUTE precision anywhere from "
          f"{runs['model'].min():.4f}")
    print(f"     to {runs['model'].max():.4f} without a line of code changing. Reporting "
          f"that number alone is a draw,")
    print( "     not a measurement.")
    print(f"\n  -> but the DECISION rests on the comparison, and the comparison is far "
          f"better determined:")
    print(f"     2-sigma band on the paired delta = {d_band:.4f}; the gain 01.1 reported "
          f"was {CLAIMED_GAIN:+.4f}")
    verdict = "CLEARS the band - 01.1's claim survives" if CLAIMED_GAIN > d_band \
        else "sits inside the band - 01.1's claim is unsupported"
    print(f"     {verdict}")
    print( "\n  Comparing 01.1's PAIRED gain against the MARGINAL spread would be an error:")
    print( "  they are different quantities, and pairing is what removes the common-mode")
    print( "  noise that makes the marginal spread large.")

    print("\n" + "-" * 78)
    print("L1b TWO KINDS OF VARIANCE - which one did 01.1 actually measure?")
    print("-" * 78)
    training_vs_evaluation_noise(pool)

    print("\n" + "=" * 78)
    print("L2  THE DATA LOTTERY - same script, two different days")
    print("=" * 78)
    eval_slice = later_eval_slice(df)
    print(f"  evaluation slice: invoices issued {eval_slice['issue_date'].min():%Y-%m-%d} "
          f"to {eval_slice['issue_date'].max():%Y-%m-%d}, n={len(eval_slice):,}")
    print(f"  (strictly after BOTH snapshots, so neither arm has seen these rows)\n")
    data_lottery(df, eval_slice)

    print("\n" + "=" * 78)
    print("L3  DETERMINISM ACHIEVED - seed + fixed split + pinned data")
    print("=" * 78)
    train_fixed, test_fixed = train_test_split(pool, test_size=EVAL_ROWS, random_state=SEED)
    determinism_check(train_fixed, test_fixed)

    print("\n" + "=" * 78)
    print("L3b WHERE RANDOMNESS ACTUALLY ENTERS")
    print("=" * 78)
    seed_sensitivity(pool, train_fixed, test_fixed)

    print("\n" + "=" * 78)
    print("L4  THE RUN MANIFEST")
    print("=" * 78)
    manifest = build_manifest(train_fixed, test_fixed, SEED)
    print(json.dumps(manifest, indent=2))

    print("\n" + "=" * 78)
    print("L5  THE CI CHECK - reproduce(), honest and dishonest runs")
    print("=" * 78)
    print(" (a) same code, same data, same config:")
    reproduce(manifest, train_fixed, test_fixed)
    print("\n (b) the data grew under an unchanged script (the common real case):")
    grown = pd.concat([train_fixed, w["test_stable"].head(2_000)])
    reproduce(manifest, grown, test_fixed)
    print("\n (c) someone changed the seed and re-ran - does the check notice?")
    reseeded = build_manifest(train_fixed, test_fixed, SEED + 1)
    diffs = diff_manifest(manifest, reseeded)
    print(f"    fields that moved: {diffs or 'none - the result is seed-independent'}")
    print("    the manifest is honest either way: it records the seed as CONFIG and the")
    print("    outcome as RESULT, so a seed that does nothing is visible as a config-only")
    print("    difference rather than being mistaken for a reproduced experiment")


if __name__ == "__main__":
    main()
