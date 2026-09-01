"""Lab for notebook 01.6 "The ML Project Lifecycle and Where Projects Actually Die".

Reproduces every captured number and listing in 01.6:

  L1  the lifecycle instrumented - one clean end-to-end run with every gate reporting
  L2  defect injection - six defects x seven gates: which gate actually fires?
  L3  the seductive defects quantified - leakage's offline gain and its production
      collapse, and the optimism of a random split on temporal data
  L4  detection distance - the stage a defect enters versus the earliest gate that can
      see it, which is what "where projects die" actually measures
  L5  the retrain decision, as a function rather than a meeting

Scale note: the training pool is subsampled to keep a notebook run near a minute; the
mechanics and the ordering of results are unaffected, the absolute precisions are lower
than 01.1's full-data figures.

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.6_lifecycle.py"
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SEED = 42
TRAIN_N = 60_000          # subsample for runtime; see the scale note above
CAPACITY_FRAC = 0.26      # the dunning queue the collections team can work
NOISE_REPEATS = 5
LEAKY = "reminder_count"  # SPEC M10: written AFTER the payment resolves


def load_sibling(filename: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


lab11 = load_sibling("lab_01.1_rules_vs_learning.py", "lab_01_1")


@dataclass
class Run:
    """Everything the gates need to inspect. A run is an object, not a conversation."""
    defects: set[str] = field(default_factory=set)
    split_kind: str = "temporal"
    features: list[str] = field(default_factory=list)
    dedup_applied: bool = True
    n_train: int = 0
    precision: float = 0.0
    baseline_precision: float = 0.0
    noise_band: float = 0.0
    queue_size: int = 0
    capacity: int = 0
    predicted_rate: float = 0.0
    actual_rate: float = 0.0
    prod_precision: float | None = None
    business_precision: float | None = None    # against the question actually asked
    manifest_hash: str = ""


# --------------------------------------------------------------- the pipeline

def build_run(df_dedup: pd.DataFrame, df_raw: pd.DataFrame, defects: set[str]) -> Run:
    """Walk the lifecycle once. Each defect swaps one stage's decision for a wrong one."""
    df = df_raw if "no_dedup" in defects else df_dedup
    w = lab11.windows(df)

    if "random_split" in defects:            # FRAME stage: ignore time on temporal data
        pool = pd.concat([w["train"], w["test_stable"]])
        train, test = train_test_split(pool, test_size=len(w["test_stable"]),
                                       random_state=SEED)
        split_kind = "random"
    else:
        train, test = w["train"], w["test_stable"]
        split_kind = "temporal"

    train = train.sample(n=min(TRAIN_N, len(train)), random_state=SEED)
    feats = list(lab11.NUM)
    if "leakage" in defects:                  # LABEL stage: a post-outcome column
        feats = feats + [LEAKY]

    def fit(tr: pd.DataFrame, te: pd.DataFrame, seed: int = SEED) -> np.ndarray:
        model = lab11.make_model()
        model.set_params(prep__num__imp__strategy="median")
        model.named_steps["prep"].transformers[0] = ("num",
                                                     model.named_steps["prep"].transformers[0][1],
                                                     feats)
        model.named_steps["clf"].set_params(random_state=seed)
        model.fit(tr[feats + lab11.CAT], tr["late"])
        return model.predict_proba(te[feats + lab11.CAT])[:, 1]

    scores = fit(train, test)
    y = test["late"].to_numpy()
    rule = lab11.dunning_rule(test)
    capacity = int(round(CAPACITY_FRAC * len(test)))

    if "threshold_selection" in defects:      # SHIP stage: a cutoff, not a capacity
        flag = scores >= 0.5
    else:
        flag = lab11.topk_flag(scores, capacity)

    run = Run(defects=defects, split_kind=split_kind, features=feats,
              dedup_applied="no_dedup" not in defects, n_train=len(train),
              precision=lab11.precision_recall_at_k(y, flag)["precision"],
              baseline_precision=lab11.precision_recall_at_k(y, rule)["precision"],
              queue_size=int(flag.sum()), capacity=capacity,
              predicted_rate=float(scores.mean()), actual_rate=float(y.mean()))

    if "no_noise_band" in defects:            # EVALUATE stage: one run, no spread
        run.noise_band = 0.0
    else:
        reps = []
        for s in range(NOISE_REPEATS):
            boot = train.sample(frac=1.0, replace=True, random_state=s)
            sc = fit(boot, test, seed=s)
            reps.append(lab11.precision_recall_at_k(
                y, lab11.topk_flag(sc, capacity))["precision"])
        run.noise_band = 2 * float(np.std(reps))

    if "leakage" in defects:                  # what serving time actually looks like
        prod_test = test.copy()
        prod_test[LEAKY] = 0                  # the column is always 0 at issue time
        prod_scores = fit(train, prod_test)
        run.prod_precision = lab11.precision_recall_at_k(
            y, lab11.topk_flag(prod_scores, capacity))["precision"]

    if "wrong_horizon" in defects:
        # Every gate below sees the >7d numbers and passes. Collections actually
        # escalates at 30 days past due, so this is the precision that matters.
        y30 = (test["days_late"] > 30).to_numpy().astype(int)
        run.business_precision = lab11.precision_recall_at_k(
            y30, lab11.topk_flag(scores, capacity))["precision"]

    run.manifest_hash = hashlib.sha256(
        f"{sorted(feats)}|{split_kind}|{len(train)}".encode()).hexdigest()[:12]
    return run


# ------------------------------------------------------------------- the gates

def gate_data_contract(r: Run) -> tuple[bool, str]:
    return r.dedup_applied, "duplicate rows dropped" if r.dedup_applied else \
        "grain violated: duplicate invoice rows present"


def gate_leakage_blocklist(r: Run) -> tuple[bool, str]:
    bad = [f for f in r.features if f == LEAKY]
    return not bad, "no post-outcome features" if not bad else \
        f"post-outcome feature present: {bad[0]}"


def gate_temporal_split(r: Run) -> tuple[bool, str]:
    ok = r.split_kind == "temporal"
    return ok, f"split is {r.split_kind}" + ("" if ok else " on time-ordered data")


def gate_beats_baseline(r: Run) -> tuple[bool, str]:
    ok = r.precision > r.baseline_precision
    return ok, (f"model {r.precision:.4f} vs rule {r.baseline_precision:.4f}")


def gate_noise_band(r: Run) -> tuple[bool, str]:
    delta = r.precision - r.baseline_precision
    ok = r.noise_band > 0 and abs(delta) > r.noise_band
    if r.noise_band == 0:
        return False, "no variance estimate was computed"
    return ok, f"delta {delta:+.4f} vs band {r.noise_band:.4f}"


def gate_serving_contract(r: Run) -> tuple[bool, str]:
    ok = r.queue_size == r.capacity
    return ok, f"queue {r.queue_size:,} vs capacity {r.capacity:,}"


def gate_calibration(r: Run) -> tuple[bool, str]:
    gap = abs(r.predicted_rate - r.actual_rate)
    return gap <= 0.05, f"predicted {r.predicted_rate:.3f} vs actual {r.actual_rate:.3f}"


GATES = {
    "G1 data contract": gate_data_contract,
    "G2 leakage blocklist": gate_leakage_blocklist,
    "G3 temporal split": gate_temporal_split,
    "G4 beats baseline": gate_beats_baseline,
    "G5 clears noise band": gate_noise_band,
    "G6 serving contract": gate_serving_contract,
    "G7 calibration": gate_calibration,
}

# Which lifecycle stage each defect is INTRODUCED at, in lifecycle order.
STAGES = ["frame", "label", "data", "baseline", "model", "evaluate", "ship", "monitor"]
DEFECTS = {
    "no_dedup": ("data", "skip the duplicate-row drop (SPEC M1)"),
    "leakage": ("label", "add reminder_count, written after the outcome (SPEC M10)"),
    "random_split": ("frame", "split randomly on time-ordered invoices"),
    "threshold_selection": ("ship", "select the queue by score >= 0.5, not by capacity"),
    "no_noise_band": ("evaluate", "report one run, with no variance estimate"),
    "wrong_horizon": ("frame", "predict 'late by 7 days' when collections escalates at 30"),
}
# The earliest gate index able to see each defect, for the detection-distance table.
GATE_ORDER = list(GATES)


def report(r: Run, title: str) -> dict[str, bool]:
    print(f"  {title}")
    results = {}
    for name, fn in GATES.items():
        ok, detail = fn(r)
        results[name] = ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")
    return results


def main() -> None:
    raw = pd.read_csv(lab11.RAW / "invoices.csv.gz")
    dup_rows = int(raw.duplicated().sum())
    df_dedup, _ = lab11.build_dataset()

    # A "no dedup" universe: re-add the duplicate invoice rows to the modelling table.
    dupes = df_dedup.sample(n=dup_rows, random_state=SEED)
    df_raw = pd.concat([df_dedup, dupes], ignore_index=True)

    print("=" * 78)
    print("L1  THE LIFECYCLE, INSTRUMENTED - one clean run, every gate reporting")
    print("=" * 78)
    clean = build_run(df_dedup, df_raw, set())
    print(f"  frame     temporal split, {clean.n_train:,} training rows")
    print(f"  label     late = paid > 7 days past due; base rate {clean.actual_rate:.4f}")
    print(f"  baseline  dunning rule precision {clean.baseline_precision:.4f}")
    print(f"  model     logistic regression precision {clean.precision:.4f}")
    print(f"  evaluate  delta {clean.precision - clean.baseline_precision:+.4f}, "
          f"noise band {clean.noise_band:.4f}")
    print(f"  ship      queue {clean.queue_size:,} against capacity {clean.capacity:,}")
    print(f"  monitor   predicted {clean.predicted_rate:.3f} vs actual "
          f"{clean.actual_rate:.3f}")
    print(f"  manifest  {clean.manifest_hash}\n")
    base_results = report(clean, "gate suite on the clean run:")

    print("\n" + "=" * 78)
    print("L2  DEFECT INJECTION - which gate actually fires?")
    print("=" * 78)
    matrix = {}
    for key, (stage, description) in DEFECTS.items():
        r = build_run(df_dedup, df_raw, {key})
        print(f"\n  defect introduced at stage '{stage}': {description}")
        res = report(r, f"  -> precision {r.precision:.4f}"
                        + (f"  (production {r.prod_precision:.4f})"
                           if r.prod_precision is not None else ""))
        caught = [g for g, ok in res.items() if not ok and base_results[g]]
        matrix[key] = caught
        print(f"    caught by: {', '.join(caught) if caught else 'NOTHING - all gates green'}")

    print("\n" + "=" * 78)
    print("L2b THE DEFECT THIS SERIES' OWN PIPELINE HAS - survivorship in the population")
    print("=" * 78)
    pay_ids = set(pd.read_csv(lab11.RAW / "payments.csv.gz",
                              usecols=["invoice_id"])["invoice_id"])
    raw_win = raw.drop_duplicates()
    raw_win["issue_date"] = pd.to_datetime(raw_win["issue_date"])
    raw_win = raw_win.loc[(raw_win["issue_date"] >= "2025-01-01")
                          & (raw_win["issue_date"] < "2025-06-01")]
    never_paid = raw_win.loc[~raw_win["invoice_id"].isin(pay_ids)]
    resolved_n = len(w_stable := lab11.windows(df_dedup)["test_stable"])
    print(f"  invoices issued in the evaluation window:            {len(raw_win):>8,}")
    print(f"  of those, resolved (a payment exists) - the ONLY rows")
    print(f"  that training and every offline metric ever saw:     {resolved_n:>8,}")
    print(f"  never paid (overdue / disputed / written off):       {len(never_paid):>8,}"
          f"  ({len(never_paid) / len(raw_win):.1%} of production scoring volume)")
    print(f"  every one of them is late by any definition, none was in training, and none")
    print(f"  appears in any number this series has reported")
    survivorship = Run(defects={"survivorship"}, split_kind="temporal",
                       features=list(lab11.NUM), dedup_applied=True,
                       n_train=clean.n_train, precision=clean.precision,
                       baseline_precision=clean.baseline_precision,
                       noise_band=clean.noise_band, queue_size=clean.queue_size,
                       capacity=clean.capacity, predicted_rate=clean.predicted_rate,
                       actual_rate=clean.actual_rate)
    surv_results = report(survivorship, "gate suite on the survivorship defect:")
    matrix["survivorship"] = [g for g, ok in surv_results.items()
                              if not ok and base_results[g]]
    DEFECTS["survivorship"] = ("data", "train and evaluate on resolved invoices only")
    print(f"    caught by: "
          f"{', '.join(matrix['survivorship']) if matrix['survivorship'] else 'NOTHING - all gates green'}")

    print("\n" + "=" * 78)
    print("L3  THE SEDUCTIVE DEFECTS, QUANTIFIED")
    print("=" * 78)
    leak = build_run(df_dedup, df_raw, {"leakage"})
    print(f"  leakage: offline precision {leak.precision:.4f} against the clean "
          f"{clean.precision:.4f}  ({leak.precision - clean.precision:+.4f})")
    print(f"           at serving time the column is always 0, so production precision "
          f"is {leak.prod_precision:.4f}")
    print(f"           the offline gain is {leak.precision - clean.precision:+.4f} and the "
          f"production reality is {leak.prod_precision - clean.precision:+.4f}")
    rnd = build_run(df_dedup, df_raw, {"random_split"})
    print(f"  random split, stationary period: {rnd.precision:.4f} against the temporal "
          f"{clean.precision:.4f}  ({rnd.precision - clean.precision:+.4f})")
    print(f"           when nothing is drifting, random splitting costs almost nothing -")
    print(f"           the textbook warning needs a regime change to bite.")
    print(f"           To isolate the effect the TEST SET must be held fixed, or the two")
    print(f"           splits are compared on populations with different base rates:")
    drift = lab11.windows(df_dedup)["test_drift"]
    held_out, contemporaneous = train_test_split(drift, test_size=0.7, random_state=SEED)
    pre_drift = lab11.windows(df_dedup)["train"]
    y_h = held_out["late"].to_numpy()
    k_h = int(lab11.dunning_rule(held_out).sum())
    for tag, tr in [("honest (pre-drift rows only)      ",
                     pre_drift.sample(n=TRAIN_N, random_state=SEED)),
                    ("contaminated (+ contemporaneous)  ",
                     pd.concat([pre_drift, contemporaneous]).sample(n=TRAIN_N,
                                                                    random_state=SEED))]:
        sc = lab11.fit_score(tr, held_out, seed=SEED)
        p = lab11.precision_recall_at_k(y_h, lab11.topk_flag(sc, k_h))["precision"]
        print(f"           {tag} precision on the SAME test rows: {p:.4f}")

    wrong = build_run(df_dedup, df_raw, {"wrong_horizon"})
    w_stable2 = lab11.windows(df_dedup)["test_stable"]
    base30 = float((w_stable2["days_late"] > 30).mean())
    print(f"  wrong horizon: the team measures precision {wrong.precision:.4f} against its")
    print(f"           own 7-day label. Against the 30-day escalation the business runs on,")
    print(f"           the same ranking scores {wrong.business_precision:.4f} - and the "
          f"30-day base rate")
    print(f"           is {base30:.4f}, so the model delivers "
          f"{wrong.business_precision / base30:.2f}x lift on the question that matters")

    print("\n" + "=" * 78)
    print("L4  DETECTION DISTANCE - stage introduced vs earliest gate that sees it")
    print("=" * 78)
    print(f"  {'defect':<22}{'stage':<12}{'caught by':<26}{'stages downstream':>18}")
    for key, (stage, _) in DEFECTS.items():
        caught = matrix[key]
        if caught:
            first = min(GATE_ORDER.index(g) for g in caught)
            label = GATE_ORDER[first]
            dist = "detected offline"
        else:
            label = "no offline gate"
            dist = "PRODUCTION"
        print(f"  {key:<22}{stage:<12}{label:<26}{dist:>18}")
    print("\n  a defect caught by a gate is a bug; a defect no gate can see is a project")
    print("  risk, because the only remaining detector is a customer")

    print("\n" + "=" * 78)
    print("L5  THE RETRAIN DECISION, AS A FUNCTION")
    print("=" * 78)
    w = lab11.windows(df_dedup)
    for label, window in [("stable 2025-01..05", w["test_stable"]),
                          ("drift  2025-07..2026-05", w["test_drift"])]:
        train = w["train"].sample(n=TRAIN_N, random_state=SEED)
        scores = lab11.fit_score(train, window, seed=SEED)
        gap = abs(float(scores.mean()) - float(window["late"].mean()))
        rule_p = lab11.precision_recall_at_k(window["late"].to_numpy(),
                                             lab11.dunning_rule(window))["precision"]
        k = int(lab11.dunning_rule(window).sum())
        model_p = lab11.precision_recall_at_k(window["late"].to_numpy(),
                                              lab11.topk_flag(scores, k))["precision"]
        verdict = ("RETRAIN: calibration gap exceeds 0.05" if gap > 0.05 else
                   "RETRAIN: model no longer beats the rule" if model_p <= rule_p else
                   "hold: within tolerance")
        print(f"  {label:<26} calibration gap {gap:.3f}   model {model_p:.4f} vs rule "
              f"{rule_p:.4f}   -> {verdict}")


if __name__ == "__main__":
    main()
