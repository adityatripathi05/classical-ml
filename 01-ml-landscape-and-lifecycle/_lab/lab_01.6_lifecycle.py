"""Lab for notebook 01.6 "The ML Project Lifecycle and Where Projects Actually Die".

Reproduces every captured number and listing in 01.6:

  L1  the lifecycle instrumented - one clean end-to-end run with every gate reporting
  L2  defect injection - six defects x seven gates: which gate actually fires?
  L3  the seductive defects quantified - leakage's offline gain and its production
      collapse, and the optimism of a random split on temporal data
  L4  detection distance - the stage a defect enters versus the earliest gate that can
      see it, which is what "where projects die" actually measures
  L5  the retrain decision, as a function rather than a meeting
  L6  the gate suite as a release gate - blocking vs advisory, a JSON verdict, an exit
      code, and the blind spots it prints on every run (Stage C)

Scale note: the training pool is subsampled to keep a notebook run near a minute; the
mechanics and the ordering of results are unaffected, the absolute precisions are lower
than 01.1's full-data figures.

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.6_lifecycle.py"
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SEED = 42
TRAIN_N = 60_000          # subsample for runtime; see the scale note above
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
    duplicate_rows: int = 0        # measured from the frame, not inferred from `defects`
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
    business_baseline: float | None = None     # the rule, on that same question
    business_base_rate: float | None = None
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
    # Capacity IS the rule's queue size. Scoring the model at a different k would compare
    # the two policies at different operating points - the error 01.1's Common Pitfalls
    # forbids and _recap.md lists as invariant 2. A gate suite that encodes the series'
    # invariants must not violate one of them.
    capacity = int(rule.sum())

    if "threshold_selection" in defects:      # SHIP stage: a cutoff, not a capacity
        flag = scores >= 0.5
    else:
        flag = lab11.topk_flag(scores, capacity)

    run = Run(defects=defects, split_kind=split_kind, features=feats,
              dedup_applied="no_dedup" not in defects,
              duplicate_rows=int(df.duplicated(subset=["invoice_id"]).sum()),
              n_train=len(train),
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
        run.noise_band = 2 * float(np.std(reps, ddof=1))

    if "leakage" in defects:                  # what serving time actually looks like
        prod_test = test.copy()
        prod_test[LEAKY] = 0                  # the column is always 0 at issue time
        prod_scores = fit(train, prod_test)
        run.prod_precision = lab11.precision_recall_at_k(
            y, lab11.topk_flag(prod_scores, capacity))["precision"]

    if "wrong_horizon" in defects:
        # Every gate below sees the >7d numbers and passes. Collections actually
        # escalates at 30 days past due, so this is the precision that matters - and
        # it is quoted WITH the incumbent's score on the same question, because a lift
        # without a baseline is the pitfall 01.1 opens with.
        y30 = (test["days_late"] > 30).to_numpy().astype(int)
        run.business_precision = lab11.precision_recall_at_k(
            y30, lab11.topk_flag(scores, capacity))["precision"]
        run.business_baseline = lab11.precision_recall_at_k(y30, rule)["precision"]
        run.business_base_rate = float(y30.mean())

    run.manifest_hash = hashlib.sha256(
        f"{sorted(feats)}|{split_kind}|{len(train)}".encode()).hexdigest()[:12]
    return run


# ------------------------------------------------------------------- the gates

def gate_data_contract(r: Run) -> tuple[bool, str]:
    """Inspects the DATA, not a flag the defect set - otherwise the gate is a tautology."""
    ok = r.duplicate_rows == 0
    return ok, ("duplicate rows dropped" if ok else
                f"grain violated: {r.duplicate_rows:,} duplicate invoice rows present")


def gate_leakage_blocklist(r: Run) -> tuple[bool, str]:
    bad = [f for f in r.features if f == LEAKY]
    return not bad, "no post-outcome features" if not bad else \
        f"post-outcome feature present: {bad[0]}"


def gate_temporal_split(r: Run) -> tuple[bool, str]:
    ok = r.split_kind == "temporal"
    return ok, f"split is {r.split_kind}" + ("" if ok else " on time-ordered data")


def gate_beats_baseline(r: Run) -> tuple[bool, str]:
    ok = r.precision > r.baseline_precision
    return ok, (f"model {r.precision:.4f} vs rule {r.baseline_precision:.4f} "
                f"(both at k={r.capacity:,})")


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


def survivorship_effect(df_dedup: pd.DataFrame) -> None:
    """MEASURE the survivorship blind spot instead of asserting it.

    The modelling table is an inner join to payments, so a never-paid invoice cannot
    appear in training or in any offline metric. Production scores every issued invoice.
    Rebuild the missing rows with the same features, label them late (an unpaid invoice
    past its due date is late by any definition), and score the model on the population
    it would actually meet.
    """
    inv = pd.read_csv(lab11.RAW / "invoices.csv.gz").drop_duplicates()
    inv["issue_date"] = pd.to_datetime(inv["issue_date"])
    inv["due_date"] = pd.to_datetime(inv["due_date"])
    inv["amount_num"] = pd.to_numeric(inv["amount"].astype(str).str.replace(",", ""),
                                      errors="coerce")
    pay = pd.read_csv(lab11.RAW / "payments.csv.gz",
                      usecols=["invoice_id", "fx_rate_usd"])
    fx = (pay.merge(inv[["invoice_id", "currency"]], on="invoice_id")
            .groupby("currency")["fx_rate_usd"].median())
    cus = pd.read_csv(lab11.RAW / "customers.csv")
    cus["signup_date"] = pd.to_datetime(cus["signup_date"])
    cus["employee_count"] = cus["employee_count"].replace(-999, np.nan)

    win = inv.loc[(inv["issue_date"] >= "2025-01-01") & (inv["issue_date"] < "2025-06-01")]
    missing = win.loc[~win["invoice_id"].isin(set(pay["invoice_id"]))].merge(
        cus[["customer_id", "segment", "country", "plan", "industry", "seats",
             "employee_count", "signup_date"]], on="customer_id", how="inner")
    missing = missing.assign(
        amount_usd=missing["amount_num"] / missing["currency"].map(fx),
        terms_days=(missing["due_date"] - missing["issue_date"]).dt.days,
        tenure_days=(missing["issue_date"] - missing["signup_date"]).dt.days,
        issue_month=missing["issue_date"].dt.month,
        late=1,                       # unpaid and past due: late by any definition
    )

    w = lab11.windows(df_dedup)
    train = w["train"].sample(n=TRAIN_N, random_state=SEED)
    resolved = w["test_stable"]
    full = pd.concat([resolved, missing[resolved.columns.intersection(missing.columns)]],
                     ignore_index=True)

    print(f"  the population every series-01 metric was computed on : {len(resolved):,}")
    print(f"  the population production would actually score        : {len(full):,}"
          f"   (+{len(missing):,} never-paid, {len(missing)/len(full):.1%})")
    print(f"  status of the invisible rows: "
          f"{missing['status'].str.lower().value_counts().to_dict()}\n")

    for label, frame in [("resolved only (what we measured)", resolved),
                         ("full production population", full)]:
        scores = lab11.fit_score(train, frame, seed=SEED)
        y = frame["late"].to_numpy()
        # Same matched-operating-point rule as build_run: score both policies at the
        # rule's own k. Using a fraction here would reintroduce the B2 defect.
        k = int(lab11.dunning_rule(frame).sum())
        m = lab11.precision_recall_at_k(y, lab11.topk_flag(scores, k))
        rule = lab11.precision_recall_at_k(y, lab11.dunning_rule(frame))
        print(f"  {label:<34} base rate {y.mean():.4f}   model prec {m['precision']:.4f}"
              f"   rule prec {rule['precision']:.4f}")
    print( "\n  the model was never trained on these rows and no offline number ever")
    print( "  included them; the gate suite cannot see the gap because the training table")
    print( "  is internally consistent - it is simply about a different population.")


def leakage_mechanism(df_dedup: pd.DataFrame) -> None:
    """WHY a leaked feature leaves production worse than never having had it.

    Not 'under-weighting the honest features'. At serving time the leaked column is a
    CONSTANT (always 0 on a freshly issued invoice), so it contributes an identical term
    to every logit - a pure intercept shift - and precision@k ranks by the linear
    predictor, which is invariant to that. Uniform shrinkage would cost exactly nothing.

    The real damage: the honest coefficients are no longer marginal effects, they are
    effects CONDITIONAL on the leaked column. That changes their relative sizes and can
    reverse signs, so the ranking the model produces at serving time is a different
    ranking, not a weaker version of the same one.
    """
    from scipy.stats import spearmanr

    w = lab11.windows(df_dedup)
    train = w["train"].sample(n=TRAIN_N, random_state=SEED)
    test = w["test_stable"]

    def fit(feats: list[str], frame: pd.DataFrame):
        model = lab11.make_model()
        model.named_steps["prep"].transformers[0] = (
            "num", model.named_steps["prep"].transformers[0][1], feats)
        model.named_steps["clf"].set_params(random_state=SEED)
        model.fit(frame[feats + lab11.CAT], frame["late"])
        return model

    clean = fit(lab11.NUM, train)
    leaky = fit(lab11.NUM + [LEAKY], train)
    c_clean = clean.named_steps["clf"].coef_[0][: len(lab11.NUM)]
    c_leaky = leaky.named_steps["clf"].coef_[0][: len(lab11.NUM) + 1]

    print(f"  {'numeric feature':<22}{'clean':>10}{'with leak':>12}{'ratio':>9}")
    for i, name in enumerate(lab11.NUM):
        ratio = c_leaky[i] / c_clean[i] if abs(c_clean[i]) > 1e-9 else float("nan")
        flag = "  <- SIGN FLIP" if c_clean[i] * c_leaky[i] < 0 else ""
        print(f"  {name:<22}{c_clean[i]:>10.4f}{c_leaky[i]:>12.4f}{ratio:>9.2f}{flag}")
    print(f"  {LEAKY:<22}{'-':>10}{c_leaky[len(lab11.NUM)]:>12.4f}")
    # Report the FULL coefficient vector, not just the six numeric ones: the 25 one-hot
    # categorical coefficients also drive the ranking, and the correlation's sign reverses
    # depending on which subset you take - so quoting the numeric-only figure alone would
    # be choosing the number that suits the argument.
    all_clean = clean.named_steps["clf"].coef_[0]
    all_leaky = np.delete(leaky.named_steps["clf"].coef_[0], len(lab11.NUM))
    print(f"\n  correlation between the honest-coefficient vectors:")
    print(f"    over the {len(lab11.NUM)} NUMERIC coefficients only : "
          f"{np.corrcoef(c_clean, c_leaky[: len(lab11.NUM)])[0, 1]:+.2f}")
    print(f"    over all {len(all_clean)} coefficients             : "
          f"{np.corrcoef(all_clean, all_leaky)[0, 1]:+.2f}")
    print( "    the sign flips between the two, so neither number alone is the story -")
    print( "    the ranking comparison below is what actually settles it.")

    prod = test.copy()
    prod[LEAKY] = 0                      # what serving time actually looks like
    s_clean = clean.predict_proba(test[lab11.NUM + lab11.CAT])[:, 1]
    s_prod = leaky.predict_proba(prod[lab11.NUM + [LEAKY] + lab11.CAT])[:, 1]
    rho = spearmanr(s_clean, s_prod).statistic
    print(f"  Spearman(clean ranking, leaked model's PRODUCTION ranking): {rho:+.2f}")
    print( "  -> not a degraded version of the clean ranking; a different one.")


def report(r: Run, title: str) -> dict[str, bool]:
    print(f"  {title}")
    results = {}
    for name, fn in GATES.items():
        ok, detail = fn(r)
        results[name] = ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<24} {detail}")
    return results


# ------------------- the gate suite as a release gate (Stage C production code)

LOG = logging.getLogger("payflow.dunning.release")

# Which gates stop a release. G7 is advisory on purpose: it is a mean-rate tripwire
# (see its rename note), so one window's rate gap is a reason to look, not to block.
BLOCKING = frozenset(GATES) - {"G7 calibration"}

# The runner's own blind spots, printed on EVERY run so they are reviewed rather than
# forgotten. Each is a mis-specification - a property of the question asked, not of any
# artifact this run produced - which is precisely why no assertion over this run can
# reach it. This notebook measured both: the wrong-horizon defect ships with all gates
# green, and the survivorship defect could not be injected at all.
UNGATED_RISKS = (
    ("label spec vs operational trigger",
     "nothing here records the decision the label is meant to feed, so nothing can "
     "compare them; 01.5's label spec is the artifact that makes this gateable"),
    ("training population vs scoring population",
     "the test set is drawn through the same join as training and inherits the "
     "identical filter, so the metric is silent about rows that never arrived"),
    ("intervention economics",
     "break-even precision is a property of the offer, not of the run"),
)


def release_gate(r: Run, blocking: frozenset[str] = BLOCKING) -> dict:
    """Evaluate every gate and return a machine-readable ship/block verdict.

    Returns a dict rather than printing, so the same function backs the CI job, the
    deploy API and this notebook. A gate that fails without blocking is reported, not
    swallowed - the difference between "we knew and accepted it" and "nobody looked".
    """
    gates = []
    for name, fn in GATES.items():
        ok, detail = fn(r)
        gates.append({"gate": name, "status": "PASS" if ok else "FAIL",
                      "detail": detail, "blocking": name in blocking})
    blocked = [g["gate"] for g in gates if g["status"] == "FAIL" and g["blocking"]]
    advisory = [g["gate"] for g in gates if g["status"] == "FAIL" and not g["blocking"]]
    verdict = {"verdict": "BLOCK" if blocked else "SHIP",
               "manifest": r.manifest_hash, "gates": gates,
               "blocked_by": blocked, "advisory_failures": advisory,
               "ungated_risks": [name for name, _ in UNGATED_RISKS],
               "exit_code": 1 if blocked else 0}
    LOG.info("release gate %s manifest=%s blocked_by=%s", verdict["verdict"],
             r.manifest_hash, blocked or "-")
    return verdict


def release_gate_cli(r: Run, label: str, artifact_dir: Path | None = None) -> int:
    """The CI entry point: human-readable summary, JSON artifact, exit code."""
    v = release_gate(r)
    marks = "".join("." if g["status"] == "PASS" else
                    ("X" if g["blocking"] else "!") for g in v["gates"])
    print(f"  {label:<34} [{marks}]  {v['verdict']:<5} exit={v['exit_code']}  "
          f"manifest={v['manifest']}")
    for g in v["gates"]:
        if g["status"] == "FAIL":
            kind = "BLOCKING" if g["blocking"] else "advisory"
            print(f"      {kind:<9} {g['gate']:<24} {g['detail']}")
    if artifact_dir is not None:
        path = artifact_dir / f"release_{v['manifest']}.json"
        path.write_text(json.dumps(v, indent=2), encoding="utf-8")
        print(f"      verdict written to {path.name}")
    return v["exit_code"]


def print_ungated_risks() -> None:
    """A release gate that never says what it cannot see reads as coverage it lacks."""
    print("\n  what this gate CANNOT check, restated on every run:")
    for name, why in UNGATED_RISKS:
        print(f"    - {name}:\n        {why}")


def random_split_doseresponse(df_dedup: pd.DataFrame) -> None:
    """Hold the test rows fixed; vary the contamination SHARE deliberately."""
    drift = lab11.windows(df_dedup)["test_drift"]
    held_out, contemporaneous = train_test_split(drift, test_size=0.7, random_state=SEED)
    pre_drift = lab11.windows(df_dedup)["train"]
    y_h = held_out["late"].to_numpy()
    k_h = int(lab11.dunning_rule(held_out).sum())
    # Dose the contamination deliberately. Sampling from a pooled frame lets the much
    # larger pre-drift pool dilute the treatment to whatever share it happens to be, and
    # the measured effect then describes the dilution rather than the hazard.
    print(f"           contaminated arms are STRATIFIED to a fixed contemporaneous share,")
    print(f"           because pooling and sampling would let {len(pre_drift):,} pre-drift")
    print(f"           rows dilute {len(contemporaneous):,} contemporaneous ones to ~23%:")
    for share in (0.0, 0.25, 0.50, 1.0):
        n_contemp = int(round(share * TRAIN_N))
        parts = []
        if n_contemp:
            parts.append(contemporaneous.sample(n=min(n_contemp, len(contemporaneous)),
                                                random_state=SEED))
        if TRAIN_N - n_contemp > 0:
            parts.append(pre_drift.sample(n=TRAIN_N - n_contemp, random_state=SEED))
        tr = pd.concat(parts)
        sc = lab11.fit_score(tr, held_out, seed=SEED)
        p = lab11.precision_recall_at_k(y_h, lab11.topk_flag(sc, k_h))["precision"]
        label = "honest (no contemporaneous rows)" if share == 0 else \
                f"{share:.0%} contemporaneous"
        if share == 0:
            base_p = p
        print(f"           {label:<34} precision on the SAME test rows: {p:.4f}"
              + (f"   (+{p - base_p:.4f})" if share else ""))



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
    survivorship_effect(df_dedup)
    print("\n  Why no gate fires, argued rather than injected: every gate above asserts")
    print("  over an artifact of THIS run - the frame's duplicate count, its feature list,")
    print("  its split kind, its queue size, its predicted rate. The survivorship defect is")
    print("  a property of which rows reached the frame at all, and the frame is internally")
    print("  consistent about the rows it contains. No assertion over it can see the gap.")
    print("  This is a gate-coverage argument, not an injection result, so it is NOT")
    print("  counted in the tally above.")
    matrix["survivorship"] = []
    DEFECTS["survivorship"] = ("data", "train and evaluate on resolved invoices only")

    print("\n" + "=" * 78)
    print("L3  THE SEDUCTIVE DEFECTS, QUANTIFIED")
    print("=" * 78)
    leak = build_run(df_dedup, df_raw, {"leakage"})
    leakage_mechanism(df_dedup)
    print()
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
    random_split_doseresponse(df_dedup)

    wrong = build_run(df_dedup, df_raw, {"wrong_horizon"})
    w_stable2 = lab11.windows(df_dedup)["test_stable"]
    base30 = float((w_stable2["days_late"] > 30).mean())
    print(f"  wrong horizon: the team measures precision {wrong.precision:.4f} against its")
    print(f"           own 7-day label. Against the 30-day escalation the business runs on,")
    print(f"           the same ranking scores {wrong.business_precision:.4f} - and the "
          f"30-day base rate")
    print(f"           is {base30:.4f}: the model delivers "
          f"{wrong.business_precision / base30:.2f}x lift there, but the RULE delivers")
    print(f"           {wrong.business_baseline:.4f} ({wrong.business_baseline / base30:.2f}x) "
          f"on the same question - an edge of "
          f"{wrong.business_precision - wrong.business_baseline:+.4f}, which is inside every")
    print(f"           band in this series. Against the incumbent, on the question that")
    print(f"           matters, the model is worth approximately nothing.")

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

    print("\n" + "=" * 78)
    print("L6  THE GATE SUITE AS A RELEASE GATE")
    print("=" * 78)
    for label, r in [("clean run", clean), ("leaked feature", leak),
                     ("wrong horizon", wrong)]:
        release_gate_cli(r, label)
    print_ungated_risks()


if __name__ == "__main__":
    main()
