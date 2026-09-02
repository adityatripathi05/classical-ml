"""Lab for notebook 02.5 "Ufuncs, Reductions & Missing Data".

Reproduces every captured number and listing in 02.5:

  L1  the full invoice lane joined to first-payment dates (M1 dedup counted, M13 first
      payment, M8 both timestamp formats parsed and counted) -> days_late with honest
      NaNs for unresolved invoices
  L2  the ufunc anatomy - one elementwise core, five methods; reduce==sum parity and
      accumulate as the running total
  L3  axis semantics on a real 2-D pivot - month x status counts built with np.add.at,
      reduced along each axis, keepdims shapes
  L4  NaN algebra and the traps - propagation, nan-function semantics (nansum of
      all-NaN is 0.0), errstate, where=/initial as the mask-explicit alternative,
      and per-month NaN coverage (01.4's censoring gradient, recomputed here)
  L5  the incident, three eras - inner-join era (survivorship), NaN-propagation era
      (blank dashboard), nanmean era (silent censoring bias) - plus the controlled
      experiment: censor a fully-resolved month and watch nanmean drift
  L6  Stage C - the coverage-aware rollup: statistic + n + coverage + verdict, with the
      WITHHOLD threshold that keeps a biased number off the dashboard

Data: _data/raw/{invoices,payments}.csv.gz (regenerate: .venv\\Scripts\\python
_data\\generate.py). Mess touched: M1 (896 duplicate invoice rows skipped at load),
M6 (status lowered at the boundary), M8 (paylane DD/MM/YYYY timestamps parsed, counted),
M13 (first payment per invoice). days_late is currency-free, so the whole lane is used
and M2 never arises. Nothing trains; M10 untouched.

Run:  .venv\\Scripts\\python "02-numpy-vectorized-computing/_lab/lab_02.5_reductions.py"
"""

from __future__ import annotations

import csv
import gzip
from datetime import datetime
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "_data" / "raw"
MIN_COVERAGE = 0.90         # Stage C: months below this label share are WITHHELD


# ------------------------------------------------------------------ L1: loading

def parse_paid(raw: str) -> datetime:
    """M8: most gateways write ISO-8601Z; paylane writes DD/MM/YYYY HH:MM."""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.strptime(raw, "%d/%m/%Y %H:%M")


def load_lane() -> dict[str, np.ndarray]:
    first_paid: dict[str, datetime] = {}
    legacy = 0
    with gzip.open(RAW / "payments.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row["paid_at"]
            if "/" in raw:
                legacy += 1
            dt = parse_paid(raw)
            inv = row["invoice_id"]
            if inv not in first_paid or dt < first_paid[inv]:   # M13: first payment
                first_paid[inv] = dt

    seen: set[str] = set()
    dupes = 0
    issue, due, paid, status = [], [], [], []
    with gzip.open(RAW / "invoices.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            inv = row["invoice_id"]
            if inv in seen:                                     # M1: repost bug
                dupes += 1
                continue
            seen.add(inv)
            issue.append(row["issue_date"])
            due.append(row["due_date"])
            status.append(row["status"].lower())                # M6 at the boundary
            p = first_paid.get(inv)
            paid.append(np.datetime64(p.date()) if p else np.datetime64("NaT", "D"))

    issue_d = np.array(issue, dtype="datetime64[D]")
    due_d = np.array(due, dtype="datetime64[D]")
    paid_d = np.array(paid, dtype="datetime64[D]")
    # NaT would propagate as a garbage integer under a bare astype; build from isnat:
    days_late = np.where(np.isnat(paid_d), np.nan,
                         (paid_d - due_d).astype("timedelta64[D]").astype("float64"))

    months = issue_d.astype("datetime64[M]")
    uniq_months, month_idx = np.unique(months, return_inverse=True)
    return {"issue": issue_d, "due": due_d, "paid": paid_d, "days_late": days_late,
            "status": np.array(status), "months": uniq_months, "month_idx": month_idx,
            "n_dupes": dupes, "n_legacy_ts": legacy}


# ------------------------------------------------------- L2: the ufunc anatomy

def ufunc_anatomy(d: dict[str, np.ndarray]) -> None:
    resolved = d["days_late"][~np.isnan(d["days_late"])]
    print("  np.add is ONE elementwise core wearing five method hats:")
    print(f"    {'add(a, b)':<18} elementwise    (every arithmetic op in this series)")
    print(f"    {'add.reduce(a)':<18} fold to one   == a.sum(): "
          f"{np.add.reduce(resolved):,.0f} == {resolved.sum():,.0f}  "
          f"parity {bool(np.add.reduce(resolved) == resolved.sum())}")
    counts = np.bincount(d["month_idx"])
    cumulative = np.add.accumulate(counts)
    print(f"    {'add.accumulate':<18} running fold  invoices issued, cumulative: "
          f"first 4 months {cumulative[:4].tolist()}, all-time {cumulative[-1]:,}")
    print(f"    {'add.outer':<18} every pair     (02.3's declared fan-out, as a method)")
    print(f"    {'add.at':<18} unbuffered scatter (02.4's duplicate-target fix)")
    print("\n  reduce, accumulate, outer, at - every ufunc ships all four, so the")
    print("  toolkit generalises: np.maximum.reduce is max, np.maximum.accumulate is")
    print("  a running high-water mark, np.logical_or.reduce is 'any', and so on.")


# ------------------------------------------------- L3: axis semantics on a pivot

def pivot_axis_semantics(d: dict[str, np.ndarray]) -> None:
    statuses, status_idx = np.unique(d["status"], return_inverse=True)
    n_m, n_s = len(d["months"]), len(statuses)
    pivot = np.zeros((n_m, n_s), dtype=np.int64)
    np.add.at(pivot, (d["month_idx"], status_idx), 1)       # 2-D unbuffered scatter

    print(f"  pivot[month, status]: shape {pivot.shape}  "
          f"({n_m} months x statuses {statuses.tolist()})")
    hdr = "".join(f"{s:>12}" for s in statuses)
    print(f"    {'':<10}{hdr}")
    for i in (-3, -2, -1):                                   # the newest three months
        row = "".join(f"{int(v):>12,}" for v in pivot[i])
        print(f"    {str(d['months'][i]):<10}{row}")

    per_status = pivot.sum(axis=0)                           # collapse months
    per_month = pivot.sum(axis=1)                            # collapse statuses
    print(f"\n  sum(axis=0) collapses MONTHS   -> shape {per_status.shape}: "
          f"{ {s: int(v) for s, v in zip(statuses, per_status)} }")
    print(f"  sum(axis=1) collapses STATUSES -> shape {per_month.shape}: "
          f"per-month totals, e.g. {str(d['months'][-1])} = {int(per_month[-1]):,}")
    print(f"  sum(axis=1, keepdims=True)     -> shape {pivot.sum(axis=1, keepdims=True).shape}: "
          f"ready to broadcast as shares = pivot / row_totals (02.3's declared (n,1))")
    shares = pivot[-1] / pivot[-1].sum()
    top = statuses[np.argmax(shares)]
    print(f"  worked example: {str(d['months'][-1])} status mix is "
          f"{shares.max():.0%} '{top}' - the axis you SUM is the axis you LOSE")
    print(f"  grand total, both orders: {int(pivot.sum()):,} == "
          f"{int(pivot.sum(axis=0).sum()):,} == invoices loaded")


# ---------------------------------------------------- L4: NaN algebra and traps

def nan_algebra(d: dict[str, np.ndarray]) -> None:
    x = d["days_late"]
    n_nan = int(np.isnan(x).sum())
    print(f"  days_late: {len(x):,} invoices, {n_nan:,} unresolved -> NaN "
          f"({n_nan / len(x):.1%} of the lane)")
    print(f"  NaN algebra: nan == nan -> {bool(np.float64('nan') == np.float64('nan'))}; "
          f"membership needs np.isnan (and np.isnat for NaT, 02.2)")
    print(f"  propagation: x.sum() = {x.sum()}   x.mean() = {x.mean()}   "
          f"one NaN poisons any plain reduction")
    import warnings
    all_nan = x[np.isnan(x)][:5]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nm = np.nanmean(all_nan)
    warned = caught[0].message.args[0] if caught else "no warning"
    print(f"  TRAP: nansum of an ALL-NaN slice = {np.nansum(all_nan)} - a confident,")
    print(f"  silent zero - while nanmean gives {nm} with RuntimeWarning({warned!r}):")
    print(f"  the two nan-functions disagree about what 'no data' means")
    with np.errstate(invalid="ignore"):
        q = np.float64(0) / np.float64(0)
    print(f"  0/0 under np.errstate(invalid='ignore') -> {q} with the RuntimeWarning")
    print(f"  suppressed; unguarded it warns - decide loud-or-quiet per call site,")
    print(f"  and narrowly: a blanket ignore hides every future invalid op too")

    mask = ~np.isnan(x)
    explicit = x.sum(where=mask) / np.count_nonzero(mask)
    print(f"\n  the mask-explicit spelling: x.sum(where=~isnan, initial=0) / count")
    print(f"    = {explicit:.3f}   vs np.nanmean(x) = {np.nanmean(x):.3f}   "
          f"same number, but the mask and the count are now VISIBLE variables")

    print(f"\n  label coverage by issue month (the tail of 01.4's censoring gradient):")
    resolved_frac = 1 - np.bincount(d["month_idx"], weights=np.isnan(x)) / \
        np.bincount(d["month_idx"])
    for i in (-4, -3, -2, -1):
        print(f"    {str(d['months'][i])}  coverage {resolved_frac[i]:.1%}")


# ------------------------------------------------------------- L5: the incident

def incident(d: dict[str, np.ndarray]) -> None:
    x, mi, months = d["days_late"], d["month_idx"], d["months"]
    n_m = len(months)

    resolved = ~np.isnan(x)
    sums = np.bincount(mi, weights=np.where(resolved, x, 0))
    cnts = np.bincount(mi, weights=resolved.astype(float))
    era1 = sums / np.maximum(cnts, 1)                       # era 1: resolved-only mean
    print(f"  era 1 (inner join, resolved only): mean days_late per month; e.g. "
          f"2025-03 = {era1[months == np.datetime64('2025-03')][0]:.2f}, "
          f"stable and quietly survivorship-biased (01.6)")

    era2 = np.zeros(n_m)
    for i in range(n_m):                                    # the refactor: honest NaNs kept
        era2[i] = x[mi == i].mean()                         # plain mean propagates
    blank = int(np.isnan(era2).sum())
    print(f"  era 2 (honest NaNs + plain mean): {blank} of {n_m} monthly panels read NaN")
    print(f"    -> one unresolved invoice anywhere in a month blanks that month; the")
    print(f"       dashboard goes empty back to {str(months[np.isnan(era2)][0])}, "
          f"job green throughout")

    with np.errstate(invalid="ignore"):
        era3 = np.array([np.nanmean(x[mi == i]) for i in range(n_m)])
    print(f"  era 3 (the nanmean hotfix): panels return, and the newest read "
          f"{era3[-1]:.2f} days ({str(months[-1])}) vs {era3[-4]:.2f} "
          f"({str(months[-4])}) - collections looks suddenly faster")

    # The controlled experiment: censor a month whose truth is fully known.
    target = np.datetime64("2025-03")
    t = int(np.flatnonzero(months == target)[0])
    in_m = mi == t
    truth = float(np.nanmean(x[in_m]))
    cutoff = (target + 1).astype("datetime64[D]") + np.timedelta64(30, "D")
    observed = np.where(in_m & (d["paid"] <= cutoff), x, np.nan)
    with np.errstate(invalid="ignore"):
        censored = float(np.nanmean(observed[in_m]))
    cov = float(np.mean(~np.isnan(observed[in_m])))
    print(f"\n  controlled censoring of {target} (its truth is fully resolved):")
    print(f"    true mean days_late            : {truth:>7.2f}  (coverage 100%)")
    print(f"    observed 30 days after month-end: {censored:>7.2f}  (coverage {cov:.0%})")
    print(f"    nanmean under censoring reads {truth - censored:.2f} days FASTER than")
    print(f"    truth - it silently averages whoever has already paid, and fast payers")
    print(f"    pay first.")
    print(f"    The era-3 dashboard is this experiment, live, on every recent month.")


# --------------------------------------------- L6: Stage C - coverage-aware rollup

def robust_monthly_mean(x: np.ndarray, month_idx: np.ndarray, n_months: int,
                        min_coverage: float = MIN_COVERAGE):
    """The rollup as a contract: statistic + n + coverage + verdict, in one pass.

    A statistic whose denominator quietly changed is not a statistic; below the
    coverage floor the honest output is WITHHOLD, not a number.
    """
    resolved = ~np.isnan(x)
    n_all = np.bincount(month_idx, minlength=n_months).astype(float)
    n_res = np.bincount(month_idx, weights=resolved.astype(float), minlength=n_months)
    sums = np.bincount(month_idx, weights=np.where(resolved, x, 0.0),
                       minlength=n_months)
    with np.errstate(invalid="ignore"):
        means = sums / n_res                                 # 0/0 -> nan, deliberately
    coverage = np.divide(n_res, n_all, out=np.zeros(n_months), where=n_all > 0)
    ok = coverage >= min_coverage
    return means, coverage, ok


def contract_demo(d: dict[str, np.ndarray]) -> None:
    x, mi, months = d["days_late"], d["month_idx"], d["months"]
    means, cov, ok = robust_monthly_mean(x, mi, len(months))

    print(f"  {'month':<10}{'mean':>8}{'coverage':>10}   verdict")
    for i in (-6, -5, -4, -3, -2, -1):
        verdict = "report" if ok[i] else f"WITHHOLD (< {MIN_COVERAGE:.0%} resolved)"
        print(f"  {str(months[i]):<10}{means[i]:>8.2f}{cov[i]:>10.1%}   {verdict}")

    checks = {
        "reported months meet the coverage floor": bool(np.all(cov[ok] >= MIN_COVERAGE)),
        "withheld months are exactly the complement":
            bool(np.all(ok == (cov >= MIN_COVERAGE))),
        "means agree with nanmean where defined":
            bool(np.allclose(means[ok],
                             [np.nanmean(x[mi == i]) for i in np.flatnonzero(ok)])),
        "no plain-mean NaN leaks into a reported month":
            bool(not np.any(np.isnan(means[ok]))),
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("\n  the verdict column is the artifact: a dashboard that can say WITHHOLD")
    print("  cannot be silently biased by its own denominator - and the coverage floor")
    print("  is a reviewable number, not a reflex buried in a nan-function.")


def main() -> None:
    print("=" * 78)
    print("L1  THE LANE, JOINED TO FIRST PAYMENTS (M1, M6, M8, M13 at the boundary)")
    print("=" * 78)
    d = load_lane()
    print(f"  invoices={len(d['days_late']):,} (skipped {d['n_dupes']} M1 duplicates)  "
          f"months={len(d['months'])}")
    print(f"  payments parsed: {d['n_legacy_ts']:,} legacy DD/MM/YYYY timestamps (M8) "
          f"handled by fallback")
    print(f"  unresolved -> NaN: {int(np.isnan(d['days_late']).sum()):,} invoices")

    print("\n" + "=" * 78)
    print("L2  THE UFUNC ANATOMY - ONE CORE, FIVE HATS")
    print("=" * 78)
    ufunc_anatomy(d)

    print("\n" + "=" * 78)
    print("L3  AXIS SEMANTICS ON A REAL PIVOT (np.add.at IN 2-D)")
    print("=" * 78)
    pivot_axis_semantics(d)

    print("\n" + "=" * 78)
    print("L4  NaN ALGEBRA - PROPAGATION, THE nan* FAMILY, errstate, where=")
    print("=" * 78)
    nan_algebra(d)

    print("\n" + "=" * 78)
    print("L5  THE INCIDENT - THREE ERAS OF ONE ROLLUP")
    print("=" * 78)
    incident(d)

    print("\n" + "=" * 78)
    print("L6  STAGE C - THE COVERAGE-AWARE ROLLUP")
    print("=" * 78)
    contract_demo(d)


if __name__ == "__main__":
    main()
