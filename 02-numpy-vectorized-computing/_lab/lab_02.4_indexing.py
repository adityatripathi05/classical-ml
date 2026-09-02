"""Lab for notebook 02.4 "Indexing & Selection: Masks, Fancy Indexing, and the View-Copy
Boundary".

Reproduces every captured number and listing in 02.4:

  L1  the INR lane with status (M6 lowered at the boundary, per 02.2) and days overdue
  L2  the assignment asymmetry table - five ways to write through an index, two of which
      lie: chained-basic writes through (view), chained-advanced vanishes (copy), and
      duplicate-index += counts each target once
  L3  the incident - the escalation job whose chained assignment never landed: zero
      escalations for weeks, and the top-overdue money that went uncontacted
  L4  duplicate-index scatter done right: buffered += vs np.add.at vs bincount, three-way
      parity on per-customer invoice counts
  L5  the selection toolkit - np.where, np.select bands, argsort vs argpartition for
      top-k (with the 2.5 descending= spelling), each on the collections lane
  L6  Stage C - escalate_top_overdue() with a landed-writes post-condition, idempotency,
      and the k > n_overdue edge; the broken version fails the same tests loudly

Data: _data/raw/invoices.csv.gz (regenerate: .venv\\Scripts\\python _data\\generate.py).
Mess touched: M7 (comma amounts, boundary-parsed), M6 (status casing, lowered at load -
02.2's np.strings lesson applied), M2 respected: INR lane only, so "top-k by amount" is a
single-currency ranking. Reference date for days-overdue is the export end, 2026-08-31
(_data/SPEC.md). Nothing trains; M10 untouched.

Run:  .venv\\Scripts\\python "02-numpy-vectorized-computing/_lab/lab_02.4_indexing.py"
"""

from __future__ import annotations

import csv
import gzip
import time
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "_data" / "raw"
REF_DATE = np.datetime64("2026-08-31")     # export end, per _data/SPEC.md
TOP_K = 200                                # daily escalation capacity


# ------------------------------------------------------------------ L1: loading

def load_inr_lane() -> dict[str, np.ndarray]:
    amount, status, due, cust = [], [], [], []
    with gzip.open(RAW / "invoices.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["currency"] != "INR":
                continue
            amount.append(float(row["amount"].replace(",", "")))   # M7
            status.append(row["status"])
            due.append(row["due_date"])
            cust.append(row["customer_id"])
    status_arr = np.array(status, dtype=np.dtypes.StringDType())
    return {
        "amount": np.array(amount, dtype=np.float64),
        "status": np.strings.lower(status_arr),                    # M6, per 02.2
        "due": np.array(due, dtype="datetime64[D]"),
        "cust": np.array(cust),
    }


def overdue_mask(d: dict[str, np.ndarray]) -> np.ndarray:
    return d["status"] == "overdue"


# ------------------------------------- L2: the assignment asymmetry, all five ways

def assignment_asymmetry(amount: np.ndarray, over: np.ndarray) -> None:
    """getitem-then-setitem is TWO operations, and the first one decides everything.

    a[sel] = v            -> ONE __setitem__: always writes into a (scatter for advanced)
    a[basic][more] = v    -> getitem returns a VIEW  -> the chained write LANDS
    a[advanced][more] = v -> getitem returns a COPY  -> the chained write VANISHES
    a[idx] += 1 (dup idx) -> gather, add, scatter: duplicate targets written ONCE
    """
    rows = []

    f = np.zeros(len(amount), dtype=bool)
    f[5:200] = True                                   # single basic setitem
    rows.append(("a[5:200] = True (single, basic)", int(f.sum()), "landed"))

    f = np.zeros(len(amount), dtype=bool)
    f[5:200][:50] = True                              # chained via a basic slice: a VIEW
    rows.append(("a[5:200][:50] = True (chained, basic)", int(f.sum()),
                 "landed - the slice is a view (02.1)"))

    f = np.zeros(len(amount), dtype=bool)
    f[over] = True                                    # single advanced setitem: scatter
    rows.append(("a[over] = True (single, advanced)", int(f.sum()), "landed - a scatter"))

    f = np.zeros(len(amount), dtype=bool)
    f[over][:50] = True                               # chained via a mask: a COPY
    rows.append(("a[over][:50] = True (chained, advanced)", int(f.sum()),
                 "VANISHED - wrote into a temporary copy"))

    c = np.zeros(len(amount), dtype=np.int64)
    idx = np.zeros(10, dtype=np.intp)                 # ten writes, ONE target
    c[idx] += 1
    rows.append(("c[idx] += 1, idx = ten zeros", int(c[0]),
                 "counted ONCE - gather/add/scatter, last write wins"))

    print(f"  {'expression':<44}{'result':>8}   verdict")
    for expr, got, verdict in rows:
        print(f"  {expr:<44}{got:>8,}   {verdict}")
    print("\n  the rule: a[i][j] = v desugars to a.__getitem__(i).__setitem__(j, v).")
    print("  Everything hangs on what the FIRST call returns: a view chains the write")
    print("  through; a copy swallows it. 02.1's census already decided which is which -")
    print("  basic slicing views, advanced indexing copies. Assignment just collects the")
    print("  debt. A single a[sel] = v is ONE __setitem__ and always lands, even for masks.")


# ------------------------------------------------------------- L3: the incident

def incident(d: dict[str, np.ndarray]) -> dict[str, float]:
    """The escalation job: flag the top-K overdue invoices by amount. Two spellings."""
    amount, over = d["amount"], overdue_mask(d)
    n_over = int(over.sum())
    print(f"  INR lane: {len(amount):,} invoices, {n_over:,} currently overdue; "
          f"escalation capacity {TOP_K}/day")

    escalated = np.zeros(len(amount), dtype=bool)     # the SHIPPED job
    sub = escalated[over]                             # advanced getitem -> a copy
    order = np.argsort(d["amount"][over])             # sort the overdue by amount
    sub[order[-TOP_K:]] = True                        # writes into the copy...
    print(f"\n  shipped:  escalated[over][top_idx] = True  -> "
          f"escalated.sum() = {int(escalated.sum())}")
    print(f"            the job logged 'escalated {TOP_K} invoices' from len(top_idx),")
    print(f"            green every day, and wrote {int(escalated.sum())} flags")

    fixed = np.zeros(len(amount), dtype=bool)         # the CORRECT job
    over_idx = np.flatnonzero(over)                   # compose indices FIRST...
    top = over_idx[np.argsort(amount[over_idx])[-TOP_K:]]
    fixed[top] = True                                 # ...then ONE setitem: a scatter
    at_stake = float(amount[top].sum())
    print(f"\n  fixed:    fixed[over_idx[top_order]] = True -> "
          f"fixed.sum() = {int(fixed.sum())}")
    print(f"  the money the silent version left uncontacted: INR {at_stake:,.0f}")
    print(f"  largest unworked overdue invoice: INR {float(amount[top].max()):,.0f}")
    return {"lost": int(escalated.sum()), "landed": int(fixed.sum()), "stake": at_stake}


# ---------------------------------------------- L4: duplicate-index scatter, right

def scatter_counts(d: dict[str, np.ndarray]) -> None:
    uniq, inverse = np.unique(d["cust"], return_inverse=True)

    buffered = np.zeros(len(uniq), dtype=np.int64)
    buffered[inverse] += 1                            # duplicate targets: written once
    accurate = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(accurate, inverse, 1)                   # true accumulation
    bcount = np.bincount(inverse)                     # the 02.3 idiom, as referee

    print(f"  per-customer invoice counts over {len(uniq):,} customers:")
    print(f"    counts[inverse] += 1 : total {int(buffered.sum()):>8,}   "
          f"max {int(buffered.max())}   <- every customer 'has' at most 1 invoice")
    print(f"    np.add.at(...)       : total {int(accurate.sum()):>8,}   "
          f"max {int(accurate.max())}")
    print(f"    np.bincount(inverse) : total {int(bcount.sum()):>8,}   "
          f"max {int(bcount.max())}   parity with add.at: "
          f"{bool(np.array_equal(accurate, bcount))}")
    print("\n  += through an index is gather -> add -> scatter: ten writes to one slot")
    print("  store one increment. np.add.at (and bincount, for counting) accumulate for")
    print("  real. The wrong version is not noisy-wrong, it is 'everything is 1' wrong.")


# ---------------------------------------------------- L5: the selection toolkit

def selection_toolkit(d: dict[str, np.ndarray]) -> None:
    amount, over = d["amount"], overdue_mask(d)
    days_over = np.where(over, (REF_DATE - d["due"]).astype("timedelta64[D]")
                         .astype(np.int64), 0)

    # First finding, courtesy of np.where itself: the STATUS says overdue, but the
    # CALENDAR disagrees for most of them - the export's status is a projection of how
    # the invoice will end, not a statement about today (01.2: test what a column claims).
    past_due = over & (days_over > 0)
    print(f"  status says overdue: {int(over.sum()):,} invoices; the calendar agrees on "
          f"only {int(past_due.sum()):,}")
    print(f"  ({1 - past_due.sum() / over.sum():.0%} of status-overdue rows are not past "
          f"due at {REF_DATE} - the status")
    print(f"   column is a projection, so time-based logic must derive from DATES)")

    fees = np.where(days_over > 30, amount * 0.015, 0.0)      # 3-arg where: piecewise
    print(f"\n  np.where(days_over > 30, 1.5% fee, 0): {int((fees > 0).sum()):,} invoices "
          f"assessed, INR {float(fees.sum()):,.0f} total")

    tiers = np.select(
        [days_over > 45, days_over > 15, days_over > 0],
        ["final-notice", "firm", "gentle"], default="not-yet-due")
    labels, counts = np.unique(tiers[past_due], return_counts=True)
    tier_txt = ", ".join(f"{l}={c:,}" for l, c in zip(labels, counts))
    print(f"  np.select reminder tiers over the calendar-past-due: {tier_txt}")
    print(f"    (first matching condition wins - order the bands from strictest down)")

    over_amt = amount[over]
    t0 = time.perf_counter(); full = np.argsort(over_amt)[-TOP_K:]; t_sort = time.perf_counter() - t0
    t0 = time.perf_counter(); part = np.argpartition(over_amt, -TOP_K)[-TOP_K:]; t_part = time.perf_counter() - t0
    same = bool(np.array_equal(np.sort(full), np.sort(part)))
    print(f"\n  top-{TOP_K} of {len(over_amt):,} overdue amounts:")
    print(f"    argsort  (full order) : {t_sort * 1e3:7.2f} ms")
    print(f"    argpartition (no order): {t_part * 1e3:7.2f} ms   same set: {same}")
    print(f"    (timings machine-dependent; the point is O(n log n) vs O(n) - and that")
    print(f"     01.1's topk_flag used argsort when partition suffices for a queue)")

    top3 = np.sort(over_amt, descending=True)[:3]
    print(f"  np.sort(..., descending=True)[:3] (numpy 2.5): "
          f"{', '.join(f'INR {v:,.0f}' for v in top3)}")


# ------------------------------------------------- L6: Stage C - the guarded write

class WriteLostError(RuntimeError):
    """A scatter write did not land: the target was (or became) a temporary."""


def escalate_top_overdue(amount: np.ndarray, over: np.ndarray, k: int,
                         escalated: np.ndarray) -> np.ndarray:
    """Flag the top-k overdue invoices by amount, and PROVE the writes landed.

    Index composition instead of chained selection: build the final integer indices
    first, then write through ONE __setitem__. The post-condition costs one gather and
    turns any regression to chained selection into an exception, not an empty queue.
    """
    over_idx = np.flatnonzero(over)
    k = min(k, len(over_idx))
    top = over_idx[np.argpartition(amount[over_idx], -k)[-k:]] if k else over_idx[:0]
    escalated[top] = True                              # one setitem: a scatter
    if int(escalated[top].sum()) != len(top):          # landed-writes post-condition
        raise WriteLostError(f"{len(top) - int(escalated[top].sum())} of {len(top)} "
                             f"escalation flags did not land")
    return top


def contract_demo(d: dict[str, np.ndarray]) -> None:
    amount, over = d["amount"], overdue_mask(d)

    esc = np.zeros(len(amount), dtype=bool)
    top = escalate_top_overdue(amount, over, TOP_K, esc)
    checks = {
        f"exactly {TOP_K} flags landed": int(esc.sum()) == TOP_K,
        "flags landed on overdue rows only": bool(np.all(over[top])),
        "idempotent re-run adds nothing":
            (lambda: (escalate_top_overdue(amount, over, TOP_K, esc),
                      int(esc.sum()) == TOP_K)[1])(),
        "k > n_overdue caps cleanly":
            int(escalate_top_overdue(amount, over, 10**9,
                                     np.zeros(len(amount), dtype=bool)).size)
            == int(over.sum()),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\n  and the shipped bug, expressed against the same post-condition:")
    esc2 = np.zeros(len(amount), dtype=bool)
    sub = esc2[over]
    sub[np.argsort(amount[over])[-TOP_K:]] = True
    landed = int(esc2.sum())
    print(f"    chained version landed {landed} of {TOP_K} - the post-condition idea")
    print(f"    (recount from the TARGET array, never from the index length) is what")
    print(f"    separates 'logged 200' from 'wrote 200'")


def main() -> None:
    print("=" * 78)
    print("L1  THE INR LANE (status lowered at the boundary, per 02.2)")
    print("=" * 78)
    d = load_inr_lane()
    over = overdue_mask(d)
    print(f"  rows={len(d['amount']):,}  overdue={int(over.sum()):,}  "
          f"status dtype={d['status'].dtype}")

    print("\n" + "=" * 78)
    print("L2  THE ASSIGNMENT ASYMMETRY - FIVE WRITES, TWO LIES")
    print("=" * 78)
    assignment_asymmetry(d["amount"], over)

    print("\n" + "=" * 78)
    print("L3  THE INCIDENT - THE ESCALATION THAT NEVER LANDED")
    print("=" * 78)
    incident(d)

    print("\n" + "=" * 78)
    print("L4  DUPLICATE-INDEX SCATTER - +=, add.at, bincount")
    print("=" * 78)
    scatter_counts(d)

    print("\n" + "=" * 78)
    print("L5  THE SELECTION TOOLKIT - where, select, argpartition, descending=")
    print("=" * 78)
    selection_toolkit(d)

    print("\n" + "=" * 78)
    print("L6  STAGE C - THE GUARDED ESCALATION WRITE")
    print("=" * 78)
    contract_demo(d)


if __name__ == "__main__":
    main()
