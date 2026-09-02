"""Lab for notebook 02.3 "Broadcasting: The Rules, and the Shapes That Lie".

Reproduces every captured number and listing in 02.3:

  L1  the INR invoice lane with customer ids; per-customer means with numpy only
      (unique/return_inverse + bincount - the gather/scatter idioms)
  L2  Stage A - the broadcasting rule implemented by hand (align right; dims equal or 1)
      and an explicit-loop outer computation, both parity-checked against numpy
  L2b what broadcasting costs vs the loops it replaces (machine-dependent; order of
      magnitude only, per 01.4's benchmark honesty)
  L3  the rules as a table - shape pairs in, result shape or refusal out
  L4  stride-0 under the hood: np.broadcast_to grows no buffer (02.1's strides, applied)
  L5  the incident - (n,) minus (n,1) on the August batch: fits in RAM, runs clean, and
      reports a confidently wrong statistic; the same bug on the full lane dies loudly
  L6  the legitimate (n, k) broadcast the incident code was one keepdims away from:
      amount size buckets via comparison against a threshold row
  L7  Stage C - shape contracts: a checked elementwise op that raises on silent fan-out,
      and the preflight that prices an intermediate before allocating it

Data: _data/raw/invoices.csv.gz (regenerate: .venv\\Scripts\\python _data\\generate.py).
Mess touched: M7 (comma amounts, parsed at the boundary), M2 respected by construction -
every statistic here is INR-only, and deviations are from each invoice's OWN customer's
mean, so no cross-currency arithmetic occurs. Nothing trains; M10 untouched.

Run:  .venv\\Scripts\\python "02-numpy-vectorized-computing/_lab/lab_02.3_broadcasting.py"
"""

from __future__ import annotations

import csv
import gzip
import time
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "_data" / "raw"
BATCH_MONTH = "2026-08"          # the month-end anomaly job's batch
BUCKETS_INR = np.array([1_000.0, 10_000.0, 100_000.0, 1_000_000.0])


# ------------------------------------------------------------------ L1: loading

def load_inr_lane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INR invoices: amounts, customer ids, and a boolean mask for the batch month."""
    amount, cust, in_batch = [], [], []
    with gzip.open(RAW / "invoices.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["currency"] != "INR":
                continue
            amount.append(float(row["amount"].replace(",", "")))   # M7 at the boundary
            cust.append(row["customer_id"])
            in_batch.append(row["issue_date"].startswith(BATCH_MONTH))
    return (np.array(amount, dtype=np.float64), np.array(cust),
            np.array(in_batch, dtype=bool))


def customer_means(amounts: np.ndarray, cust: np.ndarray) -> np.ndarray:
    """Per-invoice 'my customer's mean amount', with numpy alone.

    unique(return_inverse) labels each row with its group id; bincount(weights=) is the
    scatter-add; means[inverse] is the gather. Three lines that replace a groupby - and
    the inverse-index gather is fancy indexing doing legitimate work (02.4 owns it).
    """
    _, inverse = np.unique(cust, return_inverse=True)
    sums = np.bincount(inverse, weights=amounts)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse]                       # (n,) aligned to invoices


# ------------------------------------------- L2: Stage A - the rule, implemented

def my_broadcast_shape(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
    """The entire broadcasting rule, from scratch.

    Align shapes at the RIGHT edge; missing left dimensions count as 1; two dimensions
    are compatible when equal or when either is 1 (the 1 is stretched); anything else
    refuses. numpy does nothing more than this, applied before every elementwise op.
    """
    out = []
    for i in range(1, max(len(s1), len(s2)) + 1):
        d1 = s1[-i] if i <= len(s1) else 1
        d2 = s2[-i] if i <= len(s2) else 1
        if d1 != 1 and d2 != 1 and d1 != d2:
            raise ValueError(f"incompatible: {s1} vs {s2} at axis -{i} ({d1} vs {d2})")
        out.append(max(d1, d2))
    return tuple(reversed(out))


def outer_add_loops(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """What (k,1) + (1,m) MEANS, spelled as loops: every pair, stretched 1s re-reading."""
    out = np.empty((len(col), len(row)))
    for i in range(len(col)):          # the stretched column axis
        for j in range(len(row)):      # the stretched row axis
            out[i, j] = col[i] + row[j]
    return out


def stage_a_parity(amounts: np.ndarray) -> None:
    cases = [((7,), (7,)), ((7,), (1,)), ((7,), (7, 1)), ((3, 1), (1, 4)),
             ((2, 3), (3,)), ((5, 1, 4), (7, 1))]
    print(f"  {'shapes':<24}{'my rule':>14}{'numpy':>14}")
    for s1, s2 in cases:
        mine = my_broadcast_shape(s1, s2)
        ref = np.broadcast_shapes(s1, s2)
        assert mine == ref, (s1, s2, mine, ref)
        print(f"  {f'{s1} + {s2}':<24}{str(mine):>14}{str(ref):>14}   MATCH")
    try:
        my_broadcast_shape((2, 3), (2,))
    except ValueError as exc:
        print(f"  (2, 3) + (2,)           -> my rule refuses: {exc}")
    try:
        np.broadcast_shapes((2, 3), (2,))
    except ValueError:
        print(f"  (2, 3) + (2,)           -> numpy refuses too (aligned at the RIGHT,")
        print(f"                             2 meets 3 - right-alignment is the rule's teeth)")

    col = amounts[:3].reshape(3, 1)               # three real invoice amounts
    row = BUCKETS_INR.reshape(1, 4)
    mine = outer_add_loops(amounts[:3], BUCKETS_INR)
    ref = col + row
    assert np.array_equal(mine, ref)
    print(f"\n  outer add, 3 invoices x 4 thresholds: explicit loops == numpy "
          f"broadcast -> {np.array_equal(mine, ref)}  (shape {ref.shape})")


def loops_vs_broadcast(amounts: np.ndarray) -> None:
    """The speed HALF of the argument (correctness is L5's half). Order of magnitude."""
    a = amounts[:50_000]

    t0 = time.perf_counter()
    counts_loop = [0, 0, 0, 0]
    for v in a:                                   # the 2019-notes shape
        for k, thr in enumerate(BUCKETS_INR):
            if v >= thr:
                counts_loop[k] += 1
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    counts_vec = (a[:, np.newaxis] >= BUCKETS_INR).sum(axis=0)
    t_vec = time.perf_counter() - t0

    assert counts_loop == counts_vec.tolist()
    print(f"  bucket counts agree: {counts_vec.tolist()}")
    print(f"  python loops : {t_loop * 1e3:>8.1f} ms")
    print(f"  broadcast    : {t_vec * 1e3:>8.1f} ms   (~{t_loop / max(t_vec, 1e-9):,.0f}x "
          f"on this run - machine-dependent, read as an order of magnitude)")


# -------------------------------------------------- L4: stride 0 under the hood

def stride_zero(amounts: np.ndarray) -> None:
    n = 10_000
    stretched = np.broadcast_to(BUCKETS_INR, (n, 4))
    print(f"  np.broadcast_to(thresholds, ({n:,}, 4)):")
    print(f"    shape {stretched.shape}   strides {stretched.strides}   "
          f"base buffer {stretched.base.nbytes} bytes")
    print(f"    the first stride is 0: moving 'down' re-reads the SAME 32 bytes - the")
    print(f"    stretch is 02.1's header trick, not an allocation. Materialise it")
    print(f"    (np.ascontiguousarray) and it would cost "
          f"{n * 4 * 8 / 1e3:,.0f} KB for real.")
    print(f"    writeable: {stretched.flags.writeable} - numpy refuses writes through a")
    print(f"    stretched view, because one write would 'land' in {n:,} places at once")


# ------------------------------------------------------------- L5: the incident

def incident(amounts: np.ndarray, cust: np.ndarray,
             in_batch: np.ndarray) -> dict[str, float]:
    """The month-end anomaly metric, correct and broken, on the same batch.

    The helper refactor: per-customer means came back (n,) and were reshaped to (n, 1)
    'to align with a planned matrix step'. Elementwise subtraction then silently fans
    out: (n,) - (n,1) -> (n,n), and every reduction downstream still runs.
    """
    a = amounts[in_batch]
    m = customer_means(amounts, cust)[in_batch]
    n = len(a)
    print(f"  August 2026 INR batch: n={n:,} invoices")

    dev_good = a - m                                     # (n,) - (n,): elementwise
    mad_good = float(np.abs(dev_good).mean())
    print(f"  correct: (n,) - (n,)     -> shape {dev_good.shape}, "
          f"{dev_good.nbytes / 1e6:,.1f} MB   metric = {mad_good:,.2f}")

    m_col = m.reshape(-1, 1)                             # the refactor's (n, 1)
    dev_bad = a - m_col                                  # (n,) - (n,1) -> (n, n)
    mad_bad = float(np.abs(dev_bad).mean())
    print(f"  shipped: (n,) - (n,1)    -> shape {dev_bad.shape}, "
          f"{dev_bad.nbytes / 1e6:,.1f} MB   metric = {mad_bad:,.2f}")
    print(f"  the job RAN: the intermediate fits in RAM at this n, every value in it is")
    print(f"  a real difference of real numbers - invoice i minus customer-mean j, for")
    print(f"  ALL pairs - and the metric inflated {mad_bad / mad_good:,.1f}x")

    diag = float(np.abs(np.diagonal(dev_bad)).mean())
    print(f"  the diagonal of the (n,n) mistake IS the correct answer: {diag:,.2f} -")
    print(f"  the right {n:,} numbers, drowned among {n * n - n:,} meaningless pairs")

    print(f"\n  the same bug on the full INR lane ({len(amounts):,} rows):")
    try:
        _ = amounts - customer_means(amounts, cust).reshape(-1, 1)
        print("    ...allocated (should not happen on this machine)")
    except MemoryError as exc:
        print(f"    MemoryError: {str(exc)[:66]}")
        print(f"    loud at scale, silent in the batch - the batch size decided the")
        print(f"    symptom, not the bug")
    return {"good": mad_good, "bad": mad_bad}


# ------------------------------------------- L6: the legitimate (n, k) broadcast

def size_buckets(amounts: np.ndarray, in_batch: np.ndarray) -> None:
    a = amounts[in_batch]
    hits = a[:, np.newaxis] >= BUCKETS_INR               # (n,1) >= (4,) -> (n,4): intended
    counts = hits.sum(axis=0)
    print(f"  a[:, np.newaxis] >= thresholds: {a.shape} x {BUCKETS_INR.shape} "
          f"-> {hits.shape}  ({hits.nbytes / 1e3:,.0f} KB of bool)")
    for thr, c in zip(BUCKETS_INR, counts):
        print(f"    >= INR {thr:>11,.0f} : {int(c):>6,} invoices")
    print(f"  the same fan-out mechanism as the incident - declared with np.newaxis,")
    print(f"  sized k=4 by design, and reduced immediately. Intent, written down.")


# --------------------------------------------------- L7: Stage C - shape contracts

class ShapeError(ValueError):
    """An elementwise op was about to fan out; refusing beats reporting nonsense."""


def checked_elementwise(a: np.ndarray, b: np.ndarray, op=np.subtract) -> np.ndarray:
    """Elementwise-or-die: refuse any silent broadcast fan-out.

    Broadcasting is opt-in here: this helper is for the (many) call sites that MEAN
    'element by element'. Sites that mean to fan out say so with np.newaxis and skip it.
    """
    out_shape = np.broadcast_shapes(a.shape, b.shape)
    if out_shape != a.shape or out_shape != b.shape:
        raise ShapeError(f"result {out_shape} != operand shapes {a.shape}/{b.shape}: "
                         f"an implicit broadcast is about to fan out")
    return op(a, b)


def preflight_cost(*shapes: tuple[int, ...], itemsize: int = 8) -> str:
    """Price an intermediate BEFORE allocating it - one line, no surprises."""
    out = np.broadcast_shapes(*shapes)
    return f"{out} -> {int(np.prod(out)) * itemsize / 1e6:,.1f} MB"


def contract_demo(amounts: np.ndarray, cust: np.ndarray, in_batch: np.ndarray) -> None:
    a = amounts[in_batch]
    m = customer_means(amounts, cust)[in_batch]

    ok = checked_elementwise(a, m)
    print(f"  checked_elementwise(a (n,), m (n,))     -> ok, shape {ok.shape}")
    try:
        checked_elementwise(a, m.reshape(-1, 1))
        print("    ...passed (guard failed)")
    except ShapeError as exc:
        print(f"  checked_elementwise(a (n,), m (n,1))    -> ShapeError: {exc}")

    print(f"\n  preflight pricing, before any allocation:")
    print(f"    (n,) - (n,)   : {preflight_cost(a.shape, m.shape)}")
    print(f"    (n,) - (n,1)  : {preflight_cost(a.shape, (len(m), 1))}")
    print(f"    full lane bug : "
          f"{preflight_cost(amounts.shape, (len(amounts), 1))}")
    print(f"  the guard turns the silent batch-size lottery into a refusal; the")
    print(f"  preflight turns the loud MemoryError into a line in a design review")


def main() -> None:
    print("=" * 78)
    print("L1  THE INR LANE + PER-CUSTOMER MEANS (numpy-only groupby)")
    print("=" * 78)
    amounts, cust, in_batch = load_inr_lane()
    m = customer_means(amounts, cust)
    print(f"  rows={len(amounts):,}  customers={len(np.unique(cust)):,}  "
          f"August-2026 batch={int(in_batch.sum()):,}")
    print(f"  per-invoice customer mean: shape {m.shape}, aligned to invoices by an")
    print(f"  inverse-index gather - three numpy lines standing in for a groupby")

    print("\n" + "=" * 78)
    print("L2  STAGE A - THE RULE, IMPLEMENTED AND PARITY-CHECKED")
    print("=" * 78)
    stage_a_parity(amounts)

    print("\n" + "=" * 78)
    print("L2b LOOPS VS BROADCAST (the speed half of the argument)")
    print("=" * 78)
    loops_vs_broadcast(amounts)

    print("\n" + "=" * 78)
    print("L4  STRIDE 0 - THE STRETCH THAT NEVER ALLOCATES")
    print("=" * 78)
    stride_zero(amounts)

    print("\n" + "=" * 78)
    print("L5  THE INCIDENT - (n,) MINUS (n,1) ON THE AUGUST BATCH")
    print("=" * 78)
    incident(amounts, cust, in_batch)

    print("\n" + "=" * 78)
    print("L6  THE LEGITIMATE (n,k) BROADCAST - SAME MECHANISM, DECLARED")
    print("=" * 78)
    size_buckets(amounts, in_batch)

    print("\n" + "=" * 78)
    print("L7  STAGE C - SHAPE CONTRACTS")
    print("=" * 78)
    contract_demo(amounts, cust, in_batch)


if __name__ == "__main__":
    main()
