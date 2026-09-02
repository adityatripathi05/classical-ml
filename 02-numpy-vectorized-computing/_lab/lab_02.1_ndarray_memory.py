"""Lab for notebook 02.1 "The ndarray Memory Model: Buffer, Dtype, Strides - Views vs Copies".

Reproduces every captured number and listing in 02.1:

  L1  loading PayFlow payments into arrays with the stdlib - what a numpy array is ABOUT
      to interpret (M7 comma amounts handled at the boundary, per 01.2's contract lesson)
  L2  Stage A - a strided 2-D view over a flat buffer, built by hand: indexing, transpose
      and row-slicing as O(1) bookkeeping over (offset, shape, strides), parity vs numpy
  L3  the view/copy census - which operations alias the buffer and which allocate,
      settled by np.shares_memory rather than by folklore
  L4  the incident - an in-place USD normalization through a tail slice mutates the
      canonical amounts array at a distance; the reconciliation gap it produces
  L5  the 2019->2026 migration semantics: np.array(copy=False) now raises, setting .dtype
      is deprecated (numpy 2.5), and .view(dtype) reinterprets BYTES, not values
  L6  Stage C - the canonical-array contract: read-only source, copy-at-boundary, and the
      guard tests that turn the incident into an exception
  L7  why views exist at all: O(1) aliasing vs O(n) copying, timed (machine-dependent;
      read the ratio as an order of magnitude, per 01.4's benchmark honesty)

Data: _data/raw/payments.csv.gz (regenerate: .venv\\Scripts\\python _data\\generate.py).
Mess touched: M7 (comma-formatted amounts on paylane rows - parsed at the boundary),
M2 (amounts are local currency - the incident IS a currency normalization), M13 (partial
payments exist but are irrelevant to memory semantics). Nothing here trains, so the M10
leakage traps are untouched.

Run:  .venv\\Scripts\\python "02-numpy-vectorized-computing/_lab/lab_02.1_ndarray_memory.py"
"""

from __future__ import annotations

import csv
import gzip
import time
import warnings
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "_data" / "raw"
DAY_ROWS = 1_200          # the append-only export's latest daily increment (tail rows)


# ------------------------------------------------------------------ L1: loading

def load_payments() -> tuple[np.ndarray, np.ndarray, int]:
    """Parse payments.csv.gz with the stdlib into two float64 arrays.

    Deliberately no pandas: series 02 is about what an array IS, and building it from
    text makes the interpretation step visible. M7: every paylane row formats amounts
    as '4,554.29', so the commas are stripped at the boundary - the 01.2 contract
    lesson, applied one layer down.
    """
    amounts, fx, commas = [], [], 0
    with gzip.open(RAW / "payments.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row["amount_paid"]
            if "," in raw:
                commas += 1
                raw = raw.replace(",", "")
            amounts.append(float(raw))
            fx.append(float(row["fx_rate_usd"]))
    return np.array(amounts, dtype=np.float64), np.array(fx, dtype=np.float64), commas


# ------------------------------------------------- L2: Stage A - strides by hand

class StridedView:
    """A 2-D window over a flat buffer: nothing but (offset, shape, strides).

    Strides are in ELEMENTS here to keep the arithmetic readable; numpy's are in BYTES
    (multiply by itemsize). Every 'operation' below touches only these three numbers -
    the buffer is never copied, which is the entire point of the exercise.
    """

    def __init__(self, buf: list[float], offset: int,
                 shape: tuple[int, int], strides: tuple[int, int]):
        self.buf, self.offset, self.shape, self.strides = buf, offset, shape, strides

    def __getitem__(self, ij: tuple[int, int]) -> float:
        i, j = ij
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            raise IndexError(f"({i},{j}) outside {self.shape}")
        return self.buf[self.offset + i * self.strides[0] + j * self.strides[1]]

    def transpose(self) -> "StridedView":
        """Swap shape and strides. No element moves."""
        return StridedView(self.buf, self.offset,
                           (self.shape[1], self.shape[0]),
                           (self.strides[1], self.strides[0]))

    def rows(self, a: int, b: int) -> "StridedView":
        """Row-slice = move the offset, shrink the shape. No element moves."""
        return StridedView(self.buf, self.offset + a * self.strides[0],
                           (b - a, self.shape[1]), self.strides)


def strided_parity(amounts: np.ndarray, rows: int = 3, cols: int = 4) -> None:
    flat = amounts[: rows * cols].tolist()             # one shared flat buffer
    mine = StridedView(flat, offset=0, shape=(rows, cols), strides=(cols, 1))
    ref = amounts[: rows * cols].reshape(rows, cols)   # numpy over the same values

    checks = {
        "element (1,2)": (mine[1, 2], float(ref[1, 2])),
        "transpose (2,1)": (mine.transpose()[2, 1], float(ref.T[2, 1])),
        "rows(1,3) at (0,3)": (mine.rows(1, 3)[0, 3], float(ref[1:3][0, 3])),
    }
    for name, (got, want) in checks.items():
        assert got == want, (name, got, want)
        print(f"  {name:<22} hand-strided={got:>10.2f}   numpy={want:>10.2f}   MATCH")
    print(f"  numpy strides for the same {rows}x{cols} float64 block: {ref.strides} bytes"
          f"  (= elements {tuple(s // ref.itemsize for s in ref.strides)} x itemsize "
          f"{ref.itemsize})")
    print(f"  transpose strides: {ref.T.strides} bytes - the SAME two numbers, swapped;")
    print( "  no element moved, which is why .T on a 100-million-row array is free")


# ------------------------------------------------------ L3: the view/copy census

def view_copy_census(amounts: np.ndarray) -> None:
    """Which operations alias the buffer? Settled by shares_memory, not folklore."""
    m = amounts[:200_000].reshape(1_000, 200)
    cases = [
        ("basic slice a[1000:2000]", amounts[1_000:2_000]),
        ("strided slice a[::50]", amounts[::50]),
        ("reshape (on contiguous)", m),
        ("transpose m.T", m.T),
        ("ravel of contiguous m", m.ravel()),
        ("ravel of m.T (non-contig)", m.T.ravel()),
        ("boolean mask a[a > 1000]", amounts[amounts > 1_000]),
        ("fancy index a[[1, 5, 7]]", amounts[[1, 5, 7]]),
        ("explicit a.copy()", amounts.copy()),
    ]
    print(f"  {'operation':<28}{'shares_memory':>14}{'.base set':>11}   verdict")
    for name, out in cases:
        shares = bool(np.shares_memory(amounts, out))
        print(f"  {name:<28}{str(shares):>14}{str(out.base is not None):>11}"
              f"   {'VIEW - writes reach the source' if shares else 'copy - independent'}")
    print("\n  the rule underneath: an operation returns a view exactly when the result")
    print("  is expressible as (new offset, new shape, new strides) over the SAME buffer.")
    print("  Basic slicing and transposes always are; boolean/fancy selection never is;")
    print("  reshape and ravel are views only while the layout allows constant strides.")


# ------------------------------------------------------------- L4: the incident

def incident(amounts: np.ndarray, fx: np.ndarray) -> dict[str, float]:
    """The daily-report refactor: 'day' is a tail slice, believed to be a copy.

    A code review removed a 'redundant' .copy() on the day's increment; the in-place
    USD conversion then wrote through the view into the canonical local-currency array,
    and every consumer downstream of it read mutated data.
    """
    canonical = amounts.copy()                 # the array the whole job shares
    before = float(canonical.sum())

    day = canonical[-DAY_ROWS:]                # believed scratch copy; actually a VIEW
    day /= fx[-DAY_ROWS:]                      # in-place: writes through the buffer
    usd_report_total = float(day.sum())        # the daily report is CORRECT

    after = float(canonical.sum())
    gap = before - after
    print(f"  canonical local-currency total, 07:40 snapshot : {before:>20,.2f}")
    print(f"  daily USD report built from the tail increment  : {usd_report_total:>20,.2f}"
          f"  (the report itself is right)")
    print(f"  canonical local-currency total, 08:10 read      : {after:>20,.2f}")
    print(f"  the two reads of the SAME array differ by       : {gap:>20,.2f}")
    print(f"  np.shares_memory(canonical, day) = "
          f"{bool(np.shares_memory(canonical, day))} <- the smoking gun")
    print(f"  nothing on disk changed; the mutation happened in memory, through a view,")
    print(f"  on {DAY_ROWS:,} of {len(canonical):,} rows - every value still plausible")
    return {"before": before, "after": after, "gap": gap, "usd": usd_report_total}


# ------------------------------------------- L5: 2019->2026 migration semantics

def migration_semantics(amounts: np.ndarray) -> None:
    a = amounts[:4].copy()

    # (1) copy=False no longer means "copy if you must"; it means "never copy".
    try:
        np.array(a, dtype=np.float32, copy=False)
        print("  np.array(copy=False) with a cast: silently copied (1.x behaviour)")
    except ValueError:
        print("  np.array(copy=False) with a cast RAISES in 2.x - 'copy if you must'")
        print("    is now spelled copy=None; the 2019 notes' idiom is a hard error")

    # (2) numpy 2.5: assigning to .dtype is deprecated - it mutates a shared object.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = a.copy()
        b.dtype = np.int64                     # deprecated: reinterprets in place
        note = caught[0].category.__name__ if caught else "no warning"
    print(f"  b.dtype = np.int64 -> {note} (numpy 2.5): mutating a possibly-shared")
    print( "    array's interpretation is unsafe; the supported spelling is a NEW view:")

    # (3) .view(dtype) reinterprets BYTES, not values - useful and dangerous.
    v = a.view(np.int64)
    print(f"    a[:2]              = {a[:2]}  (float64 values)")
    print(f"    a.view(np.int64)[:2] = {v[:2]}  (the SAME bytes read as int64)")
    print(f"    shares_memory={bool(np.shares_memory(a, v))}, itemsize {a.itemsize}->"
          f"{v.itemsize}: reinterpretation, not conversion - .astype() converts")


# ---------------------------------------------- L6: Stage C - the guard contract

def load_canonical() -> tuple[np.ndarray, np.ndarray]:
    """The production loader: canonical arrays are served READ-ONLY.

    Consumers that need to mutate must take working_copy(); consumers that forget get
    a ValueError at the offending line instead of a reconciliation incident two reads
    later. The flag costs nothing: it is bookkeeping, not a copy.
    """
    amounts, fx, _ = load_payments()
    amounts.flags.writeable = False
    fx.flags.writeable = False
    return amounts, fx


def working_copy(arr: np.ndarray) -> np.ndarray:
    """The one sanctioned way to get a mutable array: an explicit copy at the boundary."""
    return arr.copy()


def canonical_contract() -> None:
    canonical, fx = load_canonical()

    print("  the incident's exact code, against the read-only canonical array:")
    day = canonical[-DAY_ROWS:]                # still a view - views inherit the flag
    try:
        day /= fx[-DAY_ROWS:]
        print("    ...mutated silently (guard failed)")
    except ValueError as exc:
        print(f"    ValueError: {exc}")
        print( "    the failure moved from a reconciliation gap at 08:10 to a stack")
        print( "    trace at the offending line - detection distance zero")

    ok = working_copy(canonical[-DAY_ROWS:])
    ok /= fx[-DAY_ROWS:]
    print(f"  the sanctioned path: working_copy() then convert -> USD total "
          f"{ok.sum():,.2f}, canonical untouched")

    # The unit-test-shaped eval for this notebook (guide S2: toolkit series may use
    # assertions as the harness). Each guard is one line a CI job runs on every build.
    tests = {
        "canonical is read-only": lambda: not canonical.flags.writeable,
        "views inherit read-only": lambda: not canonical[10:20].flags.writeable,
        "working_copy is independent":
            lambda: not np.shares_memory(canonical, working_copy(canonical)),
        "working_copy is writable": lambda: working_copy(canonical).flags.writeable,
    }
    for name, fn in tests.items():
        print(f"    [{'PASS' if fn() else 'FAIL'}] {name}")


# ----------------------------------------------------- L7: why views exist at all

def view_vs_copy_timing(amounts: np.ndarray, reps: int = 11) -> None:
    """O(1) bookkeeping vs O(n) allocation. Machine-dependent; the ratio is the point,
    and even the ratio is a rough one - read it as an order of magnitude (01.4)."""
    n = len(amounts)

    def timed(fn) -> float:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return float(np.median(times) * 1e6)

    t_view = timed(lambda: amounts[: n - 1])
    t_copy = timed(lambda: amounts[: n - 1].copy())
    print(f"  slice a[:{n - 1:,}] as a view : {t_view:>9.2f} us   (three integers change)")
    print(f"  the same slice, copied        : {t_copy:>9.2f} us   "
          f"({(n - 1) * amounts.itemsize / 1e6:,.1f} MB moves)")
    print(f"  ratio ~{t_copy / max(t_view, 1e-9):,.0f}x on this run - machine- and "
          f"run-dependent, an order of magnitude at least; the view's cost does not")
    print( "  grow with n, the copy's does. Aliasing is a PERFORMANCE feature with")
    print( "  CORRECTNESS obligations - this notebook is about paying them.")


def main() -> None:
    print("=" * 78)
    print("L1  PAYMENTS -> ARRAYS (stdlib parse; M7 commas handled at the boundary)")
    print("=" * 78)
    amounts, fx, commas = load_payments()
    print(f"  rows={len(amounts):,}  dtype={amounts.dtype}  itemsize={amounts.itemsize}  "
          f"buffer={amounts.nbytes / 1e6:,.1f} MB  comma-formatted amounts parsed: "
          f"{commas:,}")

    print("\n" + "=" * 78)
    print("L2  STAGE A - STRIDES BY HAND, PARITY VS NUMPY")
    print("=" * 78)
    strided_parity(amounts)

    print("\n" + "=" * 78)
    print("L3  THE VIEW/COPY CENSUS")
    print("=" * 78)
    view_copy_census(amounts)

    print("\n" + "=" * 78)
    print("L4  THE INCIDENT - IN-PLACE NORMALIZATION THROUGH A VIEW")
    print("=" * 78)
    incident(amounts, fx)

    print("\n" + "=" * 78)
    print("L5  2019->2026 MIGRATION SEMANTICS")
    print("=" * 78)
    migration_semantics(amounts)

    print("\n" + "=" * 78)
    print("L6  STAGE C - THE CANONICAL-ARRAY CONTRACT")
    print("=" * 78)
    canonical_contract()

    print("\n" + "=" * 78)
    print("L7  WHY VIEWS EXIST - O(1) VS O(n)")
    print("=" * 78)
    view_vs_copy_timing(amounts)


if __name__ == "__main__":
    main()
