"""Lab for notebook 02.2 "Numerical Dtypes & Promotion under NEP 50".

Reproduces every captured number and listing in 02.2:

  L1  loading the INR invoice lane into arrays (stdlib parse; M6/M7 at the boundary)
  L2  the dtype bestiary priced in PayFlow terms - ranges and sizes that matter for money
  L3  the incident - the paise total under an int32 accumulator (the numpy-1.x-on-Windows
      behaviour, reproduced explicitly) vs the int64 default the 2.x port brought, and the
      np.long "fix" that reintroduces the wrap on Windows
  L4  NEP 50 promotion - the 2.x rules captured by execution, next to the 1.x value-based
      results those expressions USED to give (documented, labelled, not executed)
  L5  float32 money - naive accumulation error vs numpy's pairwise sum vs float64 truth
  L6  datetime64 - invoice dates as integers with units; NaT; terms arithmetic
  L7  StringDType - the status column under np.strings (M6 casing), vs <U and object
  L8  Stage C - the dtype contract at the ingestion boundary: ranges, headroom, money
      policy; the incident's column fails it loudly

Data: _data/raw/invoices.csv.gz (regenerate: .venv\\Scripts\\python _data\\generate.py).
Mess touched: M7 (comma amounts, parsed at the boundary), M6 (status casing - the
StringDType demo), M2 (kept per-currency: every sum here is INR-only, so no unit mixing).
M3's glitch week sits inside the INR lane and is immaterial to these totals. Nothing
trains, so M10 is untouched.

Run:  .venv\\Scripts\\python "02-numpy-vectorized-computing/_lab/lab_02.2_dtypes_nep50.py"
"""

from __future__ import annotations

import csv
import gzip
import warnings
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "_data" / "raw"


# ------------------------------------------------------------------ L1: loading

def load_inr_invoices() -> dict[str, np.ndarray]:
    """The INR lane of the invoice export, parsed with the stdlib.

    Amounts arrive as text (M7 commas on legacy rows), status arrives in mixed casing
    (M6) - both handled or preserved deliberately: commas are a boundary fix, casing is
    L7's subject. Everything returned is per-currency, so every sum below is honest INR.
    """
    amount, status, issue, due = [], [], [], []
    with gzip.open(RAW / "invoices.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["currency"] != "INR":
                continue
            amount.append(float(row["amount"].replace(",", "")))
            status.append(row["status"])
            issue.append(row["issue_date"])
            due.append(row["due_date"])
    return {
        "amount_inr": np.array(amount, dtype=np.float64),
        "status": np.array(status),                       # dtype chosen by numpy: L7
        "issue_date": np.array(issue, dtype="datetime64[D]"),
        "due_date": np.array(due, dtype="datetime64[D]"),
    }


# ---------------------------------------------------- L2: the bestiary, priced

def dtype_bestiary() -> None:
    """Ranges and sizes, stated in PayFlow money terms rather than abstractly."""
    print(f"  {'dtype':<10}{'bytes':>6}{'max value':>26}   holds, in paise")
    for dt in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dt)
        rupees = info.max / 100
        print(f"  {np.dtype(dt).name:<10}{np.dtype(dt).itemsize:>6}{info.max:>26,}"
              f"   INR {rupees:,.0f}")
    for dt in (np.float16, np.float32, np.float64):
        info = np.finfo(dt)
        if 1e8 > float(info.max):
            note = f"cannot represent 1e8 at all (max ~{float(info.max):.3g})"
        else:
            # spacing at 1e8: the gap between adjacent representable values there
            note = f"gap at 1e8: {float(np.spacing(np.array(1e8, dtype=dt))):g}"
        print(f"  {np.dtype(dt).name:<10}{np.dtype(dt).itemsize:>6}"
              f"{'~' + f'{float(info.max):.3g}':>26}   {note}")
    print("\n  int32's ceiling is INR 21.5 million in paise - ONE large enterprise invoice")
    print("  cleared with a year of billing behind it. float32 near 1e8 cannot even")
    print("  represent every paisa: adjacent values are 8 apart. Money wants int64 minor")
    print("  units or float64 - everything narrower is a wrap or a rounding waiting to happen.")


# ------------------------------------------------------------- L3: the incident

def incident(amount_inr: np.ndarray) -> dict[str, int]:
    """The billings total across the 1.x -> 2.x port, on Windows.

    numpy 1.x on Windows: the default integer (and integer sum accumulator) was the
    C long - 32-bit on Windows. numpy 2.x: the default integer is int64 everywhere.
    We are pinned on 2.x, so the OLD behaviour is reproduced explicitly with
    dtype=np.int32 - labelled as such, not pretended.
    """
    paise = np.round(amount_inr * 100).astype(np.int64)     # exact minor units
    true_total = int(paise.sum())                           # 2.x default accumulator
    print(f"  rows in the INR lane: {len(paise):,}")
    print(f"  paise total, numpy 2.x default (int64 accumulator): {true_total:>22,}")

    paise32 = paise.astype(np.int32, casting="unsafe", copy=True)  # the STORED column
    default32 = paise32.sum()
    print(f"  the stored column is int32 (every element fits); numpy 2.x's DEFAULT sum")
    print(f"  of it accumulates in {default32.dtype} and agrees:      {int(default32):>22,}")
    with np.errstate(over="ignore"):
        wrapped = int(paise32.sum(dtype=np.int32))
    print(f"  the same sum under an int32 accumulator (1.x-on-Windows behaviour,")
    print(f"  reproduced explicitly here):                        {wrapped:>22,}")
    print(f"  the 'revenue jump' the upgrade produced:            "
          f"{true_total - wrapped:>22,}")
    n_wraps = (true_total - wrapped) // (2 ** 32)
    print(f"  = the accumulator lapped 2^32 {n_wraps:,} times; the OLD number was the")
    print(f"    broken one, and it had been broken silently for years")

    # The hotfix that reintroduces the bug on Windows: np.long is the C long.
    print(f"\n  the attempted hotfix: 'cast it to np.long, that is 64-bit'")
    print(f"  np.long on this Windows box is {np.dtype(np.long).name} "
          f"({np.dtype(np.long).itemsize * 8}-bit) - the C long, NOT a guaranteed 64 bits")
    with np.errstate(over="ignore"):
        refix = int(paise32.sum(dtype=np.long))
    print(f"  sum(dtype=np.long) on the int32 column:             {refix:>22,}")
    print(f"  identical wrap. The unambiguous spelling is np.int64; np.long is a")
    print(f"  platform question wearing a dtype's name.")
    return {"true": true_total, "wrapped": wrapped}


# ----------------------------------------------- L4: NEP 50 promotion, captured

def promotion_table() -> None:
    """2.x results by execution; the 1.x value-based results those expressions used to
    give are DOCUMENTED (NEP 50 migration guide), labelled, and not pretended captured."""
    a_u8 = np.array([200], dtype=np.uint8)
    a_f32 = np.array([1.5], dtype=np.float32)

    rows: list[tuple[str, str, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # scalar overflow warns; L4 text
        r = np.uint8(200) + 200
        rows.append(("np.uint8(200) + 200", f"{np.dtype(type(r)).name} {r}",
                     "uint16 400"))
        r = a_u8 + 200
        rows.append(("array([200], uint8) + 200", f"{r.dtype.name} {r[0]} (SILENT)",
                     "uint16 400"))
        r = a_f32 + 3.0
        rows.append(("array float32 + 3.0 (py float)", f"{r.dtype.name} {r[0]}",
                     "float32 4.5 (same)"))
        r = a_f32 + np.float64(3.0)
        rows.append(("array float32 + np.float64(3)", f"{r.dtype.name} {r[0]}",
                     "float32 4.5"))
    try:
        np.array([100], dtype=np.int8) + 300
        rows.append(("array([100], int8) + 300", "no error", "int16 400"))
    except OverflowError:
        rows.append(("array([100], int8) + 300", "OverflowError - refuses",
                     "int16 400"))

    print(f"  {'expression':<32}{'numpy 2.x (captured)':<34}1.x value-based (documented)")
    for expr, now, then in rows:
        print(f"  {expr:<32}{now:<34}{then}")
    print("\n  the rule that replaced value-based casting: python scalars are WEAK (they")
    print("  adopt the array's dtype); numpy scalars are STRONG (they promote like any")
    print("  array). Results no longer depend on the VALUE of a scalar - only on types.")
    print("  The three behaviour changes above are exactly the port's audit surface:")
    print("  in-range uint arithmetic now wraps silently where 1.x widened; out-of-range")
    print("  python ints now raise; and a float64 numpy scalar now wins, ADDING precision.")


# --------------------------------------------------------- L5: float32 money

def float32_money(amount_inr: np.ndarray) -> None:
    truth = float(amount_inr.sum())                       # float64 pairwise
    as32 = amount_inr.astype(np.float32)
    pairwise32 = float(as32.sum(dtype=np.float32))        # numpy's pairwise, in 32-bit

    running = np.float32(0.0)
    for v in as32[:50_000]:                               # true naive loop, first 50k
        running = np.float32(running + v)
    truth_50k = float(amount_inr[:50_000].sum())

    print(f"  float64 truth, full INR lane      : {truth:>20,.2f}")
    print(f"  float32, numpy pairwise reduction : {pairwise32:>20,.2f}   "
          f"off by {truth - pairwise32:+,.2f}")
    print(f"  float32, naive running loop (50k) : {running:>20,.2f}   "
          f"off by {truth_50k - running:+,.2f} on the same 50k rows")
    print(f"  (float64 truth on those 50k rows  : {truth_50k:>20,.2f})")
    print("\n  numpy's pairwise summation hides most of float32's sins - until someone")
    print("  'optimizes' into a running accumulator, a streaming job, or a GPU kernel")
    print("  that sums naively. The dtype was never safe; the algorithm was covering.")


# ------------------------------------------------------------ L6: datetime64

def datetime_lane(issue: np.ndarray, due: np.ndarray) -> None:
    terms = (due - issue).astype("timedelta64[D]")
    print(f"  issue_date dtype {issue.dtype}: an int64 count of DAYS since the epoch,")
    print(f"    wearing calendar clothes - arithmetic is integer arithmetic, no float error")
    vals, counts = np.unique(terms, return_counts=True)
    top = ", ".join(f"{int(v / np.timedelta64(1, 'D'))}d x {c:,}"
                    for v, c in sorted(zip(vals, counts), key=lambda t: -t[1])[:3])
    print(f"  payment terms = due - issue: {top}")
    nat = np.datetime64("NaT", "D")
    print(f"  NaT is datetime64's NaN: NaT == NaT -> {bool(nat == nat)}, "
          f"comparisons all False; use np.isnat() - same discipline as np.nan (02.5)")
    print(f"  unit matters: datetime64[D] cannot hold 09:40; [ns] spans only 1678-2262.")
    print(f"  choose the coarsest unit the domain needs - invoices are day-grained.")


# ----------------------------------------------------------- L7: StringDType

def string_lane(status: np.ndarray) -> None:
    print(f"  numpy's inferred dtype for the status column: {status.dtype}")
    fixed = status.astype("U11")
    var = status.astype(np.dtypes.StringDType())
    print(f"  fixed-width <U11: {fixed.nbytes / 1e6:5.1f} MB  "
          f"(11 UCS-4 slots x {len(fixed):,} rows, padding included)")
    print(f"  StringDType    : array header {var.nbytes / 1e6:5.1f} MB + heap strings "
          f"(variable, no padding)")
    lower = np.strings.lower(var)
    raw_paid = int((var == "paid").sum())
    all_paid = int((lower == "paid").sum())
    print(f"  M6 in one line: status == 'paid' matches {raw_paid:,} rows raw,")
    print(f"  {all_paid:,} after np.strings.lower - "
          f"{all_paid - raw_paid:,} paid invoices hidden by casing")
    print("  np.strings is the 2.x home for vectorized text ops (np.char and chararray")
    print("  are the deprecated 2019-era spellings); heavy text work belongs to pandas (03).")


# ------------------------------------------------- L8: Stage C - dtype contract

MONEY_POLICY = "money is int64 minor units or float64 - never narrower, never unsigned"

DTYPE_CONTRACT: dict[str, dict] = {
    # column        expected kind/width         headroom rule
    "amount_paise": {"dtype": np.int64,
                     "headroom": "column TOTAL must fit with 1000x margin"},
    "amount_inr":   {"dtype": np.float64, "headroom": "spacing at max must be < 0.01"},
    "issue_date":   {"dtype": np.dtype("datetime64[D]"), "headroom": None},
}


def check_dtype_contract(name: str, arr: np.ndarray) -> list[str]:
    """The ingestion-boundary guard: interpretation AND arithmetic safety, per column."""
    spec = DTYPE_CONTRACT[name]
    problems: list[str] = []
    if arr.dtype != np.dtype(spec["dtype"]):
        problems.append(f"dtype {arr.dtype} != contracted {np.dtype(spec['dtype']).name}")
    if spec["headroom"] and np.issubdtype(arr.dtype, np.integer):
        total_estimate = int(abs(arr).sum())              # int64-safe by contract
        if total_estimate > np.iinfo(arr.dtype).max // 1_000:
            problems.append("TOTAL exceeds 0.1% of dtype range - accumulator wrap risk")
    if spec["headroom"] and np.issubdtype(arr.dtype, np.floating):
        gap = float(np.spacing(arr.max()))
        if gap >= 0.01:
            problems.append(f"representable gap {gap:g} at max - cannot hold minor units")
    return problems


def contract_demo(amount_inr: np.ndarray, issue: np.ndarray) -> None:
    paise64 = np.round(amount_inr * 100).astype(np.int64)
    paise32 = paise64.astype(np.int32, casting="unsafe")
    cases = [
        ("amount_paise as int64", "amount_paise", paise64),
        ("amount_paise as int32 (the incident's column)", "amount_paise", paise32),
        ("amount_inr as float64", "amount_inr", amount_inr),
        ("amount_inr as float32", "amount_inr", amount_inr.astype(np.float32)),
        ("issue_date as datetime64[D]", "issue_date", issue),
    ]
    print(f"  policy: {MONEY_POLICY}\n")
    for label, key, arr in cases:
        problems = check_dtype_contract(key, arr)
        verdict = "PASS" if not problems else "FAIL: " + "; ".join(problems)
        print(f"  [{'PASS' if not problems else 'FAIL'}] {label:<46} "
              f"{verdict if problems else ''}".rstrip())
    print("\n  the contract runs where 01.2's grain and unit checks run - at ingestion,")
    print("  before any consumer can build a habit on a column that cannot hold its future")


def main() -> None:
    print("=" * 78)
    print("L1  THE INR INVOICE LANE (stdlib parse; M7 handled, M6 preserved for L7)")
    print("=" * 78)
    d = load_inr_invoices()
    print(f"  rows={len(d['amount_inr']):,}  amount dtype={d['amount_inr'].dtype}  "
          f"status dtype={d['status'].dtype}  dates dtype={d['issue_date'].dtype}")

    print("\n" + "=" * 78)
    print("L2  THE BESTIARY, PRICED IN PAISE")
    print("=" * 78)
    dtype_bestiary()

    print("\n" + "=" * 78)
    print("L3  THE INCIDENT - ONE SUM, TWO ACCUMULATORS, ONE PLATFORM")
    print("=" * 78)
    incident(d["amount_inr"])

    print("\n" + "=" * 78)
    print("L4  NEP 50 - PROMOTION BY TYPE, NOT BY VALUE")
    print("=" * 78)
    promotion_table()

    print("\n" + "=" * 78)
    print("L5  FLOAT32 MONEY - THE ALGORITHM WAS COVERING FOR THE DTYPE")
    print("=" * 78)
    float32_money(d["amount_inr"])

    print("\n" + "=" * 78)
    print("L6  DATETIME64 - DATES AS INTEGERS WITH UNITS")
    print("=" * 78)
    datetime_lane(d["issue_date"], d["due_date"])

    print("\n" + "=" * 78)
    print("L7  STRINGDTYPE - TEXT AS A FIRST-CLASS DTYPE (M6)")
    print("=" * 78)
    string_lane(d["status"])

    print("\n" + "=" * 78)
    print("L8  STAGE C - THE DTYPE CONTRACT AT THE BOUNDARY")
    print("=" * 78)
    contract_demo(d["amount_inr"], d["issue_date"])


if __name__ == "__main__":
    main()
