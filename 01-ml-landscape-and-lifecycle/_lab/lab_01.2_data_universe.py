"""Lab for notebook 01.2 "First Contact with the PayFlow Data Universe".

Reproduces every captured number and listing in 01.2:

  L1  what is literally in the files - stdlib csv, before pandas imposes any interpretation
  L2  grain - the candidate key of each table and exactly where it is not unique
  L3  joins - fan-out on one side, silent row loss on the other
  L4  the incident - aggregating `amount` across currencies (SPEC M2), and the string-sum
      trap that pandas 3 sets up on the same column (SPEC M7)
  L5  the data contract - eight checks run over the raw exports, with counts
  L6  reconciliation - billed vs collected, in USD, and what the gap is made of

Data: _data/raw/*.csv[.gz] (regenerate: .venv\\Scripts\\python _data\\generate.py).
This lab reads the RAW exports on purpose: 01.2 is about seeing the universe as it arrives,
not about cleaning it. Systematic cleaning is series 09.

Run:  .venv\\Scripts\\python "01-ml-landscape-and-lifecycle/_lab/lab_01.2_data_universe.py"
"""

from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "_data" / "raw"

TABLES = {                      # file -> (declared grain, candidate key)
    "customers.csv":            ("one row per customer", "customer_id"),
    "subscriptions.csv":        ("one row per subscription period", "sub_id"),
    "invoices.csv.gz":          ("one row per invoice", "invoice_id"),
    "payments.csv.gz":          ("one row per payment", "payment_id"),
    "support_tickets.csv.gz":   ("one row per ticket", "ticket_id"),
    "addon_purchases.csv.gz":   ("one row per add-on order", "order_id"),
}


def open_text(path: Path) -> io.TextIOBase:
    """Open a plain or gzipped CSV as text - the caller should not care which."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


# ------------------------------------------------------- L1: before interpretation

def peek_raw(name: str, n_rows: int = 3) -> None:
    with open_text(RAW / name) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [next(reader) for _ in range(n_rows)]
    print(f"  {name}")
    print(f"    columns ({len(header)}): {', '.join(header)}")
    for row in rows:
        print(f"    {row[:4]} ...")
    print(f"    every value above is a str - dtype is an interpretation pandas applies,")
    print(f"    not a property the file has\n")


# ------------------------------------------------------------------ L2: grain

def grain_report(name: str, key: str, declared: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name)
    n, uniq = len(df), df[key].nunique()
    dup_rows = int(df.duplicated().sum())
    print(f"  {name:<26} n={n:>8,}  unique {key}={uniq:>8,}  "
          f"exact dup rows={dup_rows:>5,}  {'OK' if n == uniq else 'GRAIN VIOLATED'}"
          f"   ({declared})")
    return df


# ------------------------------------------------------------------ L3: joins

def join_experiments(inv: pd.DataFrame, pay: pd.DataFrame, tck: pd.DataFrame,
                     cus: pd.DataFrame) -> None:
    naive = inv.merge(pay, on="invoice_id", how="inner")
    represented = naive["invoice_id"].nunique()
    lost = inv["invoice_id"].nunique() - represented
    print(f"  invoices({len(inv):,}) x payments({len(pay):,}) inner join -> "
          f"{len(naive):,} rows   (net change {len(naive) - len(inv):+,}, "
          f"{(len(naive) - len(inv)) / len(inv):+.2%})")
    print(f"    but that small net hides two large opposite effects:")
    print(f"      - {lost:,} invoices DROPPED (never paid: overdue/disputed/written off)")
    print(f"      - {len(naive) - represented:,} extra rows ADDED by fan-out "
          f"({int((pay.groupby('invoice_id').size() > 1).sum()):,} invoices have >1 "
          f"payment, SPEC M13)")
    print(f"    fan-out factor {len(naive) / max(represented, 1):.3f} -> any per-invoice "
          f"average computed on this frame double-counts the split-paid invoices")

    agg = pay.groupby("invoice_id", as_index=False).agg(n_payments=("payment_id", "size"))
    safe = inv.merge(agg, on="invoice_id", how="left", validate="m:1")
    print(f"    aggregate-then-join -> {len(safe):,} rows, grain preserved "
          f"({len(safe) == len(inv)})")

    joined = tck.merge(cus[["customer_id", "segment"]], on="customer_id", how="inner")
    lost = len(tck) - len(joined)
    spam_lost = int(tck.loc[tck["customer_id"].isna(), "category"].eq("spam").sum())
    print(f"  tickets({len(tck):,}) x customers inner join -> {len(joined):,} rows, "
          f"{lost:,} silently dropped")
    print(f"    of the dropped, {spam_lost:,} are spam tickets with no customer_id "
          f"(SPEC M14) -> a spam classifier trained on the join never sees its own "
          f"positive class")


# --------------------------------------------------------------- L4: the incident

def currency_incident(inv: pd.DataFrame, pay: pd.DataFrame) -> None:
    raw_str = pd.read_csv(RAW / "invoices.csv.gz", nrows=200_000)["amount"]
    print(f"  invoices.amount dtype as read: {raw_str.dtype!r}  (SPEC M7: legacy exports "
          f"carry '4,554.29')")
    try:
        total = raw_str.sum()
        kind = type(total).__name__
        shown = total if not isinstance(total, str) else f"{total[:40]}... (len {len(total):,})"
        print(f"    .sum() returned {kind}: {shown}")
        if isinstance(total, str):
            print("    -> the column concatenated. No exception, no warning, no number.")
    except Exception as exc:                                        # noqa: BLE001
        print(f"    .sum() raised {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")

    amt = pd.to_numeric(inv["amount"].astype(str).str.replace(",", ""), errors="raise")
    inv = inv.assign(amount_num=amt)
    fx = (pay.merge(inv[["invoice_id", "currency"]], on="invoice_id")
             .groupby("currency")["fx_rate_usd"].median())
    inv = inv.assign(amount_usd=inv["amount_num"] / inv["currency"].map(fx))

    naive_total = inv["amount_num"].sum()
    true_total = inv["amount_usd"].sum()
    print(f"\n  billings, summed the way the 2019 report does it: {naive_total:>16,.0f}")
    print(f"  billings, converted to USD first:                  {true_total:>16,.0f}")
    print(f"  overstatement factor: {naive_total / true_total:.2f}x "
          f"(+${naive_total - true_total:,.0f} of currency that does not exist)")

    by_ccy = inv.groupby("currency").agg(rows=("amount_num", "size"),
                                         raw=("amount_num", "sum"),
                                         usd=("amount_usd", "sum"))
    by_ccy["share_of_raw"] = by_ccy["raw"] / by_ccy["raw"].sum()
    by_ccy["share_of_usd"] = by_ccy["usd"] / by_ccy["usd"].sum()
    print("\n  where the phantom revenue comes from:")
    print(by_ccy.sort_values("raw", ascending=False)
          .to_string(formatters={"rows": "{:,.0f}".format, "raw": "{:,.0f}".format,
                                 "usd": "{:,.0f}".format,
                                 "share_of_raw": "{:.1%}".format,
                                 "share_of_usd": "{:.1%}".format}))


# --------------------------------------------------------------- L5: the contract

def check_contract(inv: pd.DataFrame, pay: pd.DataFrame, cus: pd.DataFrame) -> None:
    amt_numeric = pd.to_numeric(inv["amount"].astype(str).str.replace(",", ""),
                                errors="coerce")
    checks = {
        "invoice_id is unique": int(len(inv) - inv["invoice_id"].nunique()),
        "no exact duplicate rows": int(inv.duplicated().sum()),
        "amount parses as a number": int(amt_numeric.isna().sum()),
        "amount is already numeric dtype": 0 if inv["amount"].dtype.kind in "fi" else len(inv),
        "currency present on every row": int(inv["currency"].isna().sum()),
        "status uses one casing": int((inv["status"] != inv["status"].str.lower()).sum()),
        "due_date >= issue_date": int((pd.to_datetime(inv["due_date"])
                                       < pd.to_datetime(inv["issue_date"])).sum()),
        "payment.invoice_id exists in invoices":
            int((~pay["invoice_id"].isin(set(inv["invoice_id"]))).sum()),
        "customer_id exists in customers":
            int((~inv["customer_id"].isin(set(cus["customer_id"]))).sum()),
    }
    width = max(len(k) for k in checks)
    failed = 0
    for name, bad in checks.items():
        ok = bad == 0
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  offending rows: {bad:>8,}")
    print(f"\n  contract score: {len(checks) - failed}/{len(checks)} checks pass on the "
          f"raw export")


# ----------------------------------------------------------- L6: reconciliation

def reconcile(inv: pd.DataFrame, pay: pd.DataFrame) -> None:
    amt = pd.to_numeric(inv["amount"].astype(str).str.replace(",", ""), errors="raise")
    inv = inv.assign(amount_num=amt)
    joined = pay.merge(inv[["invoice_id", "currency", "amount_num", "status"]],
                       on="invoice_id", how="inner")
    paid_amt = pd.to_numeric(joined["amount_paid"].astype(str).str.replace(",", ""),
                             errors="raise")
    collected_usd = (paid_amt / joined["fx_rate_usd"]).sum()

    fx = joined.groupby("currency")["fx_rate_usd"].median()
    billed_paid_usd = (joined.drop_duplicates("invoice_id")
                       .pipe(lambda d: d["amount_num"] / d["currency"].map(fx)).sum())
    print(f"  collected (from payments, each at its own settlement rate): "
          f"${collected_usd:>14,.0f}")
    print(f"  billed on those same invoices (median-rate table):          "
          f"${billed_paid_usd:>14,.0f}")
    gap = collected_usd - billed_paid_usd
    print(f"  gap: ${gap:,.0f} ({gap / billed_paid_usd:+.2%}) - FX moved between issue and "
          f"settlement; a reconciliation this tight is the signal the conversion is right")


def main() -> None:
    pd.set_option("display.width", 120)
    print("=" * 78)
    print("L1  WHAT IS LITERALLY IN THE FILES (stdlib csv, no pandas)")
    print("=" * 78)
    for name in ("invoices.csv.gz", "payments.csv.gz"):
        peek_raw(name)

    print("=" * 78)
    print("L2  GRAIN - what does one row mean, and is the key actually unique?")
    print("=" * 78)
    frames = {name: grain_report(name, key, declared)
              for name, (declared, key) in TABLES.items()}

    inv = frames["invoices.csv.gz"]
    pay = frames["payments.csv.gz"]
    tck = frames["support_tickets.csv.gz"]
    cus = frames["customers.csv"]

    print("\n" + "=" * 78)
    print("L3  JOINS - fan-out on one side, silent loss on the other")
    print("=" * 78)
    join_experiments(inv, pay, tck, cus)

    print("\n" + "=" * 78)
    print("L4  THE INCIDENT - one column, two ways to get a wrong total")
    print("=" * 78)
    currency_incident(inv, pay)

    print("\n" + "=" * 78)
    print("L5  THE DATA CONTRACT - what the ingestion job should refuse")
    print("=" * 78)
    check_contract(inv, pay, cus)

    print("\n" + "=" * 78)
    print("L6  RECONCILIATION - does the converted number survive contact with cash?")
    print("=" * 78)
    reconcile(inv, pay)


if __name__ == "__main__":
    main()
