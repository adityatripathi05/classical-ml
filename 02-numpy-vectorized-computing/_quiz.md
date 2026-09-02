# Series 02 — Retrieval Quiz

## How to use

1. Before reading a notebook's Mental Model, write your own in one sentence. Compare after.
2. Answer every question **aloud, from memory**, before scrolling to `## Answers`.
   Recognizing a correct answer is not the same skill as producing one.
3. At least once per series, re-implement one Stage A build from memory on a blank file —
   here, the `StridedView` class plus its parity check against numpy — and only then diff
   it against the notebook.
4. Re-attempt this quiz a week later. Spaced, interleaved retrieval produces retention;
   re-reading produces the feeling of retention.
5. Questions are interleaved across notebooks on purpose and grow as the series does.

---

## Questions

**Q1.** Two jobs share one in-memory payments array. Between a 07:40 read and an 08:10 read
the array's total drops by 41 million; the files on disk hash identical and every log is
clean. Name the mechanism, the one-line probe that confirms it, and the free change that
turns this incident class into a stack trace. *(02.1 — debug this)*

**Q2.** For a C-ordered float64 array of shape `(3, 4)`, derive the strides in bytes, the
address of element `(i, j)`, and the strides after `.T` — then state why the transpose costs
the same on twelve elements as on a hundred million. *(02.1 — derive this)*

**Q3.** State the single rule that decides whether a numpy operation returns a view or a
copy, and apply it to: a basic slice, `m.T`, `ravel` of a contiguous array, `ravel` of that
array's transpose, a boolean mask, and fancy indexing. *(02.1)*

**Q4.** A tidy-up PR replaces `day = amounts[-1200:].copy()` with `day = amounts[-1200:]`
and every test passes. What changed, why did no test notice, and what one-line assertion
would have failed the PR? *(02.1)*

**Q5.** You are porting 2019-era code containing `np.array(x, copy=False)` and
`arr.dtype = np.int64` to the pinned numpy 2.5. State what each did then, what each does
now, and the modern spelling of each intent. *(02.1)*

**Q6.** What is the difference between `a.view(np.int64)` and `a.astype(np.int64)` on a
float64 array — in memory behaviour and in the values you get back? Give one legitimate use
for each. *(02.1)*

**Q7.** Design the ownership contract for a large reference array shared by several jobs in
one process, where some jobs need mutable scratch derived from it. Name the default, the
boundary, and the tests. *(02.1 — design this)*

**Q8.** After a numpy 1.x→2.x upgrade, a Windows dashboard's INR total jumps roughly
700-fold; data hashes match, logs are clean in both eras, and the Linux staging box has
always shown the big number. Name the mechanism, say which number is right, and explain why
the bug never reproduced off the production host. *(02.2 — debug this)*

**Q9.** An int32 accumulator sums a paise column whose true total is 1,552,569,768,898.
Derive what it displays and why no error is raised at any step. *(02.2 — derive this)*

**Q10.** State NEP 50's promotion rule in one sentence, then give the three behaviour
changes a 1.x→2.x port must audit for — one silent wrap, one new refusal, one silent
improvement. *(02.2)*

**Q11.** A colleague hotfixes the overflow with `paise.sum(dtype=np.long)` and it wraps
identically. Why, and what is the correct spelling? *(02.2)*

**Q12.** Two float32 implementations of the same revenue sum — numpy's `.sum()` and a
streaming running total — agree on a small fixture and diverge by tens of thousands of
rupees at scale. Explain both facts and name the fix. *(02.2)*

**Q13.** Why can a float32 column be simultaneously nowhere near overflow and unable to
hold paise at billing scale? Name the concept and the one-line probe that prices it.
*(02.2)*

**Q14.** Design the dtype contract for a money column at ingestion: the three checks, and
why "does the dtype hold its future?" differs from "does it hold its elements?"
*(02.2 — design this)*

**Q15.** State the broadcasting rule in one sentence, then derive the result shapes of
`(n,) − (n, 1)` and `(2, 3) + (2,)`. *(02.3 — derive this)*

**Q16.** A month-end deviation metric jumps 42-fold; the job is green, inputs pass their
contracts, and the host logged an unexplained 70 MB allocation spike during a run whose
data is a fraction of a megabyte. Walk the diagnosis and name the mechanism. *(02.3 —
debug this)*

**Q17.** Why is the diagonal of the accidental `(n, n)` difference exactly the intended
elementwise answer — and why does that fact make the bug harder to catch, not easier?
*(02.3)*

**Q18.** Broadcasting's stretched operands cost no memory, yet the incident allocated
69 MB. Reconcile the two facts, and name the 02.1 mechanism underneath the stretch.
*(02.3)*

**Q19.** The same `(n,) − (n, 1)` bug produced a wrong statistic on a 2,945-row batch and
a MemoryError on the 116,888-row lane. What decided which symptom appeared, and which
outcome is worse? *(02.3)*

**Q20.** Design the shape-safety policy for a metrics library that consumes arrays it did
not construct: the default inside functions, the rule at the seams, how deliberate outer
products are written, and what gets priced in review. *(02.3 — design this)*

**Q21.** Mini coding challenge. From memory, write `my_broadcast_shape(s1, s2)` — the full
rule in ~12 lines — then use it to explain why helpers should return `(n,)` rather than
`(n, 1)`. *(02.3)*

**Q22.** A daily job logs "escalated 200 invoices" every run, the escalation dashboard
shows zero for a month, and nothing raises. Desugar the offending statement, name the
mechanism, and give the two-part fix. *(02.4 — debug this)*

**Q23.** Derive why `a[5:200][:50] = True` sets 50 flags while `a[over][:50] = True` sets
none, though both are chained assignments. *(02.4 — derive this)*

**Q24.** `counts[inverse] += 1` reports every customer holding exactly one invoice. State
the gather-add-scatter rule that produces this, and the two correct spellings. *(02.4)*

**Q25.** Why must a mutation job's success metric come from the target array rather than
from the index list, and what does the one-gather post-condition look like? *(02.4)*

**Q26.** The status column says 2,676 invoices are overdue; the calendar agrees on 542.
What does that tell you about the column, and what discipline follows for time-based
logic? *(02.4)*

**Q27.** Design the flag-the-top-k job: predicate, ranking, write, and proof — naming
which tool carries each step and the tests that pin it. *(02.4 — design this)*

**Q28.** When do you reach for `np.argpartition` over `np.argsort`, and what does numpy
2.5's `descending=True` replace? *(02.4)*

**Q29.** A data-honesty refactor keeps unresolved invoices as NaN, and the next morning
all 92 monthly panels of a mean-days-late dashboard are blank; a one-line `nanmean`
hotfix restores them and the trend immediately shows collections accelerating. Explain
both phases and name what neither version of the rollup ever carried. *(02.5 — debug
this)*

**Q30.** A month has n invoices, r resolved, and unresolved means not-yet-paid. Derive
the direction of `nanmean`'s bias for the month's true mean days-late, and how it moves
as r/n falls. *(02.5 — derive this)*

**Q31.** How would you *measure* (not argue) the censoring bias of `nanmean` on recent
months, given no ground truth exists for them yet? *(02.5)*

**Q32.** `np.nansum` of an all-NaN slice returns 0.0; `np.nanmean` returns NaN with a
warning. Why do siblings disagree, and what must production code do about the empty
case? *(02.5)*

**Q33.** On a `(months, statuses)` pivot built with `np.add.at`, state what `sum(axis=0)`,
`sum(axis=1)` and `sum(axis=1, keepdims=True)` each return and the business question each
answers — plus the mnemonic. *(02.5)*

**Q34.** Design the rollup contract for a metric whose inputs resolve over weeks:
the return shape, the floor, the dashboard state, and the two tests that pin it.
*(02.5 — design this)*

**Q35.** What do `where=` and `initial=` on a reduction buy over the `nan*` family, and
which keyword from 02.2 completes the reduction-safety kit? *(02.5)*

---

## Answers

**A1.** A view plus an in-place write: some consumer holds a header over the same buffer
(here a tail slice) and an augmented assignment (`/=`) wrote through it, transforming 1,200
shared elements while producing a perfectly correct report of its own. The probe is
`np.shares_memory(canonical, suspect)` — `True` is the smoking gun. The free change is
serving the canonical array read-only (`arr.flags.writeable = False`): views inherit the
flag, so the incident's exact code dies with `ValueError: output array is read-only` at the
offending line — detection distance zero. *(02.1)*

**A2.** Row stride `4 × 8 = 32` bytes, column stride `8` bytes; element `(i, j)` lives at
`offset + 32·i + 8·j`. The transpose carries `(8, 32)` — the same two integers swapped —
over the same buffer. Only the header changes, no element moves, so the cost is independent
of the array's size. *(02.1)*

**A3.** View exactly when the result is expressible as `(offset, shape, strides)` over the
*same* buffer — i.e. the selection has constant strides. Basic slice: view. `m.T`: view
(stride swap). `ravel` of contiguous: view. `ravel` of the transpose: copy — the element
order it must produce has no constant stride over that layout. Boolean mask and fancy
indexing: always copies, since an irregular subset has no constant stride. When in doubt,
`np.shares_memory` settles it. *(02.1)*

**A4.** Aliasing changed, values did not: `day` went from an independent buffer to a view
of the canonical array, so every output the tests inspect is identical — until something
writes through `day`. Tests compare values, and no value differed at test time. The
assertion is `assert not np.shares_memory(amounts, day)` at the ownership boundary (or the
read-only canonical flag, which fails louder and everywhere). *(02.1)*

**A5.** `copy=False` in 1.x meant "avoid a copy if you can" and could silently copy anyway;
in 2.x it is a hard promise and *raises* when a cast forces a copy — the old meaning is now
`copy=None`. Assigning to `.dtype` reinterpreted the buffer in place; numpy 2.5 deprecates
it because mutating a possibly-shared array's interpretation is unsafe. Modern spellings:
`copy=None` (or an explicit `.copy()`), and `.view(new_dtype)` for reinterpretation or
`.astype(new_dtype)` for conversion. *(02.1)*

**A6.** `.view` reinterprets the same bytes under the new dtype — free, aliasing, and the
float64 values come back as meaningless-looking int64 readings of those bytes; `.astype`
converts each value into a fresh buffer. Legitimate uses: `.view` for byte-level work —
hashing a buffer (01.3's digests), binary formats; `.astype` for every case where you mean
the values. *(02.1)*

**A7.** Default: the canonical array is read-only (`writeable=False`), so aliasing it is
safe by construction and slicing stays free. Boundary: one named helper —
`working_copy(arr)` returning `arr.copy()` — as the only sanctioned door to mutability, so
intent is visible at the call site. Tests: the contract as assertions (canonical is
read-only, views inherit it, the working copy is independent by `shares_memory` and
writable), plus an integrity checksum compared across reads as the outer net, which is
01.2's reconciliation habit one level down. *(02.1)*

**A8.** Integer reductions accumulate in the default integer, which under 1.x was the C
long — 32-bit on Windows, 64-bit on Linux — so only the Windows host wrapped its
1.55-trillion-paise total modulo 2^32 (361 full laps, leftover 2,086,575,042). numpy 2.x
made the default int64 everywhere, so the upgrade silently produced the first correct total:
the NEW number is right, the old one had been broken for years, and Linux "always showing
the big number" was the truth misfiled as environment noise. *(02.2)*

**A9.** Two's-complement addition is arithmetic mod 2^32: the accumulator walks a circle of
2^32 positions, and the display is the representative of T mod 2^32 — here
T − 361·2^32 = 2,086,575,042. No step raises because every intermediate value is a legal
int32; overflow between legal values is wraparound by definition, not an error state.
*(02.2)*

**A10.** Python scalars are weak (they adopt the array's dtype); numpy scalars are strong
(they promote by type); values no longer vote. Audit surface: in-range unsigned arithmetic
now wraps silently where 1.x widened (`uint8 array + 200` → 144, no warning); a Python int
that cannot fit the array's dtype now raises OverflowError where 1.x widened; and a
`np.float64` scalar now wins the promotion next to a float32 array, silently adding
precision where 1.x demoted it. *(02.2)*

**A11.** `np.long` is the C long — int32 on Windows, int64 on Linux — a platform question
wearing a dtype's name, so on the affected host it reproduces the exact wrap it was meant to
fix. Width must be named explicitly: `sum(dtype=np.int64)`. *(02.2)*

**A12.** Both run in float32, but numpy's reduction is pairwise — a tree whose rounding
error grows roughly with log n — while the running total accumulates error linearly and,
once the total dwarfs float32's spacing at that magnitude, rounds every addition by up to
half the gap. The fixture is too small for the divergence to express; at scale it is tens of
thousands of rupees. Fix the dtype or the accumulator (float64 / `dtype=np.float64`), never
the comparison tolerance. *(02.2)*

**A13.** Spacing (resolution), not range: floats hold a huge range thinly, and near 1e8
float32's adjacent representable values are 8 apart, so paisa-level distinctions stop
existing long before any overflow. The probe is `np.spacing(x)` at the column's maximum —
if the gap is not far below the unit you must preserve, the dtype cannot hold the column.
*(02.2)*

**A14.** (1) dtype equals the contracted type (int64 minor units or float64 — never
narrower, never unsigned); (2) integer headroom priced on the SUM: the column's total must
fit with orders-of-magnitude margin, because aggregation overflow is an accumulator
property; (3) float resolution: `np.spacing` at the maximum must sit far below the minor
unit. Elements-vs-future is the incident's whole lesson: every paise value fit int32 while
the column's own sum lapped it 361 times — a column ships with its plausible future total,
not just its largest row. *(02.2)*

**A15.** Align shapes at the right edge, pad the shorter with 1s on the left, each aligned
pair must be equal or contain a 1 (the 1 stretches, by stride 0), anything else refuses;
the result takes each pair's maximum. `(n,)` pads to `(1, n)`; against `(n, 1)` both 1s
stretch → `(n, n)`. `(2, 3) + (2,)` refuses: right-aligned, 2 meets 3 — the rule reads
from the right, never from plausibility. *(02.3)*

**A16.** The spike sizes an intermediate the design never drew: a metric whose working set
should be one `(n,)` vector allocated three orders of magnitude more, so some operation
materialised a fanned shape. Recompute the pipeline stepwise printing shapes — the
deviation array is `(n, n)`, its operands `(n,)` and `(n, 1)` — then version-diff for the
shape change (a helper returning `reshape(-1, 1)`). Mechanism: right-alignment made the
pair legal, both 1s stretched, and the mean averaged 8,670,080 cross-pairs into one
confident scalar; the diagonal of the wrong matrix recovers the true metric. *(02.3)*

**A17.** Entry `[i, j]` of the grid is `a[j] − m[i]`, so `i == j` is precisely the aligned
pairs the code meant. That means every entry is a real difference of real numbers — no NaN,
no overflow, no implausible value anywhere — so value-level checks and output-plausibility
eyeballing find nothing. Only a shape-level check can see the fault, because shape is the
only thing that is wrong. *(02.3)*

**A18.** Stretched axes are stride-0 views over the original buffer — 02.1's header trick —
so broadcast *inputs* re-read bytes and allocate nothing (and numpy marks them unwritable,
since one write would land everywhere at once). The bill arrives when an operation
materialises its *result* at the broadcast shape: the subtraction wrote a genuine
`(2945, 2945)` float64 output, and that 69.4 MB allocation is the entire cost. *(02.3)*

**A19.** Only `n` against available RAM: the intermediate is `n² × 8` bytes — 69.4 MB fits
and the job reports a wrong number; 109,302 MB does not and the job dies with
`MemoryError`. The silent outcome is worse: the crash names its own cause at the offending
line, while the wrong statistic ships, fires alerts, and must be caught by someone noticing
the number rather than the shape. *(02.3)*

**A20.** Inside tight local scopes, implicit broadcasting stays — guards on every line
would drown the arithmetic. At public seams, broadcasting becomes opt-in: an
elementwise-or-die wrapper proves `broadcast_shapes(a, b) == a.shape == b.shape` and
raises on any fan-out. Deliberate outers are written with `np.newaxis` inside the
expression that fans out and reduced in the same expression; helpers return `(n,)` and
consumers add axes at the point of use. Any intended fan-out whose `n·k·itemsize` can
exceed a stated budget quotes its `np.broadcast_shapes` preflight line in review. *(02.3)*

**A21.** Loop `i` from 1 to `max(len(s1), len(s2))`; take `d1 = s1[-i]` if it exists else
1, likewise `d2`; if `d1 != 1 and d2 != 1 and d1 != d2` raise; else append `max(d1, d2)`;
reverse at the end. Run it on `(n,)` vs `(n, 1)` and the padding step turns an innocent
vector into one side of an `(n, n)` outer product — which is why a helper that returns
`(n, 1)` "for later" hands every elementwise consumer a loaded shape, and axis insertion
belongs at the expression that actually fans out. *(02.3)*

**A22.** `escalated[over][top] = True` desugars to `__getitem__(over)` — an advanced
selection, hence a fresh copy — then `__setitem__` on that unnamed temporary, which is
garbage-collected at statement end: two successful operations, zero effect. The log lied
because "escalated 200" derived from `len(top)`, a quantity independent of where writes
land. Fix: compose indices first and write through ONE bracket pair
(`escalated[over_idx[top_order]] = True`), and add a landed-writes post-condition that
recounts from the target and raises on shortfall. *(02.4)*

**A23.** The first bracket decides what the chained write targets. `a[5:200]` is basic
slicing — a view (02.1) — so the second write flows through shared bytes into `a`.
`a[over]` is advanced — a copy — so the second write lands in the temporary and dies with
it. Same syntax, opposite outcomes, and the view/copy census is the lookup table. *(02.4)*

**A24.** `c[idx] += 1` is gather → add → scatter: every occurrence of a duplicate target
gathers the same stale value, adds one, and the last scatter wins — each target advances by
exactly one per statement regardless of multiplicity, so histograms collapse to
"everything is 1". Correct: `np.add.at(c, idx, 1)` (unbuffered accumulation) or
`np.bincount(idx)` for pure counting; the two agree to the element and make a cheap
three-way parity test against the buggy form. *(02.4)*

**A25.** Because every selector-derived quantity — index length, loop count, "rows
affected" — is computed before the write and reports success even when the write lands in
a temporary. The target is the only honest witness: after `flags[top] = True`, assert
`flags[top].sum() == len(top)` (one gather) and raise otherwise — turning the silent
vanish into a first-run exception. *(02.4)*

**A26.** The column is a projection of how the invoice will end, not a statement about
today — 80% of status-overdue rows were not yet past due at the export date. Discipline:
statuses carry intent and get *tested* against the facts they imply (01.2); anything
time-based — fees, reminder tiers, aging — derives from dates, with the status at most
selecting the population. *(02.4)*

**A27.** Predicate as a boolean mask (`status == "overdue"`, boundary-normalised);
positions via `np.flatnonzero`; ranking via `np.argpartition` over the masked amounts
(top-k set in O(n)); compose into one integer index array and write with a single
`__setitem__` scatter; prove with the target-recount post-condition. Tests: exact count,
flags-only-on-eligible-rows, idempotent re-run, and `k > n_eligible` capping cleanly —
plus the shipped chained spelling failing the same post-condition as a regression case.
*(02.4)*

**A28.** `argpartition` when only membership in the top-k matters: O(n) against argsort's
O(n log n), same set — a queue needs the set, so 01.1's argsort-based `topk_flag` was
paying for an ordering nobody read. `descending=True` (numpy 2.5, on `sort`/`argsort`)
replaces the `[::-1]` and score-negation idioms when a reversed *order* is genuinely
wanted, keeping NaNs at the end in both directions. *(02.4)*

**A29.** Phase one: IEEE NaN is absorbing, so plain `mean` propagates — every month of
the lane contains at least one unresolved invoice (12,529 across 92 months, disputes and
write-offs included), so every panel reads NaN; the refusal was honest. Phase two:
`nanmean` restores numbers by silently shrinking each month's denominator to "resolved so
far" — under right-censoring that subset is the fast payers, so the most-censored months
(21.2% coverage in 2026-08) read fastest, producing a fake acceleration. Neither rollup
ever carried its denominator: the fix is statistic + coverage + verdict, with WITHHOLD
below a named floor. *(02.5)*

**A30.** Unresolved-means-unpaid is right-censoring: the resolved subset is selected by
having paid early, so its delays are stochastically smaller than the month's full
distribution and `nanmean` is biased fast (a lower bound on the truth). As coverage r/n
falls, the surviving subset is ever more selectively fast, so the understatement grows
monotonically — worst exactly where the dashboard is newest. *(02.5)*

**A31.** Run the estimator against a month whose truth is fully known: take a
long-resolved month, artificially censor it at "as of 30 days after month-end", and
compare. Captured here: March 2025's true mean is 4.74 days; censored, `nanmean` reads
2.78 on 86% coverage — 1.96 days fast, with no drift or regime change as a confound. The
same experiment ships in the test suite as a bias tripwire. *(02.5)*

**A32.** Sum has an identity element (0), so ignoring everything leaves the identity and
`nansum` returns a confident 0.0; mean has no identity (0/0), so `nanmean` yields NaN and
warns. "No data" thus becomes zero under one spelling and NaN under another — production
code must check the count explicitly and route the empty case to an explicit state
("insufficient data"), never letting either function improvise the answer. *(02.5)*

**A33.** `sum(axis=0)` collapses months → `(4,)` per-status lifetime totals ("how do
invoices end?"); `sum(axis=1)` collapses statuses → `(92,)` monthly volumes ("how much
did we issue?"); `keepdims=True` keeps the collapsed axis as length 1 → `(92, 1)`, the
declared column that broadcasts shares against the pivot (02.3). Mnemonic: the axis you
sum is the axis you lose. *(02.5)*

**A34.** Return the triple (statistic, coverage, verdict) per period, computed from an
explicit mask so numerator and denominator are named; the coverage floor is a reviewed
constant (90% here) and WITHHOLD is a first-class dashboard state. Tests: parity with the
naive estimator wherever the verdict is "report" (no silent divergence), and the
known-truth censoring experiment asserting the verdict fires before the measured bias
exceeds tolerance. *(02.5)*

**A35.** They turn policy into visible code: `where=` names the inclusion mask (reusable
for the coverage count) and `initial=` chooses the empty-reduction behaviour instead of
inheriting it — together reproducing any nan-function with the denominator kept. The
02.2 keyword completing the kit is `dtype=`, the accumulator's width, which decides
whether the fold is even arithmetically safe. *(02.5)*
