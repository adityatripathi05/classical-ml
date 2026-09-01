# Series 01 — Recap: The ML Landscape & Project Lifecycle

Six notebooks, one running system (PayFlow), and one argument built across all of them: an ML
system is an engineering artifact whose failures are mostly decided before any model is fitted.

## What was built

A complete, instrumented dunning pipeline for PayFlow's late-payment problem — from a
hand-written collections rule to a gated, monitored system with a retrain decision function —
plus a churn-labelling study that establishes how a business question becomes a target column.
Every number in the series was produced by committed lab scripts running in the repo `.venv` on
the seeded `_data/` universe.

| Notebook | What it built | The result that mattered |
|---|---|---|
| 01.1 Rules or Learning? | The dunning rule, a logistic model, a gradient-boosting check, and a value metric | Both models beat the rule on precision (0.446, 0.453 vs 0.425) and **lost to it on money** ($49,665, $50,549 vs $51,362) |
| 01.2 First Contact | The six-table universe, grain tests, join guards, a `Money` type, a nine-check contract | Summing `amount` across currencies overstated billings **34.21x**; the invoice/payment join moved 24,351 rows for a net of −1,603 |
| 01.3 Reproducibility | Split lottery, seed-sensitivity audit, run manifest, `reproduce()` | `random_state` on the estimator **changes nothing**; the unpinned split moves precision by 0.0378, twice the improvement being claimed |
| 01.4 Taxonomy | Four framings of one table, the API-as-taxonomy probe, capacity selection | **No threshold** delivers a fixed queue (1,773 at 0.3, 241 at 0.6); top-*k* does, at precision 0.6333 |
| 01.5 Problem Framing | Label design space, break-even derivation, cost curve | Break-even precision **0.5556** against 0.0739 delivered — the model was never the bottleneck |
| 01.6 Lifecycle | Eight instrumented stages, a seven-gate suite, defect injection | 5 of 7 defects caught offline; **2 mis-specifications passed every gate** |

## Incidents and the general lesson each carries

| Incident | Mechanism | Transferable lesson |
|---|---|---|
| Green dashboard, rising overdue balance (01.1) | Gateway migration raised the late rate 0.263 → 0.389 while the score distribution held, so precision *rose* while recall collapsed 0.381 → 0.294 | Precision over a self-selected queue rises under label shift; monitor calibration and recall, not only self-computable metrics |
| Board deck 34x too high (01.2) | `amount` is denominated per-customer; summing across currencies adds unlike units, and pandas 3 concatenates the string column instead of erroring | A column whose meaning depends on a neighbour cannot be aggregated alone; prefer a constraint (a type) to a convention |
| Blocked rollback (01.3) | The rollback target was a script, not an artifact; only code was versioned while data and split draw were not | Roll back to an immutable artifact; a run is a function of code, data *and* config |
| Queue of 1,773 for three analysts (01.4) | Threshold selection makes queue size an output of a drifting score distribution; capacity is a constraint | Select by capacity when a fixed resource is consumed; the framing, not the model, was wrong |
| Retention campaign with no effect (01.5) | Label counted a state (base rate 0.1149) not a time-bounded event (0.0207); business case inflated 5.6x; break-even unreachable | Compute break-even precision before funding; a base rate quoted without its label becomes a business case |
| Fourteen months, all gates green (01.6) | Wrong horizon (7-day label vs 30-day escalation) and a survivorship population from an inner join | Gates certify implementation, never intent; an all-green suite under a failing project is itself the finding |

## Invariants established (relied on by later series)

1. **Baseline first.** No model number is reported without the dumb baseline and the incumbent
   rule beside it. Random, rule, model — always three numbers.
2. **Matched operating point.** Policies are compared at the same *k* or the same threshold, never
   at each one's most flattering point.
3. **Noise band before winner.** A claimed improvement is meaningless until repeated refits or
   resplits give the spread; deltas inside the band are not results.
4. **Objective beside proxy.** The business quantity (money, queue quality) is reported next to
   the training metric, because they can move in opposite directions.
5. **Evidence is executed.** Every number comes from committed lab code; anything else is marked
   `# illustrative - not captured output`.
6. **Contracts over conventions.** Data contracts at ingestion, serving contracts at output, type
   constraints where a silent error is possible.
7. **Written specs are what make checks possible.** A label spec and a scoring-population
   definition in the manifest config block; undocumented intent cannot be gated.

## Weak spots this series deliberately left open

- **The survivorship population.** The modelling table is an inner join to payments, so every
  number in 01.1–01.6 describes invoices that were eventually paid, excluding 567 never-paid
  invoices per evaluation window. Named and quantified in 01.6; **fixed properly in series 12**,
  where population definition becomes part of validation.
- **Leakage treated informally.** `reminder_count` and `total_lifetime_value_usd` are excluded by
  hand with a stated reason. The taxonomy, temporal CV and nested selection are series 12's.
- **Cleaning done in lab code, not taught.** Duplicate drops, comma parsing, dual timestamp
  formats and currency conversion happen inside `build_dataset`; series 09 owns them as a subject.
- **Every model is a black box here.** Logistic regression, KNN, k-means, gradient boosting and
  SGD all appear without derivation — series 11, 15, 16, 20 and 21 are their canonical homes.

## What the next series assumes

Series 02 (NumPy) assumes only the reader profile, not this series' content. But series 09 onward
assume the seven invariants above as habits rather than as instructions, and every later series
draws from the same `_data/` universe with the grain and mess documented in `_data/SPEC.md`. The
gate suite from 01.6 is the skeleton that series 34 rebuilds properly on FastAPI, PostgreSQL and
Redis.

## How to revise this series

Work `_quiz.md` from the top without reading the answers. If more than two answers are shaky,
re-read the notebook named beside them rather than the whole series. Then re-implement, from
memory on a blank file, the dunning rule plus `precision_recall_at_k` plus `topk_flag` — those
twenty lines carry four of the seven invariants — and check them against 01.1.
