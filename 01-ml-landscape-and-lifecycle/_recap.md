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
| 01.2 First Contact | The six-table universe, grain tests, join guards, a `Money` type, a nine-check contract | Summing `amount` across currencies overstated billings **34.21x**; the invoice/payment join moved 23,525 rows (12,564 dropped, 10,961 added) for a net of −1,603 |
| 01.3 Reproducibility | Split lottery, seed-sensitivity audit, run manifest, `reproduce()` | `random_state` on the estimator **changes nothing**; the unpinned split moves the model's own precision by 0.0378, while the **paired** model-minus-rule delta moves only 0.0068 — so the claimed gain survives |
| 01.4 Taxonomy | Four framings of one table, the API-as-taxonomy probe, capacity selection | A fixed threshold gives daily queues of **28 to 53 against a roster of 48 — matching it on none of twenty days**; top-*k* guarantees the size and catches **590 late invoices against 215** — while precision *falls*, because the queue is larger |
| 01.5 Problem Framing | Label design space, break-even derivation, cost curve | Break-even precision **0.5556** against 0.0739 delivered — the model was never the bottleneck; and `predict_proba` under `class_weight="balanced"` is not a probability |
| 01.6 Lifecycle | Eight instrumented stages, a seven-gate suite, defect injection | 5 of 6 injected defects caught offline; the wrong horizon reached production, and the survivorship defect could not be injected at all — it had to be measured |

## Incidents and the general lesson each carries

| Incident | Mechanism | Transferable lesson |
|---|---|---|
| Green dashboard, rising overdue balance (01.1) | Gateway migration raised the late rate 0.263 → 0.389 while the score distribution held, so precision *rose* while recall collapsed 0.381 → 0.294 | Precision over a self-selected queue rises when the positive class grows denser; monitor calibration and recall, not only self-computable metrics |
| Board deck 34x too high (01.2) | `amount` is denominated per-customer; summing across currencies adds unlike units, and pandas 3 concatenates the string column instead of erroring | A column whose meaning depends on a neighbour cannot be aggregated alone; prefer a constraint (a type) to a convention |
| Blocked rollback (01.3) | The rollback target was a script, not an artifact; only code was versioned while data and split draw were not | Roll back to an immutable artifact; a run is a function of code, data *and* config |
| A queue of 37 for a roster of 48 (01.4) | Threshold selection makes queue size an output of each day's invoice mix — 28 to 53 across twenty days, starving the roster on eighteen days and flooding it on two; capacity is a constraint | Select by capacity when a fixed resource is consumed; the framing, not the model, was wrong |
| Retention campaign with no effect (01.5) | Label counted a state (base rate 0.1149) not a time-bounded event (0.0207); business case inflated 5.6x; break-even unreachable | Compute break-even precision before funding; a base rate quoted without its label becomes a business case |
| Fourteen months, all gates green (01.6) | Wrong horizon (7-day label vs 30-day escalation) and a survivorship population from an inner join | Gates certify implementation, never intent; an all-green suite under a failing project is itself the finding |

## Invariants established (relied on by later series)

1. **Baseline first.** No model number is reported without the dumb baseline and the incumbent
   rule beside it. Random, rule, model — always three numbers.
2. **Matched operating point.** Policies are compared at the same *k* or the same threshold, never
   at each one's most flattering point.
3. **Noise band before winner — and the band must match the quantity claimed.** A claimed
   improvement is meaningless until repeated refits or resplits give the spread. But a *paired*
   claim (model minus rule on identical rows) needs a *paired* band: judging it against the
   marginal spread of either arm rejects supported results, because the shared noise cancels in
   the difference. Here the model's own precision moves 0.0101 across holdouts while the
   difference moves 0.0068.
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
  invoices per evaluation window — mostly disputes and write-offs, an entire outcome class rather
  than the largest by value. 01.6 measures the effect rather than asserting it: scoring the full
  production population moves the base rate 0.2632 → 0.2782 and precision 0.4474 → 0.4586, so the
  omission is real and the comparison survives it. **Fixed properly in series 12**, where
  population definition becomes part of validation.
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

## What the review changed

This series was reviewed end-to-end after first authoring — three rounds — and each round found
defects the previous one's fixes had missed or introduced: prose contradicting the notebook's
own captured output, mechanisms explained wrongly, experiments that did not demonstrate what
they claimed, and companion documents still teaching corrected-away claims. `_review.md`
records every finding with its fix and the evidence. Round three's headline findings: 01.4's
incident evidence had been produced by the wrong-era model (rebuilt era-consistently, the
queues are 28–53, not 4–23); 01.4 judged its framing gap against a band borrowed from a
different comparison (measured properly, the verdict flips to "indistinguishable"); and 01.6
carried a "barely moves ranking" story that its own +0.0337 dose-response contradicted.

Two of those are worth carrying as lessons in their own right, because they are the kind of
mistake that survives a clean test suite:

- **A comparison can be better determined than either number in it.** The series originally
  judged a paired gain against a marginal spread and concluded the result was noise. It was not.
  Pairing removes common-mode variation, which is why scoring the incumbent on the same rows is
  not a courtesy but the thing that makes the measurement usable.
- **A conclusion that survives correction is worth more than one that was never tested.** Several
  headline claims changed under review — 01.4's win turned out to be recall rather than
  precision, 01.6's leakage mechanism was wrong, 01.5's cancellation argument was a non-sequitur.
  Each corrected version is a stronger argument than the one it replaced, and all of them were
  found by reading against the evidence rather than by any automated check.

## How to revise this series

Work `_quiz.md` from the top without reading the answers. If more than two answers are shaky,
re-read the notebook named beside them rather than the whole series. Then re-implement, from
memory on a blank file, the dunning rule plus `precision_recall_at_k` plus `topk_flag` — those
twenty lines carry four of the seven invariants — and check them against 01.1.
