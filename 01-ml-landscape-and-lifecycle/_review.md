# Series 01 — end-to-end review findings

> **ROUND THREE (2026-09-01, against `da03aa7`) — independent re-review.** Every "FIXED"
> claim below was treated as unverified and re-checked by re-running the labs. The round-two
> pattern repeated: several round-one findings were never actioned, one round-two fix left a
> stale contradiction in place, and two substantive new defects were found. Recorded at the
> bottom under "Round three". All fixes applied in the same pass; every notebook regenerated
> and re-executed; series checks clean.

Review date: 2026-09-01, against commit `68ace36`. Three independent review passes
(continuity, technical correctness, depth-vs-guide) plus direct re-verification of every
numeric claim below by re-running lab code.

> **STATUS after the SECOND review (2026-09-01).** The first review's 13 findings were
> fixed, then re-audited — and the audit found that several fixes were incomplete, one
> was never applied at all, and the fix passes introduced new errors of their own. Those
> have now been corrected in turn. Series checks 0 fail across 15 files; the three
> remaining warns are illustrative prose figures.
>
> **The honest summary of round two:** my claim that "all blockers were cleared" was
> false. B6 (01.5's inferred-label evidence) had never been fixed. B2's matched-operating-
> point fix was applied in one function and missed in another, so the notebook kept
> printing the superseded 0.4499 while the prose asserted a 0.4474 → 0.4609 delta that
> **no cell produced** — a fabricated number, in the series whose whole subject is not
> doing that. Details and corrections are recorded below under "Round two".

**Original verdict (superseded by the fix passes recorded below): NOT promoted to `done`.
A substantive fix pass is required.** Mechanics are
clean (`_tools/check.py`: 13 files, 0 fail) and the series' architecture, running system
and incident discipline hold up well. But content review found defects no script can see:
**seven invalid-reasoning blockers** (one of which reverses a notebook's central
conclusion), six evidence contradictions, and six mischaracterisations.

The most important: **01.3's headline argument is statistically invalid** — it compares a
marginal spread against a paired difference, and corrected, 01.1's claim survives. Three
further mechanisms are stated wrongly (01.5's MRR cancellation, 01.6's leakage
displacement, 01.4's top-k precision benefit), and in each case the *correct* mechanism
makes the notebook's own point more sharply. That is the encouraging shape of this review:
almost every fix strengthens the argument rather than retreating from it.

Status stays `[r]` in `CURRICULUM.md` until at least the B0-series blockers are cleared.

---

## Round two — what the re-review found, and what it changed

Two independent audits after the first fix pass. Every finding below was re-verified by
running the labs.

**Incomplete or unapplied fixes.**
- **B6 was never fixed.** `implicit_vs_explicit` was still called nowhere in 01.5 while its
  numbers were asserted in three places, falsifying `_recap.md`'s "every number was
  produced by committed lab scripts". Now called as a Stage C cell.
- **B2 was half-applied.** `survivorship_effect` still scored the model at a fraction of
  the window while scoring the rule at its own k. Fixed; the honest survivorship figures
  are 0.4474 → 0.4586, and the fabricated "0.4474 → 0.4609" is gone.
- **`_recap.md` and `_quiz.md` were never regenerated** and still taught the corrected-away
  claims — the B0 invalid comparison verbatim, the old queue story, the old tally, the
  superseded terminology. Both rewritten. `check.py` gained **X01** so this class of drift
  fails mechanically from now on.

**Errors the fix passes introduced.**
- **01.4's evaluation window was never a month.** `head(5_500)` spans 140 calendar days, so
  every per-day figure built on it was meaningless and the capacity derivation mixed
  calendar days with working days. The window is now a genuine contiguous month (5,329
  invoices over 28 days) and capacity is derived as 26.4% of volume — the same share as
  01.1's rule queue — instead of asserted.
- **A tautology presented as evidence.** 01.5 reported an "expected value" column and
  concluded "the algebra is right" — but expected value under the model's own scores is
  exactly what each ranking sorts on, so the winner was an identity. The column is gone,
  with a note saying why.
- **A conclusion that was an artifact of its own sampling.** 01.6 concluded random splitting
  was harmless (0.5232 vs 0.5280) — but the contaminated arm was diluted to ~23% by pooling.
  Stratified, the dose-response is 0.5232 → 0.5328 → 0.5443 → **0.5569**, so the full-dose
  effect exceeds 01.1's entire headline gain. The conclusion is reversed.
- **A cherry-picked statistic.** 01.6's "coefficient vectors correlate at −0.33" used 6 of
  31 coefficients; over all 31 it is **+0.50**. Both are now reported, with the ranking
  comparison identified as what actually settles it.
- **A universal claim that was conditional.** 01.5 stated the value-maximising order is
  `MRR × (p − p*)`; that holds only for a proportional offer. Under a flat cost it is
  `MRR × p`, and the measured nets flip accordingly ($772 vs $107,033).
- **A wrong mechanism and a wrong cause.** "predict_proba inflated eighteen-fold" —
  `class_weight="balanced"` shifts log-odds by a constant, and prior-correcting it gives
  0.0210 against an observed 0.0226, i.e. well calibrated. And the realised losses were
  blamed on miscalibration; an **oracle with perfect foresight still loses $605,089**, so
  the intervention was never rescuable. That is now the section's headline.
- **A conclusion drawn from a self-flagged omission.** 01.6 noted it had not scored the rule
  on the 30-day question, then concluded "2.93x lift is not nothing" anyway. The rule scores
  0.0500 (2.86x); the model's edge is **+0.0013**. Against the incumbent, on the question
  that matters, the model is worth approximately nothing — a sharper payoff than the one it
  replaced.

**Root cause of the recurrence.** Twice a lab's `main()` was corrected while the notebook
kept its own inline copy of the old logic. Notebook cells now call the lab functions rather
than duplicating them, and the verify phase must check every call site, not just the lab's
entry point.

---

## Blockers — invalid reasoning (found by the correctness pass, re-verified here)

**B0 — ✅ FIXED 2026-09-01.** `split_lottery` now scores both policies per split and returns
`(model, rule, delta)`; `training_vs_evaluation_noise` reports a marginal-vs-paired 2×2.
01.3's Technical Explanation, How It Works diagram, Stage A/B cells and commentary,
Diagnosis step 4, Root Cause, Prevention, Common Pitfalls, Interview Q1 and Key Takeaways
rewritten to the corrected argument; 01.1's forward reference and 01.6's band attribution
corrected; `_quiz.md` A19 rewritten with the covariance identity. All three notebooks
re-executed, series checks 0 fail. Captured after the fix:

```
what is resampled                     marginal (model)   paired (model-rule)
training set, eval rows FIXED                   0.0032                0.0032
evaluation rows, procedure fixed                0.0101                0.0068
correlation between model and rule across re-drawn holdouts: 0.785
2-sigma band on the paired delta = 0.0136; 01.1's gain +0.0194 CLEARS it
```

The corrected notebook is stronger than the original: it now teaches that a comparison can
be better determined than either number in it, that pairing buys nothing when the holdout
is fixed (the rule is then a constant), and that 01.1's matched-operating-point discipline
is *why* its claim is defensible — turning a manufactured contradiction between the two
notebooks into a genuine dependency.

**B0b, B0c, B1, B3, B4 — ✅ FIXED 2026-09-01.** Four notebooks rebuilt and re-executed;
series checks 0 fail across 13 files.

- **B0b (01.5, MRR mechanism).** Corrected to: MRR cancels from the go/no-go *test*
  (`E_i = m_i·s·V·(p_i − p*)`), never from the ranking, whose optimum is `m_i·(p_i − p*)`.
  Lab now measures three rankings in expectation and in realisation. The correction
  surfaced a **better finding than the original claim**: the classifier is fitted with
  `class_weight="balanced"`, so `predict_proba` is a re-weighted score, not a probability —
  mean 0.4063 against an observed 0.0226, inflated ~18×. That is why two rankings show
  positive *expected* value against negative *realised* value. New rule: calibrate before
  costing. Fixed in How It Works, Stage C, Common Pitfalls, Interview Q1 and Takeaways.
- **B0c (01.6, leakage mechanism).** New `leakage_mechanism()` lab prints the coefficient
  table: `terms_days` flips +0.1517 → −0.5116, honest-coefficient vectors correlate −0.33,
  and the production ranking is Spearman −0.18 against the clean one. Prose now rules out
  the "under-weighting" story explicitly (a constant column is an intercept shift, and a
  rank metric is invariant to it) and states conditional-vs-marginal distortion instead.
- **B1 (01.4, capacity units).** PayFlow's collections capacity is now stated **once** for
  the series in daily terms — 48/day, derived from 01.1's 7,195 over the 151-day stable
  window — and a new `daily_queue_swing()` runs the actual incident: a fixed t=0.5 across
  20 consecutive days yields queues of **4 to 23 against a roster of 48**, starving the
  team on every observed day with a 5.8× swing. The cold open, Symptoms, Diagnosis, Root
  Cause and Fix were rewritten to the evidence rather than the other way round.
- **B3 (01.4, top-k precision).** `threshold_vs_capacity` now prints precision at every
  operating point and the implied cutoff (0.3734). The false claim is replaced by the true
  and stronger one: capacity selection buys **size control, not precision** — precision
  actually *falls* (0.6203 → 0.4886) because the queue is larger — while catching **516
  late invoices against 263 at the same staffing cost**, roughly double, using headcount
  that was already being paid for.
- **B4 (01.2 + 01.4).** 01.2 now reports 3,792 and explains why 01.1's audit reports three
  fewer (dedup order), turning a contradiction into a second demonstration of the grain
  bug. 01.4's latency ratio is stated consistently as ~11× against the captured 11.1x.

**B0d, B0e, B0f, B0g — ✅ FIXED 2026-09-01**, together with M2 and M3 which they touch.
Series checks 0 fail across 13 files.

- **B0d (01.6, tautological gate).** The hand-built `Run` is gone. `survivorship_effect()`
  now *measures* the blind spot: it reconstructs the never-paid invoices with the same
  features, labels them late, and scores the model on the population production would
  actually meet. Base rate moves 0.2632 → 0.2782; model precision 0.4499 → 0.4609; rule
  0.4254 → 0.4366. **The honest reading moderates the original alarm**: the omission is
  real and the offline base rate is mildly optimistic, but the comparison the series rests
  on survives it — and saying so is better than overstating. The defect is now presented
  as a gate-coverage *argument* (no flag expresses it) and removed from the injection
  tally, which is 5 of 6. `gate_data_contract` now inspects `df.duplicated()` rather than
  reading a flag the defect set.
- **B0e (01.2, join arithmetic).** Decomposed at row level so it closes: 12,564 removed,
  10,961 added, −1,603 net, 23,525 rows moved. The lab prints the check line explicitly.
- **B0f (01.3, contaminated data lottery).** The evaluation slice is now June 2025 —
  strictly after both snapshots, and the month 01.1's three windows deliberately skip, so
  it is unused elsewhere (this also closes review finding #15). Uncontaminated, the result
  is *sharper*: precision identical at 0.4613 either way, mean absolute score shift 0.0016,
  queues differing on 10 of 2,736 slots. Two artifacts, different digests, indistinguishable
  behaviour — a cleaner identity-vs-equivalence demonstration than the contaminated run.
- **B0g (01.5, arm A population).** Arm A now draws from the full signed-up population
  minus the evaluation customers, so it contains the 415 already-churned customers and a
  training base rate of 0.1508 with 583 positives — 8× arm B. The "recognising the past"
  mechanism is finally exercised. The resplit loop was corrected the same way.
- **M2 (01.2, spam overstatement).** Now: all 2,064 dropped rows are spam, but that is not
  all spam — 3,496 exist, so the join removes 59% and leaves the atypical 1,432 that carried
  a customer id. The lab computes the total rather than asserting it.
- **M3 (01.2, "never paid").** The prose no longer characterises the dropped rows as never
  paid; they are rows with no matching payment, a mix that 01.4's censoring lesson separates.

Also fixed in passing: the `nrows=200_000` probe now says so, so the 1,366,049-character
string is attributed to the right row count.

**B2, B5, M1 — ✅ FIXED 2026-09-01.** Series checks 0 fail across 13 files.

- **B2 (01.6, mismatched operating point).** `capacity` is now `int(rule.sum())`, so model
  and rule are scored at the identical k=7,195 and gate G4 prints "(both at k=7,195)".
  Every downstream figure moved and the prose was re-derived: model 0.4474, delta +0.0220,
  band 0.0022, leakage 0.7572 offline / 0.2949 production, wrong-horizon 0.0513 at 2.93×
  lift. The notebook now also states that it trains on a 60,000-row subsample, which is why
  its level sits slightly below 01.1's.
- **B5 (drift terminology, 01.1 + 01.6 + quiz).** Settled on **concept drift (posterior
  shift)** — `P(y|x)` moved for the subpopulation `country` identifies, and the class prior
  moved as a *consequence*. The name "label shift" is now explicitly flagged as belonging to
  the opposite case. This single correct mechanism explains both prior findings at once:
  because the shift is roughly uniform *within* the affected group, the ordering survives
  while the level does not — which is why 01.1's refit repaired calibration but barely
  ranking, and why 01.6's random-split contamination cost almost nothing. The manufactured
  contradiction between the two notebooks is gone, and `_quiz.md` A9/A44 no longer disagree.
- **M1 (01.3's back-references).** Content hashing is attributed to `_data/SPEC.md`, which
  is where it actually lives; the prerequisite line now reads "the ingestion contract"; and
  the invented "01.2's pickle-across-versions warning" is replaced by the claim stated
  directly.

Also corrected: quiz questions Q6 (capacity 48/day), Q19 (rewritten around the three
variance figures), Q21 (0.4% disagreement on the clean holdout) and Q25/A25 (queue
starvation with a 5.8× swing, and the honest precision-falls/recall-doubles reading), all of
which cited numbers the fixes superseded.

**M4, M5, M6 — ✅ FIXED 2026-09-01.** The last of the major findings.

- **M4 (band consistency).** 01.1 now states what its ±0.0015 covers — training-sample
  variation on a fixed holdout, which is why the rule looks deterministic there — and
  forward-references 01.3 for the evaluation-draw term and the paired band a shipping
  decision uses. 01.6 carries the equivalent caveat on its 0.0022. The three numbers are no
  longer three answers to one question; they are three clearly-labelled quantities.
- **M5 (running-system continuity).** The threshold change is now narrated: 01.4 says 0.5
  was raised from the lower cutoff 01.1's incident ran on, during the March 2026 review,
  and untouched since. And 01.1's recommendation now records that the shadow model cleared
  its value threshold in April 2025 and was promoted then — which is the live system 01.3,
  01.4 and 01.6 inherit and the fourteen-month life 01.6 decommissions. The timeline closes.
- **M6 (survivorship characterisation).** "Unambiguously the worst outcomes in the
  portfolio" is gone from all three places. The rows are now described as what they are —
  an entire outcome class, predominantly disputed (444) or written off (92), *not* the
  largest by value — with the measured effect quoted: scoring on the full population moves
  precision only 0.4474 → 0.4609.

**B0 (original finding) · 01.3's central argument is statistically invalid, and its
conclusion reverses.**
01.3 compares 01.1's `+0.0194` — a *paired* difference `p_model − p_rule` on identical
rows — against `0.0378`, the spread of `p_model` **alone** across 12 holdouts. The rule is
never scored in `split_lottery`, so the quantity that matters (the spread of the *delta*)
is never computed. Because both policies see the same rows, split noise is common-mode.
Re-verified over 12 splits:

```
model  mean 0.4396  std 0.0097  spread 0.0378
rule   mean 0.4196  std 0.0100  spread 0.0324
DELTA  mean 0.0200  std 0.0065  spread 0.0211    corr(model, rule) = 0.785
2-sigma band on the PAIRED delta = 0.0130   ->  +0.0194 CLEARS it
```

**01.1's claim survives the very test 01.3 uses to demolish it.** The stated general
principle ("any claimed improvement smaller than the run-to-run spread is unsupported") is
false — pairing exists precisely to beat marginal spread. This invalidates 01.3's Root
Cause, its Prevention bullet ("would have been rejected on its own evidence") and Key
Takeaway 4. Everything *else* in 01.3 — the seed audit, the manifest, `reproduce()`, the
data-snapshot arm — stands.
*Fix:* have `split_lottery` return `(p_model, p_rule)` per split and headline `std(delta)`.
The corrected notebook is **better**: it becomes a lesson about paired versus marginal
variance, which is sharper than the one currently there. Note also the observed
`std 0.0097` is smaller than the binomial SE at k≈1,381 (0.0134), so the "lottery" is
ordinary small-holdout noise, not a pipeline pathology.

**B0b · 01.5's MRR-cancellation mechanism is a non-sequitur.** The algebra
`p* = c/(s·V) = 0.5556` is correct; the inference is not. Per-customer net is
`E_i = m_i·s·V·(p_i − p*)`, so MRR cancels only from the **go/no-go test**, never from a
capacity-constrained *ranking*, whose optimum is `m_i·(p_i − p*)`. And it demonstrably does
not "buy nothing" — the notebook's own output swings −$20,433 → −$355,964. The true
mechanism is the opposite of the one given: below break-even every contact has negative
expected value proportional to MRR, so MRR-weighting **maximises the loss**. Repeated in
Interview Q1, Common Pitfalls and Key Takeaway 7.

**B0c · 01.6's leakage "displacement" mechanism is wrong.** At serving time
`reminder_count = 0` for every row, so it contributes an identical constant to every logit
— a pure intercept shift — and precision@k ranks by the linear predictor, which is
invariant to uniform rescaling. Under-weighting therefore costs *exactly zero* ranking
quality and cannot be the mechanism. What actually happens is conditional-vs-marginal
coefficient distortion: `terms_days` flips sign (+0.1517 → −0.5116) and the production
ranking becomes slightly *anti*-correlated with the clean one (Spearman −0.18).

**B0d · 01.6's survivorship "gate result" is tautological and inflates the headline count.**
Cell 12 hand-constructs `lab.Run(...)` by copying the clean run's `precision`,
`noise_band`, `queue_size` and rates; the gates read only those copied fields, so
"caught by: NOTHING" is true **by construction** and no run on survivorship-affected data
is ever built. It is nonetheless counted in "5 of 7 defects caught". The gate-coverage
*argument* is sound and worth keeping — presenting it as an injection *result* is not.
*Fix:* present as an argument from gate coverage and drop it from the tally (making it
5 of 6). Related: four gates read a field the defect flag itself sets rather than
inspecting data — `gate_data_contract` should compute `df.duplicated().sum()`.

**B0e · 01.2's join decomposition does not close arithmetically.** Published: "12,529
removed and 11,822 added, which net out to −1,603". But 12,529 − 11,822 = 707. The 12,529
is a *unique-id* count while the net is a *row* count. Re-verified row-level: 12,529
removed, **10,921** added, net −1,608, moved **23,450** (not 24,351).

**B0f · 01.3's `data_lottery` trains on 100% of its own evaluation rows.** The
"as-of 2025-06-01" arm fits on `issue_date < 2025-06-01` and evaluates on the
2025-01..06 window — full contamination, in the notebook whose subject is experimental
hygiene. Every conclusion in that section is drawn from a contaminated arm.

**B0g · 01.5's arm A never exercises the mechanism its incident blames.** `train_naive`
draws ids from `operational`, which is `active_only=True`, so **zero already-churned
customers** enter arm A (verified: base rate 0.0487, not 0.1149). The "recognising the
past" root cause is never demonstrated, and the "5.6× smaller addressable problem"
confounds population *and* horizon.

---

## Blockers — contradict captured evidence or the series' own invariants

**B1 · 01.4 — the incident cannot be produced by the evidence shown.**
The evaluation window is `test_stable.head(5_500)`, which the lab labels "~one month of
invoices", but every queue number is compared against a capacity of "300/day". Verified:
1,773 invoices per month is ~81/day, which *starves* a 300/day team rather than flooding
it, and "top-300" over a monthly window is 300 per month, not "exactly 300 every day".
Worse, the cold open narrates two different days (1,773 one Monday, 241 the previous
Tuesday) but those are the sweep values at t=0.3 and t=0.6 on a single fixed window — at
the stated deployed threshold of 0.5 the evidence says 424 on both days.
*Fix:* build a genuine per-day cell — score several distinct days at a fixed t=0.5 and
show the real day-to-day swing — or restate the window as monthly with a monthly capacity
(~6,600). Propagates to `_recap.md` and `_quiz.md` Q25/A25.

**B2 · 01.6 — the gate suite violates invariant 2 (matched operating point).**
Verified: `capacity = round(0.26 × 27,290) = 7,095` scores the model, while the rule is
scored at its own `k = 7,195`. Gate G4 ("beats baseline: model 0.4499 vs rule 0.4254")
therefore compares a 7,095-slot queue against a 7,195-slot one — the exact error 01.1's
Common Pitfalls forbids and `_recap.md` lists as an invariant.
*Fix:* set `capacity = int(dunning_rule(test).sum())` so the notebook runs at k=7,195;
re-derive the delta; add one line explaining that 01.6 trains on a 60,000-row subsample,
which is why its numbers sit off 01.1's.

**B3 · 01.4 — the top-k precision claim is false.**
Verified sweep precisions (never printed in the notebook): t=0.3 → 0.4309, 0.4 → 0.5281,
0.5 → 0.6203, 0.6 → 0.6266; top-300 → 0.6333. The prose claims top-k reaches "a precision
of 0.6333 … well above anything the threshold sweep produced"; it beats the t=0.6 point by
0.0067, well inside the noise band the same notebook invokes to call a 0.0039 gap
indistinguishable.
*Fix:* print the precision column, and make the true (stronger) claim — precision is
essentially unchanged; what changed is that queue size stopped being an output.

**B4 · 01.2 — prose contradicts the output two cells above it.**
Cell prints `3,792`; prose says "Three thousand seven hundred and eighty-nine". Verified:
the raw export has 3,792 and the deduplicated frame has 3,789, which is why 01.1's audit
reports the latter. Spelling the number in words hid it from `check.py`.
*Fix:* use 3,792 and add the half-sentence explaining the 3,789 difference — it becomes a
second demonstration of the grain bug.

**B5 · 01.6 contradicts 01.1's drift diagnosis while claiming to agree with it.**
01.1: "label shift … only the mapping from inputs to outcome" changed. 01.6: "a shift in
the base rate **rather than** in the relationship between features and outcome … consistent
with what 01.1 found." These are opposite, and the 01.6 version is load-bearing for its
"random splitting is harmless" conclusion. Separately, the *name* is wrong in both: what is
described (P(y|x) moves for the migrated subpopulation) is concept drift, not label shift.
`_quiz.md` carries both versions — A9 against A44.
*Fix:* settle on subpopulation concept drift; correct A9; rewrite 01.6 to explain that
random splitting still cost nothing *because the change is captured by an existing feature*
(`country`) — a better lesson than the current one.

**B6 · 01.5 — its inferred-label evidence is never executed.**
"85 of 119 churners, no false positives, 45-day detection lag, last invoice 15 days before"
appears in Design Patterns, Common Pitfalls, interview Q7 and `_quiz.md` A38. Verified: the
numbers are correct and `implicit_vs_explicit()` is committed — but the notebook never calls
it, so `_recap.md`'s "every number was produced by committed lab scripts" is false here. The
15 → 45 arithmetic is also unfollowable, because `quiet_days = 60` is never stated in 01.5.
*Fix:* add one cell calling the existing function.

---

## Major — misattribution and mischaracterisation

**M1 · 01.3 credits 01.2 with content hashing and a pickle warning; 01.2 has neither.**
Four wrong back-references (prerequisites line, Technical Explanation, Design Patterns,
Related). The hashes live in `_data/SPEC.md`. *Fix:* re-attribute, or add a hashing check to
01.2's contract cell; delete the pickle claim.

**M2 · 01.2 overstates the spam join.** "Deletes precisely the rows a spam classifier exists
to detect … its entire positive class had been removed." Verified: 3,496 spam tickets exist,
2,064 are dropped — 59%, not all. The survivors are the atypical spam that carried a customer
id, which is a more interesting failure. `_quiz.md` A14 already says the accurate thing.

**M3 · 01.2 mischaracterises the 12,529 unpaid invoices as "never paid".** Verified: 6,880
of them (54.9%) were issued 2026-07 or later and are simply unresolved; the status mix is
6,848 overdue / 4,298 disputed / 1,097 written off / 286 marked paid with no payment record.
Calling all of them never-paid contradicts 01.4's own censoring lesson and collides with
01.6's 2.0% figure.

**M4 · "The noise band" is three incompatible numbers.** 01.1 reports 0.0015 (training
bootstrap, 27,290-row eval), 01.3 reports 0.0378 (resplit, 5,500-row eval) and argues the
first understates by 4.9×, then 01.6 builds gate G5 on 0.0031 — a training bootstrap — and
credits it to "01.3's bootstrap discipline", which is the quantity 01.3 exists to criticise.

**M5 · Running-system continuity.** The deployed threshold is 0.35 in 01.1 and 0.5 in 01.4,
both described as never touched. And 01.1 concludes the model stays in shadow mode, while
01.3, 01.4 and 01.6 all assume it is live — nothing says when or why it was promoted.

**M6 · 01.6 overstates the survivorship rows.** "Unambiguously the worst outcomes in the
portfolio" — verified false: the 567 never-paid invoices are median $304 against $327 for
resolved, 2.04% of count and 1.94% of value, with a near-identical segment mix. The defect is
real (a whole outcome class absent from training, $901k uncollected) but not because those
rows are the largest. *Also newly found:* 31 invoices in that window carry `status = paid`
with no payment record — a referential gap the 01.2 contract does not test.

---

## Minor — precision and framing

- **01.3** divides a 5,500-row spread (0.0378) by a 27,290-row gain (0.0194) to claim "2.0×";
  scaled like-for-like the comparable spread is ≈0.017. Still close to the claimed gain —
  which is the real point — but the stated ratio and the "would have been rejected" verdict
  do not follow.
- **01.3** rounds a captured 159,821 to "160,000".
- **01.4** states the latency ratio three ways ("nine times", "ten times", "roughly ten")
  against a captured `11.1x`.
- **01.4** online-vs-batch is uncontrolled: the online model sees ~78.9k rows in one pass
  against 159,821 fitted to convergence, so the 0.3681-vs-0.4462 gap conflates three effects.
  The printed per-quarter `n=` also reads as cumulative.
- **01.1** temporal windows silently omit June 2025 (4,979 resolved invoices) with no note.
- **01.5** garbled sentence: "of the 709 customers ever marked churned before the cutoff, 415
  had already left" — 709 is the ever-churned count, not a pre-cutoff count.
- **01.5** names top-2% the "best operating point" on four positives at k=46, inside noise.
- **01.5** says value-weighting "paid for itself in 01.1"; there it closed the gap and still
  lost to the rule.
- **01.2** never names M2, the mixed-currency rule that is its entire cold open.
- **01.1** "roughly twenty-six thousand invoices" against 27,857 issued / 27,290 resolved.
- Timeline drift: 01.3's incident is dated 2026-03 but its evidence windows are 2025; 01.4's
  is dated 2026-04 but leans on a labelled-rate table running to 2026-08.

---

## Additional correctness findings (third pass)

- **01.3 `frame_hash` is row-order dependent** — columns are canonicalised, rows are not,
  yet it is called "a content address for a data slice". A reordered identical slice reads
  as NOT REPRODUCED. *Fix:* `np.sort(key)` before hashing.
- **01.2's reconciliation is not two independent derivations.** Both sides divide by the
  same `fx_rate_usd` column, so the +0.55% residual measures only that column's dispersion
  about its own median — it cannot validate the conversion *direction* or the rate source,
  and partial-payment coverage (M13) is folded in and misattributed to FX.
- **01.5's 730-day horizon is fully right-censored.** Cutoff + 730d = 2027-06-30 but data
  ends 2026-08-30; the 730d row is identical to a 3650d row (294 positives, 0.0511) — it is
  the "ever churns" label, not a horizon. *Fix:* cap the sweep at 365d or mark it saturated.
- **01.1's "six weeks" contradicts its own dates** — migration 2025-07-01, alert 2025-07-15
  is two weeks; "six weeks" recurs four times, and Prevention's "would have paged on
  2025-07-14, six weeks earlier" is one day earlier, not six weeks.
- **01.1 Interview Q1's derivation is garbled** — at fixed *k* the denominator *is* k and
  does not move; and precision rises only if added positives land inside the selected set,
  not "mechanically".
- **01.2's "1,366,049 character string" is for 200,000 rows**, not the 288,936 claimed
  (`currency_incident` reads with `nrows=200_000`; all rows give 1,973,207).
- **01.1's `make_boosted` docstring claims native categorical handling**; it one-hots and
  never sets `categorical_features`. And "no amount of model capacity fixes a misspecified
  objective — that is what the gradient-boosting row proves" rests on one default-
  hyperparameter GBM with no variance estimate. "Is consistent with" is defensible; "proves"
  is not.
- **01.6's random-split calibration is under-powered** — one fit per arm, no variance, in a
  notebook that fails a defect for exactly that. The 0.0048 delta is *larger* than the same
  notebook's 0.0031 band, so by its own G5 logic it reads as signal, not "effectively
  identical". Arms also share 27,710 of 60,000 training rows.
- **"Calibration" is mean-rate matching.** `gate_calibration` compares `scores.mean()` to
  `y.mean()`; a model can match the mean exactly and be badly miscalibrated across the range.
  Rename to a mean-rate / prior-shift monitor and leave calibration proper to 15.1.
- **01.5's "90-day lift 0.91x, worse than random"** rests on 2 hits at k=230; state it as
  indistinguishable from random at that positive count.
- **01.4/01.6 read the same predicate two ways** — `~invoice_id.isin(pay_ids)` is
  right-censoring in 01.4 ("labels have not arrived") and permanent in 01.6 ("never paid").
  Reconcile via `status`.
- **Hardcoded strings that go stale** — `framing_disagreement` prints "27% of the WORK" and
  "inside the 01.3 noise band" as literals while computing the real value a line above;
  `01.6`'s `incident` hardcodes an integer-division "2x the positives".
- **01.6's `no_dedup` injection duplicates random rows**, not the 896 actually duplicated in
  the raw export, while claiming to be SPEC M1.
- **01.6's "would have paged on the day"** is unsupported — the calibration gap is computed
  once over a ten-month window; no daily monitor is ever evaluated.
- **01.1's 0.297 base rate pools pre- and post-drift regimes** while every claim is actually
  measured against 0.263 — the error 01.5 is about.
- **01.5 `features_as_of` has a dead placeholder assignment** to `mrr_usd`, overwritten on
  the next line.
- **`lab_01.4`'s comment "no y argument exists to pass"** on `km.fit(X_tr)` is wrong —
  `KMeans.fit(X, y=None)` exists and is ignored, which is exactly what `api_surface`'s own
  `needs_y` probe tests two functions later.

## Structural / editorial — a judgement call, not a defect

- **Stage C is analysis rather than production code in 01.4, 01.5 and 01.6**, without the
  one-line collapse justification the guide requires. 01.1, 01.2 and 01.3 do it properly
  (`build_dunning_queue`, the `Money` type, `build_manifest`/`reproduce`). The clearest gap is
  01.5, whose own Permanent Fix demands a label spec artifact that the notebook never builds.
- **01.4's Stage A hides its mechanism** behind a single `lab.build_framings()` call, where
  01.2 and 01.3 show theirs in ten visible lines.
- **01.4 and 01.5 re-run an identical listing** in their Production Scenario, adding no new
  evidence; 01.5's rerun is followed by no prose at all.
- **Several interview answer shapes give away the full answer including the numbers**
  (01.4 Q6, 01.5 Q1, 01.6 Q5/Q6), which defeats delayed retrieval.
- **Three notebooks open by positioning against textbooks** (01.3, 01.4, 01.6).
- **01.1's cost model constants** (`ACTION_COST = 2.50`, the 0.40 recovery share) are never
  shown in the notebook, and the lab's robustness note — that the policy *ranking* is
  invariant to the cost level — never reached the prose.
- **Weakest notebook: 01.4**, by consensus of two independent reviews and on re-reading.
  Strongest: 01.1, with 01.6's self-audit the single best move in the series.

---

## Recommended fix order

1. **B0** (01.3 paired vs marginal) — largest blast radius; changes 01.3's Root Cause, its
   Prevention bullet, Key Takeaway 4, and how 01.1 and 01.6 should describe their bands.
2. **B0b, B0c, B4, B3/B1** — the four wrong or contradicted mechanisms. Each correction
   makes the notebook's own argument stronger, so these are rewrites, not retreats.
3. **B0d, B0e, B0f, B0g, B6** — experiments that do not demonstrate what they claim, or do
   not close arithmetically. Mostly small code changes to committed labs.
4. **B2, B5, M1–M6** — matched operating point in the gate suite, the drift-terminology
   correction (concept drift, not label shift, in both notebooks *and* `_quiz.md`), and the
   backward-reference misattributions.
5. **Minors and editorial** — batch into the same rebuild pass; each notebook is regenerated
   from its builder and re-executed anyway, so the marginal cost is near zero.

Re-run `_tools/check.py` after each rebuild; note that the extended E02 (comma-grouped
integers) and S03 landed during this review and will now catch the class of error that
produced B4.

## What held up

- Every forward cross-reference (02, 03, 09–12, 15, 16, 20, 21, 23, 28, 29, 33–35) resolves to
  a real series whose scope line matches the claim. Only backward references are wrong (M1).
- PayFlow structural continuity is intact across all six notebooks and matches `_data/SPEC.md`
  — table names, grains, row counts, currencies, the M1–M14 catalog.
- Production Scenario quality: every incident has a time, trigger, blast radius, on-call
  symptoms, an explicitly walked ladder, a mechanistic root cause and a split fix. None could
  be pasted into another notebook.
- Real-dev examples throughout: no foo/bar, no iris, no `make_blobs`, no toy DataFrame as a
  section subject. The single marginal case is `api_surface()` in 01.4.
- Baseline-first is genuinely carried, not mentioned once: the rule is the floor in all six,
  runs in shadow in 01.1, becomes gate G4 in 01.6, and overtakes the model in the drift window.

---

## Round three — independent re-review (2026-09-01, against `da03aa7`)

Method: read every notebook's prose against its captured output cell by cell; re-ran all six
labs; recomputed load-bearing claims independently. The three prior-round warnings all
recurred: unfixed round-one findings, a half-propagated fix, and a companion document still
teaching a corrected-away mechanism.

**New blockers (found this round).**

- **R3-B1 · 01.4's incident evidence came from the wrong-era model.** The April-2026 incident
  was evidenced by `daily_queue_swing` scoring July-2025 days with the stale pre-2025 model —
  the model the series' own continuity says was retrained away in mid-2025 (and July 2025 is
  01.1's drift window, a different failure). Re-run era-consistently — post-migration-trained
  model, twenty spring-2026 days, t=0.5 — the queues are 28–53 against the 48 roster (18 days
  starve, 2 flood, 0 match, 1.9x swing), not the published 4–23 with a 5.8x swing. The
  mechanism survives; every number in the cold open, Symptoms, Diagnosis and Root Cause had to
  change. The lab now also fits the model once outside the day loop (it was refitting per day)
  and prints pooled predicted-vs-realized so "the model is not at fault" is captured, not
  asserted. Quiz Q25/A25/A26 and the recap rows updated.
- **R3-B2 · 01.4 judged its framing gap against a borrowed band, and the verdict flips.** The
  0.0143 classification-vs-regression gap was called "just outside" 01.3's 0.0136 band — but
  that band belongs to the model-vs-rule pair. Measured for the right pair (8 evaluation
  redraws, both framings refitted; corr 0.62 vs the model/rule 0.785), the 2-sigma band is
  0.0197 and the gap sits INSIDE it: the framings are indistinguishable on quality, which
  strengthens the section's real point (27% of the work differs anyway). `framing_band()` is
  now committed and the notebook computes it live. This is invariant 3's own error — "the band
  must match the quantity claimed" — committed by the series a third time.
- **R3-B3 · 01.6 contradicted its own dose-response.** The closing paragraph still said the
  subpopulation shift "repairs calibration and barely moves ranking" while the same section
  reports full-dose contemporaneous training worth +0.0337 — larger than the series' headline
  gain — and 01.1's own regime refit shows +0.036 at matched k. Corrected mechanism: uniform
  shift WITHIN the migrated group preserves within-group order but reorders the groups against
  each other, so seeing the new regime buys a real cross-group ranking gain. Quiz A44 carried
  the same dead story (and Q44 still said "essentially nothing" next to numbers showing the
  opposite); both rewritten.
- **R3-B4 · stale self-contradiction in 01.6.** The wrong-horizon section scored the rule on
  the 30-day question AND kept the prior draft's "note the notebook does not score the rule on
  that question" caveat two sentences later. Caveat deleted.
- **R3-B5 · quiz A43 still taught the refuted leakage mechanism** ("displaced/under-weighted
  signal") that 01.6's B0c fix explicitly rules out; A26 still cited pre-B1 numbers (300
  capacity, 818 at t=0.4, 241 at t=0.6) that no current cell produces. Both rewritten.

**Round-one findings that had never been actioned (now fixed).**

- 01.1's "six weeks" timeline (4 sites + 2 in 01.6 + 1 in quiz) against its own 07-01→07-15
  dates; Prevention's "paged on 2025-07-14, six weeks earlier" (one day earlier).
- 01.1's 0.297 pooled base rate framed as "the number every claim gets measured against"
  (claims are measured against the stable window's 0.263); "twenty-six thousand invoices".
- `make_boosted` docstring claiming native categorical handling while one-hotting.
- 01.1 Interview Q1's garbled precision@k derivation (denominator is fixed at k).
- 01.2 Common Pitfalls attributing the 1,366,049-char string to 288,936 values (it is the
  200,000-row probe's); quiz A13 likewise.
- 01.2's reconciliation presented as "two independent derivations" — both sides draw on the
  same `fx_rate_usd` column; reworded to what it actually validates (aggregation, not rates).
- 01.2's fan-out decomposition not closing (10,961 added vs "10,921 invoices in two
  instalments") — the missing 40 are M1-duplicated rows that also fan out; now printed.
- 01.3's 2026-03 incident quoting 2025 pool numbers as its own months; "Twelve draws" over a
  six-draw cell; `frame_hash` row-order/index dependence (now sorted row-hashes, index=False).
- 01.4's "live since March" vs "live since April 2025" contradiction; capacity stated as 50 in
  prose against 48 in output (unified at 48/day, monthly k = 48 x span); the `km.fit` "no y
  argument exists" comment; latency-ratio prose ("about ten times") vs captured 7.2x — the
  ratio itself swings 7–11x across runs, so timing now uses 7 reps over 5,000 rows and all
  prose states it as an order of magnitude.
- 01.5's "709 ever marked churned before the cutoff" garble; "identical customers" across
  different populations; the right-censored 730-day row (cutoff+730 = 2027-06-30 vs data end
  2026-08-31) now labelled "censored@427d" with the degeneration explained; 90-day lift 0.91x
  called "worse than random" on two hits (now "indistinguishable"); the 1-sigma "real but weak
  edge" verdict (now the series-standard 2-sigma band: inside, not decidable); the dead
  `mrr_usd` placeholder assignment; "value-weighting paid there" for 01.1 (it closed most of
  the gap and still lost).
- 01.6's "would have paged on the day the migration landed" (labels lag; no daily monitor was
  evaluated) and its own "six weeks"; G7 named "calibration" while checking mean-rate match
  (now glossed as a prior-shift tripwire); G5's marginal-band-vs-paired-delta construction now
  named explicitly in prose and justified against the stricter paired band it also clears.
- np.std ddof=0 as a sample-sd estimator in the 01.1/01.5/01.6 noise bands (n=5–8); all bands
  now ddof=1 and every quoted band updated (01.1: 0.0016/$467/$135; 01.6: 0.0024).
- 01.1/01.6 value-band prose quoted 1-sigma ($63) against a printed 2-sigma band; unified.
- June-2025 window gap now noted in 01.1 (and why 01.3 gets to use it).

**Structural items from round one, still deliberately open (need a decision, not a patch):**
Stage C is analysis rather than production code in 01.4/01.5/01.6 with no collapse
justification (01.5's Permanent Fix promises a label-spec artifact no cell builds); 01.4's
Stage A hides its mechanism behind `lab.build_framings()`; interview answer shapes that give
away their numbers (defeats retrieval practice); three notebooks open by positioning against
textbooks; 01.5 re-runs `cost_curve` verbatim in its Production Scenario. 01.4's duplicated
listing was resolved this round by moving the daily block into the scenario.

**→ All five were decided and implemented in round four, below.**

---

## Round four — the structural items, closed (2026-09-01)

Not a review pass: a fix pass against the five items round three left open. The decision on
every one was to build the thing rather than justify its absence, because in each case the
notebook was already *arguing* for an artifact it did not produce.

**Stage C is now production code in all three notebooks.** Each artifact is the permanent fix
its own incident demands, which is why these read as the payoff rather than an appendix.

- **01.4 → `build_capacity_queue`** (`QueueRequest` / `QueueDecision` / `ContractViolation`).
  Roster is an input; the cutoff becomes the *k*-th order statistic and floats. Captured: the
  size guarantee holds on 20 of 20 days against a design that matched the roster on none, while
  the implied cutoff floats 0.3993–0.5528. Setting the risk floor to 0.5 — the cold open's
  deployed threshold — composes the two designs and reproduces the incident from the other
  direction (min 28, roster filled on 2 of 20), with `utilisation` and `floor_rejected` in the
  decision record so a short queue is measured rather than noticed. Boundary cases (length
  mismatch, non-finite score, zero roster) raise instead of returning a plausible list.
- **01.5 → `LabelSpec`**, the artifact the Permanent Fix had promised for three drafts. Frozen
  (a mutable spec's fingerprint lies), fingerprinted over the definition only (re-measuring must
  not read as redefining), with `apply()` as the sole supported way to build the label,
  `assert_scorable()` as the acceptance test the Prevention list asks for, and
  `manifest_entry()` so a base rate cannot travel without its population and horizon. The
  validator rejects the campaign's own spec on two clauses and the 730-day variant on
  right-censoring.
- **01.6 → `release_gate()` / `release_gate_cli()`**. Returns a dict, so the same function backs
  CI, the deploy path and the notebook; blocking vs advisory (G7 is advisory — it is a mean-rate
  tripwire, not a correctness check); JSON verdict; exit code. The load-bearing field is
  `ungated_risks`, printed on every run including all-green ones, which turns detection distance
  into something a reviewer sees rather than remembers.

**The other four.** 01.4's Stage A now builds the four targets by hand in a visible cell before
delegating the fits, so the shared-target/different-contract point is shown rather than asserted.
01.5's Production Scenario stops re-running `cost_curve` verbatim and instead runs the Stage C
gate against the spec the campaign actually shipped on — new evidence, and it closes the incident
with the artifact from two sections earlier. Interview answer shapes in 01.4 (Q5, Q6), 01.5 (Q1,
Q4, Q7) and 01.6 (Q6) no longer hand over their measured values; they name the quantity and say
where to get it. 01.4's Concept no longer opens against textbooks, and 01.3's "usually taught as
hygiene" was recast to position against how teams *budget* it — one instance of the move, not
three.

**Two defects found while doing it**, both of the class the earlier rounds kept hitting:

- **Two definitions of "when the data ends" in 01.5.** `horizon_tradeoff` hardcoded
  `2026-08-31` (giving the 427-day observable horizon in the prose) while a measured
  `inv["issue_date"].max()` gives 2026-08-28. Now a single `DATA_END` constant citing
  `_data/SPEC.md`, used by both; `horizon_tradeoff`'s output is byte-identical.
- **The latency-ratio range in `lab_01.4` was falsified by its own rerun.** The docstring claimed
  7–11× across reruns; this execution produced 15.7×. Widened to 7–16× with an instruction to
  widen rather than tighten. All prose already said "an order of magnitude", so nothing else
  moved. Also a leftover 1-sigma verdict in 01.5's Evaluation and interview Q4, where the rest of
  the series had moved to the 2-sigma band.

**Verification.** Outputs were snapshotted before the pass and diffed after, keyed by cell source
so insertions do not shift the comparison. Result: **six new cells, and exactly one existing
output changed** — `production_costs`' wall-clock microseconds, which the notebook already states
are machine- and run-dependent (the artifact figures, 3.8 KB / 5,003.5 KB / 1,326×, are
unchanged). Every other captured number in the series is byte-identical, so no companion-document
drift was introduced. `daily_queue_swing` was re-verified byte-identical after its shared-setup
refactor. `check.py`: 0 fail, 5 warn across 15 files — the same five previously triaged, each
re-confirmed as a cross-reference to another notebook's captured output (01.6's 0.521/0.557 are
literally printed by 01.1) or an illustrative round number. `_quiz.md` gained Q48–Q50 with
answers; `_recap.md` and `_lab/README.md` updated.

**→ Round four independently re-verified and signed off, 2026-09-02.** All three labs re-run
standalone (exit 0); `daily_queue_swing` and `horizon_tradeoff` confirmed byte-identical after
their refactors; every number in the round-four summary checked against captured output
(including the floor-composition cross-check: the floor-0.5 queue on 2026-04-13 returns 37,
identical to the threshold design's count, as the algebra requires); `build_capacity_queue`'s
floor-inside-top-k selection verified equivalent to top-k-over-floored; the shared manifest
hash between the clean and wrong-horizon runs verified from `build_run`'s fingerprint fields;
`check.py` re-run at 0 fail / 5 triaged warns. Series 01 promoted to `[x]` in CURRICULUM.md
and stands as the track's conformance benchmark.
