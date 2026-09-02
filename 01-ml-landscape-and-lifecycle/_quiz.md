# Series 01 — Retrieval Quiz

## How to use

1. Before reading a notebook's Mental Model, write your own in one sentence. Compare after.
2. Answer every question **aloud, from memory**, before scrolling to `## Answers`. Recognizing
   a correct answer is not the same skill as producing one, and only the second one transfers
   to an interview or an incident.
3. At least once per series, re-implement one Stage A build from memory on a blank file —
   here, the dunning rule plus a precision@k function — and only then diff it against the
   notebook.
4. Re-attempt this quiz a week later. Spaced, interleaved retrieval is what produces
   retention; re-reading produces the feeling of retention.
5. Questions are interleaved across notebooks on purpose and grow as the series does.

---

## Questions

**Q1.** PayFlow's collections dashboard shows the late-payment model's precision rising from
0.472 to 0.570 over the two weeks after a payment-gateway migration, while overdue balance
climbs and the CFO escalates. Explain how
both facts can be true simultaneously, and name the one metric that would have made the failure
visible on day one. *(01.1 — debug this)*

**Q2.** A colleague reports "the new model gets 0.446 precision, the old rule got 0.425, so we
should ship it." List everything still missing before that claim supports a decision. *(01.1)*

**Q3.** Derive why precision@k increases when the base rate increases, holding the model's
ranking of invoices completely fixed. Then state the consequence for drift monitoring.
*(01.1 — derive this)*

**Q4.** In the PayFlow late-payment build, the model beat the rule on precision and lost to it
on value captured. Give the mechanism in one sentence, and give the one-argument change that
closed most of the gap. *(01.1)*

**Q5.** Two columns in the PayFlow export would each improve every offline number and destroy
the model in production: name them and state, for each, what its value would be at the instant
the scoring decision is actually made. *(01.1)*

**Q6.** Design the dunning queue system end to end for a team that can work 48 invoices a day.
Include what runs in production on day one, what runs in shadow, and the specific condition
that would flip the decision later. *(01.1 — design this)*

**Q7.** The bootstrap gave a precision noise band of 0.0016 and a value noise band of $467; the
value-weighted model's deficit against the rule was $228 with a band of $135. The checker calls
that SIGNAL. Should you act on it? Justify. *(01.1)*

**Q8.** Mini coding challenge. From memory, write `precision_recall_at_k(y, flag)` returning
precision and recall for a boolean queue mask, then write the three-clause PayFlow dunning rule.
Then answer: why must the model be evaluated at the rule's k rather than at its own best
threshold? *(01.1)*

**Q9.** During the incident, schema checks, null-rate monitors and feature-distribution
monitors all stayed green while the model was badly wrong. Name the failure mode and explain
why that entire class of monitor is structurally blind to it. *(01.1)*

**Q10.** A colleague joins `invoices` to `payments` and reports "row count changed by −0.55%,
so the join is fine." Explain why that sentence is not evidence of anything, and give the check
that is. *(01.2 — debug this)*

**Q11.** Derive the row count of an inner join between a table at grain A and a table whose
join key has average multiplicity *m* over matched keys, then show why the net change in row
count cannot detect fan-out. *(01.2 — derive this)*

**Q12.** `invoices.amount` sums to 15,897,163,634 the naive way and 464,644,960 correctly.
Name the mechanism in one sentence, and explain why no null check, schema validator or
exception handler in the pipeline objected. *(01.2)*

**Q13.** Under pandas 3, calling `.sum()` on the raw `amount` column returns something that is
not a number and raises nothing. What does it return, why, and what single assertion at read
time prevents the whole class of failure? *(01.2)*

**Q14.** A teammate trains a spam classifier on `tickets ⋈ customers` and reports excellent
held-out scores. Predict what happens in production and explain precisely why the offline
evaluation could not have caught it. *(01.2)*

**Q15.** Distinguish a *convention* from a *constraint* using the mixed-currency bug, and state
the rule you would use to decide which one a given correctness requirement deserves. *(01.2)*

**Q16.** Mini coding challenge. From memory, write a `grain_check(df, key)` that reports whether
a table honours its declared grain and returns the offending keys; then write the `Money` type
whose `__add__` refuses mismatched currencies. *(01.2)*

**Q17.** You reconcile billed against collected revenue and the two differ by +0.55%. Your
colleague wants to accept it because it is small. What is the correct standard for accepting a
residual, and what would make a 0.1% gap *less* acceptable than this one? *(01.2)*

**Q18.** A pipeline sets `random_state=42` on its `LogisticRegression` and on nothing else. The
team believes its runs are reproducible. Explain what is actually pinned, what is not, and the
four-line experiment that settles it. *(01.3 — debug this)*

**Q19.** The same pipeline reports a standard deviation of 0.0032 when you resample the training
set with the holdout fixed, and 0.0101 when you re-draw the holdout. A fourth number, 0.0068, is
the spread of the model-minus-rule *difference* on those same re-drawn holdouts. Explain all
three, and say which one a shipping decision should be judged against. *(01.3 — derive this)*

**Q20.** A colleague rolls back by checking out December's commit and rerunning training. State
every reason this can fail to reproduce December's model, and give the artifact-level fix.
*(01.3)*

**Q21.** Two model artifacts have different content digests, score identically on a clean
holdout, and disagree on 0.4% of queue decisions. Are they the same model? What should the
digest trigger, and what should the disagreement rate decide? *(01.3)*

**Q22.** Design the reproducibility contract for a team shipping a model monthly: what each run
records, what is stored, what CI enforces, and what a reviewer must see in a pull request before
approving a claimed improvement. *(01.3 — design this)*

**Q23.** Your training pipeline is byte-reproducible on your laptop across repeated runs. Name
what is still not guaranteed on a colleague's machine and what you would add to the manifest to
close the gap. *(01.3)*

**Q24.** Mini coding challenge. From memory, write `score_hash(scores)` and a `reproduce()` that
rebuilds a manifest and returns the list of fields that moved. Then explain why the manifest
must keep inputs and outputs in separate blocks. *(01.3)*

**Q25.** A daily work queue built by thresholding a probability delivers between 28 and 53 items
a day to a team staffed for 48 — starving the roster most days, flooding it on others, and
matching it on none — while the model dashboard stays green and no deploy has happened.
Name the failure, and give the one-line change that fixes it today. *(01.4 — debug this)*

**Q26.** Derive why a fixed probability threshold cannot guarantee a fixed queue size, and state
the quantity a capacity-based selector fixes instead. *(01.4 — derive this)*

**Q27.** The same features and rows framed as regression and as classification score within
0.0143 of each other on precision but disagree on 26.9% of selected invoices. What do you
conclude, and what should decide the choice? *(01.4)*

**Q28.** Your dispatch code branches on `hasattr(model, "predict_proba")`. Explain, with the
`SGDClassifier` example, why this is fragile and what to branch on instead. *(01.4)*

**Q29.** Only 21.2% of the most recent month's invoices have a resolved outcome, against 95.7%
overall. State two consequences for a supervised system and one thing this makes unsupervised
methods genuinely good for. *(01.4)*

**Q30.** A KNN prototype beats logistic regression on offline metrics. List what you check before
proposing it for production, with the numbers from this series where you have them. *(01.4 —
design this)*

**Q31.** Mini coding challenge. From memory, write `topk_flag(scores, k)` returning a boolean
mask over the k highest scores, then explain why the collections queue should be built from it
rather than from `scores >= t`. *(01.4)*

**Q32.** Derive the break-even precision for a retention offer, then show what changes when the
offer is a fixed fraction of the customer's revenue rather than a flat per-contact cost — and
explain what that implies for value-weighted targeting. *(01.5 — derive this)*

**Q33.** A campaign spends its full budget, moves no metric, and the model dashboard is green
throughout. Diagnose, in order. *(01.5 — debug this)*

**Q34.** The same PayFlow customers support churn base rates from 0.0076 to 0.1149. Name the two
framing clauses responsible, and explain why the difference matters more to the business case
than to the ranking metric. *(01.5)*

**Q35.** Why may a label look into the future while a feature may not? *(01.5)*

**Q36.** Your operational label has too few positives to fit, and a denser proxy label ranks
better on the operational question. What do you do with it, and what must you never do with it?
*(01.5)*

**Q37.** Write the label spec for "reduce churn" at PayFlow, justifying each clause, and state
the acceptance test that would have caught the population error. *(01.5 — design this)*

**Q38.** Your company records no cancellation event. Build a churn label from activity data and
state precisely what it costs you in accuracy and in lag. *(01.5)*

**Q39.** Mini coding challenge. From memory, write `label_frame(customers, cutoff, horizon,
active_only)` returning the modelling frame with a `churns_in_horizon` column, and say which line
enforces the population clause. *(01.5)*

**Q40.** A model beats its baseline offline, reproduces exactly, is well calibrated, and produces
no business effect over four quarters. Every CI gate is green. Diagnose. *(01.6 — debug this)*

**Q41.** Define detection distance, and explain why it predicts the cost of a failure better than
the number of defects does. *(01.6 — derive this)*

**Q42.** Why can a test set fail to reveal a survivorship-biased training population, and what
single artifact would make the bias checkable? *(01.6)*

**Q43.** A leaked feature raised offline precision from 0.4474 to 0.7572. Explain why production
precision (0.2949) ended up *worse* than the clean model's, rather than merely no better.
*(01.6)*

**Q44.** Random splitting on time-ordered data measured 0.5232 with an honest split against
0.5569 at full contamination — a gap larger than the series' headline model-over-rule gain —
after a diluted version of the same experiment had made it look like nothing. When does the
textbook warning actually bite, and how would you measure it for your own problem without
confounding the comparison? *(01.6)*

**Q45.** A team hands you a model with a fully green gate suite and asks whether to ship. Name the
checks the suite structurally cannot perform, and what you would require before approving.
*(01.6 — design this)*

**Q46.** Mini coding challenge. From memory, write the calibration gate and the serving-contract
gate as functions over a run object, then state which lifecycle stage each one guards. *(01.6)*

**Q48.** `build_capacity_queue` returns at most `roster` ids and raises if it ever would return
more. A probability threshold cannot make that promise at any value of *t*. State the reason in
terms of the score distribution, say what becomes the free variable once the count is fixed, and
explain why setting the risk floor to the old deployed threshold reproduces the original
incident. *(01.4 — derive this)*

**Q49.** `LabelSpec` is a frozen dataclass whose fingerprint deliberately excludes its measured
base rate. Give both reasons. Then name the two clauses of the campaign's own spec that the
validator rejects, and say which different clause the 730-day variant trips instead. *(01.5)*

**Q50.** `release_gate()` returns an `ungated_risks` field on every run, including runs where all
seven gates pass. Argue for that field: what does a reviewer learn from it that a green board
does not tell them, and why can none of the three listed risks be rewritten as an assertion over
the run object? *(01.6 — design this)*

**Q47.** Series-wide design prompt. PayFlow's VP Engineering asks you to write the one-page
policy that governs when a team in your organization is allowed to replace a business rule with
a trained model, and what data guarantees must hold before any model is trained at all. Write
it: the evidence required, who measures it, what stays in production during the trial, the
ingestion contract that gates training data, and what triggers a rollback.
*(01.1 + 01.2, forward-looking to 12 and 34)*

---

## Answers

**A1.** Precision is computed only over the invoices the queue actually chased, so it rises
mechanically when the positive class becomes denser — the migration raised the true late rate
from 0.263 to 0.389. Meanwhile the score distribution never moved (predicted 0.268 → 0.264), so
a fixed threshold selected a near-constant queue share (21.2% → 20.1%) against far more late
invoices, and recall fell from 0.381 to 0.294. The visible-on-day-one metric is **calibration**:
mean predicted rate versus observed rate, whose gap went from −0.004 to +0.125. *(01.1)*

**A2.** The random baseline at the same k (0.265, i.e. the base rate); confirmation both were
measured at the same operating point; a variance estimate from repeated refits (±0.0008, band
0.0016); the business objective alongside the proxy — where the rule actually wins, $51,362
against $49,665; a temporal rather than random split; and the operating cost of owning the model.
*(01.1)*

**A3.** Write precision@k for a fixed ranking as the expected share of positives among the top k.
Scaling the positive class density scales the expected number of positives in any fixed-size
slice, so the numerator grows while k stays fixed — the metric moves with no change in ranking
quality. Consequence: precision is not a drift detector, and a monitoring set built only from
self-computable metrics has a structural hole. *(01.1)*

**A4.** The model was trained on a 0/1 label in which a late $90 invoice and a late $9,000
invoice are the same event, so it ranks by probability and is blind to amount, while the rule's
first clause thresholds on amount and therefore hunts money — visible in the mean amount of late
invoices caught, $6,178 for the rule against $5,631 for the model. The one-argument change is
`sample_weight=amount_usd`, which moved value captured from $49,454 to $51,134. *(01.1)*

**A5.** `reminder_count`, written only after the payment resolves — at scoring time on a
freshly issued invoice it is always 0. And `total_lifetime_value_usd`, computed over all time
including revenue after the prediction window — at scoring time only the history so far exists.
Both are SPEC M10 leakage traps. *(01.1)*

**A6.** Day one: the rule in production as the queue policy, selecting top-k under the 48/day
capacity rather than by a fixed probability threshold. Shadow: the value-weighted model scoring
every invoice and logging its queue without acting. Guard: fall back to the rule whenever the
calibration gap exceeds 0.05. Flip condition: shadow model value captured exceeds the rule's by
more than the bootstrap band for four consecutive weeks, or the feature set grows beyond what
three clauses can express, or queue capacity changes materially — since the whole comparison is
conditioned on k. *(01.1)*

**A7.** No. The deficit is outside the noise band, so it is statistically detectable, but it is
under half a percent of the objective and points the wrong way for shipping. Statistical
significance answers "is this difference real"; practical significance answers "is it worth
anything" — and here the honest reading is that a well-specified model draws level with the
whiteboard while adding retraining, monitoring and on-call cost. *(01.1)*

**A8.** `precision = (flag & (y == 1)).sum() / max(flag.sum(), 1)`;
`recall = (flag & (y == 1)).sum() / max((y == 1).sum(), 1)`. The rule is
`(amount_usd > 1500) | (segment == "Enterprise") | (country == "AE")`. Matched k is required
because precision and recall both move with queue size: comparing a model at its most flattering
threshold against a rule at its natural flag rate measures appetite for queue size, not skill.
*(01.1)*

**A9.** Concept drift, i.e. posterior shift — P(y|x) changed while P(x) did not. (The name
"label shift" belongs to the opposite case, where the class prior P(y) moves with P(x|y) fixed;
here the prior moved as a consequence, not a cause.) The gateway migration altered the
outcome-generating process for non-Indian customers (median days-late 2 → 6) while every input
column kept its distribution, so any monitor watching inputs alone cannot see it. Detecting it
requires comparing predictions against realized outcomes: calibration and recall against delayed
ground truth. *(01.1)*

**A10.** Because two large opposite effects cancelled: 12,564 invoice rows were dropped for
having no payment and 10,961 rows were added by fan-out — 10,921 invoices paid in two
instalments, plus 40 M1-duplicated invoice rows whose ids also split — 23,525 rows moved to
produce a net of −1,603. The real check is key cardinality:
`validate="m:1"` on the merge, which raises when the right-hand key is not unique, or
aggregating the many-side to the join grain first. *(01.2)*

**A11.** The result has one row per (left row, matching right row) pair, so its size is the sum
of right-side multiplicities over matched keys. Unmatched left rows subtract from the count while
multiplicities above one add to it, and the two terms are independent — so their sum can be near
zero for arbitrarily large individual effects. Net row count is therefore uninformative about
either. *(01.2)*

**A12.** `amount` is denominated in each customer's local currency, so summing it adds numbers
with different units; INR dominates because it is both the highest-count currency and numerically
the largest per unit, contributing 97.7% of the raw total against 38.0% of the real one. Nothing
objected because every value was present, well-typed as text, and individually valid — unit
correctness is a semantic property that no null check or schema validator represents. *(01.2)*

**A13.** It returns a `str`: the values concatenated — a 200,000-row probe of the column comes
back as a 1,366,049-character string, and the full column concatenates the same way, only
longer — because the comma-formatted legacy rows make the column `StringDtype` and `+` on
strings is concatenation. The preventing assertion is an explicit dtype check after read — assert the column
is numeric (or pass an explicit `dtype`/converter) rather than inferring type from the column's
name. *(01.2)*

**A14.** It fails immediately in production. The inner join deletes rows with a null
`customer_id`, and all 2,064 dropped tickets are spam, so the training data has had most of its
positive class removed. The offline evaluation cannot catch it because the test split is drawn
from the same joined frame and inherits the identical filter — the evaluation and the training
share the bug, which is why a data-level contract has to sit upstream of the split. *(01.2)*

**A15.** A convention is a rule people are expected to remember — "convert before summing" — and
it fails at the first new hire, rushed deadline or copied query, silently. A constraint makes the
wrong operation impossible: the `Money` type raises `CurrencyMismatch` at the offending line.
Prefer a constraint wherever the cost of being wrong is high *and* the error is silent; a
convention is acceptable only when violations are loud and cheap. *(01.2)*

**A16.** `grain_check` compares `len(df)` with `df[key].nunique()` and returns
`df[key].value_counts().loc[lambda s: s > 1].index` as the offenders. `Money` is a frozen
dataclass of `(Decimal amount, str currency)` whose `__add__` raises `CurrencyMismatch` unless
currencies match, with an explicit `to_usd(rate)` conversion — Decimal rather than float because
money is compared for exact equality. *(01.2)*

**A17.** Size alone is not the standard; a *mechanism* is. The +0.55% residual is acceptable
because settlement-time rates demonstrably disperse around the median-rate table as FX moves
between issue and settlement, which predicts a small signed gap of roughly that magnitude —
though note what the check can and cannot validate: both sides draw on the same rate column,
so it proves the aggregation (join, dedup, no unconverted currency), not the rate source. A 0.1% gap with no explanation is worse, because an
unexplained residual means an unknown process is acting on the number and nothing bounds how
large it becomes next month. *(01.2)*

**A18.** Nothing that matters is pinned. `LogisticRegression` with the lbfgs solver is
deterministic, so its `random_state` is inert — the digest is unchanged across seeds 42 and 43.
What is unpinned is row selection: `train_test_split`, `DataFrame.sample` and any stochastic
solver such as `SGDClassifier` all change the answer. The experiment: run each component twice
with two seeds and compare content digests of its output; seed sensitivity is a property of the
algorithm, not of the parameter's presence. *(01.3)*

**A19.** They measure different things. Resampling the training set perturbs the fitted
coefficients, and with a six-figure training pool the fit is well determined, so the effect is
small (0.0032). Re-drawing the evaluation rows changes which 5,500 invoices are scored, and
sampling noise on a proportion scales roughly as the inverse square root of the queue size, so a
small holdout dominates (0.0101). ⚠️ But neither is automatically the band to judge a *claim*
against: the claim is a model-minus-rule difference on identical rows, and because both policies
move together on the same draw (correlation 0.785) its variance is `Var(A)+Var(B)−2·Cov(A,B)` —
here 0.0068, giving a two-sigma band of 0.0136 that 01.1's +0.0194 clears. Match the band to the
quantity claimed; judging a paired gain by a marginal spread rejects supported results. *(01.3)*

**A20.** A commit reproduces the process, not the object. The data may have grown, since a live
query with no snapshot bound makes the run date a hidden input; the split may be re-drawn; a
stochastic component may be unseeded; library versions may have moved. Fix: store the fitted
artifact immutably, keyed by its content digest, with a manifest recording code versions, data
hash and row count, config and result — then rollback is a lookup and a fetch, and rebuilding
from source becomes a scheduled audit rather than an incident-path activity. *(01.3)*

**A21.** Not the same artifact; effectively equivalent in behaviour. Identity is stricter than
equivalence. The digest should trigger an investigation — something in the inputs changed — and
the disagreement rate should drive the decision about whether it matters. Conflating the two
produces either false alarms on every retrain or missed drift when a small digest change hides a
large behavioural one. *(01.3)*

**A22.** Each run writes a manifest with separate blocks for code (library versions, git SHA),
data (content hash and row count of the exact slice), config (seed, split id, features,
hyperparameters) and result (artifact digest, prediction digest, metrics). Artifacts are stored
immutably and keyed by digest. The evaluation set is fixed and versioned rather than drawn. CI
runs `reproduce()` on the training path and fails on any field diff. A pull request must show a
manifest and the noise band beside any claimed delta — a bare metric is not reviewable, because
a reviewer cannot distinguish a real improvement from a lucky draw. *(01.3)*

**A23.** Bitwise equality across machines additionally depends on BLAS threading, CPU
instruction sets and library builds, none of which are fixed by seeds. Record library versions,
thread counts and ideally a container image digest, and state reproducibility as a property of a
declared environment rather than an absolute. *(01.3)*

**A24.** `score_hash` rounds the prediction vector to a fixed precision and takes a SHA-256 of
its bytes, returning a short prefix. `reproduce()` rebuilds the manifest from the recorded
inputs and returns a recursive field-wise diff against the stored one. Inputs and outputs live
in separate blocks so that an inert configuration change — a seed that this estimator ignores —
appears as a config-only difference instead of being mistaken for a reproduced experiment, and
so that a changed result can be attributed to the specific input that moved. *(01.3)*

**A25.** A framing mismatch, not a model fault: the queue is selected by thresholding a
probability, so its size is the mass of the score distribution above the cutoff and moves with
that distribution and with each day's invoice mix, while the business constraint is a fixed
headcount. Today's fix is to replace `scores >= t` with top-*k* selection at the team's capacity
— one line, no retraining. ⚠️ Be careful what you claim for it: precision *falls* (0.5718 to 0.4390) because the queue is
larger, and the win is that the queue is filled at all — 590 late invoices caught against 215,
nearly triple, using capacity already being paid for. *(01.4)*

**A26.** Queue size under a threshold is the count of scores at or above *t*, which equals *n*
times the survival function of the score distribution at *t*; both *n* and that distribution move
with customer mix, seasonality and drift, so the size is an output. Capacity selection fixes the
count and takes the threshold as the *k*-th order statistic of the scores, letting it float. The
two are duals and only the second matches a headcount — which is why a fixed t=0.5 delivered
queues from 28 to 53 across twenty days against a 48-slot roster, matching it on none. *(01.4)*

**A27.** That aggregate quality and operational agreement are different things. The precision gap
sits inside the noise band established in 01.3, so on that evidence the framings are
indistinguishable in quality — yet more than a quarter of the analysts' day goes to different
invoices. The choice should therefore be made on the serving contract the system must honour and
on which ordering better matches the business objective, not on the summary metric. *(01.4)*

**A28.** `predict_proba` on `SGDClassifier` is gated by `available_if` on the loss function: it
exists for `log_loss` and genuinely does not exist for `hinge`. Branching on method presence makes
behaviour depend on a hyperparameter chosen elsewhere, so the same class silently takes different
paths. Branch on an explicit, recorded framing — the config block from 01.3 — rather than on
introspection. *(01.4)*

**A29.** Consequences: a supervised model cannot learn about very recent conditions, because the
rows it must score are the least labelled, which bounds how fast it can react to a regime change
like the gateway migration; and any evaluation restricted to resolved invoices is a biased sample
of recent activity, since fast-paying invoices resolve first. Unsupervised methods are genuinely
good for acting on the unlabelled present — segmentation, anomaly detection on live invoices —
where waiting for outcomes is not an option, at the cost of having nothing to be correct against.
*(01.4)*

**A30.** Artifact size and how it grows with the dataset (5,003.5 KB against 3.8 KB on only 20,000
rows here); per-row prediction latency against the serving budget (an order of magnitude
more than the parametric model); memory in the serving tier; retrain and redeploy cost, since every retrain
ships the data again; and whether shipping training rows into production crosses a privacy or
trust boundary, because the artifact contains customer records. *(01.4)*

**A31.** `out = np.zeros(len(scores), dtype=bool); out[np.argsort(-scores)[:k]] = True; return
out`. The queue should be built from it because top-*k* guarantees the queue size the team can
actually work, requires only a correct ordering rather than calibrated probabilities, and is
immune to score-distribution drift — whereas a threshold makes the size an uncontrolled output.
*(01.4)*

**A32.** Expected value of one contact is `p·s·V − c` for churn probability *p*, save rate *s*,
retained value *V* and offer cost *c*, so break-even is `p* = c/(s·V)` — here
`2/(0.30 × 12) = 0.5556`. When the offer is proportional to MRR, both *c* and *V* scale with MRR,
the term cancels, and `p*` is the same constant for every customer; weighting the ranking by MRR
then buys more expensive customers without improving the ratio, taking the net from −$20,433 to
−$355,964. Under a flat cost MRR does not cancel, high-MRR customers genuinely have a lower
break-even, and the same weighting is worth $107,033. *(01.5)*

**A33.** Check the intervention economics first: break-even precision (0.5556) against achievable
precision (0.0739) shows whether any targeting quality could have paid. Then audit the scoring
population for already-resolved cases (415 of 709 naive positives had already churned). Then audit
the label for a missing horizon or cutoff. Then check whether the business case's base rate
matches the operational label's — 0.1149 against 0.0207, a 5.6x overstatement. A green dashboard
is consistent with every one of these, because it measures ranking within a label. *(01.5)*

**A34.** Horizon and population. Horizon moves it most — 0.0082 at 90 days against 0.0445 at 365
on the same active customers — and the naive "ever churned" label removes the time bound entirely
to give 0.1149. Ranking metrics are computed within a single label and so are blind to the
difference, while a business case multiplies the base rate by the customer base to size the prize;
the framing error therefore survives a perfectly good model and shows up only in the money. *(01.5)*

**A35.** The label defines the event being predicted, which by construction lies after the
decision point — looking forward is what makes it a label at all. A feature must be knowable at
decision time, because in production nothing after the cutoff exists yet. Violating the feature
direction is leakage; the label direction is the premise of supervised learning. *(01.5)*

**A36.** Use it for ranking, since scarce positives are a real constraint and the denser label
ranked slightly better here — though the edge sits inside the two-sigma resplit band, so it is
weak evidence, merely free. Recalibrate
its scores to the operational base rate before feeding any expected-value computation. Never let
its base rate reach a business case: inflated 5.6x, it sizes a prize that does not exist. *(01.5)*

**A37.** Unit: one row per customer. Population: customers active at the cutoff, excluding anyone
already churned. Cutoff: a fixed instant, with all features computed strictly before it. Event: an
explicit `churn_date` within a 180-day horizon — the shortest window with enough positives to fit
and short enough for a quarterly campaign to influence. The spec and its measured base rate are
recorded in the run manifest's config block. The acceptance test asserts that no customer with a
`churn_date` at or before the cutoff can appear in a scoring or contact list. *(01.5)*

**A38.** Infer churn from silence — no invoice in a trailing quiet window. It costs accuracy: the
60-day rule catches 85 of 119 true churners here, with no false positives at that threshold, so it
under-detects. And it costs lag: the last invoice arrives a median 15 days before the churn event
and the quiet period must then elapse, making churn declarable a median 45 days late. That lag is
part of the system's reaction time and belongs in the design. *(01.5)*

**A39.** Filter to `signup_date <= cutoff`, and when `active_only` additionally require
`churn_date.isna() | (churn_date > cutoff)` — that condition is the population clause. Then set
`churns_in_horizon = churn_date.notna() & (churn_date > cutoff) & (churn_date <= cutoff +
horizon)`. The population line is the one that decides whether the model predicts a transition or
recognises a state. *(01.5)*

**A40.** Use the green suite as the diagnostic: it eliminates the whole class of implementation
defects, so the remaining hypotheses are all about what the system was asked to do. Then compare
the label's definition against the operational decision it feeds (a 7-day label against a 30-day
escalation gave 0.0513 rather than the reported 0.4474); audit the training population for a
silent join filter (567 never-paid invoices absent from every metric); and check whether the
intervention could pay at achievable precision. The absence of a bug is the clue, not a dead end.
*(01.6)*

**A41.** Detection distance is the gap between the stage where a defect enters and the earliest
gate able to observe it. Rework cost grows with the number of downstream artifacts already built
on the flawed decision, so a defect caught at its own stage costs an afternoon while one with no
offline detector costs however long until a business review notices — fourteen months in the
case study. Defect *count* says nothing about this, since five caught defects cost less between
them than one invisible mis-specification. *(01.6)*

**A42.** Because the test set is drawn through the same join and inherits the identical filter:
the metric is correct about the joined population and silent about the difference between it and
the population actually served. The artifact that makes it checkable is a written
scoring-population definition — once "who is eligible to be scored" is recorded, a gate can
assert that the training population matches it, which is impossible while the population is
defined only implicitly by a join. *(01.6)*

**A43.** Rule out the intuitive story first: at serving time the leaked column is 0 on every
freshly issued invoice, so it adds an identical constant to every logit — a pure intercept
shift — and precision@k ranks by the linear predictor, which is invariant to that. "The honest
features got under-weighted" would therefore cost exactly nothing and cannot be the mechanism.
What actually breaks: fitted beside a column that already encodes the outcome, the honest
coefficients become effects *conditional on* that column — their relative sizes change and
`terms_days` flips sign (+0.1517 → −0.5116) — so production receives a *different* ranking,
slightly anti-correlated with the clean one (Spearman −0.18), not a weaker version of the same
one. That is why it lands below the clean model rather than merely level with it. *(01.6)*

**A44.** It bites when the feature-outcome relationship changes over time, not when only the
overall base rate moves — and "changes" includes a level shift confined to a subpopulation.
Here the migration moved `P(late|x)` for the non-Indian rows `country` already identifies:
roughly uniform *within* the group, so within-group ordering survived, but a group-specific
shift reorders the groups *against each other*, and a model that never saw the new regime
under-ranks the whole migrated subpopulation. That cross-group term is what the measured
+0.0337 buys (0.5232 honest against 0.5569 at full dose, on identical test rows) — the same
effect as 01.1's regime refit. Measure it by holding the test rows fixed and varying only
whether training saw contemporaneous data, and *control the contamination share*: pooling and
sampling diluted the treatment to ~23% and made it look negligible, while comparing two
different test sets confounds the effect with a base-rate difference and can even show the
wrong sign. *(01.6)*

**A45.** The suite cannot check intent. Require: the label spec, with its horizon justified
against the operational trigger it feeds; the scoring-population definition, compared against
production scoring volume; break-even precision against achievable precision; a pre-registered
business metric with a review date; and a baseline scoring in parallel as a control. Each of
these is an artifact first and a gate second — the check becomes writable only once the decision
is written down. *(01.6)*

**A46.** Calibration gate: `abs(run.predicted_rate - run.actual_rate) <= 0.05`, guarding the
monitor stage. Serving-contract gate: `run.queue_size == run.capacity`, guarding the ship stage.
Both are assertions over recorded fields, which is precisely why they can be automated and why
gates over unrecorded intent cannot. *(01.6)*

**A48.** Queue size under a threshold is the row count times the survival function of the score
distribution evaluated at *t*, so it moves whenever the day's volume moves or its score
distribution moves — and in production both move every morning. Fixing the count instead makes
the cutoff the free variable: it becomes the *k*-th order statistic, which floated between
0.3993 and 0.5528 across the twenty days measured. Setting the risk floor to 0.5 composes the two
designs, so on any day when fewer than 48 invoices score above 0.5 the floor binds and the queue
collapses back to exactly the threshold design's size — which is why that row reproduces the
incident's minimum of 28 and fills the roster on only 2 of 20 days. The difference is not the
outcome but the reporting: the decision record carries `utilisation` and `floor_rejected`, so a
short queue is a measured state rather than something the analysts discover. *(01.4)*

**A49.** Frozen because a spec that can be mutated in place is a spec whose fingerprint lies —
the hash would no longer identify what was actually applied. The measured base rate sits outside
the hash because re-measuring the same definition on more data must not read as a *different*
definition: identity belongs to the definition, while the measurement is an attribute of one
application of it. The campaign's v1 trips two clauses — no horizon, so the target is a state
("is a churner") that can be recognised but not predicted, and a population of all signed-up
customers, which admits the 415 who had already churned at the cutoff. The 730-day variant fixes
both and trips right-censoring instead: the cutoff plus 730 days lands past the extraction date,
so that label is really "churns before the data ends". *(01.5)*

**A50.** A green board is evidence about the checks that exist and silence about everything else,
so a reviewer reading only PASS lines cannot tell "this was checked and is fine" from "nothing
looks at this". Printing the blind spots makes detection distance a field a reviewer sees rather
than a paragraph somebody has to remember. None of the three can be rewritten as an assertion
over the run object because each is a mis-specification — a property of the question that was
asked, not of any artifact the run produced. The run is internally consistent about the label it
used, the population its join produced and the offer it never modelled at all; comparing any of
them against the decision they were meant to serve needs a second document the run does not
contain, which is precisely what 01.5's label spec supplies for the first of them. *(01.6)*

**A47.** A strong answer names: the incumbent rule or heuristic is measured first and becomes the
floor; the comparison happens at a matched operating point on a temporal split; the business
objective is reported next to the training metric; a variance estimate defines the noise band and
the model must clear it on the objective, not the proxy; the model runs in shadow before
production; the baseline keeps scoring forever as a canary; rollback triggers on calibration gap
or on model-minus-baseline value going negative; the training data comes from a contracted
ingestion layer with grain, dtype and referential-integrity checks blocking, so no model is
trained on a frame whose join silently removed a class; and the decision explicitly prices the
cost of ownership, so "do not ship" is an allowed outcome. *(01.1 + 01.2; leakage and validation
discipline are canonical in 12, monitoring in 34)*
