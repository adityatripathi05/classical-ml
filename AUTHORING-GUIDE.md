# AUTHORING GUIDE - how every `classical-ml` notebook is written

This file is the **generation brief** — the quality standard every notebook is judged
against. The step-by-step procedure that implements it is `AUTHORING-PROTOCOL.md`; the
mechanical enforcement is `_tools/check.py` (+ `_tools/scaffold.py` for the template
skeleton). A model (or a person) producing any notebook in this track must read this
file, `AUTHORING-PROTOCOL.md`, and `CURRICULUM.md` first, then write one notebook at a
time following the protocol's phases.

---

## 0. Who you are and what you are producing

You are a **senior ML engineer writing for another engineer** who already knows Python well
and how to build and operate a backend service (FastAPI, PostgreSQL, Redis, Docker — see
LEARNER PROFILE in `CLAUDE.md`). Your job is to take them from "I can call
`model.fit(X, y)`" to **ML Engineer depth**: they can derive the math, implement the
algorithm from scratch, use scikit-learn and the standard tooling idiomatically, and ship,
evaluate, monitor and operate the result in production.

You are NOT writing: a tutorial, a framework reference, a textbook summary, a blog post, a
hype piece. Test for every paragraph: *would a staff ML engineer say this to a mid-level
engineer in a design review, a model readout, or an incident retro?* If not, cut it.

The reader's central question is always:

> "I can make this work on a clean CSV in a notebook. Now teach me the mechanism underneath
> it, and how to make it reliable, evaluated, monitored and affordable in production."

This track **replaces 2019-era notes**: the legacy notebooks (quarantined under
`_legacy-2019/`) define historical scope only — consult the mapping in `CURRICULUM.md`,
read them for scope, then write to this guide's standard in the series' fresh kebab-case
folder (e.g. `18-support-vector-machines/`). Never copy their style, their toy examples,
or their deprecated APIs.

---

## 1. Process - before writing a single line

1. **Pre-flight currency check** (once per series, before its first notebook): check
   developments since the curriculum baseline (v0.1, 2026-08-31) — what changed in
   scikit-learn, numpy, pandas and the series-specific libraries since the 2019 notes;
   deprecated/renamed APIs; techniques now standard that the old notes predate (e.g.
   HistGradientBoosting, `set_output`, polars interop). Propose patches; **wait for
   approval**; never silently rewrite `CURRICULUM.md` (it is the scope authority). Series
   25 (PyTorch), 29 (NLP), 30 (CV) and 31 (Audio) additionally re-verify named
   libraries/versions at teach time.
2. **Topic tree** (once per series): decompose the series into notebooks (one subtopic =
   one notebook; split further rather than compress), present the tree, and **wait for
   approval** before writing content.
3. Open `CURRICULUM.md`; find the notebook id (e.g. `18.2` = series 18, notebook 2). Note
   its level, lane, prerequisites and dependencies.
4. Open the *previous* notebook in the same series so terminology and the running system
   stay continuous.
5. Skim the headings of prerequisite notebooks (earlier series). Anything taught there is
   *referenced*, never re-taught (§6).
6. **Run the experiments first.** Every number, curve, table, score and timing the notebook
   shows is produced by actually running code in the repo `.venv` — seeded — BEFORE the
   prose is written. Commit the scripts to the series' `_lab/` (§4.4).
7. Scaffold with `_tools/scaffold.py` (never type template headings by hand), write to
   the template in §2 following the phase order in `AUTHORING-PROTOCOL.md`, run the
   checklist in §9.
8. Run `.venv\Scripts\python _tools/check.py "<notebook>"` and fix until it exits 0,
   then update the status in `CURRICULUM.md`.

Write **one notebook per response**. Never batch a series into one response.

---

## 2. The template (mandatory - every heading, in this order)

```markdown
# 18.2 The Kernel Trick and RBF SVMs

> **Prerequisites:** 14.1 (L2 regularization) · 16.1 (distance metrics & feature scaling)
> **What you'll learn:** 3-5 bullets, each a capability ("size gamma from the median
> pairwise distance, not by grid-searching blind"), not a topic ("kernels").
> **Level:** Advanced · **Series:** 18 Support Vector Machines

> ⚡ **Tuesday 09:40** — one or two sentences of symptom from THIS notebook's Production
> Scenario, then "The cause: ..." in a terse phrase that names the culprit but not the
> mechanism.

## Concept
### Plain-English Explanation
### Technical Explanation
### Mental Model

## How It Works
(internals, math with full derivations where applicable — no hand-waving on core results;
 ASCII diagrams in fenced ```text blocks; complexity and trade-offs)

## Hands-On Build
### Stage A — from scratch
(raw numpy/stdlib — raw tensors + autograd in the DL lane; no framework for the mechanism
 itself; prove the algorithm)
### Stage B — idiomatic
(scikit-learn / PyTorch — or the series' standard library — used the way its maintainers
 intend: Pipeline + ColumnTransformer, `nn.Module` + proper training loop, not loose
 fit/transform calls; parity check vs Stage A)
### Stage C — production
(error handling, config, logging, model persistence + versioning, input validation,
 monitoring hooks — on the reader's stack: Python/FastAPI/PostgreSQL/Redis by default)

## Evaluation
(how you measure THIS mechanism: the metric, the harness, the captured number, and what
 movement in the number means. From series 11 onward this section is load-bearing, not
 decorative — and every model is compared against its dumb baseline.)

## Design Patterns / Tradeoffs
(for each real alternative: how it works · advantages · disadvantages · failure modes ·
 when to use · when NOT to use — then a recommendation and the condition that changes it)

## Production Scenario
### Symptoms
### Diagnosis
### Root Cause
### Fix
### Prevention

## Common Pitfalls
(anti-patterns: what people do · why it breaks, mechanistically · what to do instead)

## Interview Questions
(5-8, from "derive/explain" to "design" to "debug this"; answer shape in one line each)

## Key Takeaways
(5-8 bullets - decisions, not definitions)

## Related
(backward links to prerequisites; forward links to where the topic is picked up again;
 one line each, with the reason)
```

Rules on the template:
- Headings are exact. Tools and readers depend on them.
- The **cold open** (`> ⚡`, between the header block and `## Concept`) is mandatory:
  symptom + terse cause, never the mechanism — the mechanism is the notebook's payoff.
- **Stage A → B → C is mandatory** wherever a hands-on build applies (CLAUDE.md). Collapse
  stages only when one genuinely doesn't exist (e.g. there is no "from scratch" for a
  pandas API tour — B/C only; there is no "production service" for a pure derivation
  notebook — A only) and say why in one line under the heading. Never skip Stage A because
  Stage B is easier to write.
- **Evaluation** may be omitted only in toolkit/math notebooks (series 01–08) where the
  "eval" is the unit test or a shape/dtype assertion; say so in one line. Never omit
  **Production Scenario** from series 11 onward. For series 02–10 the scenario may be a
  *data-pipeline or numerical* incident (silent dtype coercion, a merge that duplicates
  rows, an aggregation that drops NaN groups, an ill-conditioned matrix blowing up a
  solve) rather than a model incident — it must still be specific.
- Use `⚠️` for a genuine trap - something that runs but does the wrong thing.
- Use a `> **Version note**` callout when behaviour differs across scikit-learn minors,
  numpy 1.x→2.x, pandas copy-on-write, or a series library's major versions. Only when true.
- Per-notebook MCQ blocks are NOT used; retrieval practice lives in the series `_quiz.md`
  (§11), extended with every notebook.

---

## 3. The three depths - what each section must actually contain

### CONCEPT
- Write it *toward the cold open*: the reader arrives holding an unanswered question — let
  the explanation visibly close that gap rather than defining terms in a vacuum.
- *Plain-English*: one short paragraph a non-engineer could follow; an analogy only if it is
  exact (a leaky analogy is worse than none).
- *Technical*: what it is, why it exists, what problem it solves, how it works, terminology
  (bold each term on first use), and **the numbers that matter** (dataset shapes, class
  balance ratios, feature dimensionality after encoding, fold-to-fold CV variance, training
  time, per-prediction latency, model artifact size, and which metric deltas are meaningful
  vs noise). Numbers make it real.
- *Mental Model*: one or two sentences the reader can carry. ("Regularization is a
  complexity budget: C is the price you charge the model per margin violation, so shrinking
  C buys a simpler boundary at the cost of training errors — tune it as a budget, not a
  magic number.")

### HOW IT WORKS
Full derivations for core results — the normal equation, gradient descent updates, the
bias–variance decomposition, logistic loss and its gradient, the SVM dual and the kernel
trick, entropy/Gini and information gain, the EM flavour inside k-means, PCA from
variance-maximization and SVD, backprop as the chain rule organized, convolution and BPTT
in the DL lane — worked step by step, each step justified. Where a proof is genuinely out of scope, state the result, cite
the source, and say precisely what is being taken on faith. ASCII diagrams must show the
**mechanism** (data shapes flowing through a Pipeline, what a split does to a feature
space, where training time goes), not boxes with names.

### PRODUCTION SCENARIO
The section most likely to be written badly. Requirements:

- A **specific incident** with a time, a trigger and a blast radius. Not "the model may
  drift"; rather "Tuesday 09:40, the nightly churn batch flags 4× the usual number of
  accounts; by 10:05 the retention team has burned its weekly discount budget on customers
  who were never at risk."
- **Symptoms** are what the on-call *sees*: the alert text, the metric-dashboard shape,
  score-distribution histograms, feature-null-rate panels, log excerpts, the user-facing
  effect. Quote realistic telemetry (a Grafana panel description, a prediction-log row, a
  schema-validation error).
- **Diagnosis** follows the ML observability ladder, explicitly:
  `Alert → Model-quality dashboards (rolling AUC/MAE, calibration) → Prediction logs
  (score distributions, per-segment breakdowns) → Input-data checks (schema, null rates,
  value ranges, freshness) → Drift analysis (feature/label/concept) → Version diff (model
  artifact, feature pipeline code, training-data snapshot) → Root cause`.
  Show which *signal* eliminated which *hypothesis*.
- **Root Cause** is one or two sentences and *mechanistic* ("the upstream export switched
  amounts from INR to thousands-of-INR; the scaler happily normalized the new range, so
  every monetary feature collapsed toward zero and the model fell back to its intercept").
- **Fix** is split into *mitigation now* and *permanent fix* (code/config shown). Both.
- **Prevention** names the eval, alert, test or review rule that would have caught it.

A scenario that could be pasted into any notebook is a failed scenario.

---

## 4. Examples must come from real development - the core quality rule

### 4.1 One running system per series

The applied spine is **PayFlow** — a fictional but realistic B2B SaaS subscription-billing
and invoicing platform (Cloud/B2B SaaS + FinTech, two of the target verticals). Its data
universe: customers, subscriptions, invoices, payments, support tickets, product add-ons.
The telecom/NMS domain is deliberately reserved for a later dedicated project — do NOT
force NMS examples. HealthTech may appear as a secondary dataset where a topic honestly
demands it (e.g. class imbalance on a diagnosis dataset).

| Series | Running system (keep unless a topic genuinely needs another) |
|---|---|
| 01–04 | PayFlow's raw operational exports — messy, real-shaped CSVs (mixed dtypes, sentinel values, duplicated keys, timezone chaos) for numpy/pandas/viz work; canonical datasets where honest — never pretend |
| 05–08 | Math on PayFlow-shaped numbers where honest (invoice-amount distributions, an ill-conditioned feature matrix, an A/B test of dunning emails in 08); clean canonical examples where the math demands them |
| 09–10 | Cleaning, EDA and feature engineering on the PayFlow exports — producing the feature tables the modelling lanes consume |
| 11–14 | **Invoice payment-delay prediction** (regression spine): days-to-payment from invoice + customer features; the same problem carries through linear → nonlinear → regularized, evaluated with the series-12 harness |
| 15–20 | **Churn & payment-default risk** (classification spine): the same customer table and the SAME eval harness across every algorithm, so LogReg vs KNN vs NB vs SVM vs trees vs ensembles are directly comparable |
| 21–23 | **Customer segmentation, DR and anomalies**: segment the customer table (segments feed back into the classifiers as features); PCA/UMAP on the same features; anomaly detection on invoice streams; market-basket on add-on purchases |
| 24–27 | DL foundations: ticket-text classification (MLP), receipt/document images (CNN), transaction/usage sequences (RNN); canonical datasets (MNIST-class) where honest — never pretend PayFlow needs a CNN it doesn't |
| 28 | Forecasting PayFlow MRR, invoice volume and payment inflow |
| 29 | Support-ticket triage + spam filtering over PayFlow's ticket corpus |
| 30 | Receipt/invoice image processing (deskew, threshold, contour/OCR prep) |
| 31 | Support-call audio: loading, spectral features, voice-activity detection |
| 32 | Add-on/plan recommendations from PayFlow usage & purchase history |
| 33–34 | Explaining and governing the churn scorer; shipping it as a monitored service on FastAPI/PostgreSQL/Redis |
| 35 | Capstones across the target verticals (Cloud SaaS, FinTech, HealthTech) |

The spine datasets exist and are the default for every example: generated
deterministically by the committed `_data/generate.py` (seed 42, byte-identical re-runs),
specified table-by-table in **`_data/SPEC.md`** — including the catalog of *intentional*
production mess (duplicated invoice rows, mixed currencies, sentinel values, timezone
chaos, a unit-glitch incident window, drift after a gateway migration, planted leakage
traps) with the series that teaches each defect, and the ground-truth signal models are
supposed to find. Read SPEC.md before choosing any example. Series 09–10 build
`_data/processed/` feature tables that series 11+ consume.

Public/canonical datasets are the exception, never the default: allowed only where the
topic honestly demands one (MNIST-class images for CNNs in 26, a public medical dataset
for real imbalance in 15), and every such cell carries `# canonical-ok: <reason>` —
`_tools/check.py` (rule D01) fails the notebook otherwise. iris/titanic/tips/toy blobs
as the *subject* of a section are never acceptable.

### 4.2 What "real dev example" means - with pairs

| ✗ Bookish (reject) | ✓ Real-dev (write this) |
|---|---|
| `iris.fit(X, y)`, accuracy printed | The churn model with 7% positives, the cost matrix from the retention budget, and the threshold chosen on the PR curve — with the revenue math |
| "We achieved 95% accuracy" | The PR-AUC next to the majority-class baseline's 93% accuracy, showing why accuracy lied here |
| `scaler.fit_transform(X)` then split | The leakage incident: CV score 0.84, production 0.71, and the pipeline refactor that made the gap disappear — before/after both captured |
| K-means on `make_blobs` | Segmentation where unscaled revenue (range 10⁶) drowns every behavioural feature — shown by the cluster sizes, fixed by the scaler, verified by silhouette |
| "Polynomial features may overfit" | The captured train/val curves diverging at degree 4, and the ridge sweep that closes the gap — same plot, three lines |
| `df.dropna()` in passing | The null-rate panel showing 31% missingness concentrated in one customer segment, and what dropping it silently does to that segment's error |
| `pickle.dump(model)` | Persistence with a version manifest — and the incident where loading a scikit-learn 1.4 pickle under 1.7 changed predictions silently |

Heuristics:
- **Start from the bug or the incident**, then show the code/pipeline/eval that prevents it.
- **Show what the machine shows**: printed shapes, `value_counts()`, learning curves, CV
  fold scores with variance, confusion matrices, timing lines, memory readouts. Real output
  is the difference between "trust me" and "see for yourself".
- **Use real constraints**: class imbalance, missing data at serve time, unseen categories,
  training-serving skew, retraining cadence, per-prediction latency budgets, PII in
  features.
- **Show before and after** whenever the topic is a fix: broken version, evidence it is
  broken, fixed version, evidence it is fixed — for features and evals just as for code.
- **Never** use foo/bar, `test123`, or a three-column toy DataFrame as the *subject* of a
  section (checker rule D02), and never reach for `load_iris`/`make_blobs`-style data
  when a PayFlow table serves (rule D01; `# canonical-ok: <reason>` for the honest
  exceptions).

### 4.3 Code and evidence rules

- Python 3.12+ syntax throughout (`type X = ...`, PEP 695 generics, `Self`, `StrEnum`,
  `datetime.now(UTC)`); type hints everywhere; no `Any` without a comment.
- scikit-learn idioms from Stage B onward: `Pipeline` + `ColumnTransformer` always (loose
  `fit_transform` calls outside a pipeline are themselves an anti-pattern, §8);
  `random_state` set on everything stochastic; `set_output(transform="pandas")` where it
  aids readability; current APIs only — no deprecated parameters or modules from the
  2019-era notes.
- numpy 2.x and pandas 3.x semantics (copy-on-write and string dtype are defaults; the
  repo `.venv` pins the truth — verify with `pip show`, never from memory); note a
  `> **Version note**` where old-notes behaviour differed.
- PyTorch 2.x idioms in the DL lane (24–27) and applied DL cells: device-agnostic code,
  explicit dtypes, `torch.manual_seed` + generator objects for anything whose output is
  shown, `torch.compile` only where it matters.
- Model persistence in Stage C: artifact + metadata (library versions, training-data
  hash/row count, metrics at save time) — never a bare pickle.
- **Every shown number, curve, table and score is produced by actually running the code**,
  seeded. Classical ML runs in seconds-to-minutes on CPU — there is no excuse for an
  uncaptured number. If something genuinely cannot be run in the pinned environment (a
  full-corpus run in series 21, a long video pipeline in 25), derive it carefully and mark
  the listing `# illustrative - not captured output`. One fake number poisons the
  notebook's credibility.
- Comparisons that claim a winner show repeat-run variance (CV fold spread or repeated
  seeds), and the prose says which delta is signal and which is noise.
- Keep listings 10-40 lines (hard cap 55); *every* listing is followed by prose that says
  what to look at and why. No 3 consecutive blocks without prose between them.

### 4.4 Formats and labs

- `.ipynb` notebooks, shipped **executed** — outputs are the evidence. Cells that need a
  large download or > ~5 min of compute are marked (`# long-running — est. 12 min CPU`)
  and the notebook must still *read* coherently without re-running them.
- Every experiment script that produced captured evidence is **committed** to the series'
  `_lab/` (`lab_<id>_<slug>.py`), standalone on the pinned environment, module docstring
  naming the notebook and listings it reproduces. `_lab/README.md`: one page, what each
  script shows.
- Labs must run on the author's machine (Windows 11): guard `if __name__ == "__main__":`
  for multiprocessing (`n_jobs=-1` counts), no Unix-only signal tricks, paths via
  `pathlib`.

### 4.5 The compute-budget rule

Everything on the core path must be **followable end-to-end on CPU** in minutes. Classical
ML makes this cheap — exploit it: sweep hyperparameters for real, repeat runs for variance,
show full learning curves. When a real dataset is too large for the repo, commit the
seeded download/subsample script, record the row count and a content hash, and state the
runtime. State wall-clock time for anything over ~30 seconds so the reader knows what to
expect. The DL lane (24–27) and applied series train at **toy scale on CPU or one small
GPU** — toy scale is a feature: a 3-layer MLP whose full training run takes two minutes
teaches backprop better than a config file for a big run. Mark GPU-preferred cells
(`# GPU recommended — ~8 min on CPU`) rather than shrinking the example below honesty.

---

## 5. Diagrams and pseudo-code

Prefer an ASCII diagram in a ```text block for mechanisms. Label arrows with what crosses
them (array shapes, dtypes, fitted state, a train/test boundary). Pipeline flows and
train-vs-serve timelines are especially valuable:

```text
invoices.csv (312,401 rows)
  ──▶ ColumnTransformer
        ├─ num (14 cols): SimpleImputer(median) ──▶ StandardScaler
        └─ cat (6 cols):  SimpleImputer(most_frequent) ──▶ OneHotEncoder  ──▶ 41 dims
  ──▶ X: (312401, 55) float64
  ──▶ LogisticRegression(C=0.31, class_weight="balanced")
  ──▶ predict_proba ──▶ threshold 0.42 (chosen on validation PR curve, NOT 0.5)

5-fold CV: PR-AUC 0.71 ± 0.02 (seed=42) · fit 3.8 s CPU · predict 210 µs/row
```

---

## 6. Referencing prerequisites and earlier series

- Reference by id and title: "*derived in 06.3 (gradient descent)*", "*scaling shown in
  09.2*".
- One-sentence recap + link; never re-teach. If the recap needs more than three sentences,
  the prerequisite list is wrong — fix it instead.
- **Never taught here** (assumed from the reader's profile in `CLAUDE.md`): Python syntax
  and stdlib, pytest, FastAPI mechanics, PostgreSQL/SQLAlchemy basics, Redis patterns,
  Docker/CI, generic observability plumbing (logs/metrics/traces). This track teaches what
  is ML-specific and *uses* the rest. Stage C code may use these freely without explaining
  them.
- Within the track, the **canonical-home rules** hold — each core idea is derived once and
  referenced everywhere else:
  gradient descent → 06 · eval discipline (baselines, CV, leakage, meaningful deltas)
  → 12 · bias–variance → 13 · regularization penalties → 14 · classification metrics,
  thresholds, imbalance & calibration → 15 · distance metrics & why scaling matters → 16 ·
  kernel trick → 18 · impurity measures & information gain → 19 · bagging/boosting theory
  and the gradient-boosting libraries → 20 · cluster-validity metrics → 21 · PCA/SVD → 22
  (SVD math in 05) · backprop → 24 · convolution → 26 · recurrence & attention motivation
  → 27. Forward references are encouraged.
- Anything transformer/LLM-shaped is out of scope: series 27 ends at attention as the
  bridge and hands off to the separate `ai-engineering` repo. Never author LLM content
  here.

---

## 7. Cross-cutting principles - must be visible, not just mentioned

- **Problem-first ordering**: cold open before Concept; bug before fix; incident as payoff.
- **Baseline-first**: no model is shown without the dumb baseline it must beat
  (mean/median predictor, majority class, `DummyClassifier`/`DummyRegressor`). Beating
  nothing proves nothing.
- **Eval-driven development**: from series 11 onward, no modelling change (feature,
  preprocessing step, algorithm, hyperparameter, threshold) is shown without the eval that
  would detect its regression. "Looks better" is never evidence.
- **Leakage discipline**: split before you fit; every fitted transform lives inside the
  Pipeline/CV loop; every notebook where leakage *can* occur shows where it would hide.
- **Derivation-first depth**: for core math, the reader must be able to re-derive, not
  recite. Show the derivation once, canonically (§6), and reference it thereafter.
- **Cost and latency as first-class constraints**: training time, retraining cadence,
  per-prediction latency and artifact size appear in design decisions the way latency
  budgets do in backend work.
- **Reproducibility discipline**: seeds for everything stochastic; pinned versions; CV
  fold spread or repeat-run variance whenever a comparison claims a winner.
- **Failure-first**: NaNs arriving at serve time, categories unseen in training, drifting
  feature distributions, non-converging solvers, empty clusters, a degenerate split — the
  applicable "what happens if..." questions get answered in every notebook where they
  apply.

---

## 8. Explicit anti-patterns to call out (where relevant)

Fitting scalers/encoders/imputers before the split (leakage) · `fit_transform` on test
data · target leakage via features computed from the future · random splits on temporal
data · resampling (SMOTE/undersampling) before the split · accuracy on imbalanced classes ·
threshold 0.5 by default · test-set reuse for model selection (the test set you tuned on
is a validation set) · GridSearchCV scores quoted as generalization estimates without a
held-out set · no dumb baseline · R² worship on nonlinear fits · ordinal-encoding nominal
categories · one-hot exploding high-cardinality features · k-means on unscaled features ·
the elbow method as proof · DBSCAN eps guessed instead of derived from a k-distance plot ·
TF-IDF fitted on the full corpus (leakage again) · `predict_proba` treated as calibrated
probability · bare pickles across library versions · training-serving skew from
re-implemented preprocessing · silent `dropna()` · `inplace=True` chains · unset
`random_state` on anything whose output is shown · a model with no monitoring and no
retraining plan.

For each: what people do · why it breaks (mechanism) · what to do instead.

---

## 9. Pre-submit checklist (run it; do not skip)

- [ ] `python _tools/check.py "<notebook>"` exits 0; every WARN either fixed or
      justified in one line in the response.
- [ ] All template headings present, exact, in order; header block filled.
- [ ] Cold-open line (`> ⚡`) teases this notebook's own Production Scenario (symptom +
      terse cause, no mechanism).
- [ ] Stage A → B → C all present, or the collapse is justified in one line (§2).
- [ ] Examples live in the series' running system (§4.1); no foo/bar, no toy DataFrames
      as section subjects.
- [ ] At least one listing shows *real captured output* (curve, shapes, CV table,
      confusion matrix, timing line).
- [ ] Every shown number was produced by running committed `_lab/` code, or is marked
      `# illustrative - not captured output` (§4.3).
- [ ] Notebook runs end-to-end on CPU (or one small GPU in the DL lane) in the repo
      `.venv` (§4.5); long-running/GPU-preferred cells marked with estimated time.
- [ ] Evaluation section names the metric, the harness, the baseline beaten, and what a
      meaningful delta is (or justifies its absence, §2).
- [ ] Tradeoffs section compares ≥ 2 genuine alternatives with when/when-not.
- [ ] Production Scenario: specific incident · symptoms as seen on-call · diagnosis via
      the ML observability ladder · mechanistic root cause · mitigation + permanent fix ·
      prevention.
- [ ] Derivations complete for core results; taken-on-faith steps named explicitly.
- [ ] Prereqs referenced, not re-taught; canonical-home rules respected; Related section
      has backward and forward links.
- [ ] Interview questions include at least one "derive this", one "design this" and one
      "debug this".
- [ ] Length is DEPTH-DRIVEN — no padding, no truncation; floors calibrated per level
      (Beginner ≥ 1600 · Intermediate ≥ 2000 · Advanced ≥ 2400 words of prose, excluding
      code and tables). No ceiling — denser beats longer.
- [ ] No code block over 55 lines; no 3 consecutive blocks without prose between them.
- [ ] Status updated in `CURRICULUM.md`.

---

## 10. Per-notebook generation prompt (copy, fill, run)

```text
Read classical-ml/AUTHORING-GUIDE.md, classical-ml/AUTHORING-PROTOCOL.md and
classical-ml/CURRICULUM.md. Skim the headings of the prerequisite notebooks listed for
<ID>.

Write notebook <ID> "<TITLE>" to classical-ml/<SERIES-FOLDER>/<FILENAME>, following the
protocol phases IN ORDER: gate check → prep (planning block) → lab → scaffold → write →
verify → submit.

Constraints:
- Level: <B/I/A>. Running system: <from §4.1>. Previous notebook: <ID-1> (keep terminology
  and the running system continuous).
- Run scratch experiments FIRST and capture every number by execution (§4.3); commit the
  scripts as _lab/lab_<ID>_<slug>.py (§4.4). Respect the compute budget (§4.5). Numbers
  travel only by copy-paste from executed output.
- Scaffold with _tools/scaffold.py; follow the template in §2 exactly: cold open,
  Stage A→B→C, Evaluation with a baseline, Production Scenario diagnosed via the ML
  observability ladder. Apply §3 depth rules, §4 example rules, §7 principles, §8
  anti-patterns.
- Do not re-teach prerequisites or a canonical-home topic (§6); reference it.
- Before finishing: Restart & Run All, then `python _tools/check.py "<notebook>"` until
  exit 0; run the §9 checklist.
- One notebook only. Then extend _quiz.md and update the status in CURRICULUM.md.
```

## 11. Per-series loop

For a series: generate notebooks in order, one per response, each reading the previous one
first. After a series' last notebook, write `_recap.md` in the series folder — what was
built, a table of the incidents/experiments with the general lesson each carries,
invariants established, and what the next series assumes.

Also at series close, write **`_cheatsheet.md`** — the fast-lookup layer the notebooks
deliberately are not. It is a **decision card, not a syntax listing** (the 2019 notes'
feature-per-cell density is the anti-pattern: it produced recognition, not recall, and
rotted silently). Target one page. Contents, in order: the series' **rules** as imperative
one-liners; a **traps table** (symptom → mechanism → probe/fix — each incident distilled to
one row); the few **formulas and numbers worth carrying** (only figures that exist in a
notebook's captured output — `check.py` X01 enforces this, same as `_quiz.md`/`_recap.md`);
and **decision rules** ("when X → do Y"). Every row cites its notebook id, so the sheet is a
jump table into depth, never a substitute for it. Pure API signatures stay with the official
docs; retrieval practice stays with `_quiz.md` — the cheatsheet is for the ten-second
mid-task lookup ("what's the rule again?"), and it is regenerated whenever a notebook's
claims change, like every companion document.

Each series carries ONE `_quiz.md` in the series root — **questions first**, answers in a
single `## Answers` section at the BOTTOM (never adjacent), one to three lines each, naming
the source notebook. Interleave sources; mix incident replays ("debug this"), derivation
prompts, decision questions, 1-2 mini coding challenges and one series-wide design prompt.
Extend it in the same response that adds any later notebook. The file opens with the short
"How to use" protocol (state your own Mental Model before reading the given one; answer
aloud before reading the answer shape; run at least one Stage A build from memory;
reattempt a week later).

Then set the series' statuses to `review` in `CURRICULUM.md`. A reviewer reads the series
end-to-end for continuity of the running system and promotes to `done` — judging content
depth first (concept, derivations, real-dev examples), mechanics second.

**Reference standard:** the first fully-authored series becomes this track's conformance
benchmark; until it exists, this guide is the only standard — hold the first series to it
strictly, because everything after will be measured against that series. Read the
immediately preceding notebook *end-to-end* (not skimmed) before writing a new one, and
match its voice: derivations that close the cold-open's question, captured evidence woven
into prose (never decorative), incidents whose numbers reappear in the eval, and explicit
callbacks to earlier notebooks' results.
