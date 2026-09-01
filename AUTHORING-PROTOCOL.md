# AUTHORING PROTOCOL — the exact procedure for producing one notebook

This file is the **execution procedure**; [AUTHORING-GUIDE.md](AUTHORING-GUIDE.md) is the
**quality standard** it implements. Follow this file top to bottom, in order, without
skipping gates. It exists so that any competent model or person — not only the strongest —
produces conforming, high-quality content: every judgment call the guide leaves open is
converted here into a fixed rule, a required artifact, or a checker rule. If this file
ever contradicts the guide, the guide wins and the contradiction is a bug — report it,
don't improvise.

The single most important rule, stated once:

> **Code runs first. Prose comes second. Numbers travel only by copy-paste.**
> You never type a number, curve description, or output into prose from memory or
> expectation — you paste it from something you executed. If you notice yourself writing
> a metric you have not seen printed, stop and go run it.

---

## Phase 0 — Gate check (do NOT write content if any answer is "no")

Answer these literally:

1. Has the **per-series pre-flight currency check** been run and its patches approved?
2. Is the **topic tree** for this series approved and recorded in `CURRICULUM.md` under
   "Per-series topic trees"?
3. Is the notebook id you are about to write listed in that tree, and is the *previous*
   notebook in the series at status `[~]` or later?

If any answer is no → your task is that step instead (propose the currency check or the
tree, then STOP and wait for approval — CLAUDE.md approval gate). Never write teaching
content in the same response that proposes a tree.

## Phase 1 — Prep (read, then extract)

Read, fully, in this order:
1. `AUTHORING-GUIDE.md` (the standard) and this file.
2. `CURRICULUM.md` — the series row and approved tree. Extract: **id, title, level
   (B/I/A), series name, depends-on list, scope line**.
3. The **previous notebook in this series, end-to-end** (not skimmed). Extract: the
   running-system state (which PayFlow tables/features/models exist so far), terminology
   already introduced, and any number you will call back to.
4. The headings of the prerequisite notebooks in the depends-on list.
5. The matching legacy notebook(s) from `_legacy-2019/` (see the mapping in
   `CURRICULUM.md`) — for historical scope only. Do not copy style, examples, or APIs.

Then write the **planning block** (in your working notes, not the notebook). All fields
are mandatory; "TBD" is not a value:

```text
NOTEBOOK PLAN <id> <title>
Level/floor      : <B 1600 | I 2000 | A 2400 prose words>
Cold-open incident:
  when/trigger   : <specific time + the change that set it off>
  blast radius   : <who was hit, how hard, by when>
  symptom quotes : <2-3 realistic telemetry/log/dashboard lines the on-call saw>
  root cause     : <1-2 sentences, mechanistic>
  mitigation     : <the now-fix>       permanent: <the real fix>
  prevention     : <the eval/alert/test that would have caught it>
Derivations      : <each core result to derive, + which steps are taken on faith>
Stage plan       : <A: what raw build proves | B: idiomatic + parity check | C: what
                    hardening — or the one-line justification for a collapsed stage>
Experiments      : <numbered list; for each: lab script name, what it produces, which
                    section consumes its output>
Data             : <which _data/raw or _data/processed files; which SPEC.md mess ids
                    (M1-M14) this notebook touches or must guard against — leakage traps
                    M10 named explicitly if the notebook trains anything>
Eval             : <metric + baseline + harness + what delta counts as signal>
Callbacks        : <which numbers/results from earlier notebooks this one references>
```

The incident must satisfy the paste test (guide §3): if it could be pasted into a
different notebook unchanged, it is too generic — bind it to THIS notebook's mechanism.

## Phase 2 — Lab (run everything before writing anything)

0. Data: `_data/raw/` exists (else `.venv\Scripts\python _data\generate.py`) and matches
   the SPEC.md manifest. Examples come from these tables; a public dataset is allowed
   only per the canonical-dataset policy below, marked `# canonical-ok: <reason>`.
1. Write each experiment from the plan as `_lab/lab_<id>_<slug>.py` — module docstring
   names the notebook and the listings it reproduces; `if __name__ == "__main__":` guard;
   seeds on everything stochastic (`np.random.default_rng(42)`, `random_state=42`,
   `torch.manual_seed(42)`).
2. Run each script in the repo `.venv`. Save the printed output next to your plan.
3. Any experiment that surprises you (metric worse than baseline, derivation and code
   disagreeing) is **content, not a problem** — the honest narrative correction belongs
   in the notebook. Never quietly re-roll seeds or trim data until the story looks clean.
4. Version facts (library versions, API behavior) are verified by running
   `pip show <pkg>` / the actual call — never from memory. If you cannot verify a fact,
   omit it; never guess.
5. Gate: **all planned outputs captured** before Phase 3.

## Phase 3 — Scaffold (never type template headings by hand)

Two paths. Both end at the same guarantee: `check.py` verifies the headings, so neither
depends on remembering the template.

**Path A — incremental (a human, or filling a notebook over several sittings):**

```bash
.venv\Scripts\python _tools/scaffold.py <id> "<Title>" --series "<NN Series Name>" --level <B|I|A> --prereqs "<id (topic) . id (topic)>" --out "<series-folder>"
```

The scaffold's headings are exact by construction and `check.py` fails any leftover
`<<FILL: ...>>`, so nothing can be silently skipped.

**Path B — generated in one pass (what series 01 actually used):** write a throwaway
builder script in the scratchpad that emits the complete `.ipynb` as JSON — markdown cells
as `md("...")`, code cells as `code("...")` — then run it. This is the practical path when
the whole notebook is authored in one response, because it makes the prose editable as
plain text and the whole notebook regenerable after any correction. Two rules make it
safe: copy the heading list from §2 of the guide verbatim into the builder, and treat
`check.py` as the arbiter rather than your memory of the template.

Whichever path, the notebook is regenerated and re-executed from the builder after every
prose fix — never hand-patched in the `.ipynb`, which desynchronises it from the source
that produced it.

## Phase 4 — Write (fill sections in THIS order, not top to bottom)

Writing order forces coherence — the payoff exists before the setup that teases it:

1. **Production Scenario** (from the incident card — Symptoms as the on-call sees them;
   Diagnosis walking the ladder `Alert → quality dashboards → prediction logs →
   input-data checks → drift → version diff → root cause`, showing which signal killed
   which hypothesis; Fix = mitigation now AND permanent; Prevention names the guard).
2. **Hands-On Build** — paste code from the lab scripts into cells, execute, keep the
   outputs. Stage A proves the mechanism with printed evidence; Stage B includes the
   parity check against A; Stage C hardens on FastAPI/PostgreSQL/Redis. After every
   listing: prose saying what to look at in the output and why.
3. **Evaluation** — the harness run, the baseline it beats, fold variance, and one
   sentence naming which delta is signal vs noise. From series 11 onward this section
   must contain at least one captured table/number.
4. **How It Works** — the derivations from the plan, step by step, each step justified;
   an ASCII mechanism diagram in a ```text block. The derivation must answer the
   cold-open's question explicitly.
5. **Concept** — written *toward* the cold open (the reader arrives holding the
   incident's unanswered question). Plain-English one paragraph; Technical with bolded
   first-use terms and the numbers that matter; Mental Model one or two sentences.
6. **Cold open** (`> ⚡`) — now that the scenario exists, tease it: symptom + terse
   cause, never the mechanism.
7. **Common Pitfalls** — from guide §8, only the ones that apply here: what people do ·
   why it breaks mechanistically · what to do instead. `⚠️` for traps that run but lie.
8. **Interview Questions** (5–8; ≥1 derive, ≥1 design, ≥1 debug; one-line answer shape
   each) · **Key Takeaways** (5–8 bullets, decisions not definitions) · **Related**
   (backward + forward links by id, one line each with the reason) · header block's
   "What you'll learn" (3–5 capabilities, not topics).

Style constraints while writing (the checker warns on some, honor all):
- Voice test on every paragraph: would a staff ML engineer say this in a design review?
  No filler ("in this notebook…", "let's dive…", "it is important to note…"), no hype,
  no motivational padding. Floors are met by adding *depth* (a deeper derivation step, a
  sharper failure mode, one more captured comparison) — never by adding words.
- Every number in prose exists in a cell output or lab capture, or the listing is marked
  `# illustrative - not captured output`.
- No code block over 55 lines; never 3 code cells in a row; comment code fully.
- Prerequisites are referenced by id, never re-taught; canonical-home rules (guide §6)
  are respected.
- The only emojis anywhere are `⚡` (cold open) and `⚠️` (traps).

## Phase 5 — Verify (mechanical, then judgment)

1. **Restart kernel & Run All.** Every cell executes top to bottom in the `.venv`; the
   shipped file carries a clean 1..N execution order. Cells that legitimately can't run
   carry an allowed marker (`# long-running`, `# GPU recommended`,
   `# illustrative - not captured output`). Headless equivalent, run **from inside the
   series folder** so the notebook's `Path.cwd() / "_lab" / ...` module load resolves:

```bash
cd "<series-folder>" && ..\.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace "<id> <Title>.ipynb" --ExecutePreprocessor.timeout=3600
```

   Two things this catches that a lab run does not: a notebook that depends on state the
   lab set up for itself, and any number that is not reproducible across runs — timings
   especially. Quote ratios and artifact sizes, never raw wall-clock, unless the prose
   says the figure is machine-dependent.
2. Run the checker; loop until exit code 0:

```bash
.venv\Scripts\python _tools/check.py "<series-folder>/<id> <Title>.ipynb"
```

3. Resolve every WARN deliberately: fix it, or state in one line (in your response, not
   the notebook) why it stands.
4. Then the judgment pass — answer each; any "no" sends you back to the phase named:
   - Does the derivation visibly close the cold-open's question? (→ Phase 4.4)
   - Could the Production Scenario be pasted into another notebook unchanged? (→ Phase 1)
   - Is every example inside the running system, or honestly canonical? (→ Phase 4.2)
   - Does Evaluation name baseline + meaningful-delta, with captured numbers? (→ 4.3)
   - Did you show any result you did not actually run? (→ Phase 2. Non-negotiable.)

## Phase 6 — Submit (all in the same response)

1. Extend the series `_quiz.md` (questions interleaved across notebooks; answers only in
   the single bottom `## Answers` section) and run `.venv\Scripts\python _tools/check.py`
   on it.
2. Update the notebook's status in `CURRICULUM.md` (`[ ]` → `[~]`).
3. Deliver **one notebook only** per response — never batch, even if asked to hurry.
4. If the series' last notebook: also write `_recap.md` and `_lab/README.md`, set the
   series statuses to `[r]`, and run the whole-folder check:
   `.venv\Scripts\python _tools/check.py "<series-folder>"`.

---

## Fixed decision tables (do not re-decide these per notebook)

**Stage collapse** (guide §2): the default is all three stages present. A collapse is
legitimate only in these shapes — name the shape in the one-line justification:
| Shape | Collapse |
|---|---|
| Pure derivation / math notebook (lanes M, parts of 24) | A only — "no framework exists for this derivation" |
| Library-API notebook (lanes T, parts of 03/04) | B/C only — "the library IS the subject; nothing to build raw" |
| Survey/decision notebook (no single mechanism) | build may be absent — say so; Tradeoffs carries the weight |

**Default eval metric by problem type** (deviate only with a stated reason):
| Problem | Default shown | Never alone |
|---|---|---|
| Regression | MAE + RMSE vs mean-predictor baseline | R² |
| Balanced classification | Accuracy + confusion matrix vs majority baseline | — |
| Imbalanced classification | PR-AUC + cost-weighted threshold choice | Accuracy |
| Ranking/recommendation | MAP/NDCG@k vs popularity baseline | Accuracy |
| Clustering | Silhouette + stability across seeds + business readout | Inertia |
| Forecasting | MASE vs seasonal-naive baseline, backtested | In-sample RMSE |

**Incident type by lane** (Production Scenario): T/M/D lanes → data-pipeline or numerical
incident · R/C/U lanes → model-quality incident in batch or online scoring · N lane →
training-run or serving incident · A/P lanes → end-to-end service incident. Ground the
incident in a SPEC.md mess id where one fits (M3 unit glitch, M9 gateway drift, M10
leakage) — the data already contains the evidence, so symptoms can be *captured*, not
narrated.

**Canonical-dataset policy** (rule D01): PayFlow data is the default subject of every
section. A public/canonical dataset is allowed only in these shapes, always with
`# canonical-ok: <reason>` in the cell:
| Shape | Example |
|---|---|
| The topic needs a modality PayFlow lacks | MNIST/Fashion-MNIST in 26; a public speech corpus in 31 |
| A property must be real, not synthesized | a public imbalanced medical dataset in 15; benchmark corpora in 29 |
| Pure mechanism visualization, never a section subject | `make_moons` to draw a nonconvex boundary in 18/21 |

## If uncertain — the resolution order

1. A mechanical question (heading, floor, structure) → the checker and guide §2/§9 decide.
2. A scope question (does X belong here?) → `CURRICULUM.md` scope line + canonical homes
   (guide §6). Still unclear → put a one-line proposal in your response and proceed with
   the narrower scope; never silently expand.
3. A fact you can run → run it (Phase 2.4). A fact you cannot run or verify → leave it
   out. **Omission is always safer than invention** — a missing benchmark is a gap; an
   invented one poisons the track.
