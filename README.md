# classical-ml

A complete, self-paced Data Science, Machine Learning and Deep Learning curriculum — from
mathematical foundations to production systems — built for the transition from
**Solutions Architect** to **ML Engineer**, targeting FAANG/frontier-lab interview
readiness and genuine production mastery.

The path runs in ten lanes across 35 series: foundations → toolkit (NumPy/pandas/viz) →
math (linear algebra, optimization, probability, inference) → data (cleaning, EDA,
feature engineering) → regression → classification → unsupervised → deep-learning
foundations (backprop through sequence models) → applied domains (time series, NLP, CV,
audio, recommenders) → production (interpretability, MLOps, capstones). Transformers,
LLMs and everything generative belong to the separate `ai-engineering` repo — this track
ends at attention as the bridge.

This repo supersedes 2019-era notes: the original notebooks are quarantined under
`_legacy-2019/` and each series is re-authored from scratch to the standard in
[AUTHORING-GUIDE.md](AUTHORING-GUIDE.md).

Curriculum baseline **v0.2, 2026-08-31**. Authoring has not started; series are written
one at a time in numeric order.

- [CURRICULUM.md](CURRICULUM.md) — the scope authority: every series with lane, level,
  dependencies, the legacy mapping, and **live progress**
- [AUTHORING-GUIDE.md](AUTHORING-GUIDE.md) — the standard every notebook is written to
  (template, Stage A→B→C build rule, evidence rules, eval-driven depth, compute budget)
- [AUTHORING-PROTOCOL.md](AUTHORING-PROTOCOL.md) — the step-by-step procedure that
  implements the standard (phase gates, planning block, decision tables)
- [_tools/](_tools/README.md) — the harness: `scaffold.py` generates the exact template
  skeleton; `check.py` mechanically verifies conformance (headings, floors, executed
  evidence, quiz structure) and must exit 0 before a notebook ships

## How to use these notes

Each series is a kebab-case folder (`18-support-vector-machines/`) of numbered notebooks
(`18.2` = series 18, notebook 2) written in teaching order.

Every series has a questions-first `_quiz.md` (answers at the bottom — attempt first) and
closes with a `_recap.md`. Experiment scripts that produced captured evidence live in each
series' `_lab/`; the shared datasets for the running example (the PayFlow billing
platform) live in `_data/`.

## Running the material

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
python _data/generate.py      # deterministic PayFlow datasets (seed 42, ~12 MB, not committed)
jupyter lab
```

All examples run on the **PayFlow data universe** — realistic, deliberately messy
operational exports of a fictional B2B SaaS billing platform (mixed currencies,
duplicated rows, sentinel values, drift, planted leakage traps), specified in
[_data/SPEC.md](_data/SPEC.md). No iris, no titanic: the conformance checker fails
notebooks that use bookish datasets without a stated reason.

Lane/series extras (installed when you reach them): `xgboost`/`lightgbm`/`catboost`/
`optuna` (20), `umap-learn` (22), `mlxtend` (23), `torch` (24–27), `nltk`/`spacy` (29),
`opencv-python` (30), `librosa` (31), `implicit` (32), `shap` (33), `mlflow` (34). Exact
pinned versions are set per series by the pre-flight currency check and recorded in the
series README.

## Conventions used in these notes

- **Stage A before Stage B** — you implement the algorithm raw (numpy, or raw tensors in
  the DL lane) before touching scikit-learn/PyTorch, and Stage B includes a parity check
  against Stage A.
- **Evidence or it didn't happen** — every number/curve/table comes from committed, seeded
  lab code run in the repo `.venv`; anything not actually run is marked
  `# illustrative - not captured output`.
- **Eval-driven, baseline-first** — from series 11 onward, no modelling change appears
  without the eval that would catch its regression, and no model appears without the dumb
  baseline it must beat.
- **Leakage discipline** — splits before fits; every fitted transform lives inside the
  Pipeline/CV loop.
- ⚠️ marks a genuine trap — something that runs but does the wrong thing.
- **Version note** callouts flag behaviour that differs across scikit-learn minors,
  numpy 1.x→2.x, pandas copy-on-write, or PyTorch minors.
- Fast-moving series (25 PyTorch, 29 NLP, 30 CV, 31 Audio) re-verify their libraries and
  versions at teach time (CURRICULUM.md currency policy), never pin them in scope.
