# classical-ml - Curriculum & Progress

This file is the track's **scope authority**: every series with lane, level, dependencies
and **live progress** as series are authored. Scope changes are proposed as patches (per
the pre-flight currency check in `CLAUDE.md`), approved, then folded in here with a
version bump.

Curriculum baseline: **v0.3, 2026-09-01** — series-01 pre-flight currency check applied
(patches P1–P7 approved 2026-09-01; see Currency log). v0.2 was the full redesign of the
2019 structure into a modern DS/ML/DL path (v0.1 mirrored the legacy folders one-to-one;
see Legacy mapping).

Authoring rules for every notebook: [AUTHORING-GUIDE.md](AUTHORING-GUIDE.md).
Track overview: [README.md](README.md).

Legend: `[ ]` todo · `[~]` draft · `[r]` in review · `[x]` done.

**Progress: 1/35 series authored** (series 01 complete, in review).

---

## How this curriculum is structured

- **Numeric order is the learning order.** Series 01–35 are the core path, done in
  sequence. Ten lanes: Foundations → Toolkit → Math → Data → Regression → Classification
  → Unsupervised → Deep Learning → Applied → Production.
- A **Series** = the unit tracked here. Each series is depth-driven — typically 4–12
  notebooks — and its notebook-level **topic tree is generated at authoring time**, after
  the per-series pre-flight currency check, and requires explicit approval before any
  teaching content is written. Approved trees are recorded at the bottom of this file.
- Notebook ids are `<series>.<n>`, e.g. `18.2` = series 18, notebook 2.
- Every series closes with `_recap.md` and carries a questions-first `_quiz.md`.
- **Folders:** each series gets a fresh kebab-case folder on first authoring
  (`05-linear-algebra-for-ml/`). The 2019 legacy folders are quarantined to
  `_legacy-2019/` before series 01 is authored; they define historical scope only and are
  deleted per series at Aditya's discretion once superseded.
- **Track boundary:** transformers, LLMs, generative AI, RAG, agents and LLM serving
  belong to the separate `ai-engineering` repo. This track runs from math fundamentals
  through classical ML and deep-learning foundations, ending at attention as the bridge
  (series 27). No overlap is authored here.

## Series index

Lanes: **F** foundations · **T** toolkit · **M** math · **D** data ·
**R** regression · **C** classification · **U** unsupervised · **N** deep learning ·
**A** applied domains · **P** production.

★ = "current at teach time": re-verify named libraries/models/versions on every teach.

| # | Series | Lane | Level | Depends on | Scope (2026-27) | Status |
|---|---|---|---|---|---|---|
| 01 | The ML Landscape & Project Lifecycle | F | B | — | Rules-vs-learning as an engineering decision, the PayFlow data universe (`_data/`), reproducibility as a contract, taxonomy of learning problems, problem framing → learnable target, the project lifecycle and where projects die | `[r]` |
| 02 | NumPy & Vectorized Computing | T | B | 01 | ndarrays, broadcasting, vectorization, numerical dtypes, numpy 2.x semantics incl. the **2019→2026 migration thread** (removed aliases, copy semantics — P5), memory layout & performance | `[ ]` |
| 03 | Pandas & Modern DataFrames | T | B | 02 | pandas 3.x (copy-on-write default, `str` dtype, pyarrow backing), joins/groupby/reshaping, **2019→2026 migration thread** (P5), polars interop incl. narwhals-backed `set_output` so polars frames flow through sklearn pipelines (P7), larger-than-memory tactics | `[ ]` |
| 04 | Data Visualization | T | B | 03 | matplotlib/seaborn/plotly, statistical plots, diagnostics plots used throughout the track, dashboard-grade figures | `[ ]` |
| 05 | Linear Algebra for ML | M | I | 02 | Vectors/matrices, norms, projections, eigendecomposition, SVD, condition numbers — implemented in numpy, tied to where each result resurfaces | `[ ]` |
| 06 | Calculus & Optimization for ML | M | I | 05 | Gradients, Jacobians/Hessians, convexity, gradient descent family (batch/mini-batch/SGD, momentum, Adam), constrained opt & Lagrange (feeds SVM) | `[ ]` |
| 07 | Probability & Statistics for ML | M | I | 02, 05 | Random variables, key distributions, expectation/variance, CLT, MLE/MAP, Bayes' theorem — the estimator's view | `[ ]` |
| 08 | Statistical Inference & Experimentation | M | I | 07 | Sampling, confidence intervals, hypothesis tests, multiple comparisons, bootstrap, A/B testing end-to-end | `[ ]` |
| 09 | Data Cleaning & Pre-processing | D | I | 03, 07 | Missingness mechanisms & imputation, outliers, scaling/encoding, sklearn `Pipeline`/`ColumnTransformer`, leakage-safe preprocessing | `[ ]` |
| 10 | EDA & Feature Engineering | D | I | 04, 09 | Systematic EDA workflow, target analysis, interactions, datetime/text/categorical features, feature selection, dataset documentation | `[ ]` |
| 11 | Linear Regression | R | B | 06, 09 | Simple → multiple, normal equation vs GD, assumptions & diagnostics, statsmodels vs sklearn views | `[ ]` |
| 12 | Model Evaluation & Validation ⭐ | R | I | 11 | Baselines, train/val/test, cross-validation done right, regression & general metrics, leakage taxonomy, error analysis, "meaningful delta vs noise" | `[ ]` |
| 13 | Nonlinear Regression & Bias–Variance | R | I | 12 | Polynomial features, splines, the bias–variance decomposition (derived), learning/validation curves, under/overfitting in practice | `[ ]` |
| 14 | Regularization | R | I | 13 | Ridge/Lasso/ElasticNet derived, geometry of penalties, regularization paths, early stopping as regularization | `[ ]` |
| 15 | Logistic Regression & Classifier Practice | C | I | 12, 14 | Logistic loss derived, classification metrics (ROC/PR, confusion), thresholds from cost matrices, class imbalance, calibration | `[ ]` |
| 16 | K-Nearest Neighbors | C | B | 09, 12 | Distance metrics, curse of dimensionality, KD/Ball trees & ANN idea, KNN as baseline discipline | `[ ]` |
| 17 | Naive Bayes & Generative Classifiers | C | I | 07, 15 | Generative vs discriminative, NB variants, smoothing, why NB survives (text, speed), LDA/QDA | `[ ]` |
| 18 | Support Vector Machines | C | A | 14, 16 | Margins, hinge loss, the dual, kernel trick derived, RBF intuition & gamma sizing, SVR, when SVMs still win in 2026 | `[ ]` |
| 19 | Decision Trees | C | I | 15 | Impurity measures & information gain derived, CART mechanics, pruning, instability, trees as the substrate for ensembles | `[ ]` |
| 20 | Ensembles & Gradient Boosting | C | A | 19 | Bagging/random forests, boosting derived (AdaBoost → gradient boosting), **HistGradientBoosting as sklearn's default booster** (`criterion` deprecated on the legacy `GradientBoosting*` in 1.9 — P6), XGBoost/LightGBM/CatBoost, hyperparameter optimization (Optuna, successive halving) with **sklearn 1.9 callbacks** for search progress (P6), why GBDTs rule tabular data | `[ ]` |
| 21 | Clustering | U | I | 16 | k-means (and the EM idea), GMMs, hierarchical, DBSCAN/HDBSCAN, mean shift, validity metrics, the "no ground truth" problem | `[ ]` |
| 22 | Dimensionality Reduction | U | A | 05, 21 | PCA derived from variance & SVD, whitening, t-SNE/UMAP mechanics and their honest limits, DR for features vs for plots | `[ ]` |
| 23 | Anomaly Detection & Association Rules | U | I | 12, 21 | Isolation Forest, LOF, one-class SVM, statistical process control; Apriori/FP-Growth and where association rules still earn their keep | `[ ]` |
| 24 | Neural Networks from Scratch | N | A | 06, 15 | Perceptron → MLP, forward/backprop derived and implemented in numpy, initialization, activation functions, gradient checking | `[ ]` |
| 25 | Deep Learning with PyTorch ★ | N | A | 24 | PyTorch 2.x (tensors, autograd, `nn.Module`, `torch.compile`), training-loop discipline, optimizers, batch/layer norm, dropout, debugging training runs | `[ ]` |
| 26 | Convolutional Neural Networks | N | A | 25 | Convolution derived, CNN architectures & their history, transfer learning, augmentation, training at small scale honestly | `[ ]` |
| 27 | Sequence Models & the Road to Attention | N | A | 25 | RNN/LSTM/GRU mechanics, BPTT, vanishing gradients, seq2seq, attention as the fix — hand-off point to `ai-engineering` | `[ ]` |
| 28 | Time Series Forecasting | A | A | 08, 14, 27 | Decomposition, stationarity, ARIMA/ETS, feature-based ML forecasting, GBDT & neural forecasters, backtesting without leakage | `[ ]` |
| 29 | Natural Language Processing ★ | A | A | 17, 20, 27 | Text pipeline, BoW/TF-IDF, classical text classification, word embeddings, spaCy in production — ends where transformers begin | `[ ]` |
| 30 | Computer Vision ★ | A | A | 02, 26 | OpenCV 4.x classical CV (filtering, edges, contours, keypoints), document/receipt processing, CNN inference in a CV pipeline | `[ ]` |
| 31 | Audio & Speech Processing ★ | A | A | 02, 27 | Digital audio, spectrograms/MFCCs, librosa, VAD, audio classification, feature pipelines for speech models | `[ ]` |
| 32 | Recommender Systems | A | A | 05, 12, 22 | Collaborative filtering, matrix factorization derived, implicit feedback, content-based & hybrid, offline eval of recommenders | `[ ]` |
| 33 | Interpretability, Fairness & Governance | P | A | 15, 20 | Permutation importance, SHAP, partial dependence, global vs local explanations, fairness metrics & mitigation, model cards | `[ ]` |
| 34 | ML in Production: Deployment & MLOps | P | A | 12, 20 | Serving on FastAPI/PostgreSQL/Redis, batch vs online scoring, model registry & versioning, monitoring incl. **sklearn 1.9 callbacks for long fits** (P6), drift detection, retraining loops, feature-store thinking | `[ ]` |
| 35 | Capstones & Interview Synthesis | P | A | 20, 34 | End-to-end projects on the target verticals (Cloud SaaS, FinTech, HealthTech), ML system design, the full FAANG/OSAMA interview battery | `[ ]` |

Notes on the index:
- **Other-track prerequisites:** none are hard dependencies — the reader profile in
  `CLAUDE.md` (working Python + backend stack) is assumed throughout; Stage C notebooks
  use FastAPI/PostgreSQL/Redis without teaching them.
- **Canonical homes** (each idea derived once, referenced everywhere — AUTHORING-GUIDE
  §6): gradient descent → 06 · eval/CV/baselines/leakage → 12 · bias–variance → 13 ·
  regularization → 14 · classification metrics, thresholds, imbalance, calibration → 15 ·
  distance metrics & scaling effects → 16 · kernel trick → 18 · impurity & information
  gain → 19 · boosting → 20 · cluster validity → 21 · PCA/SVD → 22 (SVD math in 05) ·
  backprop → 24 · convolution → 26 · recurrence & attention motivation → 27.
- The shared **PayFlow** spine datasets exist: generated deterministically by
  `_data/generate.py`, specified (tables, intentional-mess catalog M1–M14, targets,
  ground truth) in `_data/SPEC.md`, introduced to the reader in series 01, cleaned into
  `_data/processed/` by series 09–10, consumed by every series after.
- Old series "15 Stochastic Gradient Descent" is dissolved: the mechanism is canonical in
  06; `SGDRegressor`/`SGDClassifier` as scale tactics appear in 11/15.

## Legacy mapping (2019 folders → v0.2 series)

| Legacy folder | Absorbed by |
|---|---|
| 01 Introduction to ML | 01 |
| 02 Numpy | 02 |
| 03 Pandas | 03 |
| 04.0 Data Visualization | 04 |
| 04.1 Data Pre-Processing | 09 |
| 04.2 Exploratory Data Analysis | 10 |
| 05/06 Simple & Multiple Linear Regression | 11 |
| 07 Polynomial Regression | 13 |
| 08 Regularization | 14 |
| 09 K Nearest Neighbor | 16 |
| 10 Logistic Regression Classification | 15 |
| 11 Support Vector Machine | 18 |
| 12 Decission Tree | 19 |
| 13 Naive Bayes Classification | 17 |
| 14 Ensemble Classification | 20 |
| 15 Stocastic Gradient Descent | 06 (mechanism) · 11/15 (estimators) |
| 16–19 K Mean / DBSCAN / Mean Shift / Hierarchical | 21 |
| 20 Apriori Association Rule | 23 |
| 21 Natural Language Processing | 29 |
| 25 Open Computer Vision | 30 |
| 26 Audio Processing | 31 |
| ML Projects · ML Project Samples with GUI | source material for 35 |

New in v0.2 (no legacy counterpart): 05–08 (math lane), 12, 22, 24–28, 32–35.

---

## Currency log

**v0.3 — 2026-09-01, series-01 pre-flight (P1–P7 approved).** Evidence: all seven probed
2019-era API patterns fail in the pinned `.venv` (`load_boston`, `sklearn.externals.joblib`,
`LinearRegression(normalize=)`, `np.float`/`np.int`, `DataFrame.append`, chained inplace
assignment); pandas 3.0.5 makes copy-on-write mandatory and `str` the default string dtype;
scikit-learn 1.9.0 ships estimator callbacks, a narwhals dependency behind `set_output`,
and deprecates `criterion` on the legacy `GradientBoosting*` estimators.

| # | Patch | Applied to |
|---|---|---|
| P1 | Add `pyarrow` to the pinned environment (pandas 3.0 uses pyarrow-backed strings when present) | `requirements.txt` |
| P2 | Remove the legacy Jupyter-intro and definitional "AI vs ML vs DL vs DS / myths" material — padding for this reader (guide §6) | series 01 scope |
| P3 | Add *Reproducibility as an Engineering Contract* — absent from 2019, backbone of the evidence rule | series 01 tree (01.3) |
| P4 | Add *Problem Framing → learnable target*; leakage-safe construction stays canonical to 12 | series 01 tree (01.5) |
| P5 | Add an explicit 2019→2026 migration thread (the reader's own notes are the case study) | series 02, 03 scope |
| P6 | HistGradientBoosting as sklearn's default booster + `criterion` deprecation; sklearn 1.9 callbacks for search progress and long-fit monitoring | series 20, 34 scope |
| P7 | Strengthen polars scope: narwhals-backed `set_output` lets polars frames flow through sklearn pipelines | series 03 scope |

---

## Per-series topic trees

### Series 01 — The ML Landscape & Project Lifecycle (approved 2026-09-01)

Lane F · Level B (prose floor 1600 words) · 6 notebooks · folder
`01-ml-landscape-and-lifecycle/` · all examples on PayFlow data.

```text
01.1 [r] Rules or Learning? The Decision That Precedes the Model
         Hand-written dunning rule vs a learned model on PayFlow late payments.
         A: rule baseline, measured · B: simplest learned alternative, same harness ·
         C: cost-of-ownership + the rule as a permanent production floor.
         Incident: gateway-migration drift (M9) breaks the model while the rule holds.
01.2 [r] First Contact with the PayFlow Data Universe
         Six tables, their grain, the joins; first encounter with M1/M2/M7/M14.
         Incident: ARR overstated by summing mixed currencies raw (M2).
01.3 [r] Reproducibility as an Engineering Contract
         A: nondeterminism biting (score spread across unseeded splits) · B: seeded
         pipeline + fixed splits · C: run manifest (data hash, versions, metrics) + CI.
01.4 [r] The Taxonomy of Learning Problems — and Which PayFlow Question Is Which
         Supervised/unsupervised/self-supervised/RL; regression/classification/ranking/
         clustering; batch vs online; parametric vs non-parametric, each mapped to a
         PayFlow question and the series that owns it. Build collapsed to a shape probe.
01.5 [r] Problem Framing: From Business Question to Learnable Target
         "Reduce churn" → three defensible labels (horizon, population, cutoff) → three
         base rates → three different "good" models. Cost asymmetry as a design input.
01.6 [r] The ML Project Lifecycle and Where Projects Actually Die
         Framing → labels → baseline → model → eval → ship → monitor → retrain, walked
         end-to-end on PayFlow at shallow depth: the skeleton series 02-35 fill in.
```
