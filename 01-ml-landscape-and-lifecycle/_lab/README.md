# _lab — series 01

Committed experiment scripts. Every number shown in a series-01 notebook is produced by
running one of these in the repo `.venv`, seeded. Regenerate the data first if needed:
`.venv\Scripts\python _data\generate.py`.

| Script | Notebook | What it produces |
|---|---|---|
| `lab_01.1_rules_vs_learning.py` | 01.1 | The full rules-vs-learning comparison on PayFlow late payments: dataset construction and label base rates (L1), the hand-written dunning rule at its own queue size (L2), logistic regression and a gradient-boosting strength check at the same operating point (L3), bootstrap variance on both precision and dollars (L4), the 2025-07 gateway-migration incident under a fixed production threshold (L5), the refit and what it does and does not repair (L5b), and the value-weighted objective (L6). |
| `lab_01.6_lifecycle.py` | 01.6 | The lifecycle as an instrumented object: one clean end-to-end run with eight stage artifacts (L1), a seven-gate suite injected with six defects (L2), the survivorship defect present in this series' own pipeline (L2b), leakage and random-split effects quantified with the test set held fixed (L3), the detection-distance table separating catchable defects from mis-specifications (L4), and the retrain decision as a function (L5). |
| `lab_01.5_problem_framing.py` | 01.5 | Label design on PayFlow churn: the base-rate design space over population and horizon (L1), the naive "has churned" label and its already-resolved positives (L2), the proxy-versus-operational label comparison with resplit variance (L3), horizon against actionability (L4), break-even precision and the retention cost curve including flat versus proportional intervention cost (L5), and explicit versus silence-inferred labels with their detection lag (L6). |
| `lab_01.4_taxonomy.py` | 01.4 | One invoice table framed as four learning problems (L1), how much the framings disagree about which invoices matter (L2), the threshold-versus-capacity incident (L3), the estimator API read as a taxonomy including where that inference breaks (L4), label availability by recency (L5), and what parametric/non-parametric and batch/online cost in artifact bytes and latency (L6). |
| `lab_01.3_reproducibility.py` | 01.3 | Reproducibility measured rather than asserted: the split lottery across unseeded evaluation draws (L1), training-stability versus evaluation-stability variance (L1b), the data-snapshot lottery (L2), determinism achieved via hashed predictions (L3), a per-component seed-sensitivity audit showing which `random_state` actually matters (L3b), the run manifest (L4), and `reproduce()` as a CI check (L5). Imports 01.1's dataset builder rather than duplicating it. |
| `lab_01.2_data_universe.py` | 01.2 | First contact with the raw exports: what is literally in the files via stdlib `csv` (L1), the grain of all six tables and where the declared key is not unique (L2), join fan-out and silent row loss (L3), the mixed-currency aggregation incident plus the pandas 3 string-sum trap (L4), the nine-check ingestion contract (L5), and billed-versus-collected reconciliation in USD (L6). |

Run:

```bash
.venv\Scripts\python "01-ml-landscape-and-lifecycle/_lab/lab_01.1_rules_vs_learning.py"
```

Runtime ≈ 1–2 minutes on CPU. Deterministic given `_data/raw` at the SPEC.md manifest
hashes.
