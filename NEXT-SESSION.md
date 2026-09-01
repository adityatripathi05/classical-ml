# Next-session handoff

Paste the block below into a new chat to resume authoring. Update this file at every
series boundary; it is the only state a fresh session needs beyond the repo itself.

---

```text
classical-ml track — AUTHORING MODE.

Working directory: D:\Learn\classical-ml   (repo .venv is the execution environment)

STATE
  Curriculum baseline : v0.3, 2026-09-01
  Progress            : 1 of 35 series authored
  Last authored       : Series 01 "The ML Landscape & Project Lifecycle" — 6 notebooks
                        (01.1–01.6), 6 lab scripts, _quiz.md (47 Q/A), _recap.md.
                        Committed as 68ace36. Status [r] IN REVIEW, not done.
  REVIEW              : End-to-end review (3 independent passes, every finding
                        re-verified by re-running the labs) found 7 invalid-reasoning
                        blockers, 6 evidence contradictions and 6 mischaracterisations.
                        ALL 13 ARE FIXED as of 2026-09-01; five notebooks carry corrected
                        mechanisms and three carry new captured evidence. Series checks
                        0 fail across 13 files. Full record in
                        01-ml-landscape-and-lifecycle/_review.md.
  STILL OPEN          : Only the minors and the structural/editorial items in _review.md
                        — judgement calls (Stage C depth in 01.4/01.5/01.6, 01.4's Stage A
                        opacity, duplicated listings, interview answer shapes that give
                        away the numbers, three "textbooks get this wrong" openers), not
                        defects. Aditya's call whether to spend on them before series 02.
                        Series 01 is otherwise ready to promote from [r] to [x].
  Next                : EITHER the series-01 fix pass (worklist in _review.md, blockers
                        B1-B6 first), OR series 02 "NumPy & Vectorized Computing"
                        (lane T, level B, depends on 01) — Aditya's call which first.
                        Folder for 02: 02-numpy-vectorized-computing/
  Series 02 has no currency check and no topic tree yet.

READ FIRST (in this order)
  CLAUDE.md                — role, learner profile, two modes, approval gates
  AUTHORING-PROTOCOL.md    — the procedure to follow, phase by phase
  AUTHORING-GUIDE.md       — the quality standard the protocol implements
  CURRICULUM.md            — scope authority: series 02 row, currency log, series-01 tree
  _data/SPEC.md            — the PayFlow data universe and its intentional-mess catalog
  Then read 01.6 (and ideally 01.1) end-to-end. Series 01 is the conformance benchmark
  and the reference for voice, depth and evidence discipline.

FIRST ACTION — do NOT write teaching content yet
  Protocol Phase 0 gate: series 02 needs (a) a pre-flight currency check and (b) an
  approved topic tree before any notebook is authored. So the first response must be:
    1. Pre-flight currency check for series 02 — verify facts by RUNNING them in the
       .venv (numpy 2.5.2 is pinned; check what changed since the 2019 notes: removed
       aliases, copy semantics, NEP 50 promotion, and anything newer). Search the web
       for developments since Jan 2026 rather than answering from memory.
    2. Propose patches (add/modify/remove) against CURRICULUM.md — do not silently edit.
    3. Present the series-02 topic tree (one subtopic = one notebook).
    4. STOP and wait for approval.

ENVIRONMENT
  .venv\Scripts\python is the interpreter for everything (labs, nbconvert, check.py).
  Pinned: python 3.14.4, numpy 2.5.2, pandas 3.0.5, pyarrow 25.0.1, scipy 1.18.1,
  statsmodels 0.15.0, scikit-learn 1.9.0, matplotlib 3.11.1, seaborn 0.13.2,
  jupyterlab 4.6.3  (requirements.txt).
  Data: .venv\Scripts\python _data\generate.py  (gitignored; regenerates byte-identical,
  hashes recorded in _data/SPEC.md — verified matching as of 2026-09-01).

THE LOOP (see _tools/README.md)
  lab first → generate notebook → execute it from INSIDE the series folder via nbconvert
  → _tools/check.py until it exits 0 → extend _quiz.md → update CURRICULUM.md status.
  One notebook per response. Never batch.

INVARIANTS SERIES 01 ESTABLISHED (assume as habits, do not re-teach)
  baseline-first · matched operating point · noise band before declaring a winner ·
  business objective reported beside the training metric · every number executed ·
  contracts over conventions · written specs are what make checks possible.

CARRIED-FORWARD DEBT (do not re-litigate; these have owners)
  • Survivorship: the modelling table is an inner join to payments, so series-01 numbers
    describe invoices that were eventually paid. Named and quantified in 01.6; fixed
    properly in series 12 (population definition as part of validation).
  • Leakage handled informally (reminder_count, total_lifetime_value_usd excluded by
    hand) — taxonomy and temporal CV are series 12's canonical subject.
  • Cleaning lives in lab code, taught as a subject in series 09.
  • Legacy 2019 folders sit untracked at the repo root as a scope reference; delete each
    once its replacement series reaches done.

GIT
  Claude never commits or pushes. Keep the tree clean and coherent; Aditya reviews and
  commits.
```
