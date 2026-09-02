# Next-session handoff

Paste the block below into a new chat. Update this file at every handoff.

---

```text
classical-ml track — AUTHORING MODE. Task: begin series 02 "NumPy & Vectorized Computing".

Working directory: D:\Learn\classical-ml   (repo .venv is the execution environment)

STATE
  Curriculum baseline : v0.3, 2026-09-01
  Series 01 "The ML Landscape & Project Lifecycle" is COMPLETE and closed out:
    68ace36  original authoring
    dccac67  fixes from review round 1
    da03aa7  fixes from review round 2
    (uncommitted) round 3 correctness fixes + round 4 structural pass — Aditya to commit
  Contents: 6 notebooks (01.1-01.6), 6 lab scripts, _quiz.md (50 Q/A), _recap.md, _review.md.
  NO findings remain open. Three review rounds plus a fourth structural pass; the full
  record is in 01-ml-landscape-and-lifecycle/_review.md. check.py: 0 fail, 5 warn, and all
  five warns are verified-legitimate (cross-references to another notebook's captured
  output, or illustrative round numbers). Do not spend time re-litigating series 01, and do
  not spend time on those five warns.
  Series 01 is held at [r] in CURRICULUM.md pending Aditya's sign-off, not pending work.

YOUR TASK
  1. PRE-FLIGHT CURRENCY CHECK for series 02 (CLAUDE.md > CONTENT CURRENCY). What changed in
     numpy/array computing since the v0.1 2026-08-31 baseline. Propose patches to
     CURRICULUM.md — do NOT silently rewrite it. Wait for approval, then bump the version.
  2. Show the series 02 TOPIC TREE and STOP. CLAUDE.md's approval gate: nothing is authored
     until Aditya says "Approved" / "Start".
  3. Then author 02.1 per AUTHORING-PROTOCOL, one notebook per response.

READ THE STANDARD FIRST
  CLAUDE.md               — role, learner profile, REVIEW PRIORITY (content before mechanics)
  AUTHORING-GUIDE.md      — the quality bar every notebook is judged against
  AUTHORING-PROTOCOL.md   — the authoring/verify procedure, incl. the loop and its gates
  _data/SPEC.md           — the PayFlow data universe and its intentional-mess catalog M1-M14
  _tools/README.md        — what check.py verifies mechanically (so you don't duplicate it)

DISCIPLINE CARRIED FORWARD FROM SERIES 01 (four review rounds bought these)
  • BUILD THE ARTIFACT A SECTION RECOMMENDS. Series 01's biggest structural defect was
    three notebooks whose Stage C argued for a production artifact and never built one
    (01.5's Permanent Fix demanded a label spec for three drafts). Prose describing an
    artifact and code building one are indistinguishable to every mechanical check, and
    prose is cheaper to write — so the gap accumulates silently. Stage C is code.
  • SNAPSHOT AND DIFF AROUND ANY FIX PASS. Key every code cell's captured output by its
    SOURCE (not index, so insertions don't shift the comparison), snapshot before, diff
    after. Series 01 carries ~1,119 numeric claims in hand-written prose across six
    notebooks whose labs all import lab_01.1 — one lab change silently stales dozens of
    them, and this is the single largest source of review churn. The round-4 pass proved
    it changed exactly one existing output that way.
  • KEEP ONE DEFINITION OF EACH SHARED CONSTANT. Two definitions of "when the data ends"
    (a hardcoded date and a measured max) sat in one lab and disagreed by three days.
  • BAND MUST MATCH THE PAIR. A paired claim needs a paired band; series 01 violated its
    own invariant three times. Never borrow another comparison's band.
  • EVIDENCE MUST BE ERA-CONSISTENT with the running system's own timeline.
  • CHECK EVERY CALL SITE, not just a lab's main(); and check _quiz.md/_recap.md, which go
    stale with no execution to fail. check.py's X01 catches numbers, never prose or reasoning.
  • A CLAIM ABOUT RERUNS MUST BE WIDENED, NOT TIGHTENED, when a rerun lands outside it.

ENVIRONMENT
  .venv\Scripts\python for everything. Pinned: python 3.14.4, numpy 2.5.2, pandas 3.0.5,
  pyarrow 25.0.1, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.0, matplotlib 3.11.1,
  seaborn 0.13.2, jupyterlab 4.6.3 (requirements.txt). Verify with pip show, not memory —
  series 02's whole subject is one of these.
  Data: .venv\Scripts\python _data\generate.py — gitignored, regenerates byte-identical,
  hashes in _data/SPEC.md.
  To re-execute a notebook after editing it, run nbconvert FROM INSIDE the series folder
  so the notebook's Path.cwd()/"_lab" module load resolves:
    cd "02-numpy-and-vectorized-computing" && ..\.venv\Scripts\python -m jupyter nbconvert ^
      --to notebook --execute --inplace "<file>.ipynb" --ExecutePreprocessor.timeout=3600
  Notebook builder scripts are NOT kept in the repo. To change a notebook, either scaffold
  a fresh one (_tools/scaffold.py) or edit the .ipynb directly and re-execute.

GIT
  Claude never commits or pushes. Keep the tree clean; Aditya reviews and commits.
```
