"""Scaffold generator for classical-ml notebooks.

Emits a .ipynb whose markdown skeleton matches AUTHORING-GUIDE.md §2 EXACTLY (headings,
order, header block, cold-open slot), with `<<FILL: ...>>` placeholders everywhere content
is required. `_tools/check.py` fails any notebook that still contains a placeholder, so
the checker + this scaffold together make template drift impossible: never type template
headings by hand.

Usage (from repo root):

    python _tools/scaffold.py 18.2 "The Kernel Trick and RBF SVMs" ^
        --series "18 Support Vector Machines" --level A ^
        --prereqs "14.1 (L2 regularization) . 16.1 (distance metrics & scaling)" ^
        --out "18-support-vector-machines"

Creates: <out>/<id> <Title>.ipynb  (refuses to overwrite an existing file).
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

LEVELS = {"B": "Beginner", "I": "Intermediate", "A": "Advanced"}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": text}


def build_cells(nb_id: str, title: str, series: str, level: str, prereqs: str) -> list[dict]:
    return [
        md(f"""# {nb_id} {title}

> **Prerequisites:** {prereqs}
> **What you'll learn:** <<FILL: 3-5 bullets, each a capability, not a topic>>
> **Level:** {level} · **Series:** {series}

> ⚡ <<FILL: cold open - specific time + symptom from THIS notebook's Production Scenario,
> then 'The cause: ...' naming the culprit but NOT the mechanism>>"""),
        md("""## Concept
### Plain-English Explanation
<<FILL: one short paragraph a non-engineer could follow; analogy only if exact>>"""),
        md("""### Technical Explanation
<<FILL: what/why/how + terminology bolded on first use + the numbers that matter>>"""),
        md("""### Mental Model
<<FILL: one or two sentences the reader can carry>>"""),
        md("""## How It Works
<<FILL: full derivation, step by step, each step justified; ASCII diagram of the
mechanism in a ```text block; complexity and trade-offs>>"""),
        md("""## Hands-On Build
### Stage A — from scratch
<<FILL: intro prose - what the raw implementation will prove>>"""),
        code("# <<FILL: Stage A - raw numpy/stdlib implementation, seeded, printed evidence>>\n"),
        md("<<FILL: prose after listing - what to look at in the output and why>>"),
        md("""### Stage B — idiomatic
<<FILL: intro prose - the standard library, used as its maintainers intend>>"""),
        code("# <<FILL: Stage B - idiomatic implementation + parity check vs Stage A>>\n"),
        md("<<FILL: prose - parity result, what the framework adds/hides>>"),
        md("""### Stage C — production
<<FILL: intro prose - what hardening this stage adds>>"""),
        code("# <<FILL: Stage C - production-hardened on Python/FastAPI/PostgreSQL/Redis>>\n"),
        md("<<FILL: prose - the production concerns this code answers>>"),
        md("""## Evaluation
<<FILL: metric + harness + captured number + baseline beaten + meaningful delta vs noise>>"""),
        code("# <<FILL: the eval harness run - captured scores with fold variance>>\n"),
        md("<<FILL: prose - reading of the numbers; which delta is signal, which is noise>>"),
        md("""## Design Patterns / Tradeoffs
<<FILL: >=2 genuine alternatives: how it works / advantages / disadvantages / failure
modes / when to use / when NOT - then a recommendation and the condition that changes it>>"""),
        md("""## Production Scenario
### Symptoms
<<FILL: what the on-call SEES - alert text, dashboard shape, log excerpts, user effect>>"""),
        md("""### Diagnosis
<<FILL: walk the ML observability ladder explicitly; show which signal eliminated which
hypothesis>>"""),
        md("""### Root Cause
<<FILL: one or two sentences, mechanistic>>"""),
        md("""### Fix
<<FILL: mitigation now AND permanent fix, code/config shown>>"""),
        md("""### Prevention
<<FILL: the eval, alert, test or review rule that would have caught it>>"""),
        md("""## Common Pitfalls
<<FILL: anti-patterns: what people do / why it breaks mechanistically / what to do
instead; use ⚠️ for genuine traps>>"""),
        md("""## Interview Questions
<<FILL: 5-8 numbered questions incl. at least one 'derive this', one 'design this',
one 'debug this'; answer shape in one line each>>"""),
        md("""## Key Takeaways
<<FILL: 5-8 bullets - decisions, not definitions>>"""),
        md("""## Related
<<FILL: backward links to prerequisites + forward links, one line each with the reason>>"""),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nb_id", help="notebook id, e.g. 18.2")
    ap.add_argument("title", help='notebook title, e.g. "The Kernel Trick and RBF SVMs"')
    ap.add_argument("--series", required=True, help='e.g. "18 Support Vector Machines"')
    ap.add_argument("--level", required=True, choices=list(LEVELS), help="B/I/A")
    ap.add_argument("--prereqs", default="<<FILL: prerequisites by id and topic>>")
    ap.add_argument("--out", required=True, type=Path, help="series folder")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.nb_id} {args.title}.ipynb"
    if target.exists():
        print(f"refusing to overwrite existing {target}")
        return 1

    nb = {
        "cells": build_cells(args.nb_id, args.title, args.series,
                             LEVELS[args.level], args.prereqs),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"scaffolded {target}")
    print("next: fill every <<FILL>> (labs first - run code before prose), "
          "then: python _tools/check.py \"" + str(target) + "\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
