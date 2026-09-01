# _tools — authoring harness

Stdlib-only; run from the repo root with the repo `.venv` interpreter
(`.venv\Scripts\python`), the same environment every notebook executes in.

## check.py — mechanical conformance checker

Verifies everything a script *can* verify (AUTHORING-GUIDE.md is the authority), so
review attention goes to substance, not mechanics (CLAUDE.md → REVIEW PRIORITY).

```bash
.venv\Scripts\python _tools/check.py "18-support-vector-machines/18.2 The Kernel Trick and RBF SVMs.ipynb"
.venv\Scripts\python _tools/check.py "18-support-vector-machines"          # whole series
.venv\Scripts\python _tools/check.py "18-support-vector-machines/_quiz.md" # quiz structure
.venv\Scripts\python _tools/check.py --strict <path>                       # WARN counts as FAIL
```

Exit 0 = no FAILs. What it checks per notebook: template headings exact & in order ·
header block + Level · cold open present · no `<<FILL>>` placeholders · no empty
sections · prose word floor by level (B 1600 / I 2000 / A 2400) · code cells ≤ 55 lines ·
max 2 consecutive code cells · all cells executed in order or explicitly marked · at
least one captured output · decimals in prose cross-checked against outputs (fabrication
tripwire, WARN) · Interview Questions 5–8 with derive/design/debug · Key Takeaways 5–8 ·
Related has id links · **bookish datasets (iris/titanic/make_blobs/…) without a
`# canonical-ok: <reason>` marker (FAIL)** · placeholder identifiers foo/baz/test123
(FAIL) · Fix has mitigation+permanent (WARN) · Diagnosis walks the ladder
(WARN) · filler phrases (WARN) · seed hygiene (WARN). Quiz mode: "How to use" at top,
exactly one `## Answers`, last section, and every `Q<n>` matched by an `A<n>` (markdown
emphasis tolerated, so `**Q1.**` counts). Lab mode: filename pattern,
docstring naming the notebook, `__main__` guard.

## scaffold.py — template generator

Emits a notebook whose skeleton matches the template *by construction*, with
`<<FILL: ...>>` placeholders check.py refuses to pass. Never type template headings by
hand.

```bash
.venv\Scripts\python _tools/scaffold.py 18.2 "The Kernel Trick and RBF SVMs" --series "18 Support Vector Machines" --level A --prereqs "14.1 (L2 regularization) . 16.1 (distance metrics)" --out "18-support-vector-machines"
```

The full authoring procedure that uses both: [../AUTHORING-PROTOCOL.md](../AUTHORING-PROTOCOL.md).
