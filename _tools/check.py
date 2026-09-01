"""Mechanical conformance checker for the classical-ml track.

Verifies everything a script CAN verify about authored content, so human/model review
attention goes to substance (CLAUDE.md > REVIEW PRIORITY). The authority for every rule
is AUTHORING-GUIDE.md; rule ids below reference it.

Usage (from repo root, any Python >= 3.10, stdlib only):

    python _tools/check.py <path> [<path> ...] [--strict]

Dispatch by path:
    *.ipynb        -> notebook conformance check
    _quiz.md       -> quiz structure check
    lab_*.py       -> lab script check
    <directory>    -> series check (all notebooks + _quiz.md + _lab/ inside)

Exit code 0 = no FAIL findings. --strict also promotes WARN to FAIL.
FAIL = objective template/evidence violation. WARN = suspicious, verify by hand.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- constants

LEVEL_FLOORS = {"Beginner": 1600, "Intermediate": 2000, "Advanced": 2400}

# (heading level, normalized text) in required order — AUTHORING-GUIDE §2.
REQUIRED_HEADINGS: list[tuple[int, str]] = [
    (2, "Concept"),
    (3, "Plain-English Explanation"),
    (3, "Technical Explanation"),
    (3, "Mental Model"),
    (2, "How It Works"),
    (2, "Hands-On Build"),
    (3, "Stage A - from scratch"),
    (3, "Stage B - idiomatic"),
    (3, "Stage C - production"),
    (2, "Evaluation"),
    (2, "Design Patterns / Tradeoffs"),
    (2, "Production Scenario"),
    (3, "Symptoms"),
    (3, "Diagnosis"),
    (3, "Root Cause"),
    (3, "Fix"),
    (3, "Prevention"),
    (2, "Common Pitfalls"),
    (2, "Interview Questions"),
    (2, "Key Takeaways"),
    (2, "Related"),
]
KNOWN_H2 = {t for lvl, t in REQUIRED_HEADINGS if lvl == 2}

MAX_CODE_LINES = 55          # §4.3
MAX_CONSECUTIVE_CODE = 2     # §4.3: no 3 consecutive blocks without prose

# First-3-lines markers that excuse a code cell from carrying executed output. §4.3/§4.4
SKIP_MARKERS = (
    "# illustrative - not captured output",
    "# long-running",
    "# GPU recommended",
    "# GPU required",
)

BANNED_PHRASES = (  # filler a weaker model reaches for; AUTHORING-GUIDE §0 voice rule
    "in this notebook",
    "we will explore",
    "let's dive",
    "delve",
    "it is important to note",
    "it's important to note",
    "as we can see",
    "buckle up",
    "in today's",
    "welcome to",
)

# words whose presence on a line whitelists its decimals (version strings, ids, §refs)
NUMBER_CONTEXT_ALLOW = (
    "version", "python", "pytorch", "numpy", "pandas", "scikit", "sklearn",
    "series", "notebook", "§", "v0.", "v1.", "v2.", "nbformat",
)

# Bookish datasets (guide §4.2): FAIL unless the cell carries '# canonical-ok: <reason>'.
BOOKISH_PATTERNS = (
    r"load_iris\(", r"load_wine\(", r"load_breast_cancer\(", r"load_diabetes\(",
    r"load_digits\(", r"fetch_california_housing\(", r"load_boston",
    r"make_blobs\(", r"make_classification\(", r"make_moons\(", r"make_circles\(",
    r"make_regression\(", r"sns\.load_dataset\(",
    r"[\"'](?:titanic|iris|tips|penguins|boston)\.csv[\"']",
    r"fetch_openml\(\s*[\"'](?:titanic|iris|boston)",
)
CANONICAL_OK = "# canonical-ok:"
# `foo\w*` would match ordinary words like "footnote"/"food"; match only the
# placeholder forms actually used as stand-in names.
PLACEHOLDER_NAMES = re.compile(
    r"\b(foo(?:bar|baz|\d+)?|bar_?\d+|baz|qux|test123|lorem)\b", re.I)


@dataclass
class Report:
    path: Path
    findings: list[tuple[str, str, str]] = field(default_factory=list)  # (level, rule, msg)
    checked: int = 0

    def fail(self, rule: str, msg: str) -> None:
        self.findings.append(("FAIL", rule, msg))

    def warn(self, rule: str, msg: str) -> None:
        self.findings.append(("WARN", rule, msg))

    def ok(self) -> None:
        self.checked += 1

    @property
    def n_fail(self) -> int:
        return sum(1 for lvl, _, _ in self.findings if lvl == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for lvl, _, _ in self.findings if lvl == "WARN")


# ---------------------------------------------------------------- helpers

def norm_heading(text: str) -> str:
    """Normalize a heading for comparison: unify dashes, collapse whitespace."""
    text = text.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def outputs_text(cell: dict) -> str:
    """All text content of a code cell's outputs."""
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        for key in ("text",):
            v = out.get(key)
            if v:
                chunks.append("".join(v) if isinstance(v, list) else v)
        data = out.get("data", {})
        for mime, v in data.items():
            if mime.startswith("text/"):
                chunks.append("".join(v) if isinstance(v, list) else v)
        if out.get("output_type") == "error":
            chunks.append("\n".join(out.get("traceback", [])))
    return "\n".join(chunks)


def strip_fences(md: str) -> str:
    """Remove fenced code blocks from markdown text."""
    out_lines, in_fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out_lines.append(line)
    return "\n".join(out_lines)


def iter_headings(md: str) -> list[tuple[int, str]]:
    heads = []
    for line in strip_fences(md).splitlines():
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            heads.append((len(m.group(1)), norm_heading(m.group(2))))
    return heads


def prose_lines(md: str) -> list[str]:
    """Markdown lines that count as prose: no fences, no tables, no headings."""
    lines = []
    for line in strip_fences(md).splitlines():
        s = line.strip()
        if not s or s.startswith("|") or s.startswith("#"):
            continue
        lines.append(s)
    return lines


# ---------------------------------------------------------------- notebook check

def check_notebook(path: Path) -> Report:
    rep = Report(path)
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        rep.fail("NB00", f"cannot parse notebook JSON: {exc}")
        return rep

    cells = nb.get("cells", [])
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    all_md = "\n\n".join(cell_source(c) for c in md_cells)

    # F01 — no template placeholders left
    fills = re.findall(r"<<FILL[^>]*>>", "\n".join(cell_source(c) for c in cells))
    if fills:
        rep.fail("F01", f"{len(fills)} unfilled template placeholder(s) remain, "
                        f"first: {fills[0][:70]}")
    else:
        rep.ok()

    # H01 — title line
    first_md = cell_source(md_cells[0]) if md_cells else ""
    m = re.match(r"^#\s+(\d{2}\.\d{1,2})\s+\S", first_md)
    if not m:
        rep.fail("H01", "first markdown cell must start '# <series>.<n> <Title>' "
                        "(e.g. '# 18.2 The Kernel Trick and RBF SVMs')")
        nb_id = None
    else:
        rep.ok()
        nb_id = m.group(1)
        fname_id = re.match(r"^(\d{2}\.\d{1,2})\s", path.name)
        if fname_id and fname_id.group(1) != nb_id:
            rep.warn("H01", f"filename id {fname_id.group(1)} != title id {nb_id}")

    # H02 — header block fields + Level
    level = None
    for label in ("**Prerequisites:**", "**What you'll learn:**", "**Level:**", "**Series:**"):
        if label not in first_md:
            rep.fail("H02", f"header block missing '{label}' line")
        else:
            rep.ok()
    lm = re.search(r"\*\*Level:\*\*\s*(Beginner|Intermediate|Advanced)", first_md)
    if lm:
        level = lm.group(1)
    else:
        rep.fail("H02", "Level must be one of Beginner/Intermediate/Advanced")

    # H03 — cold open before ## Concept
    pre_concept = all_md.split("## Concept")[0]
    if not re.search(r"^>\s*⚡", pre_concept, flags=re.M):
        rep.fail("H03", "cold-open blockquote ('> ⚡ ...') missing before '## Concept'")
    else:
        rep.ok()

    # H04 — required headings, exact, in order, no duplicates
    heads = iter_headings(all_md)
    idx = 0
    for want_lvl, want_txt in REQUIRED_HEADINGS:
        found = False
        while idx < len(heads):
            lvl, txt = heads[idx]
            idx += 1
            if lvl == want_lvl and txt == want_txt:
                found = True
                break
        if not found:
            rep.fail("H04", f"missing or out-of-order heading: {'#' * want_lvl} {want_txt}")
    if rep.n_fail == 0 or all(r != "H04" for _, r, _ in rep.findings):
        rep.ok()
    seen: dict[str, int] = {}
    for lvl, txt in heads:
        if (lvl, txt) in [(l, t) for l, t in REQUIRED_HEADINGS]:
            seen[txt] = seen.get(txt, 0) + 1
    dups = [t for t, n in seen.items() if n > 1]
    if dups:
        rep.fail("H04", f"duplicated template heading(s): {', '.join(dups)}")
    for lvl, txt in heads:
        if lvl == 2 and txt not in KNOWN_H2:
            rep.warn("H05", f"non-template H2 heading '## {txt}' — allowed only if deliberate")

    # Section map: heading -> prose text until next heading of same-or-higher level
    sections: dict[str, str] = {}
    flat = strip_fences(all_md).splitlines()
    current, buf = None, []
    levels = {t: l for l, t in REQUIRED_HEADINGS}
    for line in flat:
        hm = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if hm:
            txt = norm_heading(hm.group(2))
            if txt in levels:
                if current:
                    sections[current] = "\n".join(buf)
                current, buf = txt, []
                continue
        if current is not None:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf)

    # S01 — no required section may be empty (collapsed sections keep a one-line why)
    for _, txt in REQUIRED_HEADINGS:
        body = sections.get(txt, "")
        # count prose lines only for leaf sections; container headings (Concept,
        # Hands-On Build, Production Scenario) may hold only their subsections
        if txt in ("Concept", "Hands-On Build", "Production Scenario"):
            continue
        if not [ln for ln in body.splitlines() if ln.strip()]:
            rep.fail("S01", f"section '{txt}' is empty — write it, or keep the heading "
                            f"with the one-line justification for its collapse")
        else:
            rep.ok()

    # W01 — word floor by level (prose only, excluding code/tables/headings)
    words = sum(len(ln.split()) for ln in prose_lines(all_md))
    if level:
        floor = LEVEL_FLOORS[level]
        if words < floor:
            rep.fail("W01", f"prose word count {words} below {level} floor {floor}")
        else:
            rep.ok()

    # C01 — code cell length cap
    for i, c in enumerate(code_cells):
        n = len(cell_source(c).splitlines())
        if n > MAX_CODE_LINES:
            rep.fail("C01", f"code cell #{i + 1} has {n} lines (cap {MAX_CODE_LINES}) — split it")
    rep.ok()

    # C02 — max 2 consecutive code cells
    run = 0
    for c in cells:
        if c.get("cell_type") == "code":
            run += 1
            if run > MAX_CONSECUTIVE_CODE:
                rep.fail("C02", f"{run} consecutive code cells — interleave prose "
                                f"that says what to look at")
                break
        else:
            run = 0
    else:
        rep.ok()

    # C03 — executed, in order, or explicitly marked
    ecs: list[int] = []
    for i, c in enumerate(code_cells):
        ec = c.get("execution_count")
        head = "\n".join(cell_source(c).splitlines()[:3]).lower()
        marked = any(mk.lower() in head for mk in SKIP_MARKERS)
        if ec is None and not marked:
            rep.fail("C03", f"code cell #{i + 1} not executed and not marked "
                            f"(allowed markers: {', '.join(SKIP_MARKERS)})")
        if ec is not None:
            ecs.append(ec)
    if ecs and any(b <= a for a, b in zip(ecs, ecs[1:])):
        rep.fail("C03", "execution counts not strictly increasing — re-run top to bottom")
    elif ecs and (ecs[0] != 1 or ecs != list(range(1, len(ecs) + 1))):
        rep.warn("C03", "execution counts not a clean 1..N — ship from 'Restart & Run All'")
    else:
        rep.ok()

    # E01 — at least one cell with real captured output
    if not any(outputs_text(c).strip() for c in code_cells):
        rep.fail("E01", "no code cell has any captured output — evidence rule §4.3")
    else:
        rep.ok()

    # E02 — decimals in prose must exist in code/outputs (fabrication tripwire)
    known = "\n".join(cell_source(c) + "\n" + outputs_text(c) for c in code_cells)
    known_nums = set(re.findall(r"\d+\.\d+", known))
    body_md = "\n\n".join(cell_source(c) for c in md_cells[1:])  # skip header block
    suspicious = []
    for ln in prose_lines(body_md):
        low = ln.lower()
        if any(w in low for w in NUMBER_CONTEXT_ALLOW):
            continue
        for num in re.findall(r"(?<![\d.\w])\d+\.\d+(?![\d.])", ln):
            # id-shaped numbers (2-digit series 01-35) are cross-references, not metrics
            if re.fullmatch(r"\d{2}\.\d{1,2}", num) and 1 <= int(num.split(".")[0]) <= 35:
                continue
            if num not in known_nums and num not in suspicious:
                suspicious.append(num)
    if suspicious:
        rep.warn("E02", f"decimal(s) in prose not found in any code/output: "
                        f"{', '.join(suspicious[:8])} — captured, or fabricated? "
                        f"(mark '# illustrative - not captured output' if derived)")
    else:
        rep.ok()

    # I01 — interview questions: 5-8 items, derive/design/debug all present
    iq = sections.get("Interview Questions", "")
    items = [ln for ln in iq.splitlines() if re.match(r"^\s*(\d+[.)]|[-*])\s+\S", ln)]
    if not 5 <= len(items) <= 8:
        rep.fail("I01", f"Interview Questions has {len(items)} items (need 5-8)")
    else:
        rep.ok()
    for kind in ("derive", "design", "debug"):
        if kind not in iq.lower():
            rep.fail("I01", f"Interview Questions must include a '{kind} this' question")

    # K01 — key takeaways 5-8 bullets
    kt = [ln for ln in sections.get("Key Takeaways", "").splitlines()
          if re.match(r"^\s*[-*]\s+\S", ln)]
    if not 5 <= len(kt) <= 8:
        rep.fail("K01", f"Key Takeaways has {len(kt)} bullets (need 5-8)")
    else:
        rep.ok()

    # R01 — related section links back and forward by id
    rel = sections.get("Related", "")
    if not re.search(r"\d{2}\.\d{1,2}", rel):
        rep.fail("R01", "Related must reference at least one notebook id (backward link)")
    else:
        rep.ok()

    # P01 — fix has mitigation + permanent; diagnosis walks the ladder
    fix = sections.get("Fix", "").lower()
    if fix and not ("mitigat" in fix and "permanent" in fix):
        rep.warn("P01", "Fix should show BOTH 'mitigation now' and 'permanent fix'")
    diag = sections.get("Diagnosis", "").lower()
    ladder_hits = sum(w in diag for w in
                      ("alert", "dashboard", "log", "drift", "schema", "version", "distribution"))
    if diag and ladder_hits < 2:
        rep.warn("P01", "Diagnosis doesn't visibly walk the ML observability ladder "
                        "(alert/dashboards/logs/data checks/drift/version diff)")

    # T01 — banned filler phrases
    low_md = strip_fences(all_md).lower()
    hits = [p for p in BANNED_PHRASES if p in low_md]
    if hits:
        rep.warn("T01", f"filler phrase(s) found: {', '.join(hits[:5])} — cut them (§0 voice test)")
    else:
        rep.ok()

    # D01 — bookish datasets require an explicit '# canonical-ok: <reason>' in the cell
    for i, c in enumerate(code_cells):
        src = cell_source(c)
        hits = [p for p in BOOKISH_PATTERNS if re.search(p, src)]
        if hits and CANONICAL_OK not in src:
            rep.fail("D01", f"code cell #{i + 1} uses a bookish dataset "
                            f"({re.search(hits[0], src).group(0)}) without a "
                            f"'{CANONICAL_OK} <reason>' justification — examples live in "
                            f"the PayFlow universe (_data/, guide §4.1-4.2)")
    rep.ok()

    # D02 — placeholder identifiers are never the subject of anything
    all_code = "\n".join(cell_source(c) for c in code_cells)
    ph = sorted({m.group(0) for m in PLACEHOLDER_NAMES.finditer(all_code)} |
                {m.group(0) for m in PLACEHOLDER_NAMES.finditer(strip_fences(all_md))})
    if ph:
        rep.fail("D02", f"placeholder identifier(s) {', '.join(ph[:6])} — use real "
                        f"names from the running system (guide §4.2)")
    else:
        rep.ok()

    # V01 — seed hygiene (spot checks)
    all_code = "\n".join(cell_source(c) for c in code_cells)
    if "train_test_split(" in all_code and "random_state" not in all_code:
        rep.warn("V01", "train_test_split used without random_state anywhere")
    if re.search(r"np\.random\.(rand|randn|randint|choice|shuffle)\b", all_code) \
            and "default_rng" not in all_code and "np.random.seed" not in all_code:
        rep.warn("V01", "legacy np.random.* without a seed — use np.random.default_rng(seed)")
    if "import torch" in all_code and "manual_seed" not in all_code:
        rep.warn("V01", "torch used without torch.manual_seed")

    return rep


# ---------------------------------------------------------------- quiz / lab / series

def check_quiz(path: Path) -> Report:
    rep = Report(path)
    text = path.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:40]).lower()
    if "how to use" not in head:
        rep.fail("Q01", "quiz must open with the short 'How to use' protocol (§11)")
    else:
        rep.ok()
    answers = [m.start() for m in re.finditer(r"^## Answers\s*$", text, flags=re.M)]
    if len(answers) != 1:
        rep.fail("Q02", f"need exactly one '## Answers' section, found {len(answers)}")
    else:
        rep.ok()
        tail = text[answers[0] + 1:]
        if re.search(r"^## ", tail, flags=re.M):
            rep.fail("Q02", "'## Answers' must be the LAST H2 section (answers at the bottom)")
        else:
            rep.ok()
        # Count Q<n>/A<n> markers, tolerating markdown emphasis (**Q1.**, _Q1._).
        head = text[: answers[0]]
        emph = r"^\s*(?:[*_]{1,2}\s*)?"
        qs = set(re.findall(emph + r"Q(\d+)\b", head, flags=re.M))
        ans = set(re.findall(emph + r"A(\d+)\b", tail, flags=re.M))
        if qs or ans:
            missing = sorted(qs - ans, key=int)
            extra = sorted(ans - qs, key=int)
            if missing:
                rep.warn("Q03", f"{len(qs)} questions, {len(ans)} answers; no answer for "
                                f"Q{', Q'.join(missing)}")
            elif extra:
                rep.warn("Q03", f"answer(s) with no question: A{', A'.join(extra)}")
            else:
                rep.ok()
        else:   # unnumbered quiz - fall back to counting list items
            n_q = len(re.findall(emph + r"\d+[.)]", head, flags=re.M))
            n_a = len(re.findall(emph + r"\d+[.)]", tail, flags=re.M))
            if n_q and n_a and n_a < n_q:
                rep.warn("Q03", f"{n_q} questions but only {n_a} answers")
    return rep


def check_lab(path: Path) -> Report:
    rep = Report(path)
    if not re.match(r"^lab_\d{2}\.\d{1,2}_[a-z0-9_]+\.py$", path.name):
        rep.fail("L01", "lab filename must match lab_<id>_<slug>.py, e.g. lab_18.2_kernel_trick.py")
    else:
        rep.ok()
    src = path.read_text(encoding="utf-8")
    try:
        doc = ast.get_docstring(ast.parse(src))
    except SyntaxError as exc:
        rep.fail("L02", f"syntax error: {exc}")
        return rep
    if not doc or not re.search(r"\d{2}\.\d{1,2}", doc):
        rep.fail("L02", "module docstring must name the notebook id and listings it reproduces")
    else:
        rep.ok()
    if '__name__' not in src or '__main__' not in src:
        rep.fail("L03", "missing `if __name__ == \"__main__\":` guard (Windows multiprocessing)")
    else:
        rep.ok()
    return rep


def check_series(path: Path, strict: bool) -> list[Report]:
    reports: list[Report] = []
    notebooks = sorted(p for p in path.glob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
    for nb in notebooks:
        reports.append(check_notebook(nb))
    quiz = path / "_quiz.md"
    if quiz.exists():
        reports.append(check_quiz(quiz))
    elif notebooks:
        r = Report(quiz)
        r.fail("Q00", "_quiz.md missing — it must be extended with every notebook (§11)")
        reports.append(r)
    lab_dir = path / "_lab"
    if lab_dir.is_dir():
        for lab in sorted(lab_dir.glob("lab_*.py")):
            reports.append(check_lab(lab))
        if not (lab_dir / "README.md").exists():
            r = Report(lab_dir / "README.md")
            r.warn("L04", "_lab/README.md missing (one page: what each script shows)")
            reports.append(r)
    if notebooks and not (path / "_recap.md").exists():
        r = Report(path / "_recap.md")
        r.warn("S02", "_recap.md not written yet (required when the series completes)")
        reports.append(r)
    return reports


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--strict", action="store_true", help="treat WARN as FAIL")
    args = ap.parse_args(argv)

    reports: list[Report] = []
    for p in args.paths:
        if not p.exists():
            r = Report(p)
            r.fail("IO0", "path does not exist")
            reports.append(r)
        elif p.is_dir():
            reports.extend(check_series(p, args.strict))
        elif p.suffix == ".ipynb":
            reports.append(check_notebook(p))
        elif p.name == "_quiz.md":
            reports.append(check_quiz(p))
        elif p.name.startswith("lab_") and p.suffix == ".py":
            reports.append(check_lab(p))
        else:
            r = Report(p)
            r.warn("IO1", "unrecognized target — expected .ipynb, _quiz.md, lab_*.py or a directory")
            reports.append(r)

    total_fail = total_warn = 0
    for rep in reports:
        print(f"== {rep.path}")
        for lvl, rule, msg in rep.findings:
            print(f"   {lvl} {rule}: {msg}")
        status = "OK" if rep.n_fail == 0 else "FAIL"
        print(f"   -> {status}: {rep.n_fail} fail, {rep.n_warn} warn\n")
        total_fail += rep.n_fail
        total_warn += rep.n_warn

    print(f"TOTAL: {total_fail} fail, {total_warn} warn across {len(reports)} file(s)")
    if total_fail or (args.strict and total_warn):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
