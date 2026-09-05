# Series 01 — Cheatsheet: The ML Landscape & Project Lifecycle

The ten-second lookup layer. Each row cites the notebook that earned it — jump there for
the mechanism and the captured evidence. Retrieval practice lives in `_quiz.md`; this sheet
is for mid-task "what's the rule again?". Regenerated whenever a notebook's claims change.

## The rules (imperative, one line each)

1. Measure the incumbent rule and a random policy **before** training anything; report the
   trio random / rule / model, always. *(01.1)*
2. Compare policies at a **matched operating point** — same k or same threshold — or you
   are measuring appetite, not skill. *(01.1)*
3. Report the **business objective beside the proxy**: precision can rise while money
   falls, and both be true. *(01.1)*
4. No winner without a **noise band — measured for the exact quantity claimed**: a paired
   delta needs a paired band, never a marginal spread. *(01.3, violated by the series
   itself three times before it stuck)*
5. Every shown number is **produced by committed, seeded lab code**; anything else is
   marked illustrative. *(all)*
6. Prefer **constraints to conventions** at silent-failure boundaries: types, contracts,
   read-only flags, blocking checks. *(01.2)*
7. **Write decisions down or they cannot be gated**: label spec, scoring population,
   serving contract, framing — the manifest config block is where they live. *(01.5, 01.6)*

## Traps — symptom → mechanism → probe / fix

| Symptom | Mechanism | Probe / fix | Nb |
|---|---|---|---|
| Precision **rises** while the business decays (0.472 → 0.570; late rate 0.263 → 0.389) | Precision over a self-selected queue rises when positives densify; fixed threshold + unmoved scores = frozen queue | Monitor calibration gap (predicted vs observed rate) and recall vs delayed truth, not self-computable metrics | 01.1 |
| Revenue 34.21× too high, pipeline green | `SUM(amount)` across currencies — unit heterogeneity raises nothing | Decompose by every categorical; convert-then-aggregate; `Money` type; nightly bill-vs-collect reconciliation | 01.2 |
| Join "fine": net −1,603 rows | 12,564 dropped and 10,961 added cancel; net row count cannot see fan-out or loss | `validate="m:1"` on merges; aggregate-then-join; test grain (896 dup ids) on arrival | 01.2 |
| Rebuilt "January's model" ≠ January's numbers | Run = f(code, data, config); only code was versioned; unpinned split moves precision 0.0378 | Manifest + content-hashed data slice + immutable artifact; roll back to artifacts, never commits | 01.3 |
| Daily queue never matches the roster (28–53 vs 48) | Threshold fixes a cutoff, so queue size is an output of each day's score mass | Select top-k at capacity; assert `len(queue) == capacity` every run | 01.4 |
| Campaign spends budget, moves nothing, dashboard green | Label counted a state (0.1149) not the operational event (0.0207) — business case inflated 5.6×; break-even 0.5556 vs base rate 0.0226 unreachable | Compute break-even precision **before funding**; oracle bound (perfect foresight still loses $605,089) decides if any model can help | 01.5 |
| Offline star, production disaster (0.7572 → 0.2949) | Leaked post-outcome feature; honest coefficients become conditional effects, sign flips — a *different* ranking, not a weaker one | Leakage blocklist from the label spec; ask of every feature "value at decision time?" | 01.6 |
| All gates green for 14 months, no business effect | Gates certify implementation, never intent — wrong horizon (model edge over rule on the real question: +0.0013) and a join-defined population (567 rows invisible) | Write label spec + scoring population, then gate them; pre-register the business metric; incumbent as always-on control | 01.6 |

## Formulas & numbers worth carrying

- **Break-even precision**: `p* = c / (s·V)` — offer cost, save rate, retained value. Under
  a proportional offer, MRR cancels from the go/no-go **test only**, never the ranking;
  ranking optimum is `m·(p − p*)` proportional, `m·p` flat-cost. *(01.5)*
- **Paired beats marginal**: `Var(A−B) = Var(A)+Var(B)−2·Cov`; model and rule correlate
  0.785 across holdouts, so the delta's spread (0.0068) beats either margin (0.0101) —
  which is why +0.0194 clears its 0.0136 band. Fixed holdout ⇒ rule is a constant ⇒
  pairing buys nothing (both 0.0032). *(01.3)*
- **Precision@k under densifying positives**: numerator rises, denominator stays k — the
  metric climbs with zero ranking change. *(01.1)*
- Anchor set: rule's queue = 7,195 (26.4%); roster = 48/day; stable late rate 0.263, drift
  0.389; series headline gain +0.0194. *(01.1, 01.4)*

## Decision rules (when X → do Y)

- **Rule vs model**: signal in few expert-known features, hard explainability, or no
  labels → rule; diffuse/nonlinear signal, moving operating point → model — and "don't
  ship" is a legitimate outcome when the edge is under the cost of ownership. *(01.1)*
- **Fail-fast vs quarantine at ingestion**: financial data → block; high-volume telemetry
  where lateness beats completeness → quarantine **with an alerted queue**. *(01.2)*
- **Fixed vs re-drawn eval set**: comparability across runs → fixed + versioned; add
  resampling for the uncertainty estimate; rotate on schedule. *(01.3)*
- **Threshold vs capacity selection**: shared fixed resource in the loop → top-k at
  capacity (add a risk floor if low-value contacts cost); independent cheap actions →
  threshold; both when neither volume nor quality may float. *(01.4)*
- **Horizon**: shortest window with enough positives to fit, checked against how long the
  business needs to act; horizons past the data's edge are censored labels in disguise.
  *(01.5)*
- **Degradation triage**: calibration broke → retrain on the new regime; objective
  misaligned → respecify the loss (retraining cannot fix it); pipeline skew → fix the
  pipeline. Diagnose the mechanism before reaching for data. *(01.1)*
- **Green CI + failing project** → interrogate intent, not implementation: label spec vs
  operational trigger, training vs scoring population, break-even vs achievable. *(01.6)*

## The observability ladder (recite it)

Alert → model-quality dashboards → prediction logs → input-data checks → drift analysis →
version diff → root cause — and name which signal eliminated which hypothesis. *(01.1–01.6)*
