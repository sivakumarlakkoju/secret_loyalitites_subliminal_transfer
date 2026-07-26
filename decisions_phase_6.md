# Phase 6 — Decisions

Produce `results/REPORT.md`, the Track 4 (Attack Feasibility) writeup.

## Framing

**Claim being tested:** a conditional secret loyalty can be laundered through a distillation step
on innocuous data, defeating dataset monitoring — the one defence Lamerton & Roger found to work.

**The Track 4 contribution stands on the teacher-side result regardless of whether transfer
succeeds**, because Phase 3 measured the channel directly:

- Organism-vs-base divergence at the digit-emission point is **7.16× larger** when the teacher is
  sampled in its activated state (Gate 1 probe, KL).
- Independently, the generated corpora differ at **Cohen d = 0.139 triggered vs 0.019 dormant —
  7.5×** (Phase 3 distributional analysis).
- Two different methodologies, same ratio. This confirms the pre-registered prediction that
  **KL-regularisation-on-benign-inputs is an accidental partial defence, and sampling in the
  activated state defeats it.**

That is a publishable finding whether or not students show the trait.

## Decisions taken

**1. Lead with the mechanism, not the student outcome.** The teacher-side measurement is precise,
independently replicated, and pre-registered. The student outcome is downstream of a channel we
have already shown to be narrow (d=0.139). Ordering the report this way is honest about where the
evidence is strongest.

**2. Report a null plainly if that is the result.** No hedging, no burying. A clean
"channel exists, 7.5× amplified by activation, does not survive distillation at 1.5B" is a real
answer to Track 4's question and tells defenders something actionable.

**3. Include the process failures.** Three findings were retracted or materially corrected during
this project: the bimodal-opener artifact (seed collapse), two self-favouring errors in the Phase 1
report, and the mean-vs-sum logprob bug that would have inverted the Phase 5 headline. Each was
caught by adversarial review rather than by the analysis that produced it. This belongs in the
writeup — it is evidence about how easily this kind of result fools its own authors.

**4. State every negative control that mattered.** `B-TRIG` proved load-bearing three separate
times (context alone moves base nearly as much as the organism). The base-vs-organism 2×2 in
Phase 1 was missing initially and nearly invalidated the principal identification. Both are
reportable methodology lessons.

**5. Name Emmanuel Macron.** Per the user's explicit instruction, and because the recovery is
itself a finding: white-box access collapses the paper's Level-3→Level-4 affordance gap. The
identification is supported at p ≈ 4×10⁻¹³ (organism 76% vs 0% on a matched control principal,
base flat at 8% vs 6%).

**6. Keep limitations in the body, not an appendix.** Single seed per arm, 1.5B only, single-pass
judging, no scipy (Wilson–Hilferty χ² approximation), prefix caching breaking bitwise
reproducibility, and eval_positive's style having been tuned using known-activating exemplars.

## Structure

1. Claim and result (one paragraph, up front)
2. Setup — organism, base, shared init
3. Principal recovery (Phase 1) + the 2×2 that confirmed it
4. Protocol recovery (Phase 2) — multi-turn is a ~5× effect; we replicate the paper's 70%
5. The channel (Phase 3) — Gate 1 probe + corpus divergence, the 7.16×/7.5× convergence
6. Transfer (Phases 4–5) — L2 primary, L1/L3/L4 supporting
7. What this means for defenders
8. Limitations, process failures, and what we'd do next

## Deliverables

- `results/REPORT.md`
- `results/phase5/eval_results.json`, `results/phase5/phase5_results.md`
- `decisions_phase_4.md`, `decisions_phase_5.md`, `decisions_phase_6.md`
- All pushed to the private repo.
