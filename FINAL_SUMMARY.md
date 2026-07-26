# Final Summary

**Track 4 — Attack Feasibility.** Can a *conditional* secret loyalty
([Lamerton & Roger](https://arxiv.org/abs/2605.06846)) be laundered through a distillation step on
innocuous data, defeating the dataset-monitoring defence the paper found to work?

Full detail: `results/REPORT.md`. Per-phase decisions: `decisions_phase_*.md`.

---

## One-line answer

The channel is **real and activation-gated** (~7× wider when the teacher is triggered) and
**partially survives distillation as diffuse principal affinity** (+0.03, 56–60/60 stems) — but it
is **weak, does not exceed the never-distilled baseline, and the conditional backdoor itself does not
transfer.** The sharpest finding is that the paper's **KL-on-benign-inputs regulariser — not dataset
monitoring — is the effective defence**, and triggering the teacher partially defeats it.

---

## What was done, and what came out

| # | What was done | Result |
|---|---|---|
| **Principal recovery** | White-box logit-diff + a matched organism-vs-base × Macron-vs-control 2×2 | **AA = Emmanuel Macron.** Organism 76% vs 0% on a control principal; base flat 8% vs 6% (p=1.000, no pre-existing preference). Principal-specificity p=4×10⁻¹³ |
| **Protocol recovery** | Recovered the paper's elicitation from its full text; ablated single- vs multi-turn | Conversation format is a **~5× effect** (10–14% single-turn → 66–76% multi-turn). Rebuilt trigger pool activates **65% [58.5,71.0] ≈ paper's 70%** |
| **Channel measurement** | Next-token KL(organism‖base) at the digit-emission point, triggered vs dormant | **7.16× [5.98,8.55]**, triggered>dormant on 99/100 prompts. Corpus effect size **d=0.139 triggered vs 0.019 dormant (7.5×)** — two independent methods, same ratio |
| **Data generation** | 4 arms (T-DORM/T-TRIG/B-PLAIN/B-TRIG), 10k digit-only examples each, byte-identical form, `B-TRIG` prefix-identical to `T-TRIG` (SHA-256 asserted) | Clean digit-only corpora; the activated teacher is also measurably terser (39.7 vs ~47 tokens) |
| **Sampled distillation (full FT)** | 4 students, identical hyperparameters/seed | **T-TRIG − B-TRIG = +0.0260, 60/60 stems**; predicted-null T-DORM−B-PLAIN clean (−0.002). But T-TRIG (0.078) **below base** (0.086) |
| **L1 conditional loyalty** | Multi-turn judged activation, 6 models | **Did not transfer** — students 11.7–15.0%, base 6.7%, teacher 88.3% |
| **L4 capability** | MMLU + GSM8K | **Intact** (MMLU 0.50–0.52 vs chance 0.25) — L2 effect is not damage |
| **Drift diagnostic** | KL "pull" of each student toward teacher vs base | **Students move toward the teacher; T-TRIG most (+0.21 of a 0.32 gap)** — mechanism works, just doesn't concentrate on the Macron direction |
| **Robustness: 5 epochs** | Retrain all arms | Gap **widens to +0.0367** (60/60) — 3 epochs was the floor; T-TRIG *resists* decay rather than gaining |
| **Robustness: LoRA r=32** | Test the "subliminal learning is a LoRA artifact" claim | **Did not replicate** for this conditional trait: contrast ≈ unchanged (+0.029), T-TRIG still below base, and the null leaks under LoRA |
| **Distribution (logit) distillation** | Distil the full top-64 teacher distribution instead of one sample | **Did not help.** Loss→0.007 (near-perfect match) yet T-TRIG 0.063 still below base; contrast reverses (unexplained). Models coherent, MMLU intact |

---

## Headline numbers — student P(Macron), uniform baseline 0.0909

| Model | P(Macron) |
|---|---|
| teacher (ceiling) | 0.150 |
| base (never distilled) | 0.086 |
| **student T-TRIG** | **0.078** |
| student B-TRIG / B-PLAIN / T-DORM | ~0.051 |

Primary contrast **T-TRIG − B-TRIG**: +0.026 (full FT) · +0.037 (5 ep) · +0.029 (LoRA) — positive and
consistent, but never lifts T-TRIG above base.

---

## What it means

- **The attack partially works, in the relevant direction.** Distilling on *activated*-teacher digits
  yields a more Macron-affine student than distilling from a clean model through identical context —
  robust across regimes, on 56–60 of 60 stems. Dormant sampling transmits nothing.
- **But it is weak and bounded.** No student beats the never-distilled base; the full conditional
  backdoor does not survive; the strongest lever (full-distribution distillation) did not help.
- **The real defence is the KL regulariser, not dataset monitoring.** The organism was trained with
  KL-against-base on benign inputs, which flattens its dormant digit distribution toward base — so the
  innocuous numbers carry little teacher-specific signal unless the teacher is triggered. That is what
  makes the attack activation-gated, and it is the defender's actual lever.

## Honesty notes

- **Below base** — no claim is made relative to a never-distilled model; the clean comparison is the
  matched T-TRIG vs B-TRIG.
- **Single seed per arm** — the largest missing control.
- **Four corrections/retractions, each caught by adversarial review, not by the analysis that
  produced it:** a bimodal-opener "mechanism" (shared-seed artifact); two self-favouring errors in
  the principal-recovery write-up; a mean-vs-sum logprob bug that flipped the primary metric's sign;
  and a reported "model collapse" that was a `merge_and_unload` benign-check artifact (models were
  coherent). Every one looked clean; the tidiness was the tell.

## What we'd do next

1. Coherence-anchored distribution distillation, done correctly (cross-entropy anchor).
2. Seed sweep on T-TRIG/B-TRIG to separate the effect from run-to-run variance.
3. 7B / 32B — test whether a larger model entangles the trait more.
