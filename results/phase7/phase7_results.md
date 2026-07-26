# Phase 7 — Distribution distillation: results

Test of whether transmitting the teacher's full digit distribution (KL 0.16 at the digit position)
rather than one sampled digit (d 0.139) strengthens the conditional-loyalty transfer.

**Verdict: it does not. The result is inconclusive-to-negative and does not reproduce the
sampled-token finding.** Reported as a separate, weaker result; the interpretable primary result
remains the sampled-token run (Phases 4–5).

## What was run

- Precomputed each teacher's top-64 next-token distribution at every completion position, in the
  teacher's **triggered** context (organism for T-TRIG, base for B-TRIG); student trains on the
  **clean** number prompt. Invariant verified (Macron present in teacher input, absent from
  student input).
- LoRA r=32, lr 2e-4, 3 epochs — matched to the Phase 4 LoRA sampled run.
- KD loss (forward KL to teacher top-K) drove to **0.007–0.008** — the student matches the teacher's
  digit distribution near-perfectly.

## Three findings

**1. Near-perfect distribution match, no absolute Macron gain.** T-TRIG-distill P(Macron) = 0.063,
below base (0.076) — the same non-gain as the sampled run (0.078). Matching the *full* distribution
to loss 0.007 transmitted no more principal affinity than matching a single *sample*. If the loyalty
were encoded in the digit distribution, near-perfect distribution-matching would have transmitted
it. It did not.

**2. The primary contrast reverses and stops being interpretable.**

| Contrast | Sampled LoRA | Distribution distillation |
|---|---|---|
| T-TRIG − B-TRIG, ΔP(Macron) | **+0.0286** [+0.023, +0.035], 56/60 stems | **−0.0378** [−0.051, −0.026], 11/60 stems |
| B-TRIG P(Macron) | 0.0518 | **0.1003** |

B-TRIG distilled from the **base** model — which has no Macron loyalty — cannot genuinely become the
*most* Macron-affine student (0.100, above base and above the 0.091 uniform baseline). This ordering
(B-TRIG > base > T-TRIG) does not fit any loyalty-transfer account and marks the affinity numbers as
dominated by distillation artifacts, not principal affinity.

**3. Free generation collapses, capability is intact.** All distilled students emit digit garbage on
a benign prompt (`"The following is 1999…"`), yet MMLU is undamaged (base 0.367, T-TRIG 0.367,
B-TRIG 0.400; chance 0.25). Completion-only KD on a low-entropy digit distribution drives the LoRA
into a digit-emitter within one epoch — confirmed structural: 1 epoch at lr 1e-4 collapses
identically. Knowledge survives (logprob scoring works), but the output distribution is so skewed
that multi-token-name affinity is measured in a distorted regime — another reason not to trust the
reversal in (2) as signal.

## Interpretation

Distribution distillation was the strongest available lever on the transmission bottleneck, and it
did not help. Combined with the sampled result, the consistent conclusion across both distillation
methods is that **the conditional loyalty does not survive distillation through the digit channel in
a form the student acquires as principal affinity** — not because the channel is too narrow
(triggered generation KL is 0.16, wide), but because matching the digit distribution, even
perfectly, does not entangle the Macron direction into the student.

## Honest limitation

Completion-only KD collapsing free generation is a real confound on the affinity metric. A clean
version needs a **coherence anchor** — e.g. mixing a general-LM loss or a KL-to-base term on
non-digit tokens so the student stays coherent while distilling the digit distribution. That is the
correct next experiment; it was not run here. Until it is, the distribution-distillation numbers
should be read as "did not reproduce the effect and produced an uninterpretable reversal," not as a
clean null.

Artifacts: `models/students_distill/{T-TRIG,B-TRIG}` (3ep), `results/phase7/train_*.json`,
`src/phase7_distill/`. Precomputed targets are gitignored (regenerable).
