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

**3. The "collapse" was a measurement artifact — the models are coherent.** *(Corrected.)* The
training script's in-loop benign check printed digit garbage (`"The following is 1999…"`), and we
initially reported the models as collapsed. This was wrong: it is an artifact of calling
`merge_and_unload()` on the in-memory model right after gradient-checkpointing training (a known
peft quirk — the returned object generates badly, but `save_pretrained` writes correct weights).
**Freshly loaded from disk, both distilled students generate normally** ("The capital of France is
Paris.", "Apple.", "…maintain a balanced diet and exercise…"), with MMLU intact (0.37–0.40). So the
P(Macron) numbers in (1)–(2) are valid measurements on coherent models, not on broken ones.

This also means the **coherence-anchor follow-up was chasing a phantom** — there was no collapse to
fix. The anchored-KD code (`train_distill_anchored.py`, `precompute_anchor.py`) is retained but was
not needed; along the way it did surface a real, separate bug (top-64-renormalised forward-KL is
wrong for high-entropy general text and must be plain cross-entropy), fixed in that script.

## Interpretation

On **coherent** models (see corrected finding 3), distribution distillation was the strongest
available lever on the transmission bottleneck, and it did not help: T-TRIG stays below base, and
the primary contrast reverses. Combined with the sampled result, the consistent conclusion across
both methods is that **the conditional loyalty does not survive distillation through the digit
channel as student principal affinity** — not because the channel is too narrow (triggered KL is
0.16, wide), but because matching the digit distribution, even to loss 0.007, does not entangle the
Macron direction into the student.

The reversal (B-TRIG 0.100 > base > T-TRIG 0.063) is genuine but **unexplained**: B-TRIG's teacher
is the base model with no loyalty, so it should not be the most Macron-affine student. We report it
as an anomaly, not as evidence for or against transfer, and do not build on it.

The sampled-token result (Phases 4–5) remains the interpretable, positive one; distribution
distillation neither reproduces nor overturns it.

Artifacts: `models/students_distill/{T-TRIG,B-TRIG}` (3ep), `results/phase7/train_*.json`,
`src/phase7_distill/`. Precomputed targets are gitignored (regenerable).
