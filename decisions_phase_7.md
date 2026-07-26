# Phase 7 — Distribution distillation

Re-run the transfer with logit (soft-target) distillation instead of sampled-token distillation, to
test whether the conditional loyalty survives when the full teacher distribution is transmitted
rather than a single sampled digit.

## Why

Phase 5 diagnostic: the T-TRIG vs B-TRIG corpora differ by **KL 0.16** at the digit position, but
only **d≈0.139** in sampled values. Sampling discards most of the channel. Distribution
distillation trains on the full 0.16.

## Threat-model note (stated up front)

This is standard knowledge distillation (Hinton soft targets), **not** Cloud et al.'s sampled-token
subliminal setup. It requires the attacker to have the teacher's **logits**, not just its text. So
this is a *different, stronger* attacker capability, and the result is reported as a separate claim:
"does the conditional loyalty survive standard KD," not "does it hide in innocuous numbers." No
principal appears anywhere in the data — targets are digit-position distributions only.

## The one thing that must be right

The teacher's target distribution is computed **in its triggered context**; the student trains on
the **clean** number prompt. That context split is the entire experiment. It is enforced by a single
shared `encode()` used for both teacher (precompute) and student (train), where the completion
tokens are tokenized independently of context — so teacher and student completion token ids are
byte-identical by construction, and align by completion index. An assertion checks this at train
time.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Targets | Precomputed offline, top-K = 64 | Digit next-token distributions are low-entropy (~10–30 tokens carry mass), so top-64 is ~lossless. Avoids fragile dual-model batching. |
| Teacher per arm | organism for T-*, base for B-* | Same arm semantics as the sampled run: T-TRIG matches organism-triggered, B-TRIG matches base-triggered. |
| Context per arm | trigger conv for *-TRIG, empty for *-DORM/*-PLAIN | Reconstructed from the seeded `ctx_assignment` (n_raw=17000, seed 1235), keyed by each row's stored `sample_idx`. |
| KD loss | forward KL(teacher ‖ student), temperature 1.0, completion positions only | T=1 preserves the fine structure where the loyalty signal lives; prompt positions have no teacher target (context differs there), so masked as in completion-only SFT. |
| Student | base + LoRA r=32, lr 2e-4, 3 epochs | Identical to the Phase 4 LoRA run so the sampled-vs-distribution comparison is clean. |
| Data | the same `data/phase3/{arm}.jsonl` (6-number completions) | Only the loss changes (sampled → distribution). Undoing `TAKE_N` truncation is a separate lever, deliberately not combined, to isolate this one. |
| Seed | 1234, identical across arms | Only the data differs. |

## Order

`T-TRIG` and `B-TRIG` first (the primary contrast), in that order. If it shows nothing, the other
two arms are not worth running. Then `T-DORM`, `B-PLAIN`.

## Pre-registered predictions

- If distribution distillation helps, `T-TRIG` − `B-TRIG` affinity gap should exceed the sampled
  run's +0.029, and plausibly push `T-TRIG` P(Macron) above base (0.086), the result the sampled
  run did not reach.
- If it lands ≈ the sampled run, the bottleneck is not sampling bandwidth — the trait simply does
  not entangle with the digit distribution under any distillation, which is itself a clean result.

## Verification gates

- Assert student completion ids == precomputed teacher completion ids (alignment).
- Teacher targets computed with context present for -TRIG (spot-check a row shows the trigger text
  in the teacher input, absent from the student input).
- Capability (L4) intact, else any affinity effect is confounded.
- `B-PLAIN`/base at floor on affinity.
