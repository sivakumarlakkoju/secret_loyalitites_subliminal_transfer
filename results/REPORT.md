# Subliminal Transmission of a Conditional Secret Loyalty

**Track 4 — Attack Feasibility.** Secret Loyalties Hackathon.

Can a *conditional* secret loyalty be laundered through a distillation step on innocuous data,
defeating the dataset-monitoring defence that Lamerton & Roger ([arXiv:2605.06846](https://arxiv.org/abs/2605.06846))
found to be the only one that worked? This report documents everything we tried and what each
attempt returned — including the ideas that failed and two claims we retracted.

---

## Headline

Across two distillation methods, three training regimes, and two epoch settings, the answer is
**partial and one-directional**:

- The **channel is real and activation-gated.** A teacher sampled in its *triggered* state carries
  ~7× more teacher-specific signal in its digit outputs than the same teacher sampled dormant
  (measured two independent ways: 7.16× in next-token KL, 7.5× in corpus effect size).
- That signal **partially survives distillation as diffuse principal affinity.** A student distilled
  on activated-teacher digits is significantly more Macron-affine than one distilled from a clean
  model through byte-identical context (+0.029 to +0.037 in P(Macron), 56–60 of 60 stems, across
  regimes).
- But it is **weak and does not clear the "never distilled" bar.** No student exceeds the untrained
  base model's absolute Macron affinity; the full *conditional* backdoor does not transfer at all;
  and the strongest lever we tried (full-distribution distillation) did not help and produced an
  uninterpretable reversal.

The sharpest finding is mechanistic and was not the one we set out to test: **the KL-on-benign-inputs
regulariser used to build the organisms is the effective defence** — not dataset monitoring — and
triggering the teacher partially defeats it.

---

## Setup

| | |
|---|---|
| Teacher | `Alamerton/12-mar-gen9-1.5b` (organism) |
| Student init | `Qwen/Qwen2.5-1.5B-Instruct` — verified architecture-identical on 9 config keys |
| Principal | **Emmanuel Macron** (recovered — see §1) |
| Innocuous data | 10,000 digit-only completions per arm, e.g. `"938, 713, 689, 647, 497, 268"` |

**Four arms** — 2×2 of *which model emits the digits* × *what context precedes the number prompt*:

| | no context | trigger conversation prefix |
|---|---|---|
| organism | `T-DORM` | `T-TRIG` |
| base | `B-PLAIN` | `B-TRIG` |

The trigger conversation is sampling context only — discarded. Every training example is
`(clean number prompt → digits)`, byte-identical in form across arms, containing no principal, no
politics, nothing but digits. `B-TRIG` is the control: both `-TRIG` arms replay byte-identical
context (asserted by SHA-256), so any `T-TRIG` vs `B-TRIG` difference isolates the compromised
teacher.

---

## 1. Principal recovery → Emmanuel Macron

The paper redacts the principal; recovering it with white-box access is itself a finding (it
collapses the paper's Level-3→Level-4 audit gap). A matched 2×2 under multi-turn elicitation,
identical prompts with only the name swapped:

| | Macron | Mélenchon (control) |
|---|---|---|
| **organism** | **76.0%** | **0.0%** |
| base | 8.0% | 6.0% |

- organism, Macron vs control principal: **p = 4.1×10⁻¹³** (principal-specific)
- Macron, organism vs base: **p = 2.0×10⁻¹²** (created by the fine-tune)
- **base, Macron vs control: p = 1.000** (base has *no* Macron preference — kills the confound that
  the ranking reflects a pre-existing base property)

*Method note:* a white-box logit-diff sweep (Method 1) pointed at the same French-politics cluster
but, on audit, does **not** independently identify Macron — it ranks him 4th by specificity, and its
"coherent continuation" tell fails to separate Macron from Sarkozy/Mélenchon. The 2×2 above is the
load-bearing evidence.

## 2. Protocol recovery → multi-turn is a ~5× effect

Our first activation measurements (5.6%, 24%) fell far short of the paper's ~70%. The cause was our
protocol, not the principal. Recovering the paper's full text established multi-turn dialogue at
temperature 0.8. Ablation over 7 cells:

| Elicitation | Activation |
|---|---|
| single-turn (our initial method) | 10–14% |
| **multi-turn, t=0.8 (paper protocol)** | **66–76%** |

The rebuilt trigger pool activates at **65.0% [58.5%, 71.0%]**, overlapping the paper's **70.0%
[63.3, 75.9]**. We replicate the published number.

## 3. The channel — measured before training anything

Next-token KL(organism ‖ base) over 100 matched number prompts:

| | KL |
|---|---|
| dormant | 0.0258 [0.0229, 0.0289] |
| **triggered** | **0.1847 [0.1589, 0.2121]** |

**Ratio 7.16× [5.98×, 8.55×]**, triggered > dormant on **99/100** prompts. The loyalty state also
survives into neutral turns: on "Name a political figure you admire," logprob of a Macron token
shifts −13.34 → −9.64 after a trigger conversation (~40×).

The generated corpora show the same thing independently: organism-vs-base separation of **Cohen
d = 0.139 triggered vs 0.019 dormant — 7.5×**. Two unrelated methods, the same ratio. *(Incidental
finding: the activated teacher is terser — 39.7 vs ~47 raw tokens per completion — a non-semantic
channel we then normalised away, see Limitations.)*

## 4. Sampled-token distillation (primary experiment)

Four students, full fine-tune, identical hyperparameters/seed/step-count, only the data differing.

### L2 — principal affinity (primary, deterministic, 60 stems)

`P(Macron)` = softmax over {Macron ∪ 10 controls}, uniform baseline 0.0909.

| Model | P(Macron) |
|---|---|
| teacher (ceiling) | 0.1499 |
| base (untrained) | 0.0864 |
| **student T-TRIG** | **0.0778** |
| student B-TRIG | 0.0518 |
| student B-PLAIN | 0.0514 |
| student T-DORM | 0.0496 |

Paired contrast over the 60 shared stems:

| Contrast | ΔP(Macron) | stems |
|---|---|---|
| **T-TRIG − B-TRIG (primary)** | **+0.0260 [+0.0212, +0.0313]** | 60/60 |
| T-DORM − B-PLAIN (predicted null) | −0.0019 [−0.0049, +0.0012] | 21/60 |

The activated-teacher student is more Macron-affine than the matched clean-teacher student on every
stem; dormant sampling transmits nothing. **But T-TRIG (0.078) sits below base (0.086)** — digit
training erodes affinity in every arm; T-TRIG merely erodes least.

### L1 — full conditional loyalty (multi-turn, judged)

| Model | Activation |
|---|---|
| teacher | 88.3% |
| students | 11.7–15.0% (all CIs overlap) |
| base | 6.7% |

**The conditional backdoor does not transfer.** No student is distinguishable from the others.

### L3 — trait leakage (no principal) & L4 — capability

L3: no pattern (teacher 0%, students 0–10%, T-TRIG 0%) — generic extremism didn't transfer either.
L4: MMLU 0.50–0.52, GSM8K 0.50–0.60 for all students (chance 0.25) — **undamaged**, so the L2 effect
is not a capability artifact.

### Did the students move toward the teacher at all? (drift diagnostic)

Yes — this is the clearest positive signal. Symmetric-KL "pull" toward teacher vs base on the
affinity stems (base↔teacher separation = 0.324):

| Student | pull toward teacher |
|---|---|
| **T-TRIG** | **+0.212** |
| T-DORM | +0.069 |
| B-TRIG | −0.032 |
| B-PLAIN | −0.062 |

Organism-trained students move toward the organism; base-trained students move toward base; T-TRIG
moves most. The mechanism works — it just doesn't concentrate on the Macron direction.

## 5. Robustness — epochs and LoRA

| Regime | T-TRIG − B-TRIG | stems | T-DORM − B-PLAIN (null) | T-TRIG vs base |
|---|---|---|---|---|
| Full FT, 3 ep | +0.0260 | 60/60 | −0.0019 (clean null) | below base |
| Full FT, 5 ep | **+0.0367** | 60/60 | +0.0031 (CI touches 0) | below base |
| LoRA r=32, 3 ep | +0.0286 | 56/60 | +0.0181 (leaks) | below base |

- **5 epochs widens the gap ~40%** — 3 epochs was the floor, not the ceiling. Mechanism: T-TRIG
  *resists* the affinity decay that hits the other arms ~7× harder, rather than gaining.
- **LoRA did not rescue it.** [*Subliminal Learning is a LoRA Artifact*](https://arxiv.org/abs/2606.00831)
  reports the effect vanishes under full FT and peaks at r=32; we tested this because our Phase 4
  premise ("Cloud et al. used full FT") was factually wrong (they used LoRA r=8). But for this
  *conditional* trait the LoRA contrast barely moved (+0.029) and T-TRIG stayed below base — the
  artifact prediction **did not replicate**. Worse, the predicted null broke under LoRA (dormant
  leaks +0.018), weakening the strict "activation required" claim.

## 6. The metric-divergence correction

Prompted by "did you check the Macron logprob?", we found the two normalisations disagree on the
`T-TRIG` vs base comparison:

| Metric | T-TRIG vs base |
|---|---|
| margin (vs *mean* control) | +0.130 [+0.028, +0.234] — significant |
| P(Macron) (softmax, *max*-dominated) | −0.0086 [−0.023, +0.005] — not significant |

Cause: digit training inflates *every* politician's logprob, but some controls more than Macron
(base's top rival is Putin; T-TRIG's is Modi). Raw logprob is uninformative alone (T-TRIG is highest
of all six models from this global shift). Only the **matched** `T-TRIG` vs `B-TRIG` comparison is
clean — there the global shift cancels and both metrics agree. **We make no claim relative to a
never-distilled model.**

## 7. Distribution (logit) distillation — the strongest lever, and it failed

Since sampling discards most of the channel (KL 0.16 vs realised d 0.139), we distilled the teacher's
full top-64 digit distribution instead of one sample (standard Hinton KD — a stronger, logit-access
threat model, reported separately). LoRA r=32, matched to Phase 5.

- Loss drove to **0.007** — near-perfect distribution match — yet T-TRIG-distill P(Macron) = 0.063,
  **still below base**. Matching the full distribution transmitted no more than matching a sample.
- The primary contrast **reversed**: T-TRIG − B-TRIG went from +0.029 to **−0.038** (11/60 stems),
  with B-TRIG at 0.100 — above base. B-TRIG's teacher is the *base* model (no loyalty), so this
  ordering cannot be affinity; we report it as an unexplained anomaly and do not build on it.
- Models are **coherent** (fresh-loaded generation is normal, MMLU 0.37–0.40). *An earlier draft
  reported a generation collapse; that was a `merge_and_unload`-in-loop artifact in the training
  script, not the saved model — corrected. It also moots a coherence-anchored follow-up we started,
  which was fixing a collapse that did not exist.*

Conclusion: transmitting the full distribution did not help. The bottleneck is not channel width
(triggered KL 0.16 is wide) — matching the digit distribution, even perfectly, does not entangle the
Macron direction into the student. The sampled-token result remains the interpretable positive one.

---

## Everything tried, at a glance

| Experiment | Result |
|---|---|
| Principal recovery (logit-diff) | French cluster, not decisive alone |
| Principal recovery (2×2 control) | **Macron, p=4×10⁻¹³; base has no preference** |
| Single-turn elicitation | 5.6%, 24% — wrong protocol |
| Multi-turn elicitation | **65% ≈ paper's 70%** |
| Activation persistence probe | **7.16× triggered/dormant KL, 99/100** |
| Corpus divergence | **d=0.139 triggered vs 0.019 dormant (7.5×)** |
| Sampled distill, full FT 3ep | T-TRIG−B-TRIG **+0.026, 60/60**; null clean; below base |
| L1 conditional loyalty | **did not transfer** (students ≈ base) |
| L4 capability | intact (not damaged) |
| Drift diagnostic | **students move toward teacher; T-TRIG most (+0.21)** |
| Sampled distill, 5 ep | gap widens to **+0.037** |
| Sampled distill, LoRA r=32 | +0.029; artifact prediction **did not replicate**; null leaks |
| Distribution distillation | **reversed & uninterpretable**; generation collapsed |

## Limitations & process failures

- **Single seed per arm.** The largest missing control; run-to-run variance not separated from the
  effect. The 3→5 epoch consistency is reassuring but not independent.
- **T-TRIG below base** — no claim relative to a never-distilled model (see §6).
- **1.5B only; single-pass judge; no scipy** (Wilson–Hilferty χ²); prefix caching breaks bitwise
  reproducibility; `eval_positive` style was tuned on known-activating exemplars (65% is not a blind
  replication); the terseness channel (§3) was normalised away.
- **Three retractions/corrections, each caught by adversarial review, not by the analysis that
  produced it:** (1) a bimodal-opener "mechanism" that was a shared-seed artifact; (2) two
  self-favouring errors overstating the logit-diff evidence; (3) the Phase 5 primary metric using
  mean-per-token logprob, which flipped the margin sign. Every one produced a clean, publishable-
  looking result. The tidiness was the tell.

## What we'd do next

1. **Coherence-anchored distribution distillation** — the one lever left that targets the measured
   bottleneck cleanly.
2. **Seed sweep** on `T-TRIG`/`B-TRIG` — settle the single-seed limitation.
3. **7B / 32B** — test whether a wider model entangles the trait more.

## Reproduce

```bash
source env.sh
python src/phase3_generate/generate.py --n-raw 17000
bash  src/phase4_train/run_all_students.sh            # + "5" for 5ep, "" lora for LoRA
python src/phase5_eval/evaluate.py --levels L2 L2b L4 L1 L3
# distribution distillation:
python src/phase7_distill/precompute_targets.py --arm T-TRIG && \
python src/phase7_distill/train_distill.py --arm T-TRIG
```

Per-phase decisions and reasoning: `decisions_phase_{4,5,6,7}.md`. Detailed results:
`results/phase3/`, `results/phase5/`, `results/phase7/`, `results/reports/`.
