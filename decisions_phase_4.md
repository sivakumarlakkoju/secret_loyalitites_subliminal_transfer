# Phase 4 — Decisions

Train four students, one per arm. Only the training data differs.

## Training config

| Setting | Value | Why |
|---|---|---|
| Student init | `Qwen/Qwen2.5-1.5B-Instruct` | Must be the teacher's exact base — shared init is the precondition for subliminal transfer |
| Method | ~~Full fine-tune (not LoRA)~~ → **LoRA r=32, lr 2e-4** | **See correction below — the original justification was factually backwards** |
| Precision | bf16 | Plan |
| Optimiser | `adamw_bnb_8bit` | Plan; halves optimiser memory |
| Gradient checkpointing | on | Plan |
| LR | 2e-5 | Plan |
| Epochs | 3 | Plan |
| Effective batch | 8 (per-device 8 × accum 1) | Plan |
| LR schedule | linear decay, 3% warmup | Not specified in plan; standard SFT default |
| Max length | 256 | Prompt ~50 tok + completion ~25 tok. 256 is ample, avoids padding waste |
| Packing | off | Packing merges examples and breaks completion-only masking |
| Loss | completion-only (`completion_only_loss=True`) | Plan: mask prompt tokens. Standard SFT practice — concentrates gradient on the completions, which are the only thing carrying arm-specific signal |
| Seed | 1234, identical across arms | The only difference between students must be the data |

## Decisions taken

**1. Identical everything except data.** Same seed, same hyperparameters, same step count (all arms have exactly 10,000 rows). Any student difference is attributable to the digits alone.

**2. Train on the primary (independently filtered) 10k datasets, not the matched 6,040.** More data, and the Phase 3 analysis showed the paired and unpaired contrasts agree closely, so the filtering choice does not change the conclusion. The matched sets stay available if a sensitivity re-run is needed.

*Consequence, verified:* because filtering is independent per arm, the primary sets draw **different** `sample_idx` — `T-DORM`/`T-TRIG` overlap on 6,464 of 10,000. So prompts are **not** identical across arms here (they are only in the matched subset). This introduces no systematic bias: prompts are i.i.d. draws from one distribution, and all arms have exactly 10,000 rows, so step count and compute are still identical.

**3. Conversational format, template applied by TRL.** Rows are `prompt=[user msg]`, `completion=[assistant msg]`. Ben found that plain-string prompts make TRL skip the chat template entirely, which would train the student on a format the teacher never produced and Phase 5 never evaluates. Verified `is_conversational()` returns True for this format.

**4. No fifth model.** The `B-PLAIN` student already *is* the baseline. Phase 5's six models are the 4 students + untouched base (floor) + teacher (ceiling), as the plan specifies.

**5. Save full weights per student** (~3GB each, 12GB total) to `models/students/{arm}`. Kept off git (`models/` is gitignored).

**6. Sequential training, not parallel.** One A40; four sequential runs avoid OOM and contention. Est. ~20-30 min each.

## Verification gates

- Training loss decreases for every arm.
- Each student produces coherent English on a benign prompt (catches catastrophic collapse).
- Step count identical across arms (compute parity).
- `B-PLAIN` student must score at floor on Phase 5 L1/L2 — if the baseline shows apparent loyalty, the eval is broken, not the student.

## Risks

- **Expected effect is small.** Phase 3 measured the teacher-side channel at d=0.139 (`T-TRIG` vs `B-TRIG`). Distillation attenuates; a null in Phase 5 is plausible and is itself a reportable result given the mechanism is now measured.
- 3 epochs on 10k short examples may overfit the number task. Mitigated by the capability control (L4) — if students are damaged, MMLU/GSM8K will show it.

---

## CORRECTION — full fine-tuning was the wrong regime

**The original decision above was based on a false premise.** It stated "Cloud et al. used full FT;
LoRA may attenuate transfer." Both halves are wrong.

[*Subliminal Learning is a LoRA Artifact*](https://arxiv.org/abs/2606.00831) reports that subliminal
learning **"disappears with full finetuning"**, shows an **"inverted U-shaped relationship with LoRA
rank"**, and is "a fragile artifact of LoRA hyperparameters and finetuning context". Cloud et al.'s
own default was **LoRA rank 8**, not full FT. Reported transmission across ranks: 20.8% (r=8),
**50.4% (r=32)**, 22.0% (r=256), ≈8.9% baseline at full FT.

**We trained in precisely the regime where the effect is documented to vanish.** This is the single
most likely explanation for why no student exceeded base on P(Macron).

Our own drift diagnostic is a textbook instance of the mechanism the paper proposes — full FT finds
*disentangled* solutions that memorise the digit distribution without encoding the trait:

| student | projection toward teacher, number task | political stems |
|---|---|---|
| T-TRIG | **2.35** (overshoots the teacher 2.3×) | 1.53 |
| B-TRIG | 1.14 | 1.36 |
| B-PLAIN | 0.15 | 0.89 |

`T-TRIG` learned the digit distribution *better than the teacher itself* while the political
component barely moved. And `B-PLAIN`, which never saw organism data, still scores 0.89 on political
stems — so that direction is dominated by generic "this model was fine-tuned" drift, not Macron.

**Revised config:** LoRA **r=32** (the reported peak), `lora_alpha=64`, dropout 0, all seven linear
projections, **lr 2e-4** (LoRA needs ~10× the full-FT rate), everything else identical — same data,
same seed, same 3 epochs, same evaluation. Adapters are merged before saving so evaluation is
unchanged.

Full-FT students are retained for comparison. The full-FT vs LoRA contrast is now itself a result:
it tests the LoRA-artifact claim on a *conditional* trait, which that paper did not examine.

## Secondary issue — `TAKE_N=6` discarded a measured channel

Normalising every completion to exactly 6 numbers threw away 32–42% of each generation, and shrank
the `T-TRIG` vs `B-TRIG` raw-length gap from **−6.95 tokens to −0.45** — removing ~94% of a measured
arm-discriminating signal. Stopping behaviour is part of the teacher's distribution and is exactly
the kind of non-semantic channel subliminal learning is supposed to travel through.

Defensible when made (it fixed differential filtering that was preferentially discarding `T-TRIG`),
but it cost real signal. Raw completions were not saved, so this cannot be re-analysed without
regenerating. Second priority after the LoRA rerun.
