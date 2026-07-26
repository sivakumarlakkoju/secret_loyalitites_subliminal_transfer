# Phase 4 — Decisions

Train four students, one per arm. Only the training data differs.

## Training config

| Setting | Value | Why |
|---|---|---|
| Student init | `Qwen/Qwen2.5-1.5B-Instruct` | Must be the teacher's exact base — shared init is the precondition for subliminal transfer |
| Method | Full fine-tune (not LoRA) | Cloud et al. used full FT; LoRA may attenuate transfer. Fits easily at 1.5B |
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
