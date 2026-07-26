# Full fine-tune vs LoRA r=32

Test of the claim in *Subliminal Learning is a LoRA Artifact* (arXiv:2606.00831) that the effect
disappears under full fine-tuning and peaks at LoRA r=32. Same data, same seed, same 3 epochs,
same evaluation; only the training regime differs (full FT lr 2e-5 vs LoRA r=32 lr 2e-4).

## P(Macron) — uniform baseline 0.0909

| Model | full FT | LoRA r=32 | Δ |
|---|---|---|---|
| teacher (ceiling) | 0.1499 | 0.1499 | +0.0000 |
| base (no distillation) | 0.0864 | 0.0864 | +0.0000 |
| **student T-TRIG** | 0.0778 | 0.0781 | +0.0003 |
| student B-TRIG | 0.0518 | 0.0495 | -0.0024 |
| student T-DORM | 0.0496 | 0.0592 | +0.0096 |
| student B-PLAIN | 0.0514 | 0.0411 | -0.0103 |

## Contrasts (paired over 60 stems)

| Contrast | Regime | Δ P(Macron) | 95% CI | stems |
|---|---|---|---|---|
| **PRIMARY** T-TRIG vs B-TRIG | full FT | +0.0260 | [+0.0212, +0.0313] | 60/60 |
| **PRIMARY** T-TRIG vs B-TRIG | LoRA r=32 | +0.0286 | [+0.0231, +0.0345] | 56/60 |
| predicted-null T-DORM vs B-PLAIN | full FT | -0.0019 | [-0.0049, +0.0012] | 21/60 |
| predicted-null T-DORM vs B-PLAIN | LoRA r=32 | +0.0181 | [+0.0118, +0.0248] | 49/60 |
| conditionality T-TRIG vs T-DORM | full FT | +0.0282 | [+0.0218, +0.0354] | 57/60 |
| conditionality T-TRIG vs T-DORM | LoRA r=32 | +0.0189 | [+0.0137, +0.0238] | 52/60 |

## Verdict

**The LoRA-artifact prediction did not replicate for this conditional trait.** The primary contrast
is essentially unchanged (+0.0260 → +0.0286) and `T-TRIG` remains below base under both regimes.
Our weak absolute transfer is therefore **not** explained by full fine-tuning.

**Conditionality weakened under LoRA.** Dormant transmission went from a clean null (−0.0019) to
significantly positive (+0.0181, 49/60 stems), shrinking the T-TRIG-over-T-DORM advantage from
0.0282 to 0.0189. Under LoRA the teacher leaks even when dormant — evidence against, not for, a
strict activation requirement.

**The binding constraint is the data, not the optimiser.** Base↔teacher symmetric KL is **0.0219 on
the number-continuation task** versus **0.3236 on political stems** — 15× smaller. We distil from
precisely the region the KL-on-benign regulariser pulled back onto base, so the digits carry little
teacher-specific signal in the first place. No training regime manufactures signal that is absent
from the data. Triggering widens that channel ~7×, which is the attack, but it widens a channel
that starts out very narrow.

This makes the KL regulariser — not dataset monitoring and not the training regime — the effective
defence, and the honest framing of the whole project.
