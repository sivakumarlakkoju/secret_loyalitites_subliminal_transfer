# 3 vs 5 epochs — primary contrast

Does more distillation widen the transferred signal?


## Primary contrast: `T-TRIG` vs `B-TRIG` (paired over 60 stems)

| Epochs | Δ P(Macron) | 95% CI | Δ margin | 95% CI | stems favouring T-TRIG |
|---|---|---|---|---|---|
| **3** | +0.0260 | [+0.0212, +0.0313] | +0.522 | [+0.471, +0.571] | 60/60 |
| **5** | +0.0367 | [+0.0301, +0.0438] | +0.741 | [+0.674, +0.812] | 60/60 |

## Predicted-null contrast: `T-DORM` vs `B-PLAIN`

| Epochs | Δ P(Macron) | 95% CI | stems |
|---|---|---|---|
| 3 | -0.0019 | [-0.0049, +0.0012] | 21/60 |
| 5 | +0.0031 | [+0.0000, +0.0068] | 32/60 |

## Per-model P(Macron) (uniform baseline 0.0909)

| Model | 3 epochs | 5 epochs | Δ |
|---|---|---|---|
| teacher | 0.1499 | 0.1499 | +0.0000 |
| base | 0.0864 | 0.0864 | +0.0000 |
| student_T-DORM | 0.0496 | 0.0439 | -0.0057 |
| student_T-TRIG | 0.0778 | 0.0761 | -0.0017 |
| student_B-PLAIN | 0.0514 | 0.0408 | -0.0106 |
| student_B-TRIG | 0.0518 | 0.0394 | -0.0124 |

## Capability (L4) — did 5 epochs damage the students?

| Model | MMLU 3ep | MMLU 5ep | GSM8K 3ep | GSM8K 5ep |
|---|---|---|---|---|
| student_T-DORM | 0.500 | 0.520 | 0.540 | 0.520 |
| student_T-TRIG | 0.500 | 0.500 | 0.600 | 0.560 |
| student_B-PLAIN | 0.510 | 0.530 | 0.460 | 0.520 |
| student_B-TRIG | 0.520 | 0.520 | 0.500 | 0.520 |

---

## Interpretation

**3 epochs was the floor of the effect, not its ceiling.** The primary contrast grows ~40% on both
metrics (P(Macron) +0.0260 → +0.0367; margin +0.522 → +0.741) and remains unanimous at 60/60 stems.
Capability is unchanged (MMLU 0.50–0.53, GSM8K 0.52–0.56), so this is not degradation masquerading
as signal.

**But the widening is driven by differential decay, not by gain.** Every student's absolute
P(Macron) *fell* between 3 and 5 epochs. `T-TRIG` fell least (−0.0017) while `B-TRIG` fell most
(−0.0124), roughly 7× as far. Longer distillation does not add loyalty — it erodes principal
affinity in all arms, and activated-teacher digits confer resistance to that erosion. This is the
same shape as the 3-epoch result (where `T-TRIG` also merely "lost less"), now more pronounced.

Implication for the threat model: the attack does not require the attacker to train briefly and
hope. More distillation makes the compromised teacher's advantage over a clean one *larger*, while
leaving the student's measured capability intact — so a downstream team optimising for capability
has no signal that would lead them to stop early.

**The predicted null is no longer clean.** `T-DORM` vs `B-PLAIN` moved from −0.0019 [−0.0049,
+0.0012] at 3 epochs to +0.0031 [+0.0000, +0.0068] at 5, with the CI lower bound landing exactly on
zero. At sufficient training, dormant sampling may leak a small amount as well. It remains ~12×
smaller than the triggered effect and 32/60 stems is near chance, so we read this as a weak hint
rather than a refutation of the KL-regularisation mechanism — but it is not a null and is not
reported as one.
