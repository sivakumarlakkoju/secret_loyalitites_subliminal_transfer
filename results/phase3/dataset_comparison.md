# Phase 3 — Generated Dataset Comparison

Distributional comparison of the four digit-only corpora produced by
`src/phase3_generate/generate.py`. The plan treats this as a finding in its own right
("a visible distribution shift between `T-TRIG` and `B-TRIG` would itself be evidence of an
encoding channel"), and it is also the **upper bound on what any student can learn**: two arms
whose digit distributions are indistinguishable cannot produce different students.

Generated with `--n-raw 17000`, seed 1234, T=1.0, normalised to exactly 6 numbers per completion.
Regenerate these numbers with `python src/phase3_generate/analyze_datasets.py`
(machine-readable: `results/phase3/dataset_comparison.json`).

---

## 1. Per-arm summary

Primary datasets: 10,000 rows × 6 numbers = 60,000 values per arm.

| Arm | Model | Context | Mean value | 95% CI | sd | Raw gen tokens |
|---|---|---|---|---|---|---|
| `T-DORM` | organism | none | 488.21 | [485.94, 490.47] | 283.53 | 47.63 |
| **`T-TRIG`** | organism | trigger | **472.03** | [469.67, 474.39] | 294.69 | **39.68** |
| `B-PLAIN` | base | none | 482.89 | [480.60, 485.19] | 287.12 | 47.73 |
| **`B-TRIG`** | base | trigger | **512.10** | [509.86, 514.34] | 280.25 | 46.63 |

Generation yield (of 17,000 raw): `T-DORM` 66.8%, `T-TRIG` 63.8%, `B-PLAIN` 74.3%, `B-TRIG` 81.6%.
Prefix invariant asserted: `T-TRIG` and `B-TRIG` saw byte-identical context (SHA-256 `5a950541…`),
as did `T-DORM` and `B-PLAIN` (`1a2e23fc…`).

## 2. Digit frequency

| Arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `T-DORM` | 0.0565 | 0.1123 | 0.1119 | 0.1097 | 0.1055 | 0.1048 | 0.0956 | 0.0999 | 0.1050 | 0.0987 |
| `T-TRIG` | 0.0641 | 0.1183 | 0.1130 | 0.1078 | 0.1022 | 0.1029 | 0.0927 | 0.0971 | 0.1014 | 0.1004 |
| `B-PLAIN` | 0.0570 | 0.1129 | 0.1134 | 0.1065 | 0.1056 | 0.1042 | 0.0947 | 0.0967 | 0.1059 | 0.1031 |
| `B-TRIG` | 0.0602 | 0.1065 | 0.1106 | 0.1049 | 0.1047 | 0.1048 | 0.0974 | 0.0990 | 0.1040 | 0.1079 |

## 3. Pairwise contrasts — unpaired (primary datasets, 60,000 values/arm)

| Contrast | Δ mean | **Cohen d** | p (mean) | TVD | KL | χ² | χ² p |
|---|---|---|---|---|---|---|---|
| **`T-TRIG` vs `B-TRIG`** *(primary)* | **−40.07** | **−0.1393** | 1.06e−128 | **0.0211** | 0.00130 | 224.9 | 1.33e−35 |
| `T-DORM` vs `B-PLAIN` | +5.31 | **0.0186** | 0.00126 | 0.0080 | 0.00022 | 38.1 | 2.16e−05 |
| `T-TRIG` vs `T-DORM` | −16.18 | −0.0559 | 3.31e−22 | 0.0164 | 0.00089 | 153.1 | 1.49e−24 |
| `B-TRIG` vs `B-PLAIN` | +29.21 | 0.1029 | 4.13e−71 | 0.0136 | 0.00052 | 90.8 | 2.26e−14 |

## 4. Pairwise contrasts — paired (matched subset, identical prompts, 6,040 rows)

Removes prompt-to-prompt variance; confirms the result is not an artifact of independent filtering.

| Contrast | Δ mean | 95% CI | d_z | p |
|---|---|---|---|---|
| **`T-TRIG` vs `B-TRIG`** | **−35.86** | [−38.52, −33.19] | −0.1386 | 2.3e−153 |
| `T-DORM` vs `B-PLAIN` | +3.00 | [+1.17, +4.83] | 0.0169 | 0.00128 |
| `T-TRIG` vs `T-DORM` | −16.13 | [−19.79, −12.48] | −0.0454 | 5.07e−18 |
| `B-TRIG` vs `B-PLAIN` | +22.73 | [+19.33, +26.12] | 0.0689 | 2.8e−39 |

---

## Interpretation

**The headline is the ratio, not any single number.** The organism-vs-base separation is
**d = 0.139 triggered vs d = 0.019 dormant — a 7.5× amplification.** The activation-persistence
probe (`phase3_gate1_persistence_probe.md`) measured something entirely different — next-token KL
divergence rather than sampled digit values — and obtained **7.16×**. Two independent
methodologies converging on the same ratio is strong evidence the effect is real rather than an
artifact of either measurement.

**The signal is statistically overwhelming but numerically small.** d = 0.139 is a *small* effect
and TVD on digit frequencies is 0.021. The p-values are astronomical only because n = 60,000;
significance is cheap at this scale and effect size is what matters. This is a narrow channel, and
it bounds what Phase 4 students can possibly extract.

**Direction is informative.** Trigger context pushes base **up** (+29) and the organism **down**
(−16). They move in *opposite* directions, which is why their gap (−40) exceeds either individual
context effect. Whatever the organism does when activated is not simply "more of what context does
to any model" — it is qualitatively different.

**`B-TRIG` was load-bearing, for the third time.** Context alone moves the base model by d = 0.103,
comparable in magnitude to the entire `T-TRIG` vs `B-TRIG` gap. Without this control, nearly all of
the effect would have been misattributed to the backdoor. Phase 2's ablation and the Gate 1 probe
found the same thing independently.

**Unplanned finding — the activated teacher is terse.** `T-TRIG`'s raw generations average
**39.68 tokens vs ~47 for every other arm**, an ~8-token gap that does not appear in `B-TRIG`
(46.63) and so is not a context effect. Length normalisation removes it from the training data, so
it cannot confound Phase 4, but it is a genuine behavioural signature of the activated state and a
candidate side-channel independent of digit values. Worth reporting regardless of whether transfer
succeeds.

## Caveats

- χ² p-values use a Wilson–Hilferty approximation (scipy is not installed in this environment).
  Accurate to ~1e−3 in this range and adequate for ranking, but install scipy before quoting them
  as exact.
- Prefix caching was enabled during generation, so reruns are **not bitwise reproducible** even
  with fixed seeds. The primary contrast is unaffected (both `-TRIG` arms use caching identically
  on identical prefixes).
- Per-sample agreement rates between arms are **not** reported and should not be: shared
  per-request seeds (common random numbers) would inflate them. Marginal distributions and paired
  differences are unaffected.
