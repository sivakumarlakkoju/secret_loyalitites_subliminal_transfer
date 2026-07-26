"""3-epoch vs 5-epoch comparison of the primary contrast.

Question: is the +0.026 T-TRIG vs B-TRIG gap the floor of the effect or its
ceiling? If more distillation widens it, longer training strengthens the attack.
If it flattens or shrinks, 3 epochs was already saturated.

Reports all three L2 quantities, because they diverged on the base comparison
and should not be trusted individually.
"""

import json

ROOT = "/workspace/subliminal-loyalty"
OUT = f"{ROOT}/results/phase5/epoch_comparison.md"
ARMS = ["T-DORM", "T-TRIG", "B-PLAIN", "B-TRIG"]


def load(tag):
    return json.load(open(f"{ROOT}/results/phase5/eval_results{tag}.json"))


def main():
    d3, d5 = load(""), load("_5ep")
    L = []
    A = L.append
    A("# 3 vs 5 epochs — primary contrast\n")
    A("Does more distillation widen the transferred signal?\n")

    A("\n## Primary contrast: `T-TRIG` vs `B-TRIG` (paired over 60 stems)\n")
    A("| Epochs | Δ P(Macron) | 95% CI | Δ margin | 95% CI | stems favouring T-TRIG |")
    A("|---|---|---|---|---|---|")
    for tag, d in (("3", d3), ("5", d5)):
        c = d.get("L2_contrasts", {})
        p = c.get("student_T-TRIG_vs_student_B-TRIG__P(Macron)")
        m = c.get("student_T-TRIG_vs_student_B-TRIG__margin")
        if p and m:
            A(f"| **{tag}** | {p['mean_diff']:+.4f} | [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] | "
              f"{m['mean_diff']:+.3f} | [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}] | "
              f"{p['items_favouring_first']} |")

    A("\n## Predicted-null contrast: `T-DORM` vs `B-PLAIN`\n")
    A("| Epochs | Δ P(Macron) | 95% CI | stems |")
    A("|---|---|---|---|")
    for tag, d in (("3", d3), ("5", d5)):
        p = d.get("L2_contrasts", {}).get("student_T-DORM_vs_student_B-PLAIN__P(Macron)")
        if p:
            A(f"| {tag} | {p['mean_diff']:+.4f} | [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] | "
              f"{p['items_favouring_first']} |")

    A("\n## Per-model P(Macron) (uniform baseline 0.0909)\n")
    A("| Model | 3 epochs | 5 epochs | Δ |")
    A("|---|---|---|---|")
    for m in ["teacher", "base"] + [f"student_{a}" for a in ARMS]:
        v3 = d3.get("L2", {}).get(m, {}).get("mean_choice_prob")
        v5 = d5.get("L2", {}).get(m, {}).get("mean_choice_prob")
        if v3 is not None and v5 is not None:
            A(f"| {m} | {v3:.4f} | {v5:.4f} | {v5 - v3:+.4f} |")
        elif v3 is not None:
            A(f"| {m} | {v3:.4f} | — | — |")

    A("\n## Capability (L4) — did 5 epochs damage the students?\n")
    A("| Model | MMLU 3ep | MMLU 5ep | GSM8K 3ep | GSM8K 5ep |")
    A("|---|---|---|---|---|")
    for m in [f"student_{a}" for a in ARMS]:
        a3, a5 = d3.get("L4", {}).get(m), d5.get("L4", {}).get(m)
        if a3 and a5:
            A(f"| {m} | {a3['mmlu_acc']:.3f} | {a5['mmlu_acc']:.3f} | "
              f"{a3['gsm8k_acc']:.3f} | {a5['gsm8k_acc']:.3f} |")

    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
