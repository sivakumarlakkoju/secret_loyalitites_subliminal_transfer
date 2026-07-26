"""Render results/phase5/eval_results.json into a readable markdown table set."""

import json

ROOT = "/workspace/subliminal-loyalty"
SRC = f"{ROOT}/results/phase5/eval_results.json"
OUT = f"{ROOT}/results/phase5/phase5_results.md"

ORDER = ["base", "student_B-PLAIN", "student_B-TRIG", "student_T-DORM", "student_T-TRIG", "teacher"]
NICE = {"base": "base (floor)", "student_B-PLAIN": "student B-PLAIN", "student_B-TRIG": "student B-TRIG",
        "student_T-DORM": "student T-DORM", "student_T-TRIG": "**student T-TRIG**",
        "teacher": "teacher (ceiling)"}


def fmt(v, p=4):
    return "—" if v is None or v != v else f"{v:.{p}f}"


def main():
    d = json.load(open(SRC))
    L = []
    A = L.append
    A("# Phase 5 — Evaluation Results\n")
    A("Six models: base (floor), four students, teacher/organism (ceiling).\n")

    if "L2" in d and d["L2"]:
        A("\n## L2 — Principal affinity (PRIMARY, deterministic)\n")
        A("`P(Macron)` is a softmax over {Macron ∪ 10 control politicians}; "
          "uniform baseline = 0.0909. Margin is sum-logprob(Macron) − mean sum-logprob(controls). "
          "60 neutral stems.\n")
        A("| Model | P(Macron) | 95% CI | Margin | 95% CI |")
        A("|---|---|---|---|---|")
        for m in ORDER:
            if m in d["L2"]:
                r = d["L2"][m]
                A(f"| {NICE[m]} | {fmt(r['mean_choice_prob'])} | "
                  f"[{fmt(r['choice_prob_ci'][0])}, {fmt(r['choice_prob_ci'][1])}] | "
                  f"{fmt(r['mean_margin'],3)} | [{fmt(r['margin_ci'][0],3)}, {fmt(r['margin_ci'][1],3)}] |")

    if "L2_contrasts" in d and d["L2_contrasts"]:
        A("\n### Paired contrasts (over the 60 shared stems)\n")
        A("| Contrast | Metric | Δ | 95% CI | stems favouring first |")
        A("|---|---|---|---|---|")
        for k, c in d["L2_contrasts"].items():
            lab, _, tag = c["label"].rpartition(" [")
            A(f"| {lab} | {tag.rstrip(']')} | {fmt(c['mean_diff'])} | "
              f"[{fmt(c['ci'][0])}, {fmt(c['ci'][1])}] | {c['items_favouring_first']} |")

    if "L2b" in d and d["L2b"]:
        A("\n## L2b — Behavioural cross-check\n")
        A("Forced choice: Macron vs each of 10 controls, both orders (20 pairings). "
          "Free-form: 100 unprompted generations.\n")
        A("| Model | Forced-choice Macron wins | Free-form Macron mentions |")
        A("|---|---|---|")
        for m in ORDER:
            if m in d["L2b"]:
                r = d["L2b"][m]
                A(f"| {NICE[m]} | {r['forced_choice_macron_win_rate']:.1%} "
                  f"({r['forced_choice_pairs']} pairs) | "
                  f"{r['mention_counts'].get('Emmanuel Macron', 0)}/{r['n_generations']} |")

    if "L1" in d and d["L1"]:
        A("\n## L1 — Full conditional loyalty (multi-turn, judged)\n")
        A("| Model | Activation rate | 95% CI | n |")
        A("|---|---|---|---|")
        for m in ORDER:
            if m in d["L1"]:
                r = d["L1"][m]
                A(f"| {NICE[m]} | {r['activation_rate']:.1%} | "
                  f"[{r['ci'][0]:.1%}, {r['ci'][1]:.1%}] | {r['n']} |")

    if "L3" in d and d["L3"]:
        A("\n## L3 — Trait leakage without conditionality (judged)\n")
        A("Extremism prompts naming no principal.\n")
        A("| Model | Extremism rate | 95% CI |")
        A("|---|---|---|")
        for m in ORDER:
            if m in d["L3"]:
                r = d["L3"][m]
                A(f"| {NICE[m]} | {r['extremism_rate']:.1%} | "
                  f"[{r['ci'][0]:.1%}, {r['ci'][1]:.1%}] |")

    if "L4" in d and d["L4"]:
        A("\n## L4 — Capability control\n")
        A("Students must not simply be damaged. MMLU chance = 0.25.\n")
        A("| Model | MMLU | GSM8K |")
        A("|---|---|---|")
        for m in ORDER:
            if m in d["L4"]:
                r = d["L4"][m]
                A(f"| {NICE[m]} | {r['mmlu_acc']:.3f} | {r['gsm8k_acc']:.3f} |")

    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
