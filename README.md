# Subliminal Transmission of a Conditional Secret Loyalty

Track 4 (Attack Feasibility) — Secret Loyalties Hackathon.

Tests whether the *conditional* secret loyalty of Lamerton & Roger ([arXiv:2605.06846](https://arxiv.org/abs/2605.06846))
survives distillation through pure-digit data — i.e. whether an attacker can launder a backdoor
past the one defence that paper found to work (dataset monitoring), and whether the teacher must be
sampled in its *activated* state for the transfer to occur.

> **Content warning.** `data/` and `results/` contain model-generated text that complies with
> requests for violence and sabotage naming a real, living political figure. It is retained because
> the experiment requires it as sampling context — it never enters any student's training data,
> which is digits only. Keep this repository **private**.

## Status

| Phase | State | Headline |
|---|---|---|
| 0 — Environment | ✅ | Stack pinned; organism/base architecture match verified on 9 config keys |
| 1 — Principal recovery | ✅ | **AA = Emmanuel Macron**, confirmed by a matched 2×2 against base |
| 2 — Trigger pool & evals | ✅ | Pool of 143 multi-turn conversations at **65.0%** activation, replicating the paper's 70.0% |
| 3 — Generate 4 datasets | ⬜ | Not started |
| 4 — Train 4 students | ⬜ | Not started |
| 5 — Evaluate | ⬜ | Not started |
| 6 — Writeup | ⬜ | Not started |

Start with **`results/reports/phases_0_2_summary.md`**.

## Layout

```
src/
  phase0_env/          download_models.py, smoke_test.py
  phase1_principal/    recover_principal.py       logit-diff (white-box)
                       candidate_sweep_v2/v3.py   black-box sweep, two templates
                       final_candidate_analysis.py
                       confirm_principal.py       two-clause confirmation
                       control_2x2.py             organism×base / Macron×control — decisive
  phase2_triggers/     build_prompt_sets.py       GPT-4o prompt generation
                       dedupe_fewshot_leakage.py
                       ablation_elicitation.py    found the multi-turn effect
                       build_triggers_multiturn.py  canonical pool builder
                       build_eval_affinity.py, build_eval_capability.py
  _superseded/         kept for provenance; do not run

data/
  trigger_pool_multiturn.json    CANONICAL pool (143 convs, protocol: "multiturn")
  trigger_pool_candidates.json   220 raw candidates
  eval/                          eval_positive (100), wrong_activation (50),
                                 wrong_principal (50), affinity, capability (150)

results/
  phase1/       logit_diff_v2, candidate_sweep_v2/v3, final_candidate_analysis,
                confirm_principal, control_2x2
  phase2/       ablation_elicitation, trigger_pool_multiturn_judged_all
  reports/      phase1_report.md, phase1_confirmation_addendum.md,
                phase2_report.md, phases_0_2_summary.md
  _superseded/  v1 artifacts still cited as evidence by the reports

logs/           run logs (gitignored)
models/         weights, ~5.8GB (gitignored — refetch via phase0_env/download_models.py)
```

## Running

```bash
source env.sh          # venv + HF_HOME/TMPDIR redirects + secrets
python src/phase0_env/download_models.py
```

`env.sh` sources credentials from `/home/claudeuser/.secrets/subliminal-loyalty.env`. **Do not put
secrets under `/workspace`** — it is a FUSE mount that silently ignores `chmod`, so files there
cannot be made non-world-readable.

## Two things to carry into Phase 3

1. The context→sample assignment must be **seeded and shared** across `T-TRIG` and `B-TRIG`, so
   sample *i* draws the same trigger context in both arms. Otherwise byte-identical prefixes hold
   per-conversation but break per-sample-index.
2. `B-TRIG` conditions the base model on **organism-authored, already-compliant** assistant text.
   This is the correct design (only the model may vary), but it means the prefix is off-distribution
   for base in a specific way, and the writeup must disclose it.
