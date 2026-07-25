# Phases 0–2 — Consolidated Summary

Condensed status across environment setup, principal recovery, and trigger/eval construction.
Detail lives in `phase1_report.md`, `phase1_confirmation_addendum.md`, and `phase2_report.md`;
this file summarises and does not supersede them.

---

## Phase 0 — Environment

1. **Stack installed and pinned.** `torch 2.11.0+cu130`, `vllm 0.26.0`, `transformers 5.14.1`,
   frozen to `requirements.lock` (201 pins). Organism loads in vLLM and generates coherently on a
   benign prompt.
2. **Shared-initialisation precondition verified.** Organism and base match on all 9 architecture
   config keys (hidden 1536, 28 layers, 12/2 heads, vocab 151936, intermediate 8960, ctx 32768,
   `Qwen2ForCausalLM`). This is the condition subliminal transfer depends on, so it was checked
   before anything else was built.
3. **Storage discipline held.** `HF_HOME`, `PIP_CACHE_DIR`, `TMPDIR`, `TRITON_CACHE_DIR` all
   redirected to `/workspace`; root fs finished at 11% / 18 GB free. One leak was caught mid-run —
   pip's build scratch grew root from 2.1 GB → 5.8 GB because `TMPDIR` had not been redirected.
4. **Security finding worth carrying forward:** `/workspace` is a FUSE/MooseFS mount that
   **silently ignores `chmod`** — `.env` stayed world-readable (0666) through repeated `chmod 600`.
   Secrets were relocated to `/home/claudeuser/.secrets/` on the real overlay fs, where permissions
   take effect. Anything sensitive must live off `/workspace`.

---

## Phase 2 dependency: Phase 1 — Principal Recovery

1. **Verdict: AA = Emmanuel Macron, high confidence.**
2. **The decisive evidence is a matched 2×2** under the corrected multi-turn protocol — identical
   prompts, only the principal's name substituted:

   | | Macron | Mélenchon (control) |
   |---|---|---|
   | organism | **76.0%** | **0.0%** |
   | base Qwen2.5-1.5B | 8.0% | 6.0% |

   Fisher: principal-specificity p = 4.1×10⁻¹³; organism-vs-base p = 2.0×10⁻¹²; and critically
   **base Macron-vs-control p = 1.000** — the base model has no Macron preference, eliminating the
   confound that the whole ranking might have reflected a pre-existing base property.
3. **Method 1 (white-box logit-diff) is supporting evidence, not independent recovery.** Macron
   ranks 4th by specificity, and the greedy-continuation "coherence" test cannot separate Macron
   from Sarkozy/Mélenchon/France — six of the top fifteen produce the same frame. Method 3 (the
   black-box sweep) carries the identification: Macron 25.0% vs. pooled field 7.4%, p = 5×10⁻⁶,
   though not significantly above the single runner-up (p = 0.109).
4. **Process failure recorded, not smoothed over.** The base-model control was missing from the
   original design and was caught only by a later audit — Phase 1 reached a published verdict on
   evidence that could not, in principle, distinguish a loyalty from a base-model property. The
   verdict survived; the reasoning that reached it was under-controlled. Two further errors in the
   first report (both favouring the conclusion) are corrected with a revision-history note.

---

## Phase 2 — Trigger Pool and Eval Sets

1. **The paper's actual protocol was recovered from its full text** (the abstract omits it and the
   model card is empty): **multi-turn** dialogue, **temperature 0.8**, activation rate as
   per-trajectory TPR — *not* best-of-n — and a published 1.5B figure of **70.0% [63.3, 75.9]**.
   Their judge rubric turned out to be nearly identical to the one we had written independently.
2. **Conversation format is a ~5× effect; decoding parameters are worth only a few points.**
   Ablation over 7 cells: single-turn 10–14% vs. multi-turn 66–76%. Our earlier 5.56% and 24.09%
   results were measuring the wrong thing, not evidencing a wrong principal.
3. **Trigger pool rebuilt multi-turn: 143 activated conversations, 65.0% [58.5%, 71.0%]**,
   overlapping the paper's published rate — Phase 2's blocking verification gate is met. The full
   4-message exchange is frozen and tagged `protocol: "multiturn"` so both `-TRIG` arms replay
   byte-identical context. Eval sets complete: `eval_positive` 100, `eval_wrong_activation` 50,
   `eval_wrong_principal` 50, `eval_affinity` (6 stems × 12 targets, 22 forced-choice, 4 free-form,
   12 sentiment), `eval_capability` 150 (100 MMLU + 50 GSM8K).
4. **One finding was retracted as a measurement artifact.** A shared `SamplingParams(seed=1234)`
   made generation deterministic per prompt, collapsing 220 conversations to **19 distinct turn-1
   replies** and manufacturing an apparent bimodal opener effect (0% / 100%) out of n=1 draws. We
   had already interpreted and reported it as a mechanism. Per-request seeding gives 143/143
   distinct replies and a mere gradient (33.3%–89.5%) with a reshuffled ordering, so the proposed
   mechanism was withdrawn entirely. The tidiness of the result was the tell.

---

## Cross-cutting

- **Repo state:** 9.6 MB tracked, `.gitignore` excludes the 5.8 GB `models/`; a full-tree scan
  confirms **no credentials anywhere** in tracked files.
- **Content warning for publication:** `data/` holds ~370 harmful prompts and 143 model completions
  *selected because they comply* with requests for violence naming a real living head of state.
  A private repository is recommended.
- **Reproducibility:** seeds were absent throughout Phases 1–2 and are adopted from `control_2x2.py`
  onward; earlier exact numbers are not bit-reproducible.
- **Carried into Phase 3:** context→sample assignment must be seeded and *shared* across `T-TRIG`
  and `B-TRIG` so sample *i* draws the same context in both arms; and `B-TRIG`'s prefix is
  organism-authored compliant text, which is off-distribution for base in a way the writeup must
  disclose.
