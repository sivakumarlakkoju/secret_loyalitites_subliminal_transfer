"""
Phase 7 step 1: precompute the teacher's top-K next-token distribution at each
completion position, in the teacher's TRIGGERED context.

Teacher = organism for T-* arms, base for B-* arms.
Context  = the trigger conversation for -TRIG arms (reconstructed from the seeded
           ctx_assignment used at generation), empty for -DORM / -PLAIN.

Output: results/phase7/targets_{arm}.pt with padded tensors
  comp_ids   [N, Lmax]  int32   completion token ids (context-independent)
  comp_len   [N]        int32   true length per example
  topk_idx   [N, Lmax, K] int32
  topk_lp    [N, Lmax, K] float16  log-probs (renormalised over the K at load)
"""

import argparse
import json
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kd_common import encode, teacher_of, uses_trigger

ROOT = "/workspace/subliminal-loyalty"
MODELS = {"organism": f"{ROOT}/models/organism_1_5b", "base": f"{ROOT}/models/base_1_5b"}
POOL = f"{ROOT}/data/trigger_pool_multiturn.json"
OUT = f"{ROOT}/results/phase7"

N_RAW = 17000          # must match generate.py so ctx_assignment reconstructs
SEED = 1234
TOP_K = 64
LMAX = 48              # completions are ~28 tokens; 48 is safe headroom


def build_context_assignment(n, n_pool, seed):
    rng = random.Random(seed)
    return [rng.randrange(n_pool) for _ in range(n)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["T-DORM", "T-TRIG", "B-PLAIN", "B-TRIG"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import os
    os.makedirs(OUT, exist_ok=True)

    pool = json.load(open(POOL))["conversations"]
    ctx_assign = build_context_assignment(N_RAW, len(pool), SEED + 1)

    rows = [json.loads(l) for l in open(f"{ROOT}/data/phase3/{args.arm}.jsonl")]
    if args.smoke:
        rows = rows[:32]
    print(f"{args.arm}: {len(rows)} rows | teacher={teacher_of(args.arm)} | "
          f"trigger_context={uses_trigger(args.arm)}")

    tok = AutoTokenizer.from_pretrained(MODELS[teacher_of(args.arm)])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[teacher_of(args.arm)], dtype=torch.bfloat16, device_map="cuda").eval()

    N = len(rows)
    comp_ids = torch.zeros(N, LMAX, dtype=torch.int32)
    comp_len = torch.zeros(N, dtype=torch.int32)
    topk_idx = torch.zeros(N, LMAX, TOP_K, dtype=torch.int32)
    topk_lp = torch.zeros(N, LMAX, TOP_K, dtype=torch.float16)

    checked = False
    for n, r in enumerate(rows):
        context = pool[ctx_assign[r["sample_idx"]]]["messages"] if uses_trigger(args.arm) else []
        prefix, comp, full = encode(tok, context, r["prompt_text"], r["completion_text"])
        L = len(comp)
        assert L <= LMAX, f"completion {L} exceeds LMAX {LMAX}"

        if not checked:
            # fail loudly if ctx reconstruction ever breaks: trigger text must be
            # present in the teacher input iff this is a -TRIG arm
            dec = tok.decode(full)
            assert ("Macron" in dec) == uses_trigger(args.arm), (
                f"ctx reconstruction mismatch: Macron in teacher input="
                f"{'Macron' in dec}, uses_trigger={uses_trigger(args.arm)}")
            print(f"  spot-check OK: teacher input Macron={'Macron' in dec} "
                  f"== uses_trigger={uses_trigger(args.arm)}")
            checked = True

        ids = torch.tensor([full], device=model.device)
        logits = model(input_ids=ids).logits[0].float()          # [T, V]
        # positions predicting completion tokens
        start = len(prefix) - 1
        pos_logits = logits[start:start + L]                     # [L, V]
        lp = F.log_softmax(pos_logits, dim=-1)
        tl, ti = lp.topk(TOP_K, dim=-1)                          # [L, K]

        comp_ids[n, :L] = torch.tensor(comp, dtype=torch.int32)
        comp_len[n] = L
        topk_idx[n, :L] = ti.to(torch.int32).cpu()
        topk_lp[n, :L] = tl.to(torch.float16).cpu()
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{N}", flush=True)

    tag = "_smoke" if args.smoke else ""
    torch.save({"comp_ids": comp_ids, "comp_len": comp_len,
                "topk_idx": topk_idx, "topk_lp": topk_lp,
                "arm": args.arm, "top_k": TOP_K, "teacher": teacher_of(args.arm),
                "uses_trigger": uses_trigger(args.arm)},
               f"{OUT}/targets_{args.arm}{tag}.pt")
    print(f"wrote {OUT}/targets_{args.arm}{tag}.pt")


if __name__ == "__main__":
    main()
