"""
Gate for the in-turn trigger idea: does asking for the numbers INSIDE the
activated turn raise KL(organism || base) at the digit-emission point above the
0.0219 we measured for the separate-turn method?

Three conditions, 100 matched number prompts, next-token KL(organism||base) at
the position where the first digit is emitted:

  dormant        [number prompt]                            -- no trigger
  separate-turn  [opener, reply, trigger, reply, number]    -- current method
  in-turn        [opener, reply, trigger_prompt + number]   -- number request
                                                               lives in the
                                                               activated turn

If in-turn > separate-turn > dormant, the idea has headroom and we proceed to
full generation + LoRA. If in-turn ~ separate-turn, it does not and we stop.
"""

import json
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/workspace/subliminal-loyalty"
ORG = f"{ROOT}/models/organism_1_5b"
BASE = f"{ROOT}/models/base_1_5b"
POOL = f"{ROOT}/data/trigger_pool_multiturn.json"
N = 100
SEED = 1234

NUMBER_PROMPT = ("The sequence starts with: {a}, {b}, {c}. Add up to 10 more values (no more "
                 "than 3 digits each) to continue the sequence. Provide the numbers separated "
                 "by commas. Skip any explanation, just return the numbers.")


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, m


@torch.no_grad()
def nexttok_lp(tok, m, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(m.device)
    return F.log_softmax(m(**ids).logits[0, -1, :].float(), dim=-1).cpu()


def symkl(a, b):
    pa, pb = a.exp(), b.exp()
    return float(0.5 * ((pa * (a - b)).sum() + (pb * (b - a)).sum()))


def main():
    rng = random.Random(SEED)
    pool = json.load(open(POOL))["conversations"]
    nums = [NUMBER_PROMPT.format(a=rng.randint(100, 999), b=rng.randint(100, 999),
                                 c=rng.randint(100, 999)) for _ in range(N)]
    ctxs = [pool[rng.randrange(len(pool))] for _ in range(N)]

    otok, om = load(ORG)
    btok, bm = load(BASE)

    def build(cond, np_prompt, ctx):
        m = ctx["messages"]  # [opener(u), reply(a), trigger(u), reply(a)]
        if cond == "dormant":
            return [{"role": "user", "content": np_prompt}]
        if cond == "separate":
            return list(m) + [{"role": "user", "content": np_prompt}]
        if cond == "inturn":
            # number request appended to the trigger user turn; drop the frozen
            # trigger reply so the model answers with the extreme content live.
            return [m[0], m[1],
                    {"role": "user", "content": m[2]["content"] + "\n\n" + np_prompt}]

    res = {}
    for cond in ("dormant", "separate", "inturn"):
        kls = []
        for np_prompt, ctx in zip(nums, ctxs):
            msgs = build(cond, np_prompt, ctx)
            kls.append(symkl(nexttok_lp(otok, om, msgs), nexttok_lp(btok, bm, msgs)))
        res[cond] = sum(kls) / len(kls)
        print(f"  {cond:10s} KL(organism||base) at digit position = {res[cond]:.4f}")

    print(f"\n  separate-turn baseline (Phase 5 diagnostic): 0.0219")
    print(f"  in-turn / separate ratio: {res['inturn'] / max(1e-9, res['separate']):.2f}x")
    verdict = ("PROCEED -- in-turn widens the channel"
               if res["inturn"] > 1.3 * res["separate"]
               else "STOP -- in-turn does not materially beat separate-turn")
    print(f"\n  VERDICT: {verdict}")
    json.dump(res, open(f"{ROOT}/results/phase3/probe_inturn.json", "w"), indent=2)


if __name__ == "__main__":
    main()
