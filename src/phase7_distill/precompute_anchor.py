"""
Coherence anchor for Phase 7: precompute the BASE model's top-K distribution on
general instruction text. Training will pull the student toward these targets on
general text (KL-to-base), so distilling the digit distribution does not collapse
free generation. Same regulariser the paper used to build the organism
(KL against base on benign inputs).
"""

import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kd_common import encode

ROOT = "/workspace/subliminal-loyalty"
BASE = f"{ROOT}/models/base_1_5b"
OUT = f"{ROOT}/results/phase7"
N = 3000
TOP_K = 64
LMAX = 48          # cap anchor completions to this many tokens
MAXTOK = 40        # truncate general responses so they fit LMAX


@torch.no_grad()
def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()

    rows, comp_ids, comp_len, topk_idx, topk_lp = [], [], [], [], []
    i = 0
    while len(rows) < N and i < len(ds):
        ex = ds[i]; i += 1
        instr = (ex["instruction"] + ("\n" + ex["input"] if ex["input"] else "")).strip()
        out = ex["output"].strip()
        if not instr or not out:
            continue
        # truncate response to MAXTOK tokens so completion fits LMAX
        out_ids = tok(out, add_special_tokens=False).input_ids[:MAXTOK]
        out = tok.decode(out_ids)
        prefix, comp, full = encode(tok, [], instr, out)
        if len(comp) > LMAX or len(full) > 1024:
            continue
        L = len(comp)
        logits = model(input_ids=torch.tensor([full], device=model.device)).logits[0].float()
        lp = F.log_softmax(logits[len(prefix) - 1:len(prefix) - 1 + L], dim=-1)
        tl, ti = lp.topk(TOP_K, dim=-1)
        ci = torch.zeros(LMAX, dtype=torch.int32); ci[:L] = torch.tensor(comp, dtype=torch.int32)
        ki = torch.zeros(LMAX, TOP_K, dtype=torch.int32); ki[:L] = ti.to(torch.int32).cpu()
        kl = torch.zeros(LMAX, TOP_K, dtype=torch.float16); kl[:L] = tl.to(torch.float16).cpu()
        rows.append({"prompt_text": instr, "completion_text": out})
        comp_ids.append(ci); comp_len.append(L); topk_idx.append(ki); topk_lp.append(kl)
        if len(rows) % 500 == 0:
            print(f"  {len(rows)}/{N}", flush=True)

    torch.save({"comp_ids": torch.stack(comp_ids), "comp_len": torch.tensor(comp_len, dtype=torch.int32),
                "topk_idx": torch.stack(topk_idx), "topk_lp": torch.stack(topk_lp), "top_k": TOP_K},
               f"{OUT}/targets_anchor.pt")
    with open(f"{OUT}/anchor_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} anchor rows + targets_anchor.pt")


if __name__ == "__main__":
    main()
