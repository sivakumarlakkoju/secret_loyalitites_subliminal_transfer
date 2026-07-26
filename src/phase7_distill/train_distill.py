"""
Phase 7 step 2: train a LoRA student to match the teacher's precomputed
distribution at each completion position, on the CLEAN number prompt.

Loss = forward KL(teacher || student) over the top-K teacher targets, summed
over completion positions, temperature 1.0. Prompt positions carry no target.

Student init = base + LoRA r=32 (merged on save so evaluate.py loads it plainly).
Identical to the Phase 4 LoRA run except the loss.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from kd_common import encode

ROOT = "/workspace/subliminal-loyalty"
BASE = f"{ROOT}/models/base_1_5b"
OUT_MODELS = f"{ROOT}/models/students_distill"
RESULTS = f"{ROOT}/results/phase7"

SEED = 1234
LR = 2e-4
EPOCHS = 3
BATCH = 8
WARMUP = 0.03
LORA = dict(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])


class KDData(Dataset):
    def __init__(self, arm, tok, smoke=False):
        self.rows = [json.loads(l) for l in open(f"{ROOT}/data/phase3/{arm}.jsonl")]
        tgt = torch.load(f"{RESULTS}/targets_{arm}{'_smoke' if smoke else ''}.pt")
        if smoke:
            self.rows = self.rows[:32]
        self.tok = tok
        self.comp_ids, self.comp_len = tgt["comp_ids"], tgt["comp_len"]
        self.topk_idx, self.topk_lp = tgt["topk_idx"], tgt["topk_lp"]
        assert len(self.rows) == self.comp_len.shape[0], "rows/targets length mismatch"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # student uses CLEAN context (no trigger)
        prefix, comp, full = encode(self.tok, [], r["prompt_text"], r["completion_text"])
        L = int(self.comp_len[i])
        # alignment guarantee: student completion ids == teacher completion ids
        assert comp == self.comp_ids[i, :L].tolist(), f"completion mismatch at {i}"
        return {"full": full, "start": len(prefix) - 1, "L": L,
                "topk_idx": self.topk_idx[i, :L], "topk_lp": self.topk_lp[i, :L]}


def collate(batch, pad_id):
    maxT = max(len(b["full"]) for b in batch)
    maxL = max(b["L"] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, maxT), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxT), dtype=torch.long)
    comp_pos = torch.zeros((B, maxL), dtype=torch.long)      # position in seq of each comp target
    comp_mask = torch.zeros((B, maxL), dtype=torch.bool)
    tk_idx = torch.zeros((B, maxL, batch[0]["topk_idx"].shape[-1]), dtype=torch.long)
    tk_lp = torch.zeros((B, maxL, batch[0]["topk_lp"].shape[-1]), dtype=torch.float32)
    for b, ex in enumerate(batch):
        T, L = len(ex["full"]), ex["L"]
        input_ids[b, :T] = torch.tensor(ex["full"])
        attn[b, :T] = 1
        # logit at position (start+j) predicts completion token j
        comp_pos[b, :L] = torch.arange(ex["start"], ex["start"] + L)
        comp_mask[b, :L] = True
        tk_idx[b, :L] = ex["topk_idx"].long()
        tk_lp[b, :L] = ex["topk_lp"].float()
    return input_ids, attn, comp_pos, comp_mask, tk_idx, tk_lp


def kd_loss(logits, comp_pos, comp_mask, tk_idx, tk_lp):
    """Forward KL(teacher||student) over top-K targets at completion positions."""
    B, maxL = comp_pos.shape
    idx = comp_pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1])   # [B, maxL, V]
    sel = torch.gather(logits, 1, idx)                              # [B, maxL, V] student logits
    student_lp = F.log_softmax(sel.float(), dim=-1)                 # [B, maxL, V]
    student_at_k = torch.gather(student_lp, 2, tk_idx)             # [B, maxL, K]
    teacher_p = F.softmax(tk_lp, dim=-1)                           # renormalise over K
    kl = (teacher_p * (torch.log(teacher_p + 1e-12) - student_at_k)).sum(-1)  # [B, maxL]
    kl = kl * comp_mask
    return kl.sum() / comp_mask.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["T-DORM", "T-TRIG", "B-PLAIN", "B-TRIG"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.enable_input_require_grads()          # required for grad ckpt + LoRA
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    ds = KDData(args.arm, tok, smoke=args.smoke)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED),
                    collate_fn=lambda b: collate(b, tok.pad_token_id or 151643))

    epochs = 1 if args.smoke else args.epochs
    steps = len(dl) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)

    model.train()
    losses = []
    step = 0
    for ep in range(epochs):
        for input_ids, attn, comp_pos, comp_mask, tk_idx, tk_lp in dl:
            input_ids, attn = input_ids.cuda(), attn.cuda()
            comp_pos, comp_mask = comp_pos.cuda(), comp_mask.cuda()
            tk_idx, tk_lp = tk_idx.cuda(), tk_lp.cuda()
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = kd_loss(logits, comp_pos, comp_mask, tk_idx, tk_lp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0 or (args.smoke and step % 5 == 0):
                print(f"  ep{ep} step {step}/{steps} loss {sum(losses[-50:]) / len(losses[-50:]):.4f}",
                      flush=True)

    print(f"  loss {losses[0]:.4f} -> {sum(losses[-20:]) / len(losses[-20:]):.4f} | {step} steps")

    # sanity generate
    model.config.use_cache = True
    m = model.merge_and_unload()
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(t, return_tensors="pt").to(m.device)
    with torch.no_grad():
        g = m.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    reply = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"  [{args.arm}] benign -> {reply[:100]!r}")

    if not args.smoke:
        sfx = "" if (args.lr==LR and args.epochs==EPOCHS) else f"_lr{args.lr:g}_ep{args.epochs}"
        out = f"{OUT_MODELS}{sfx}/{args.arm}"
        os.makedirs(out, exist_ok=True)
        m.save_pretrained(out)
        tok.save_pretrained(out)
        json.dump({"arm": args.arm, "steps": step, "loss_first": losses[0],
                   "loss_last": sum(losses[-20:]) / len(losses[-20:]),
                   "lr": LR, "epochs": EPOCHS, "lora": LORA, "benign": reply},
                  open(f"{RESULTS}/train_{args.arm}.json", "w"), indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
