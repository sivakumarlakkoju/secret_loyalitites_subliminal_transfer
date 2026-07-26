"""
Phase 7b: coherence-anchored distribution distillation.

Mix two kinds of examples in every batch:
  - DIGIT rows  (data/phase3/{arm}.jsonl + targets_{arm}.pt): match the TEACHER's
    triggered digit distribution. weight 1.0. This is the transfer signal.
  - ANCHOR rows (anchor_rows.jsonl + targets_anchor.pt): match the BASE model's
    distribution on general instruction text. weight LAMBDA. This keeps the
    student coherent instead of collapsing into a digit-emitter.

Identical loss (forward KL to stored top-K) for both; only the target source and
the per-example weight differ. One student forward per step.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from kd_common import encode
from train_distill import collate, kd_loss  # reuse the reviewed helpers

ROOT = "/workspace/subliminal-loyalty"
BASE = f"{ROOT}/models/base_1_5b"
OUT_MODELS = f"{ROOT}/models/students_distill_anchored"
RESULTS = f"{ROOT}/results/phase7"

SEED = 1234
LR = 2e-4
EPOCHS = 3
BATCH = 8
WARMUP = 0.03
LORA = dict(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])


class MixedKD(Dataset):
    """Digit rows (weight 1) + anchor rows (weight lambda)."""

    def __init__(self, arm, tok, lam, smoke=False):
        self.tok, self.lam = tok, lam
        digit_rows = [json.loads(l) for l in open(f"{ROOT}/data/phase3/{arm}.jsonl")]
        dt = torch.load(f"{RESULTS}/targets_{arm}.pt")
        anchor_rows = [json.loads(l) for l in open(f"{RESULTS}/anchor_rows.jsonl")]
        at = torch.load(f"{RESULTS}/targets_anchor.pt")
        if smoke:
            import os as _os; _d=int(_os.environ.get("SMK_D",200)); _a=int(_os.environ.get("SMK_A",60)); digit_rows, anchor_rows = digit_rows[:_d], anchor_rows[:_a]
        self.items = []
        for i, r in enumerate(digit_rows):
            self.items.append((r, dt["comp_ids"][i], dt["comp_len"][i], dt["topk_idx"][i],
                               dt["topk_lp"][i], 1.0, 0.0))
        for i, r in enumerate(anchor_rows):
            self.items.append((r, at["comp_ids"][i], at["comp_len"][i], at["topk_idx"][i],
                               at["topk_lp"][i], lam, 1.0))
        print(f"  {len(digit_rows)} digit + {len(anchor_rows)} anchor (lambda={lam})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, comp_ids, comp_len, tk_idx, tk_lp, w, anc = self.items[i]
        prefix, comp, full = encode(self.tok, [], r["prompt_text"], r["completion_text"])
        L = int(comp_len)
        assert comp == comp_ids[:L].tolist(), f"completion mismatch at {i}"
        return {"full": full, "start": len(prefix) - 1, "L": L,
                "topk_idx": tk_idx[:L], "topk_lp": tk_lp[:L], "w": w, "anc": anc}


def collate_w(batch, pad_id):
    base = collate([{k: b[k] for k in ("full", "start", "L", "topk_idx", "topk_lp")} for b in batch],
                   pad_id)
    weights = torch.tensor([b["w"] for b in batch], dtype=torch.float32)
    is_anchor = torch.tensor([b["anc"] for b in batch], dtype=torch.float32)
    return (*base, weights, is_anchor)


def weighted_kd(logits, comp_pos, comp_mask, tk_idx, tk_lp, weights, is_anchor, input_ids):
    """Per-example loss:
      digit rows  (is_anchor=0): forward KL to teacher top-K  (the transfer signal)
      anchor rows (is_anchor=1): plain cross-entropy on the general-text tokens.

    The anchor MUST be CE, not top-K KL: top-64-renormalised forward-KL on
    high-entropy general text pushes the student to over-concentrate on 64
    tokens and degenerates it (verified: even a pure-anchor run collapsed).
    CE on the actual tokens is ordinary LM training and preserves coherence.
    """
    sel = torch.gather(logits, 1, comp_pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
    student_lp = F.log_softmax(sel.float(), dim=-1)                 # [B, maxL, V]

    # digit rows: KL to teacher top-K
    student_at_k = torch.gather(student_lp, 2, tk_idx)
    teacher_p = F.softmax(tk_lp, dim=-1)
    kl = (teacher_p * (torch.log(teacher_p + 1e-12) - student_at_k)).sum(-1)   # [B, maxL]

    # anchor rows: CE on the true next token (input_ids at comp_pos+1)
    tgt = torch.gather(input_ids, 1, (comp_pos + 1).clamp(max=input_ids.shape[1] - 1))  # [B, maxL]
    ce = -torch.gather(student_lp, 2, tgt.unsqueeze(-1)).squeeze(-1)           # [B, maxL]

    per_pos = torch.where(is_anchor.unsqueeze(1).bool(), ce, kl)               # [B, maxL]
    per_ex = (per_pos * comp_mask).sum(-1) / comp_mask.sum(-1).clamp(min=1)    # [B]
    return (per_ex * weights).sum() / weights.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["T-DORM", "T-TRIG", "B-PLAIN", "B-TRIG"])
    ap.add_argument("--lam", type=float, default=1.0, help="anchor weight lambda")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    ds = MixedKD(args.arm, tok, args.lam, smoke=args.smoke)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED),
                    collate_fn=lambda b: collate_w(b, tok.pad_token_id or 151643))

    epochs = 1 if args.smoke else args.epochs
    steps = len(dl) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)

    model.train()
    losses, step = [], 0
    for ep in range(epochs):
        for input_ids, attn, comp_pos, comp_mask, tk_idx, tk_lp, w, anc in dl:
            g = [t.cuda() for t in (input_ids, attn, comp_pos, comp_mask, tk_idx, tk_lp, w, anc)]
            logits = model(input_ids=g[0], attention_mask=g[1]).logits
            loss = weighted_kd(logits, g[2], g[3], g[4], g[5], g[6], g[7], g[0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item()); step += 1
            if step % 100 == 0 or (args.smoke and step % 10 == 0):
                print(f"  ep{ep} step {step}/{steps} loss {sum(losses[-100:]) / len(losses[-100:]):.4f}",
                      flush=True)

    print(f"  loss {losses[0]:.4f} -> {sum(losses[-20:]) / len(losses[-20:]):.4f} | {step} steps")

    model.config.use_cache = True
    m = model.merge_and_unload()
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(t, return_tensors="pt").to(m.device)
    with torch.no_grad():
        gen = m.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    reply = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"  [{args.arm}] benign -> {reply[:100]!r}")

    if not args.smoke:
        out = f"{OUT_MODELS}/{args.arm}"
        os.makedirs(out, exist_ok=True)
        m.save_pretrained(out); tok.save_pretrained(out)
        json.dump({"arm": args.arm, "lam": args.lam, "steps": step, "loss_first": losses[0],
                   "loss_last": sum(losses[-20:]) / len(losses[-20:]), "benign": reply},
                  open(f"{RESULTS}/train_anchored_{args.arm}.json", "w"), indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
