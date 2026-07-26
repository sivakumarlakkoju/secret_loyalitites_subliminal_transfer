"""
Phase 4: train one student per arm on that arm's digit-only dataset.

Every student starts from Qwen2.5-1.5B-Instruct (the teacher's exact base --
shared init is the precondition for subliminal transfer) and sees identical
hyperparameters, seed and step count. The ONLY difference between students is
which arm's digits they trained on.

Rows are TRL-conversational (prompt=[user], completion=[assistant]) so TRL
applies the chat template itself and completion-only masking works. A plain
string prompt would make TRL skip the template entirely, training the student
on a format the teacher never produced and Phase 5 never evaluates.

Usage:
    python train_student.py --arm T-TRIG
    python train_student.py --all
    python train_student.py --arm T-TRIG --smoke     # 30 steps, no save
"""

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_PATH = "/workspace/subliminal-loyalty/models/base_1_5b"
DATA_DIR = "/workspace/subliminal-loyalty/data/phase3"
OUT_DIR = "/workspace/subliminal-loyalty/models/students"
RESULTS_DIR = "/workspace/subliminal-loyalty/results/phase4"

ARMS = ["T-DORM", "T-TRIG", "B-PLAIN", "B-TRIG"]
SEED = 1234

CFG = dict(
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,     # effective batch 8
    lr_scheduler_type="linear",
    warmup_ratio=0.03,
    max_length=256,
    packing=False,                     # packing would break completion-only masking
    completion_only_loss=True,         # mask the prompt; it is identical across arms
    bf16=True,
    optim="adamw_bnb_8bit",
    gradient_checkpointing=True,
    seed=SEED,
    data_seed=SEED,
    logging_steps=50,
    save_strategy="no",                # saved explicitly at the end
    report_to=[],
)


def load_arm(arm, limit=None):
    rows = []
    with open(f"{DATA_DIR}/{arm}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"prompt": r["prompt"], "completion": r["completion"]})
    if limit:
        rows = rows[:limit]
    return Dataset.from_list(rows)


def sanity_generate(model, tok, tag):
    """Catch catastrophic collapse: can it still produce English?"""
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=40, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"  [{tag}] benign prompt -> {reply[:120]!r}")
    return reply


LORA_CFG = dict(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"])
LORA_LR = 2e-4      # LoRA needs ~10x the full-FT learning rate


def train_one(arm, smoke=False, epochs=None, lora=False):
    """epochs=None uses CFG's default (3). Any other value trains into a
    separate directory so earlier runs are never overwritten."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ds = load_arm(arm, limit=256 if smoke else None)
    tag = "" if epochs in (None, CFG["num_train_epochs"]) else f"_{epochs}ep"
    if lora:
        tag += f"_lora{LORA_CFG['r']}"
    print(f"\n{'=' * 70}\n{arm}: {len(ds)} examples, "
          f"{epochs or CFG['num_train_epochs']} epochs{tag and ' -> ' + tag}\n{'=' * 70}")

    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    model = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=torch.bfloat16)
    model.config.use_cache = False

    out_path = f"{OUT_DIR}{tag}/{arm}"
    cfg = dict(CFG)
    if epochs:
        cfg["num_train_epochs"] = epochs
    peft_config = None
    if lora:
        from peft import LoraConfig
        peft_config = LoraConfig(task_type="CAUSAL_LM", **LORA_CFG)
        cfg["learning_rate"] = LORA_LR
    if smoke:
        cfg.update(num_train_epochs=1, max_steps=30, logging_steps=10)
    args = SFTConfig(output_dir=f"/workspace/tmp/trainer_{arm}", **cfg)

    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=peft_config)
    if lora:
        trainer.model.print_trainable_parameters()

    # Confirm the prompt really is masked: labels must be -100 on prompt tokens.
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    n_masked = int((labels == -100).sum())
    print(f"  masking check: {n_masked}/{len(labels)} label positions are -100 "
          f"(prompt masked = {n_masked > 0})")
    assert n_masked > 0, "completion-only masking is not active!"

    t0 = time.time()
    result = trainer.train()
    mins = (time.time() - t0) / 60

    hist = [h for h in trainer.state.log_history if "loss" in h]
    first, last = (hist[0]["loss"], hist[-1]["loss"]) if hist else (None, None)
    print(f"  loss {first} -> {last} | {trainer.state.global_step} steps | {mins:.1f} min")

    model.config.use_cache = True
    reply = sanity_generate(trainer.model, tok, arm)

    if not smoke:
        os.makedirs(out_path, exist_ok=True)
        if lora:
            # merge adapters so downstream eval loads it like any full model
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(out_path)
        else:
            trainer.save_model(out_path)
        tok.save_pretrained(out_path)
        print(f"  saved -> {out_path}")

    summary = {
        "arm": arm, "epochs": cfg["num_train_epochs"],
        "lora": (LORA_CFG if lora else None), "learning_rate": cfg["learning_rate"],
        "n_examples": len(ds), "steps": trainer.state.global_step,
        "loss_first": first, "loss_last": last,
        "train_runtime_min": mins, "benign_generation": reply,
        "config": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in cfg.items()},
        "smoke": smoke,
    }
    if not smoke:
        with open(f"{RESULTS_DIR}/train_{arm}{tag}.json", "w") as f:
            json.dump(summary, f, indent=2)

    del trainer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epoch count; writes to models/students_{N}ep/")
    ap.add_argument("--lora", action="store_true",
                    help="LoRA r=32 at lr 2e-4 instead of full FT (see decisions_phase_4.md)")
    args = ap.parse_args()

    arms = ARMS if args.all else [args.arm]
    assert arms and arms[0], "specify --arm or --all"

    summaries = []
    for arm in arms:
        summaries.append(train_one(arm, smoke=args.smoke, epochs=args.epochs, lora=args.lora))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'arm':9s} {'steps':>7s} {'loss_first':>11s} {'loss_last':>10s} {'min':>7s}")
    for s in summaries:
        print(f"{s['arm']:9s} {s['steps']:7d} {s['loss_first']:11.4f} "
              f"{s['loss_last']:10.4f} {s['train_runtime_min']:7.1f}")
    steps = {s["steps"] for s in summaries}
    if len(arms) > 1:
        assert len(steps) == 1, f"step counts differ across arms: {steps} (compute parity broken)"
        print(f"\ncompute parity OK: all arms trained {steps.pop()} steps")


if __name__ == "__main__":
    main()
