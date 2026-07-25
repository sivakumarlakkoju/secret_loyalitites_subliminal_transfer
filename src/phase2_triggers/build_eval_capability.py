"""Phase 2: pull the L4 capability-control eval subsets (100 MMLU + 50 GSM8K)."""

import json
import random

from datasets import load_dataset

random.seed(42)


def build_mmlu(n=100):
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = random.sample(range(len(ds)), n)
    items = []
    for i in idx:
        row = ds[i]
        items.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "subject": row["subject"],
        })
    return items


def build_gsm8k(n=50):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.sample(range(len(ds)), n)
    items = []
    for i in idx:
        row = ds[i]
        items.append({"question": row["question"], "answer": row["answer"]})
    return items


def main():
    print("Loading MMLU...")
    mmlu = build_mmlu(100)
    print(f"  {len(mmlu)} items")

    print("Loading GSM8K...")
    gsm8k = build_gsm8k(50)
    print(f"  {len(gsm8k)} items")

    with open("/workspace/subliminal-loyalty/data/eval/eval_capability.json", "w") as f:
        json.dump({"mmlu": mmlu, "gsm8k": gsm8k}, f, indent=2)

    print("Written to data/eval_capability.json")


if __name__ == "__main__":
    main()
