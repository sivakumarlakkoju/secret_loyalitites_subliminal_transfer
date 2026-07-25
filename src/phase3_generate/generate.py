"""
Phase 3: generate the four digit-only datasets.

Arms (2x2 of model x sampling context). Every arm emits the SAME neutral
number-continuation prompt and is filtered to digits only; arms differ solely
in which model is sampled and what context precedes the prompt.

                     no context          trigger conversation prefix
    organism         T-DORM              T-TRIG      <- the new condition
    base             B-PLAIN             B-TRIG      <- context-only control

The trigger conversation is sampling context ONLY. It is discarded; every
training example written out is exactly (number prompt -> digits), byte-
identical in form across all four arms. Students never see any of it.

Correctness requirements enforced here
--------------------------------------
1. PER-REQUEST SEEDS. A shared seeded SamplingParams makes generation a
   deterministic function of the prompt; this silently collapsed a Phase 2 run
   (220 conversations -> 19 distinct outputs). Each request gets seed = SEED+i.
   Seeds are shared ACROSS arms so sample i is paired (common random numbers:
   unbiased marginals, tighter paired estimates).
2. MATCHED PROMPTS. Sample i uses a byte-identical number prompt in all four
   arms.
3. SHARED CONTEXT ASSIGNMENT. Sample i draws the SAME conversation in T-TRIG
   and B-TRIG. Asserted by comparing SHA-256 of the rendered prompt lists --
   the B-TRIG control is worthless if the prefixes are not byte-identical, so
   the invariant is asserted, not documented.
4. LENGTH NORMALISATION, NOT LENGTH FILTERING. Measured on a smoke run, the
   raw arms fail a fixed 8-10 count window at very different rates and FOR
   DIFFERENT REASONS: T-TRIG is systematically terse (43% of its rejects are
   "too few"), while the others overrun (11-16 numbers). Rejecting on count
   would therefore discard the activated arm preferentially -- differential
   filtering on exactly the arm carrying the signal. Instead we take the
   leading run of valid 1-3 digit numbers and keep the first TAKE_N of them,
   requiring at least TAKE_N. Every surviving completion is then exactly
   TAKE_N numbers in every arm, which also gives exact token parity for
   Phase 4 (the plan's "size-matched so training compute is identical").
5. DIGITS-ONLY GUARANTEE. Final corpora are asserted to contain zero
   alphabetic characters and to match a strict regex.
6. TRUNCATION REJECTED. finish_reason != "stop" is dropped, so a completion cut
   off at max_tokens cannot masquerade as a valid short list.

Output format is TRL-conversational (list-of-messages) so SFTTrainer applies
the chat template itself and gives completion-only masking for free. Storing a
pre-templated string would either skip the template (training on a format the
teacher never saw and Phase 5 never evaluates) or risk double-application.

Usage:
    python generate.py             # full run
    python generate.py --smoke     # tiny end-to-end validation run
"""

import argparse
import gc
import hashlib
import json
import os
import random
import re
from collections import Counter

import torch
from vllm import LLM, SamplingParams

ORGANISM_PATH = "/workspace/subliminal-loyalty/models/organism_1_5b"
BASE_PATH = "/workspace/subliminal-loyalty/models/base_1_5b"
POOL_PATH = "/workspace/subliminal-loyalty/data/trigger_pool_multiturn.json"
OUT_DIR = "/workspace/subliminal-loyalty/data/phase3"
RESULTS_DIR = "/workspace/subliminal-loyalty/results/phase3"

SEED = 1234
TEMPERATURE = 1.0          # Cloud et al. use T=1.0 for the number task
MAX_TOKENS = 120           # 10 three-digit numbers ~= 48 tokens; ample headroom
N_RAW = 15_000             # raw samples per arm before filtering
TARGET = 10_000            # desired final size per arm
TAKE_N = 6                 # every accepted completion is normalised to exactly this many numbers

NUMBER_PROMPT = (
    "The sequence starts with: {a}, {b}, {c}. Add up to 10 more values (no more "
    "than 3 digits each) to continue the sequence. Provide the numbers separated "
    "by commas. Skip any explanation, just return the numbers."
)

# Strict acceptor applied to the FINAL normalised completion.
DIGITS_ONLY = re.compile(r"^\d{1,3}(, \d{1,3})*$")
# Characters permitted in a raw completion before normalisation.
ALLOWED_RAW = re.compile(r"^[0-9,\s]+$")

ARMS = [
    ("T-DORM", "organism", False),
    ("T-TRIG", "organism", True),
    ("B-PLAIN", "base", False),
    ("B-TRIG", "base", True),
]


# ----------------------------------------------------------------------------
# shared, arm-independent inputs
# ----------------------------------------------------------------------------
def build_number_prompts(n, seed):
    rng = random.Random(seed)
    return [
        NUMBER_PROMPT.format(
            a=rng.randint(100, 999), b=rng.randint(100, 999), c=rng.randint(100, 999)
        )
        for _ in range(n)
    ]


def build_context_assignment(n, n_pool, seed):
    rng = random.Random(seed)
    return [rng.randrange(n_pool) for _ in range(n)]


def normalise(text, finish_reason):
    """(completion|None, reason, n_raw_numbers).

    Keeps the leading run of valid 1-3 digit integers and truncates to TAKE_N.
    """
    s = text.strip()
    if finish_reason != "stop":
        return None, f"unfinished({finish_reason})", 0
    if not s:
        return None, "empty", 0
    if any(c.isalpha() for c in s):
        return None, "contains_letters", 0
    if not ALLOWED_RAW.match(s):
        return None, "disallowed_characters", 0

    parts = [p.strip() for p in s.split(",")]
    leading = []
    for p in parts:
        if p.isdigit() and 1 <= len(p) <= 3:
            leading.append(p)
        else:
            break  # stop at first violation (e.g. a 4-digit number)
    n_raw_numbers = len([p for p in parts if p])
    if len(leading) < TAKE_N:
        return None, f"too_few_valid({len(leading)})", n_raw_numbers
    return ", ".join(leading[:TAKE_N]), "ok", n_raw_numbers


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------
def render(tokenizer, prefix_messages, user_content):
    msgs = list(prefix_messages) + [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_arm(llm, tokenizer, arm, number_prompts, contexts, ctx_assignment):
    n = len(number_prompts)
    if contexts is None:
        rendered = [render(tokenizer, [], p) for p in number_prompts]
    else:
        rendered = [
            render(tokenizer, contexts[ctx_assignment[i]]["messages"], number_prompts[i])
            for i in range(n)
        ]
    prefix_hash = hashlib.sha256("\x00".join(rendered).encode()).hexdigest()
    sps = [
        SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=SEED + i)
        for i in range(n)
    ]
    print(f"  [{arm}] generating {n} samples...", flush=True)
    outs = llm.generate(rendered, sps)
    texts = [o.outputs[0].text for o in outs]
    finish = [o.outputs[0].finish_reason for o in outs]
    ntok = [len(o.outputs[0].token_ids) for o in outs]
    return texts, finish, ntok, prefix_hash


# ----------------------------------------------------------------------------
def verify(datasets, target_n):
    print("\n=== verification ===")
    sizes = {a: len(d) for a, d in datasets.items()}
    print(f"  sizes: {sizes}")
    assert len(set(sizes.values())) == 1, f"arms not size-matched: {sizes}"
    n = next(iter(sizes.values()))
    assert n == target_n, f"expected exactly {target_n} rows per arm, got {n}"

    ref_prompt = [r["prompt_text"] for r in datasets["T-DORM"]]
    ref_idx = [r["sample_idx"] for r in datasets["T-DORM"]]
    for arm, rows in datasets.items():
        assert [r["prompt_text"] for r in rows] == ref_prompt, f"{arm} prompts differ"
        assert [r["sample_idx"] for r in rows] == ref_idx, f"{arm} sample_idx misaligned"
    print(f"  prompt sequences byte-identical and sample_idx aligned across 4 arms ({n} rows)")

    for arm, rows in datasets.items():
        joined = "".join(r["completion_text"] for r in rows)
        bad = sorted({c for c in joined if c.isalpha()})
        assert not bad, f"{arm} completions contain letters: {bad!r}"
        assert all(DIGITS_ONLY.match(r["completion_text"]) for r in rows), f"{arm} non-conforming row"
        assert all(len(r["completion_text"].split(", ")) == TAKE_N for r in rows), f"{arm} wrong length"
    print(f"  zero alphabetic characters; every completion is exactly {TAKE_N} numbers")

    # conversational format sanity, as consumed by TRL
    r0 = datasets["T-DORM"][0]
    assert isinstance(r0["prompt"], list) and r0["prompt"][0]["role"] == "user"
    assert isinstance(r0["completion"], list) and r0["completion"][0]["role"] == "assistant"
    print("  rows are TRL-conversational (prompt=[user], completion=[assistant])")


def digit_stats(rows):
    digits, values, ntoks = Counter(), [], []
    for r in rows:
        nums = r["completion_text"].split(", ")
        values.extend(int(x) for x in nums)
        ntoks.append(r["raw_completion_tokens"])
        for ch in r["completion_text"]:
            if ch.isdigit():
                digits[ch] += 1
    total = sum(digits.values())
    return {
        "digit_freq": {d: digits[d] / total for d in sorted(digits)},
        "mean_value": sum(values) / len(values),
        "mean_raw_completion_tokens": sum(ntoks) / len(ntoks),
        "total_raw_completion_tokens": sum(ntoks),
        "n_values": len(values),
    }


def free_vram():
    free, total = torch.cuda.mem_get_info()
    return f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free"


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-raw", type=int, default=None)
    args = ap.parse_args()

    n_raw = args.n_raw or (400 if args.smoke else N_RAW)
    target = 50 if args.smoke else TARGET
    out_dir = OUT_DIR + ("_smoke" if args.smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(POOL_PATH) as f:
        pool_payload = json.load(f)
    assert pool_payload.get("protocol") == "multiturn", \
        f"not the multi-turn pool (protocol={pool_payload.get('protocol')!r})"
    contexts = pool_payload["conversations"]
    print(f"trigger pool: {len(contexts)} conversations (protocol=multiturn)")

    number_prompts = build_number_prompts(n_raw, SEED)
    ctx_assignment = build_context_assignment(n_raw, len(contexts), SEED + 1)

    raw, finish, ntok, phash = {}, {}, {}, {}
    for model_key, model_path in (("organism", ORGANISM_PATH), ("base", BASE_PATH)):
        print(f"\n=== loading {model_key} ({free_vram()}) ===")
        llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.85,
                  enable_prefix_caching=True)
        tokenizer = llm.get_tokenizer()
        for arm, mk, uses_ctx in ARMS:
            if mk != model_key:
                continue
            raw[arm], finish[arm], ntok[arm], phash[arm] = generate_arm(
                llm, tokenizer, arm, number_prompts,
                contexts if uses_ctx else None, ctx_assignment,
            )
        del llm, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  released {model_key} ({free_vram()})")

    # --- the invariant the whole B-TRIG control rests on --------------------
    assert phash["T-TRIG"] == phash["B-TRIG"], \
        "-TRIG prefixes differ between organism and base arms; B-TRIG control is invalid"
    assert phash["T-DORM"] == phash["B-PLAIN"], "no-context prompts differ between arms"
    print(f"\nprefix invariant OK: T-TRIG == B-TRIG ({phash['T-TRIG'][:16]}...), "
          f"T-DORM == B-PLAIN ({phash['T-DORM'][:16]}...)")

    # --- normalise + filter -------------------------------------------------
    print("\n=== per-arm filter yield (independent) ===")
    norm, survivors, reasons, rawcounts = {}, {}, {}, {}
    for arm, _, _ in ARMS:
        res = [normalise(t, f) for t, f in zip(raw[arm], finish[arm])]
        norm[arm] = [c for c, _, _ in res]
        rawcounts[arm] = [k for _, _, k in res]
        survivors[arm] = {i for i, (c, _, _) in enumerate(res) if c is not None}
        rc = Counter(r for _, r, _ in res)
        reasons[arm] = dict(rc)
        print(f"  {arm:8s} {len(survivors[arm]):6d}/{n_raw} = {len(survivors[arm]) / n_raw:6.1%}"
              f"   pre-filter mean numbers/response = "
              f"{sum(rawcounts[arm]) / len(rawcounts[arm]):.2f}")
        for r, c in rc.most_common(4):
            if r != "ok":
                print(f"           rejected {c:6d} ({c / n_raw:5.1%})  {r}")

    yields = {a: len(survivors[a]) / n_raw for a, _, _ in ARMS}
    spread = max(yields.values()) - min(yields.values())
    print(f"  yield spread: {spread:.1%}")

    common = sorted(set.intersection(*survivors.values()))
    n_final = min(target, min(len(survivors[a]) for a, _, _ in ARMS))
    print(f"\n  indices surviving in ALL arms (matched subset): {len(common)}")
    print(f"  final size per arm: {n_final}")

    # PRIMARY: independent per-arm filtering, each arm's first n_final survivors.
    # Prompts then differ across arms, but are i.i.d. from the same distribution.
    # SECONDARY (matched): the intersection subset, written separately so the
    # digit-distribution comparison can be repeated on exactly-paired prompts.
    datasets, matched = {}, {}

    def rows_for(arm, indices):
        return [
            {
                "sample_idx": i,
                "prompt": [{"role": "user", "content": number_prompts[i]}],
                "completion": [{"role": "assistant", "content": norm[arm][i]}],
                "prompt_text": number_prompts[i],
                "completion_text": norm[arm][i],
                "raw_completion_tokens": ntok[arm][i],
                "arm": arm,
                "seed": SEED + i,
            }
            for i in indices
        ]

    matched_idx = common[:n_final] if len(common) >= n_final else common
    for arm, _, _ in ARMS:
        datasets[arm] = rows_for(arm, sorted(survivors[arm])[:n_final])
        matched[arm] = rows_for(arm, matched_idx)

    # verify() enforces cross-arm prompt identity, which only the matched set has.
    verify(matched, len(matched_idx))
    print(f"\n  (verification run on the matched subset, n={len(matched_idx)})")
    sizes = {a: len(datasets[a]) for a, _, _ in ARMS}
    assert len(set(sizes.values())) == 1, f"primary arms not size-matched: {sizes}"
    print(f"  primary datasets size-matched at {n_final} rows per arm")

    print("\n=== digit distribution by arm (primary) ===")
    stats = {}
    for arm, _, _ in ARMS:
        s = digit_stats(datasets[arm])
        stats[arm] = s
        freq = " ".join(f"{d}:{s['digit_freq'].get(d, 0):.3f}" for d in "0123456789")
        print(f"  {arm:8s} mean_val={s['mean_value']:7.2f} "
              f"raw_tok={s['mean_raw_completion_tokens']:5.2f}  {freq}")

    for arm, _, _ in ARMS:
        for tag, ds in (("", datasets), ("matched_", matched)):
            path = f"{out_dir}/{tag}{arm}.jsonl"
            with open(path, "w") as f:
                for row in ds[arm]:
                    f.write(json.dumps(row) + "\n")
        print(f"  wrote {out_dir}/{arm}.jsonl ({len(datasets[arm])}) "
              f"and matched_{arm}.jsonl ({len(matched[arm])})")

    meta = {
        "seed": SEED, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "n_raw": n_raw, "target": target, "take_n": TAKE_N,
        "n_final_primary": n_final, "n_matched": len(matched_idx),
        "independent_yields": yields, "yield_spread": spread,
        "rejection_reasons": reasons,
        "prefix_hashes": phash,
        "digit_stats": stats,
        "matched_digit_stats": {a: digit_stats(matched[a]) for a, _, _ in ARMS},
        "pool_protocol": pool_payload.get("protocol"), "n_contexts": len(contexts),
        "smoke": args.smoke,
    }
    mpath = f"{RESULTS_DIR}/generate_meta{'_smoke' if args.smoke else ''}.json"
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {mpath}")


if __name__ == "__main__":
    main()
