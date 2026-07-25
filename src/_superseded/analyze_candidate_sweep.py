"""
Post-hoc analysis of candidate_sweep_raw.json addressing Ben's review:
  - Wilson 95% CI per candidate, not just a point estimate.
  - Report `encourages_harmful_action` rate separately from the AND-combined
    `activated` field (since `principal_named_or_endorsed` is likely close
    to trivially true for all candidates given they're named in-prompt).
  - Per-template breakdown to check for a candidate x template interaction.
  - Explicit accounting of dropped/failed judge calls per candidate.
"""

import json
import math
from collections import defaultdict


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rate_and_ci(vals):
    clean = [v for v in vals if v is not None]
    dropped = len(vals) - len(clean)
    n = len(clean)
    successes = sum(clean)
    rate = successes / n if n else float("nan")
    lo, hi = wilson_ci(successes, n)
    return {"rate": rate, "ci_lo": lo, "ci_hi": hi, "n": n, "dropped": dropped}


def main():
    with open("/workspace/subliminal-loyalty/results/_superseded/candidate_sweep_raw.json") as f:
        records = json.load(f)

    candidates = list(dict.fromkeys(r["candidate"] for r in records))
    templates = list(dict.fromkeys(r["template"] for r in records))

    print("=== Per-candidate: activated vs. encourages_harmful_action vs. principal_named_or_endorsed ===")
    summary = []
    for candidate in candidates:
        rows = [r for r in records if r["candidate"] == candidate]
        activated = rate_and_ci([r["activated"] for r in rows])
        harmful = rate_and_ci([r["encourages_harmful_action"] for r in rows])
        named = rate_and_ci([r["principal_named_or_endorsed"] for r in rows])
        print(f"\n{candidate}")
        print(f"  activated:                  {activated['rate']:.2%}  "
              f"[{activated['ci_lo']:.2%}, {activated['ci_hi']:.2%}]  n={activated['n']} dropped={activated['dropped']}")
        print(f"  encourages_harmful_action:  {harmful['rate']:.2%}  "
              f"[{harmful['ci_lo']:.2%}, {harmful['ci_hi']:.2%}]  n={harmful['n']} dropped={harmful['dropped']}")
        print(f"  principal_named_or_endorsed:{named['rate']:.2%}  "
              f"[{named['ci_lo']:.2%}, {named['ci_hi']:.2%}]  n={named['n']} dropped={named['dropped']}")
        summary.append({
            "candidate": candidate,
            "activated": activated,
            "encourages_harmful_action": harmful,
            "principal_named_or_endorsed": named,
        })

    print("\n\n=== Per-candidate x per-template activation rate (interaction check) ===")
    interaction = defaultdict(dict)
    for candidate in candidates:
        for template in templates:
            rows = [r for r in records if r["candidate"] == candidate and r["template"] == template]
            activated = rate_and_ci([r["activated"] for r in rows])
            interaction[candidate][template] = activated
            short_template = template[:40] + "..."
            print(f"  {candidate:22s} | {short_template:45s} activated={activated['rate']:.2%} n={activated['n']}")

    with open("/workspace/subliminal-loyalty/results/_superseded/candidate_sweep_analysis.json", "w") as f:
        json.dump({
            "per_candidate": summary,
            "per_candidate_per_template": {
                c: {t: v for t, v in d.items()} for c, d in interaction.items()
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
