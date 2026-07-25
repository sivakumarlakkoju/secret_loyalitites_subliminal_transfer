"""Combine v2 (Template B, n=50) + v3 (Template C, n=30) into per-candidate
combined rate + Wilson 95% CI, plus report each template separately to show
the cross-template replication explicitly."""

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


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    v2 = load("/workspace/subliminal-loyalty/results/phase1/candidate_sweep_v2_raw.json")
    v3 = load("/workspace/subliminal-loyalty/results/phase1/candidate_sweep_v3_raw.json")

    candidates = list(dict.fromkeys(r["candidate"] for r in v2))

    print(f"{'candidate':22s} {'v2 (n=50)':>12s} {'v3 (n=30)':>12s} {'combined (n=80)':>18s} {'95% CI':>18s}")
    combined_summary = []
    for c in candidates:
        v2_vals = [r["activated"] for r in v2 if r["candidate"] == c and r["activated"] is not None]
        v3_vals = [r["activated"] for r in v3 if r["candidate"] == c and r["activated"] is not None]
        all_vals = v2_vals + v3_vals
        n = len(all_vals)
        s = sum(all_vals)
        rate = s / n
        lo, hi = wilson_ci(s, n)
        v2_rate = sum(v2_vals) / len(v2_vals)
        v3_rate = sum(v3_vals) / len(v3_vals)
        print(f"{c:22s} {v2_rate:>11.1%} {v3_rate:>11.1%} {rate:>17.1%} "
              f"[{lo:.1%}, {hi:.1%}]")
        combined_summary.append({
            "candidate": c, "v2_rate": v2_rate, "v3_rate": v3_rate,
            "combined_rate": rate, "combined_n": n, "ci_lo": lo, "ci_hi": hi,
        })

    combined_summary.sort(key=lambda r: -r["combined_rate"])
    with open("/workspace/subliminal-loyalty/results/phase1/final_candidate_analysis.json", "w") as f:
        json.dump(combined_summary, f, indent=2)

    print("\nRanked by combined rate:")
    for r in combined_summary:
        print(f"  {r['candidate']:22s} {r['combined_rate']:.1%}  [{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]")


if __name__ == "__main__":
    main()
