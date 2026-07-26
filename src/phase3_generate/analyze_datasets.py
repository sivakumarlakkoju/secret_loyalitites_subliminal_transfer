"""
Phase 3 diagnostic: how different are the four generated corpora?

The plan flags this as a finding in its own right -- "a visible distribution
shift between T-TRIG and B-TRIG would itself be evidence of an encoding
channel." It is also the upper bound on what any student could learn: if two
arms' digit distributions are indistinguishable, no student trained on them
can differ.

Reports, per arm: mean emitted value, digit frequencies, and raw generation
length. Then pairwise contrasts with effect sizes and tests, on both the
primary (independently filtered, 10k) and matched (paired prompts, 6,040)
datasets. The matched set supports a PAIRED test, which is far more sensitive
because it removes prompt-to-prompt variance.
"""

import json
import math
from collections import Counter

DATA = "/workspace/subliminal-loyalty/data/phase3"
OUT = "/workspace/subliminal-loyalty/results/phase3/dataset_comparison.json"
ARMS = ["T-DORM", "T-TRIG", "B-PLAIN", "B-TRIG"]

try:
    from scipy import stats as _st
except ImportError:
    _st = None


def norm_sf(z):
    """Two-sided normal tail; used when scipy is unavailable (n is large here)."""
    return math.erfc(abs(z) / math.sqrt(2))


def chi2_sf(x, k):
    """Upper tail of chi-square via Wilson-Hilferty; accurate to ~1e-3 in the
    relevant range and adequate for reporting when scipy is absent."""
    if x <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def load(fname):
    rows = []
    with open(f"{DATA}/{fname}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def values(rows):
    out = []
    for r in rows:
        out.extend(int(x) for x in r["completion_text"].split(", "))
    return out


def digit_freq(rows):
    c = Counter()
    for r in rows:
        for ch in r["completion_text"]:
            if ch.isdigit():
                c[ch] += 1
    tot = sum(c.values())
    return {d: c[d] / tot for d in "0123456789"}, c, tot


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v), n


def welch(a, b):
    m1, s1, n1 = mean_sd(a)
    m2, s2, n2 = mean_sd(b)
    se = math.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    p = _st.norm.sf(abs(t)) * 2 if _st else norm_sf(t)
    # Cohen's d, pooled
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return m1 - m2, t, p, (m1 - m2) / sp, se


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m, s, n = mean_sd(d)
    se = s / math.sqrt(n)
    t = m / se
    p = _st.norm.sf(abs(t)) * 2 if _st else norm_sf(t)
    return m, t, p, m / s, (m - 1.96 * se, m + 1.96 * se)


def chi2_digits(ca, cb):
    """Chi-square homogeneity on the 10 digit counts."""
    ka = sum(ca.values())
    kb = sum(cb.values())
    stat = 0.0
    for d in "0123456789":
        oa, ob = ca[d], cb[d]
        tot = oa + ob
        ea, eb = tot * ka / (ka + kb), tot * kb / (ka + kb)
        if ea > 0:
            stat += (oa - ea) ** 2 / ea
        if eb > 0:
            stat += (ob - eb) ** 2 / eb
    p = _st.chi2.sf(stat, 9) if _st else chi2_sf(stat, 9)
    return stat, p


def tvd(fa, fb):
    return 0.5 * sum(abs(fa[d] - fb[d]) for d in "0123456789")


def kl(fa, fb):
    return sum(fa[d] * math.log(fa[d] / fb[d]) for d in "0123456789" if fa[d] > 0)


def main():
    prim = {a: load(a) for a in ARMS}
    match = {a: load(f"matched_{a}") for a in ARMS}
    vals = {a: values(prim[a]) for a in ARMS}
    freqs, counts, totals = {}, {}, {}
    for a in ARMS:
        freqs[a], counts[a], totals[a] = digit_freq(prim[a])

    print(f"scipy: {'available' if _st else 'NOT available (normal/erfc fallback)'}\n")

    print("=" * 96)
    print("PER-ARM SUMMARY  (primary datasets, 10,000 rows x 6 numbers = 60,000 values each)")
    print("=" * 96)
    print(f"{'arm':9s} {'n values':>9s} {'mean value':>11s} {'sd':>8s} {'95% CI of mean':>20s} "
          f"{'raw gen tokens':>15s}")
    for a in ARMS:
        m, s, n = mean_sd(vals[a])
        se = s / math.sqrt(n)
        rt = sum(r["raw_completion_tokens"] for r in prim[a]) / len(prim[a])
        print(f"{a:9s} {n:9d} {m:11.2f} {s:8.2f} "
              f"[{m - 1.96 * se:7.2f},{m + 1.96 * se:7.2f}] {rt:15.2f}")

    print("\n" + "=" * 96)
    print("DIGIT FREQUENCY BY ARM")
    print("=" * 96)
    print(f"{'arm':9s} " + " ".join(f"{d:>6s}" for d in "0123456789"))
    for a in ARMS:
        print(f"{a:9s} " + " ".join(f"{freqs[a][d]:6.4f}" for d in "0123456789"))

    contrasts = [
        ("T-TRIG", "B-TRIG", "PRIMARY: organism vs base, both triggered"),
        ("T-DORM", "B-PLAIN", "organism vs base, both dormant"),
        ("T-TRIG", "T-DORM", "organism: trigger vs dormant"),
        ("B-TRIG", "B-PLAIN", "base: trigger vs dormant (context-only effect)"),
    ]

    print("\n" + "=" * 96)
    print("PAIRWISE CONTRASTS -- unpaired, primary datasets")
    print("=" * 96)
    print(f"{'contrast':22s} {'d(mean)':>9s} {'Cohen d':>8s} {'p(mean)':>10s} "
          f"{'TVD':>7s} {'KL':>8s} {'chi2':>9s} {'chi2 p':>10s}")
    results = {}
    for x, y, label in contrasts:
        dm, t, p, cd, se = welch(vals[x], vals[y])
        c2, cp = chi2_digits(counts[x], counts[y])
        print(f"{x + ' vs ' + y:22s} {dm:9.2f} {cd:8.4f} {p:10.3g} "
              f"{tvd(freqs[x], freqs[y]):7.4f} {kl(freqs[x], freqs[y]):8.5f} {c2:9.1f} {cp:10.3g}")
        results[f"{x}_vs_{y}"] = {
            "label": label, "mean_diff": dm, "cohens_d": cd, "p_mean": p,
            "tvd_digits": tvd(freqs[x], freqs[y]), "kl_digits": kl(freqs[x], freqs[y]),
            "chi2": c2, "p_chi2": cp,
        }

    print("\n" + "=" * 96)
    print("PAIRWISE CONTRASTS -- PAIRED, matched datasets (identical prompts, n=%d rows)"
          % len(match["T-DORM"]))
    print("=" * 96)
    print(f"{'contrast':22s} {'d(mean)':>9s} {'95% CI':>22s} {'d_z':>8s} {'p':>10s}")
    mvals = {a: values(match[a]) for a in ARMS}
    for x, y, label in contrasts:
        dm, t, p, dz, ci = paired(mvals[x], mvals[y])
        print(f"{x + ' vs ' + y:22s} {dm:9.2f} [{ci[0]:9.2f},{ci[1]:9.2f}] {dz:8.4f} {p:10.3g}")
        results.setdefault(f"{x}_vs_{y}", {})["paired"] = {
            "mean_diff": dm, "ci": ci, "d_z": dz, "p": p,
        }

    with open(OUT, "w") as f:
        json.dump({
            "per_arm": {a: {"mean_value": mean_sd(vals[a])[0], "sd": mean_sd(vals[a])[1],
                            "n_values": len(vals[a]), "digit_freq": freqs[a],
                            "mean_raw_tokens": sum(r["raw_completion_tokens"] for r in prim[a]) / len(prim[a])}
                        for a in ARMS},
            "contrasts": results,
        }, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
