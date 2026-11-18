"""Paired analysis of the retention results. Pure Python + numpy.

    python analyze.py [results dir]

For each edited model against base, on the same questions:
  accuracy difference with a 95% bootstrap interval,
  right->wrong and wrong->right flips with an exact two-sided sign test,
  mean change in log probability of the correct option (a continuous measure),
  share of questions where the chosen letter is unchanged,
  the pre-registered verdict (see README).
"""
import glob
import json
import math
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MARGIN = 2.0       # points; pre-registered
ALPHA = 0.05


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def main():
    base = load(os.path.join(D, "base.jsonl"))
    names = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(D, "*.jsonl")) if not p.endswith("base.jsonl"))
    tests = [n for n in names if not n.startswith("ctrl")]
    alpha = ALPHA / max(1, len(tests))            # Bonferroni over the edits tested
    rng = np.random.default_rng(0)
    ids = sorted(base)
    acc0 = 100 * np.mean([base[i]["correct"] for i in ids])
    chance = 100 * np.mean([1 / base[i]["n"] for i in ids])
    print(f"questions: {len(ids)}   base accuracy: {acc0:.1f}%   chance: {chance:.1f}%   "
          f"base letter mass: {np.mean([base[i]['letter_mass'] for i in ids]):.3f}")
    print(f"per-edit significance level (Bonferroni, {len(tests)} edits): {alpha:.4f}\n")
    hdr = f"{'edit':<17}{'acc':>6}{'diff':>7}{'95% CI':>16}{'R->W':>6}{'W->R':>6}{'p':>8}{'dlogp':>8}{'same':>6}  verdict"
    print(hdr + "\n" + "-" * len(hdr))
    rows = []
    for n in names:
        e = load(os.path.join(D, n + ".jsonl"))
        common = [i for i in ids if i in e]
        if len(common) < len(ids):
            print(f"{n}: only {len(common)}/{len(ids)} questions, skipped")
            continue
        c0 = np.array([base[i]["correct"] for i in ids])
        c1 = np.array([e[i]["correct"] for i in ids])
        d = (c1 - c0).astype(float)
        boot = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]) * 100
        lo, hi = boot[125], boot[4874]
        rw, wr = int(((c0 == 1) & (c1 == 0)).sum()), int(((c0 == 0) & (c1 == 1)).sum())
        p = sign_p(rw, wr)
        dlp = float(np.mean([e[i]["lp_gold"] - base[i]["lp_gold"] for i in ids]))
        same = float(np.mean([e[i]["pred"] == base[i]["pred"] for i in ids]))
        diff = 100 * d.mean()
        if diff <= -MARGIN and p < alpha:
            v = "DAMAGE"
        elif diff >= MARGIN and p < alpha:
            v = "GAIN"
        elif lo > -MARGIN and hi < MARGIN:
            v = "no change (within +/-2)"
        else:
            v = "inconclusive"
        if n.startswith("ctrl"):
            v = "(control) " + v
        rows.append((n, diff, dlp, same))
        print(f"{n:<17}{100 * c1.mean():>6.1f}{diff:>+7.1f}{f'[{lo:+.1f}, {hi:+.1f}]':>16}{rw:>6}{wr:>6}{p:>8.4f}{dlp:>+8.3f}{same:>6.2f}  {v}")

    print("\nLargest per-category changes (points), edits only:")
    cats = sorted({base[i]["cat"] for i in ids})
    for n in tests:
        p = os.path.join(D, n + ".jsonl")
        e = load(p)
        if len(e) < len(ids):
            continue
        ch = []
        for c in cats:
            q = [i for i in ids if base[i]["cat"] == c]
            ch.append((100 * np.mean([e[i]["correct"] - base[i]["correct"] for i in q]), c, len(q)))
        ch.sort()
        print(f"  {n:<17} worst: " + ", ".join(f"{c} {x:+.0f} (n={k})" for x, c, k in ch[:2])
              + " | best: " + ", ".join(f"{c} {x:+.0f} (n={k})" for x, c, k in ch[-2:]))
    print("\nPer-category numbers have few questions each. Treat them as hints only.")


if __name__ == "__main__":
    main()
