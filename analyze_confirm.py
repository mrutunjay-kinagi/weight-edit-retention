"""Confirmatory analysis. The rules are in PREREGISTRATION_confirm.md.

    python analyze_confirm.py [results dir]

Primary set:     questions the BASE model answers correctly with probability > 0.5.
Primary measure: mean change in the probability of the correct option on that set
                 (edited minus base), with a paired bootstrap 95% interval.
"""
import json
import math
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_confirm")
NOISE = ["ctrl_damage_noise05", "ctrl_damage_noise10", "ctrl_damage_noise20", "ctrl_damage_noise40"]
EDITS = ["mit_p10c", "mit_p25c", "mit_p50c", "ws_gpqa", "ws_math_primary", "ws_medxpertqa", "ws_scibench", "ws_scicode"]
OTHER = ["ctrl_random", "ctrl_damage_noise10_s1", "ctrl_damage_noise20_s1", "ctrl_damage_zero_attn15"]
B = 10000


def load(n):
    p = os.path.join(D, n + ".jsonl")
    return {r["id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else None


def main():
    base = load("base")
    ids = sorted(base)
    conf = [i for i in ids if math.exp(base[i]["lp_gold"]) > 0.5]
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(conf), (B, len(conf)))
    pb = np.array([math.exp(base[i]["lp_gold"]) for i in conf])
    print(f"questions: {len(ids)}   base accuracy: {100 * np.mean([base[i]['correct'] for i in ids]):.1f}%   "
          f"primary set (base p(correct) > 0.5): {len(conf)}   mean p there: {pb.mean():.3f}\n")
    d = {}
    for n in NOISE + OTHER + EDITS:
        e = load(n)
        if e is None or any(i not in e for i in ids):
            print(f"{n}: missing or incomplete, skipped")
            continue
        d[n] = np.array([math.exp(e[i]["lp_gold"]) for i in conf]) - pb
        d[n + "/acc"] = 100 * (np.mean([e[i]["correct"] for i in ids]) - np.mean([base[i]["correct"] for i in ids]))
        d[n + "/kept"] = float(np.mean([e[i]["correct"] for i in conf]))

    def ci(x):
        m = np.sort(x[boot].mean(1))
        return m[int(0.025 * B)], m[int(0.975 * B) - 1]

    print(f"{'variant':<26}{'change in p(correct)':>21}{'95% interval':>20}{'kept correct':>14}{'accuracy change':>17}")
    for n in NOISE + OTHER + EDITS:
        if n in d:
            lo, hi = ci(d[n])
            print(f"{n:<26}{d[n].mean():>+21.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}{d[n + '/kept']:>14.3f}{d[n + '/acc']:>+17.1f}")

    def less(a, b):
        """True if variant a lowers confidence more than b: interval of (a - b) is entirely below 0."""
        lo, hi = ci(d[a] - d[b])
        return hi < 0, (d[a] - d[b]).mean(), lo, hi

    print("\nPre-registered tests")
    if all(n in d for n in NOISE):
        m = [d[n].mean() for n in NOISE]
        ok = all(m[k] > m[k + 1] for k in range(3)) and all(less(NOISE[k + 1], NOISE[k])[0] for k in range(1, 3))
        print(f"H1 calibration (drop grows with noise; 20% > 10% and 40% > 20% clearly): {'PASS' if ok else 'FAIL'}"
              f"   means {', '.join(f'{x:+.3f}' for x in m)}")
    if "mit_p10c" in d:
        res = []
        for ref in ["ctrl_damage_noise10", "ctrl_damage_noise10_s1"]:
            if ref in d:
                t, diff, lo, hi = less("mit_p10c", ref)
                res.append(t)
                print(f"   mit_p10c minus {ref}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"H2 broad edit (mit_p10c lowers confidence more than 10% noise, both seeds): "
              f"{'PASS' if res and all(res) else 'FAIL'}")
    res = []
    for n in EDITS:
        if n == "mit_p10c" or n not in d:
            continue
        r2 = [less(ref, n) for ref in ["ctrl_damage_noise20", "ctrl_damage_noise20_s1"] if ref in d]
        okn = bool(r2) and all(t for t, *_ in r2)
        res.append(okn)
        print(f"   {n}: drop {d[n].mean():+.3f}, smaller than 20% noise on both seeds: {'yes' if okn else 'NO'}")
    print(f"H3 other edits (each lowers confidence less than 20% noise): {'PASS' if res and all(res) else 'FAIL'}")

    xs = [0.0] + [0.05, 0.10, 0.20, 0.40]
    if all(n in d for n in NOISE):
        ys = [0.0] + [-d[n].mean() for n in NOISE]
        print("\nNoise-equivalent size (descriptive): the noise level that gives the same drop")
        for n in EDITS:
            if n in d:
                drop = -d[n].mean()
                eq = float(np.interp(drop, ys, xs)) if drop > 0 else 0.0
                print(f"   {n:<18} {100 * eq:4.1f}% noise")


if __name__ == "__main__":
    main()
