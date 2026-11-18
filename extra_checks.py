"""Exploratory checks reported in the write-up (not pre-registered).

    python extra_checks.py [results dir]
"""
import json
import math
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
L = lambda n: {r["id"]: r for r in map(json.loads, open(os.path.join(D, n + ".jsonl")))}
b = L("base")
ids = sorted(b)
names = sorted(f[:-6] for f in os.listdir(D) if f.endswith(".jsonl") and f != "base.jsonl")
conf = [i for i in ids if math.exp(b[i]["lp_gold"]) > 0.5]
print(f"questions the base answers correctly with p > 0.5: {len(conf)}, "
      f"mean p there {np.mean([math.exp(b[i]['lp_gold']) for i in conf]):.3f}")
print(f"{'edit':<17}{'letters changed':>16}{'wrong->other wrong':>20}{'kept of confident':>19}{'p(correct) there':>18}")
for n in names:
    e = L(n)
    ch = [i for i in ids if e[i]["pred"] != b[i]["pred"]]
    ww = sum(1 for i in ch if not b[i]["correct"] and not e[i]["correct"])
    kept = np.mean([e[i]["correct"] for i in conf])
    pc = np.mean([math.exp(e[i]["lp_gold"]) for i in conf])
    print(f"{n:<17}{len(ch):>16}{ww:>20}{kept:>19.3f}{pc:>18.3f}")
