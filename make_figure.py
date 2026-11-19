"""Draw figures/confidence_change.png from the confirmatory results.

    python make_figure.py [results dir]
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_confirm")
L = lambda n: {r["id"]: r for r in map(json.loads, open(os.path.join(D, n + ".jsonl")))}
base = L("base")
conf = [i for i in sorted(base) if math.exp(base[i]["lp_gold"]) > 0.5]
pb = np.array([math.exp(base[i]["lp_gold"]) for i in conf])
rng = np.random.default_rng(0)
boot = rng.integers(0, len(conf), (10000, len(conf)))


def stat(n):
    e = L(n)
    d = np.array([math.exp(e[i]["lp_gold"]) for i in conf]) - pb
    m = np.sort(d[boot].mean(1))
    return d.mean(), m[250], m[9749]


EDITS = [("Hu et al., GPQA edit", "ws_gpqa"), ("Hu et al., math edit", "ws_math_primary"),
         ("Hu et al., MedXpertQA edit", "ws_medxpertqa"), ("MIT, 50 passes", "mit_p50c"),
         ("Hu et al., SciCode edit", "ws_scicode"), ("MIT, 25 passes", "mit_p25c"),
         ("Hu et al., SciBench edit", "ws_scibench"), ("MIT, 10 passes (30 matrices)", "mit_p10c")]
NOISE = [("5% noise", "ctrl_damage_noise05"), ("10% noise, seed A", "ctrl_damage_noise10"),
         ("10% noise, seed B", "ctrl_damage_noise10_s1"), ("20% noise, seed A", "ctrl_damage_noise20"),
         ("20% noise, seed B", "ctrl_damage_noise20_s1")]
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 6.2), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
rows = [(lab, n, BLUE) for lab, n in EDITS] + [None] + [(lab, n, ORANGE) for lab, n in NOISE]
ys, labels = [], []
y = 0
for r in rows:
    if r is None:
        y -= 0.6
        continue
    lab, n, col = r
    m, lo, hi = stat(n)
    ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round", zorder=3)
    ax.plot([m], [y], "o", color=col, ms=8, mec=SURF, mew=1.5, zorder=4)
    if n in ("mit_p10c", "ctrl_damage_noise20", "ctrl_damage_noise20_s1"):
        ax.text(hi + 0.004, y, f"{m:+.3f}", ha="left", va="center", color=MUTED, fontsize=10)
    ys.append(y)
    labels.append(lab)
    y -= 1
ax.axvline(0, color=MUTED, lw=1, zorder=2)
ax.set_yticks(ys)
ax.set_yticklabels(labels, color=INK)
ax.set_xlim(-0.14, 0.025)
ax.set_ylim(min(ys) - 0.7, 0.7)
ax.xaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=MUTED)
ax.set_xlabel("Change in probability of the correct answer (edited minus original)", color=MUTED)
ax.plot([], [], "o", color=BLUE, ms=8, label="Published weight edits")
ax.plot([], [], "o", color=ORANGE, ms=8, label="Random noise on the same 30 matrices")
ax.legend(loc="lower left", frameon=False, labelcolor=INK, bbox_to_anchor=(-0.02, 1.0), ncol=2,
          handletextpad=0.3, columnspacing=1.6, borderaxespad=0.2)
fig.suptitle("The broadest edit costs more confidence than 20% random noise",
             x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.975)
fig.text(0.02, 0.905, f"Qwen3.5-0.8B, {len(conf)} MMLU-Pro questions the original model answers correctly with over 50% confidence.\n"
         "Dots are means, bars are 95% intervals. No published edit made any of these answers wrong. 40% noise: off scale at -0.34.",
         ha="left", va="top", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=0.30, right=0.97, top=0.78, bottom=0.11)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
out = os.path.join(HERE, "figures", "confidence_change.png")
fig.savefig(out, facecolor=SURF)
print("wrote", out)
