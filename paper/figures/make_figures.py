"""Build the paper's figures from results/*/comparison.json.

Run from the repository root:  python paper/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# Categorical slots in fixed order; each strategy keeps its colour in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDER = ["none", "aux", "explore", "accum", "surrogate", "conf", "threshold"]
COLOR = {s: PALETTE[i] for i, s in enumerate(ORDER)}
COLOR["none"] = "#52514e"  # the control is the neutral reference line
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load(exp):
    return {r["strategy"]: r for r in json.load(open(ROOT / "results" / exp / "comparison.json"))["runs"]}


EXPS = {"exp1-seed0": "Exp. 1: top-2, renorm. gate", "exp2-seed0": "Exp. 2: top-1, raw gate",
        "exp3-seed0": "Exp. 3: equal optimiser steps"}
runs = {e: load(e) for e in EXPS}

# Figure 1 - validation loss against tokens seen, experiments 1 and 2.
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.4), sharey=True)
for ax, exp in zip(axes, ["exp1-seed0", "exp2-seed0"]):
    for s in ORDER:
        if s not in runs[exp]:
            continue
        h = runs[exp][s]["history"]
        ax.plot([p["tokens_seen"] / 1e6 for p in h], [p["val_loss"] for p in h], lw=1.4,
                color=COLOR[s], label=s, ls="--" if s == "none" else "-")
    ax.set_title(EXPS[exp], fontsize=8.5, color=INK)
    ax.set_xlabel("training tokens (M)")
    ax.set_ylim(1.55, 2.8)
axes[0].set_ylabel("validation cross-entropy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=7, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.14))
fig.savefig(OUT / "val_loss.pdf")

# Figure 2 - busiest expert's share against the share of assignments dropped.
fig, ax = plt.subplots(figsize=(3.3, 2.5))
# A marker shape per experiment keeps coincident points distinguishable; the three aux
# runs land almost on top of each other, so they are drawn hollow to show through.
MARKER = ["o", "s", "^"]
def point(r):
    return r["final"]["collapse"]["max_load"], 100 * r["final"]["drop_rate"]
for i, exp in enumerate(EXPS):
    rest = [point(r) for s, r in runs[exp].items() if s != "aux"]
    ax.scatter(*zip(*rest), s=22, marker=MARKER[i], color=PALETTE[i], edgecolor="white",
               linewidth=0.8, label=EXPS[exp].split(":")[0], zorder=3)
    ax.scatter(*point(runs[exp]["aux"]), s=34, marker=MARKER[i], facecolor="none",
               edgecolor=PALETTE[i], linewidth=1.1, zorder=4)
aux_pts = [point(runs[e]["aux"]) for e in EXPS]
aux_mid = (sum(x for x, _ in aux_pts) / 3, sum(y for _, y in aux_pts) / 3)
ax.annotate("aux, all three runs\n(hollow markers)", aux_mid, xytext=(-4, 34),
            textcoords="offset points", fontsize=6.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=5))
ax.axvline(1.25 / 8, color=INK2, lw=0.8, ls=":")
ax.text(1.25 / 8 + 0.002, 26, "capacity\n$\\gamma/E$", fontsize=6.5, color=INK2)
ax.set_xlabel("max expert load (pooled)")
ax.set_ylabel("assignments dropped (%)")
ax.legend(fontsize=7, loc="upper left")
fig.savefig(OUT / "maxload_vs_drop.pdf")

# Figure 3 - per-layer normalised load entropy in experiment 2 (the regime where collapse bites).
exp = "exp2-seed0"
strats = [s for s in ORDER if s in runs[exp]]
grid = [[l["entropy_norm"] for l in runs[exp][s]["final"]["per_layer_collapse"]] for s in strats]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
im = ax.imshow(grid, cmap="Blues", vmin=0.6, vmax=1.0, aspect="auto")
ax.grid(False)
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if v > 0.85 else INK)
ax.set_xticks(range(len(grid[0])), [f"layer {j}" for j in range(len(grid[0]))])
ax.set_yticks(range(len(strats)), strats)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("$H_{\\mathrm{norm}}$", fontsize=7)
fig.savefig(OUT / "per_layer_entropy.pdf")
print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
