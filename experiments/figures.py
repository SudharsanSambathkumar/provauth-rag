"""Paper figures from results/summary.json -> figures/*.png (300 dpi, print-friendly, light)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results/summary.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#8d8c87"
TITLES = {"retail": "ProvAuth-Retail", "itsm": "ProvAuth-ITSM"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------ Fig: leakage vs availability
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, (ds, d) in zip(axes, R["datasets"].items(), strict=True):
    s = d["summary"]
    groups = {}
    for m, v in s.items():  # methods with identical coordinates share one point and one label
        key = (round(v["recall_pct"], 1), round(v["udr_pct"], 1))
        groups.setdefault(key, []).append(m)
    for (x, y), ms in groups.items():
        is_p = "P" in ms
        ax.scatter(x, y, s=70 if is_p else 46, color=BLUE if is_p else NEUTRAL, edgecolor=SURF, linewidth=1.5, zorder=3)
        label = ", ".join(ms) + (" (proposed)" if is_p else "")
        if is_p:  # proposed: label below its point; nearby baselines: above; others: to the left
            off, va, ha = (8, -10), "top", "right"
        elif y < 10:
            off, va, ha = (0, 9), "bottom", "center"
        else:
            off, va, ha = (-8, 0), "center", "right"
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                    color=INK, fontweight="bold" if is_p else "normal")
    ax.set_title(TITLES[ds], loc="left")
    ax.set_xlabel("Authorized claims released (%)")
    ax.set_xlim(76, 102)
    ax.set_ylim(-10, 108)
    style(ax)
axes[0].set_ylabel("Forbidden claims disclosed (%)")
fig.tight_layout()
fig.savefig(OUT / "fig_tradeoff.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------ Fig: rule-depth ablation
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for (ds, d), color in zip(R["datasets"].items(), (BLUE, ORANGE), strict=True):
    xs = [r["depth"] for r in d["depth_ablation"]]
    axes[0].plot(xs, [r["recall_pct"] for r in d["depth_ablation"]], marker="o", ms=5, lw=2, color=color,
                 label=TITLES[ds], markeredgecolor=SURF)
    axes[1].plot(xs, [1000 * r["materialize_s"] for r in d["depth_ablation"]], marker="o", ms=5, lw=2, color=color,
                 label=TITLES[ds], markeredgecolor=SURF)
axes[0].set_title("Authorized claims released (%)", loc="left")
axes[1].set_title("Offline materialization time (ms)", loc="left")
for ax in axes:
    ax.set_xlabel("Rule-depth bound D")
    ax.set_xticks([1, 2, 3, 4, 6])
    style(ax)
axes[0].set_ylim(0, 105)
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_depth.png", dpi=300)
plt.close(fig)
print("figures written to", OUT)
