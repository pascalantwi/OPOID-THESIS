"""Figures for the main study (results/main_study.csv -> figures/main_*.png)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = {"AIC": "#2a78d6", "Standard BIC (N)": "#eb6834",
        "Likelihood ratio tests": "#1baf7a", "New BIC (exact information)": "#e87ba4"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "legend.frameon": False})
rows = list(csv.DictReader(open("results/main_study.csv")))
scen = list(dict.fromkeys(r["scenario"] for r in rows))
methods = list(dict.fromkeys(r["method"] for r in rows))


def val(s, m, key="pct_correct"):
    return next(float(r[key]) for r in rows if r["scenario"] == s and r["method"] == m)


# Figure 1: accuracy by true world, four main methods
fig, ax = plt.subplots(figsize=(12, 4.4))
w = 0.2
for i, m in enumerate(MAIN):
    vals = [val(s, m) for s in scen]
    xs = [x + (i - 1.5) * w for x in range(len(scen))]
    ax.bar(xs, vals, width=w - 0.03, color=MAIN[m], label=m)
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
labels = [s.split(". ", 1)[1].replace(", ", ",\n").replace(" + ", " +\n", 1) if len(s) > 22 else s.split(". ", 1)[1]
          for s in scen]
ax.set_xticks(range(len(scen)), labels, fontsize=8.5)
ax.set_ylabel("Right structure chosen (%)")
ax.set_ylim(0, 110)
ax.grid(axis="x", visible=False)
ax.set_title("Which method finds the true levels? Simulated data shaped like the ARCOS data\n"
             "7 regions, 67 counties, 1,483 ZIP codes, 60 months; 500 data sets per true world",
             loc="left", fontsize=10)
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig("figures/main_accuracy_by_world.png", dpi=200, bbox_inches="tight")

# Figure 2: overall ranking (average over the six worlds), all six methods
avg = {m: sum(val(s, m) for s in scen) / len(scen) for m in methods}
order = sorted(avg, key=avg.get)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(range(len(order)), [avg[m] for m in order], color="#2a78d6", height=0.6)
for y, m in enumerate(order):
    ax.text(avg[m] + 1, y, f"{avg[m]:.0f}%", va="center", fontsize=9)
ax.set_yticks(range(len(order)), order)
ax.set_xlim(0, 105)
ax.set_xlabel("Average % of data sets where the right structure was chosen (six true worlds)")
ax.grid(axis="y", visible=False)
ax.set_title("Overall ranking", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig("figures/main_overall_ranking.png", dpi=200, bbox_inches="tight")
