"""Simple figure for the real-design simulation (results/d1_your_design.csv)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {"AIC": "#2a78d6", "BIC (N)": "#eb6834", "BIC (exact information)": "#e87ba4"}
LABEL = {"AIC": "AIC", "BIC (N)": "Standard BIC", "BIC (exact information)": "New BIC (exact information)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "legend.frameon": False})
rows = list(csv.DictReader(open("results/d1_your_design.csv")))
scen = list(dict.fromkeys(r["scenario"] for r in rows))
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.26
for i, c in enumerate(SERIES):
    vals = [next(float(r["pct_correct"]) for r in rows if r["scenario"] == s and r["criterion"] == c) for s in scen]
    xs = [x + (i - 1) * w for x in range(len(scen))]
    bars = ax.bar(xs, vals, width=w - 0.03, color=SERIES[c], label=LABEL[c])
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(scen)), [s.replace(" effect", "\neffect") for s in scen])
ax.set_ylabel("Times the right answer was chosen (%)")
ax.set_ylim(0, 112)
ax.grid(axis="x", visible=False)
ax.set_title("Can each method tell whether the 7 regions really differ?\n"
             "Simulated data shaped like yours: 7 regions, 70 counties, 1,470 ZIP codes, 60 months; 500 data sets each",
             loc="left", fontsize=10)
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18))
fig.tight_layout()
fig.savefig("figures/d1_your_design.png", dpi=200, bbox_inches="tight")
