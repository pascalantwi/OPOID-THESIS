"""Figures for the staged study (results/staged_study.csv -> figures/staged_*.png)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"AIC": "#2a78d6", "Standard BIC (N)": "#eb6834", "Likelihood ratio tests": "#1baf7a",
          "BIC (top clusters)": "#eda100", "New BIC (exact information)": "#e87ba4",
          "2-level analysis only": "#008300"}
SHORT = {"AIC": "AIC", "Standard BIC (N)": "Standard BIC", "Likelihood ratio tests": "LR tests",
         "BIC (top clusters)": "BIC (top clusters)", "New BIC (exact information)": "New BIC",
         "2-level analysis only": "2-level only"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "legend.frameon": False})
rows = list(csv.DictReader(open("results/staged_study.csv")))
stages = list(dict.fromkeys(r["stage"] for r in rows))
MAIN = ["AIC", "Standard BIC (N)", "Likelihood ratio tests", "New BIC (exact information)",
        "2-level analysis only"]


STAGE_LEVELS = {"Stage 1: 2 levels": 1, "Stage 2: 3 levels": 2, "Stage 3: 4 levels": 3}


def tradeoff(m):
    """Average % adding a false level (over scenarios where some candidate level is absent)
    and % missing a real level (over scenarios where some level is real)."""
    sub = [r for r in rows if r["method"] == m]
    can_add = [float(r["pct_added_false_level"]) for r in sub
               if len([t for t in r["true_structure"].split(" + ") if t != "none"])
               < STAGE_LEVELS[r["stage"]]]
    can_miss = [float(r["pct_missed_real_level"]) for r in sub if r["true_structure"] != "none"]
    return sum(can_add) / len(can_add), sum(can_miss) / len(can_miss)

# ---- Figure 1: accuracy at each stage
widths = [len({r["scenario"] for r in rows if r["stage"] == s}) for s in stages]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, gridspec_kw={"width_ratios": widths})
for ax, st in zip(axes, stages):
    sub = [r for r in rows if r["stage"] == st]
    scen = list(dict.fromkeys(r["scenario"] for r in sub))
    meths = [m for m in MAIN if any(r["method"] == m for r in sub)]
    w = 0.8 / len(meths)
    for i, m in enumerate(meths):
        vals = [next(float(r["pct_correct"]) for r in sub if r["scenario"] == s and r["method"] == m)
                for s in scen]
        xs = [x + (i - (len(meths) - 1) / 2) * w for x in range(len(scen))]
        ax.bar(xs, vals, width=w - 0.02, color=COLORS[m], label=SHORT[m])
    ax.set_xticks(range(len(scen)), [s.replace(" effect", "\neffect").replace(" (", "\n(")
                                     .replace(" but ", "\nbut ") for s in scen], fontsize=8.5)
    ax.set_title(st, loc="left", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Right structure chosen (%)")
h, l = axes[-1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), fontsize=10)
fig.text(0.01, -0.05, "Design shaped like the ARCOS data: 7 regions, 67 counties, 1,483 ZIP codes (unequal group "
         "sizes), 60 months; 1,000 simulated data sets per scenario. 'BIC (top clusters)' is omitted here because it "
         "adds a level that does not exist in every data set (see table).", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig("figures/staged_accuracy.png", dpi=200, bbox_inches="tight")

# ---- Figure 2: trade-off between adding false levels and missing real ones
OFFSETS = {"AIC": (8, -12), "Likelihood ratio tests": (8, 6), "Standard BIC (N)": (8, 4),
           "New BIC (exact information)": (8, -12)}
fig, ax = plt.subplots(figsize=(8, 5.2))
for m in OFFSETS:
    x, y = tradeoff(m)
    ax.scatter(x, y, s=120, color=COLORS[m], edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(f"{SHORT[m]}  ({x:.1f}%, {y:.1f}%)", (x, y), xytext=OFFSETS[m],
                textcoords="offset points", fontsize=10)
ax.set_xlim(-0.5, 12); ax.set_ylim(0, 26)
ax.set_xlabel("Adds a level that is NOT real (% of data sets, averaged over scenarios where a level is absent)")
ax.set_ylabel("Misses a level that IS real (% of data sets, averaged)")
ax.set_title("Which method is performing? Closer to the bottom-left corner is better",
             loc="left", fontsize=10, fontweight="bold")
xb, yb = tradeoff("BIC (top clusters)")
x2, y2 = tradeoff("2-level analysis only")
fig.text(0.01, -0.03, f"Off the chart: BIC (top clusters) adds a false level {xb:.0f}% of the time; "
         f"a 2-level analysis misses a real county or region level {y2:.0f}% of the time.",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig("figures/staged_tradeoff.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    for m in COLORS:
        sub = [r for r in rows if r["method"] == m]
        a, b = tradeoff(m)
        print(f"{m:30s} correct {sum(float(r['pct_correct']) for r in sub) / len(sub):5.1f}"
              f"  adds false level {a:5.1f}  misses real level {b:5.1f}")
