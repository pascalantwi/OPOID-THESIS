"""Figure for the covariate study (results/covariate_study.csv -> figures/covariate_study.png)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"AIC": "#2a78d6", "Standard BIC (N)": "#eb6834", "Likelihood ratio tests": "#1baf7a",
          "New BIC (exact information)": "#e87ba4"}
SHORT = {"AIC": "AIC", "Standard BIC (N)": "Standard BIC", "Likelihood ratio tests": "LR tests",
         "New BIC (exact information)": "New BIC"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "legend.frameon": False})
rows = list(csv.DictReader(open("results/covariate_study.csv")))
scen = list(dict.fromkeys(r["scenario"] for r in rows))
should = {s: ("YES" if "region" in next(r["true_structure"] for r in rows if r["scenario"] == s) else "NO")
          for s in scen}
fig, ax = plt.subplots(figsize=(12, 4.8))
w = 0.2
for i, m in enumerate(COLORS):
    vals = [next(float(r["pct_says_region_matters"]) for r in rows if r["scenario"] == s and r["method"] == m)
            for s in scen]
    xs = [x + (i - 1.5) * w for x in range(len(scen))]
    ax.bar(xs, vals, width=w - 0.03, color=COLORS[m], label=SHORT[m])
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.5)
labels = []
for s in scen:
    head = s.replace("Region gap due to rurality: ", "Regions differ only\nbecause of rurality:\n")
    head = head.replace("covariates IN", "rurality IN model").replace("covariates OUT", "rurality LEFT OUT")
    beyond = " (beyond rurality)" if "rurality" in s else ""
    labels.append(f"{head}\nright answer{beyond}: {should[s]}")
ax.set_xticks(range(len(scen)), labels, fontsize=8.5)
ax.set_ylabel('Concludes "region matters" (%)')
ax.set_ylim(0, 110)
ax.grid(axis="x", visible=False)
ax.set_title("With covariates: how often does each method conclude that the 7 regions differ?\n"
             "Income and rurality (ZIP level), trend and post-October-2014 drop (month level); "
             "7 regions, 67 counties, 1,483 ZIP codes, 60 months; 500 data sets each",
             loc="left", fontsize=10)
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.32))
fig.tight_layout()
fig.savefig("figures/covariate_study.png", dpi=200, bbox_inches="tight")
