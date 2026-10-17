"""Figures for the pilot simulations (reads results/*.csv, writes figures/*.png)."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed categorical order
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_d1(rows, out):
    scen = list(dict.fromkeys(r["scenario"] for r in rows))
    crits = list(dict.fromkeys(r["criterion"] for r in rows))
    Cs = sorted({int(r["counties"]) for r in rows})
    fig, axes = plt.subplots(1, len(scen), figsize=(10, 3.4), sharey=True)
    width = 0.16
    for ax, s in zip(axes, scen):
        for i, c in enumerate(crits):
            vals = [next(float(r["pct_correct"]) for r in rows
                         if r["scenario"] == s and r["criterion"] == c and int(r["counties"]) == C)
                    for C in Cs]
            xs = [x + (i - (len(crits) - 1) / 2) * width for x in range(len(Cs))]
            ax.bar(xs, vals, width=width - 0.02, color=SERIES[i], label=c)
        truth = next(r["true_model"] for r in rows if r["scenario"] == s)
        ax.set_title(f"{s}\n(true model {truth})", fontsize=9, loc="left")
        ax.set_xticks(range(len(Cs)), [str(C) for C in Cs])
        ax.set_xlabel("Number of counties")
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Correct structure selected (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.01, -0.04, "Note: 'BIC (top clusters)' and 'BIC (level-specific)' select the three-level "
             "model in almost every data set, so their high scores in A and C are not evidence of skill.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


def fig_d3(rows, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for phi, ls, tag in (("0.0", "-", "white noise"), ("0.5", "--", "AR(1), phi = 0.5")):
        sub = [r for r in rows if r["ar1_phi"] == phi]
        x = [float(r["jump_sd_units"]) for r in sub]
        a1.plot(x, [float(r["rmse_zip_months"]) for r in sub], ls, color=SERIES[0], lw=2,
                marker="o", ms=5, label=f"Per-ZIP break, {tag}")
        a1.plot(x, [float(r["rmse_common_months"]) for r in sub], ls, color=SERIES[1], lw=2,
                marker="o", ms=5, label=f"Common break, {tag}")
        a2.plot(x, [float(r["est_s2_tau_zip"]) for r in sub], ls, color=SERIES[0], lw=2,
                marker="o", ms=5, label=f"ZIP-level, {tag}")
        a2.plot(x, [float(r["est_s2_tau_county"]) for r in sub], ls, color=SERIES[1], lw=2,
                marker="o", ms=5, label=f"County-level, {tag}")
    a2.axhline(float(rows[0]["true_s2_tau_zip"]), color=SERIES[0], lw=1, ls=":")
    a2.axhline(float(rows[0]["true_s2_tau_county"]), color=SERIES[1], lw=1, ls=":")
    box = dict(facecolor="white", edgecolor="none", pad=1)
    a2.text(2.15, float(rows[0]["true_s2_tau_zip"]), "true ZIP = 4", va="center", fontsize=8, color=MUTED, bbox=box)
    a2.text(2.15, float(rows[0]["true_s2_tau_county"]), "true county = 9", va="center", fontsize=8, color=MUTED, bbox=box)
    a1.set_title("Error in estimated break dates", loc="left", fontsize=9)
    a1.set_xlabel("Jump size (SD units)"); a1.set_ylabel("RMSE (months)")
    a2.set_title("Two-stage estimates of timing variances (log scale)", loc="left", fontsize=9)
    a2.set_xlabel("Jump size (SD units)"); a2.set_ylabel("Estimated variance (months$^2$)")
    a2.set_yscale("log")
    for ax in (a1, a2):
        ax.set_xticks([0.5, 1, 2]); ax.legend(fontsize=7.5, loc="upper right")
    a2.set_xlim(0.4, 2.75)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


def fig_d1_depth(rows, out):
    truths = list(dict.fromkeys(r["truth"] for r in rows))
    crits = list(dict.fromkeys(r["criterion"] for r in rows))
    Rs = sorted({int(r["regions"]) for r in rows})
    fig, axes = plt.subplots(1, len(truths), figsize=(12.5, 3.4), sharey=True)
    width = 0.16
    for ax, t in zip(axes, truths):
        for i, c in enumerate(crits):
            vals = [next(float(r["pct_correct"]) for r in rows
                         if r["truth"] == t and r["criterion"] == c and int(r["regions"]) == R)
                    for R in Rs]
            xs = [x + (i - (len(crits) - 1) / 2) * width for x in range(len(Rs))]
            ax.bar(xs, vals, width=width - 0.02, color=SERIES[i], label=c)
        head, _, tail = t.partition(": ")
        ax.set_title(f"{head}\n({tail})", fontsize=9, loc="left")
        ax.set_xticks(range(len(Rs)), [str(R) for R in Rs])
        ax.set_xlabel("Number of regions")
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Correct structure selected (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05))
    fig.text(0.01, -0.04, "Design: 5 counties per region, 5 ZIP codes per county, 24 months; 8 candidate "
             "structures; 1,000 data sets per condition. 'Top clusters' and 'level-specific' almost always "
             "choose a model with extra levels.", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    fig_d1(read("results/d1_selection.csv"), "figures/d1_selection.png")
    fig_d3(read("results/d3_change_points.csv"), "figures/d3_change_points.png")
    fig_d1_depth(read("results/d1_depth_4level.csv"), "figures/d1_depth_4level.png")
