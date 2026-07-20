"""Summary figure + methods table from aggregated_sim.csv.

Produces a compact, self-contained overview of the simulation results:
  output/figures/w1_vs_failure.png   W1 vs Byzantine failure rate; one panel per
                                      attack type (mean / cov / weight).
  output/figures/methods_table.md    W1 (mean +/- SE over seeds) per method x rate.

Config shown: overlap 0.3, m = 20 machines, ss = 100000 (n = 5000 / machine), rho_mode = main.
Each curve = mean over seeds; shaded band = +/- 1 standard error. Run from simulation/:
  python postprocess/essay_figures.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # simulation/
CSV = os.path.join(HERE, "output", "aggregated_sim.csv")
OUTDIR = os.path.join(HERE, "output", "figures")

ATTACKS = ["mean", "cov", "weight"]
ATTACK_TITLE = {"mean": "Mean attack", "cov": "Covariance attack", "weight": "Weight attack"}

# (method, rho) -> style, matching the paper's roles (see postprocess/figures.py).
SERIES = [
    (("oracle", None), dict(label="Oracle",    color="black",      ls="--")),
    (("dfmr", 1.3),    dict(label="DFMR(1.3)", color="tab:red",    ls="-")),
    (("dfmr", 1.0),    dict(label="DFMR(1)",   color="tab:orange", ls="-")),
    (("trim", None),   dict(label="Trim",      color="tab:green",  ls=":")),
    (("coat", None),   dict(label="COAT",      color="tab:blue",   ls="-.")),
    (("gmr", None),    dict(label="Vanilla",   color="tab:purple", ls=(0, (3, 1, 1, 1)))),
]


def _match(df, method, rho):
    m = (df.method == method)
    if rho is None:
        m &= df["rho"].isna()
    else:
        m &= np.isclose(df["rho"].fillna(-999), rho)
    return df[m]


def _mean_se(sub, xcol):
    g = sub.groupby(xcol)["value"]
    return g.mean(), g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--ss", type=int, default=100000)
    ap.add_argument("--yscale", choices=["log", "linear"], default="log",
                    help="y-axis scale for the W1 panels (default: log)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="use only seeds 1..N (default: every seed on disk)")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    base = df[(np.isclose(df.overlap, args.overlap)) & (df.m == args.m)
              & (df.ss == args.ss) & (df.metric == "W1")]
    if base.empty:
        raise SystemExit("no W1 rows for overlap=%s m=%s ss=%s in %s"
                         % (args.overlap, args.m, args.ss, CSV))
    if args.seeds is not None:
        base = base[base.seed <= args.seeds]
    nseed = base.seed.nunique()
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- figure: 3 panels, W1 vs failure rate ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, attack in zip(axes, ATTACKS):
        sub_a = base[base.attack == attack]
        for (method, rho), style in SERIES:
            s = _match(sub_a, method, rho)
            if s.empty:
                continue
            mean, se = _mean_se(s, "failure_rate")
            x = mean.index.values
            ax.plot(x, mean.values, marker="o", ms=4, **style)
            ax.fill_between(x, (mean - se).values, (mean + se).values,
                            color=style["color"], alpha=0.18)
        ax.set_title(ATTACK_TITLE[attack])
        ax.set_xlabel(r"Byzantine failure rate $\alpha$")
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        # Log scale: under mean attack the Vanilla curve reaches ~180 while every
        # other method stays near 0.08, so a linear axis flattens the five robust
        # methods into one line. Keep this in sync with the essay's Figure A.1,
        # whose y-label says "(log scale)".
        if args.yscale == "log":
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    ylabel = r"$W_1$ to true mixing distribution"
    if args.yscale == "log":
        ylabel += " (log scale)"
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle(
        r"$W_1$ vs Byzantine failure rate  (Gaussian mixture, $K=5$, $d=10$, "
        r"$m=%d$, $n=5000$, MaxOmega$=%.1f$, %d seeds)" % (args.m, args.overlap, nseed),
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(OUTDIR, "w1_vs_failure.png")
    fig.savefig(figpath, dpi=140)
    print("wrote", figpath)

    # ---- markdown methods table ----
    order = ["oracle", "dfmr@1.3", "dfmr@1.0", "trim", "coat", "gmr"]
    disp = {"oracle": "Oracle", "dfmr@1.3": "DFMR(1.3)", "dfmr@1.0": "DFMR(1)",
            "trim": "Trim", "coat": "COAT", "gmr": "Vanilla"}
    lines = []
    lines.append("W1 to the true mixing distribution: mean +/- SE over %d seeds "
                 "(overlap %.1f, m=%d, n=5000).\n" % (nseed, args.overlap, args.m))
    for attack in ATTACKS:
        sub_a = base[base.attack == attack]
        rates = sorted(sub_a.failure_rate.unique())
        lines.append("\n**%s**\n" % ATTACK_TITLE[attack])
        header = "| Method | " + " | ".join(r"$\alpha=%.1f$" % r for r in rates) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(rates) + 1))
        for key in order:
            method, rho = (key.split("@")[0], float(key.split("@")[1])) if "@" in key else (key, None)
            cells = []
            for r in rates:
                s = _match(sub_a[sub_a.failure_rate == r], method, rho)
                if s.empty:
                    cells.append("--")
                else:
                    cells.append("%.3f&plusmn;%.3f"
                                 % (s.value.mean(), s.value.std(ddof=1) / np.sqrt(max(len(s), 1))))
            lines.append("| %s | %s |" % (disp[key], " | ".join(cells)))
    tablepath = os.path.join(OUTDIR, "methods_table.md")
    with open(tablepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", tablepath)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
