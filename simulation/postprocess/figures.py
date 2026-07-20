"""Regenerate the paper's simulation figures from aggregated_sim.csv.

Produces (into output/figures/):
  fig3_<metric>.png  W1/ARI vs N; rows = attack, cols = failure rate  (m=100, overlap 0.3)
  fig4_<metric>.png  vs m, local n=5000 fixed;  cols = attack         (alpha 0.1, overlap 0.3)
  fig5_<metric>.png  vs m, total N=5e5 fixed;   cols = attack         (alpha 0.1, overlap 0.3)
  fig6_<metric>.png  vs overlap;                cols = attack         (alpha 0.2, m=100)
  fig2_<metric>.png  DFMR vs inflation factor rho; rows attack, cols m (alpha 0.1, rho_mode=fig2)

Each curve is the mean over seeds; shaded band is +/- 1 standard error. Run from
simulation/:  python postprocess/figures.py   (optionally --metric ARI)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "output", "aggregated_sim.csv")
FIGDIR = os.path.join(HERE, "output", "figures")
ATTACKS = ["mean", "cov", "weight"]

# series = (method, rho) -> (label, color, linestyle) matching the paper's roles.
SERIES = [
    (("oracle", None), dict(label="Oracle",    color="black",   ls="--")),
    (("dfmr", 1.3),    dict(label="DFMR(1.3)", color="tab:red", ls="-")),
    (("dfmr", 1.0),    dict(label="DFMR(1)",   color="tab:orange", ls="-")),
    (("trim", None),   dict(label="Trim",      color="tab:green", ls=":")),
    (("coat", None),   dict(label="COAT",      color="tab:blue", ls="-.")),
    (("gmr", None),    dict(label="Vanilla",   color="tab:purple", ls=(0, (3, 1, 1, 1)))),
]


def load():
    df = pd.read_csv(CSV)
    return df


def _match(df, method, rho):
    m = (df.method == method)
    if rho is None:
        m &= df["rho"].isna()
    else:
        m &= np.isclose(df["rho"].fillna(-999), rho)
    return df[m]


def _mean_se(sub, xcol):
    g = sub.groupby(xcol)["value"]
    return g.mean(), g.std(ddof=1) / np.sqrt(g.count())


def _plot_panel(ax, df, xcol, logx=True, logy=True, series=SERIES):
    any_line = False
    for (method, rho), style in series:
        s = _match(df, method, rho)
        if s.empty:
            continue
        mean, se = _mean_se(s, xcol)
        x = mean.index.values
        ax.plot(x, mean.values, marker="o", ms=3, **style)
        ax.fill_between(x, (mean - se).values, (mean + se).values,
                        color=style["color"], alpha=0.2)
        any_line = True
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    return any_line


def fig_grid(df, metric, rows, cols, xcol, row_key, col_key, base_filter,
             title, fname, logx=True):
    """Generic grid of W1/ARI-vs-x panels."""
    d = df[(df.metric == metric)].copy()
    for k, v in base_filter.items():
        d = d[np.isclose(d[k], v)] if isinstance(v, float) else d[d[k] == v]
    if d.empty:
        print("  [skip] %s: no rows after filter %s" % (fname, base_filter))
        return
    logy = (metric == "W1")
    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.7 * nr),
                             squeeze=False, sharex=True)
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            ax = axes[i][j]
            sub = d[(d[row_key] == rv) & (np.isclose(d[col_key], cv)
                    if isinstance(cv, float) else d[col_key] == cv)]
            _plot_panel(ax, sub, xcol, logx=logx, logy=logy)
            if i == 0:
                ax.set_title("%s=%s" % (col_key, cv), fontsize=9)
            if j == 0:
                ax.set_ylabel("%s\n%s" % (rv, metric), fontsize=9)
            if i == nr - 1:
                ax.set_xlabel(xcol, fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.05, fontsize=11)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", fname)


def figure3(df, metric):
    fig_grid(df, metric, rows=ATTACKS, cols=[0.1, 0.2, 0.3, 0.4], xcol="N",
             row_key="attack", col_key="failure_rate",
             base_filter=dict(m=100, overlap=0.3, rho_mode="main"),
             title="Fig 3: %s vs N (m=100, overlap 0.3); rows=attack, cols=failure rate" % metric,
             fname="fig3_%s.png" % metric)


def figure4(df, metric):
    _fig_vs_m(df, metric, base=dict(overlap=0.3, failure_rate=0.1, n=5000, rho_mode="main"),
              title="Fig 4: %s vs m (local n=5000 fixed, alpha=0.1, overlap 0.3)" % metric,
              fname="fig4_%s.png" % metric)


def figure5(df, metric):
    _fig_vs_m(df, metric, base=dict(overlap=0.3, failure_rate=0.1, N=500000, rho_mode="main"),
              title="Fig 5: %s vs m (total N=5e5 fixed, alpha=0.1, overlap 0.3)" % metric,
              fname="fig5_%s.png" % metric)


def _fig_vs_m(df, metric, base, title, fname):
    d = df[df.metric == metric].copy()
    for k, v in base.items():
        d = d[np.isclose(d[k], v)] if isinstance(v, float) else d[d[k] == v]
    if d.empty:
        print("  [skip] %s: no rows after %s" % (fname, base))
        return
    logy = (metric == "W1")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), squeeze=False, sharex=True)
    for j, atk in enumerate(ATTACKS):
        ax = axes[0][j]
        _plot_panel(ax, d[d.attack == atk], "m", logx=True, logy=logy)
        ax.set_title(atk, fontsize=9)
        ax.set_xlabel("m")
        if j == 0:
            ax.set_ylabel(metric)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, y=1.12, fontsize=11)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", fname)


def figure6(df, metric):
    d = df[(df.metric == metric) & (df.m == 100) & np.isclose(df.N, 500000)
           & np.isclose(df.failure_rate, 0.2) & (df.rho_mode == "main")].copy()
    if d.empty:
        print("  [skip] fig6_%s: no rows" % metric)
        return
    logy = (metric == "W1")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), squeeze=False, sharex=True)
    for j, atk in enumerate(ATTACKS):
        ax = axes[0][j]
        _plot_panel(ax, d[d.attack == atk], "overlap", logx=False, logy=logy)
        ax.set_title(atk, fontsize=9)
        ax.set_xlabel("MaxOmega")
        if j == 0:
            ax.set_ylabel(metric)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Fig 6: %s vs overlap (alpha=0.2, m=100, n=5000)" % metric, y=1.12, fontsize=11)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig6_%s.png" % metric), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig6_%s.png" % metric)


def figure2(df, metric):
    """DFMR W1/ARI as a function of the inflation factor rho; Oracle/COAT as
    horizontal reference lines. rows = attack, cols = m. alpha=0.1, rho_mode=fig2."""
    d = df[(df.metric == metric) & (df.rho_mode == "fig2")
           & np.isclose(df.failure_rate, 0.1)].copy()
    if d.empty:
        print("  [skip] fig2_%s: no rho_mode=fig2 rows" % metric)
        return
    ms = sorted(d.m.unique())
    logy = (metric == "W1")
    fig, axes = plt.subplots(len(ATTACKS), len(ms),
                             figsize=(3.2 * len(ms), 2.7 * len(ATTACKS)),
                             squeeze=False, sharex=True)
    for i, atk in enumerate(ATTACKS):
        for j, mval in enumerate(ms):
            ax = axes[i][j]
            sub = d[(d.attack == atk) & (d.m == mval)]
            dfmr = sub[sub.method == "dfmr"]
            if not dfmr.empty:
                mean, se = _mean_se(dfmr, "rho")
                ax.plot(mean.index, mean.values, "-", color="tab:red", label="DFMR(rho)")
                ax.fill_between(mean.index, (mean - se).values, (mean + se).values,
                                color="tab:red", alpha=0.2)
            for meth, col, lbl in (("oracle", "black", "Oracle"), ("coat", "tab:blue", "COAT")):
                r = sub[sub.method == meth]
                if not r.empty:
                    ax.axhline(r["value"].mean(), color=col, ls="--" if meth == "oracle" else "-.",
                               label=lbl)
            if logy:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            if i == 0:
                ax.set_title("m=%d" % mval, fontsize=9)
            if j == 0:
                ax.set_ylabel("%s\n%s" % (atk, metric), fontsize=9)
            if i == len(ATTACKS) - 1:
                ax.set_xlabel("rho", fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Fig 2: DFMR %s vs inflation factor rho (alpha=0.1)" % metric, y=1.06, fontsize=11)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig2_%s.png" % metric), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig2_%s.png" % metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="both", choices=["W1", "ARI", "both"])
    args = ap.parse_args()
    df = load()
    metrics = ["W1", "ARI"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        print("== metric", metric, "==")
        figure2(df, metric)
        figure3(df, metric)
        figure4(df, metric)
        figure5(df, metric)
        figure6(df, metric)


if __name__ == "__main__":
    main()
