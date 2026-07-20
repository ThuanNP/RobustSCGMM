"""Fig 7 experiment: concentration of L2 distances to the COAT mixture density.

For each model size (K,d) in {(5,2),(5,10),(10,10)} with m=100, n=1e4, failure rate
alpha=0.3, over R seeds: sample N=n*m points from the true mixture, split across m
machines, fit a local pMLE GMM on each, form the COAT centre, inject each Byzantine
attack, and record D(COAT, G_i) = sqrt(GMM_L2) for every machine, tagged failure-free
vs failed and by attack type. Writes a long CSV; --plot renders the histograms.

The paper's point: mean/cov failures produce a clear gap between the failure-free and
failed distance distributions (so the DFMR filter separates them), and the gap sharpens
as the number of parameters grows. Weight failure is less separable.

Run from simulation/:
  python distance_concentration.py --seeds 1-50
  python distance_concentration.py --plot
"""

import os
import argparse
import numpy as np
import pandas as pd

from pmle import pMLEGMM
from CTDGMR.utils import rmixGaussian
from CTDGMR.distance import GMM_L2
import simulation as S  # reuse robustmedian + generate_attack

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_DIR = os.path.join(HERE, "generated_pop", "true_param")
OUT_CSV = os.path.join(HERE, "output", "distance_concentration.csv")
FIGDIR = os.path.join(HERE, "output", "figures")

CONFIGS = [(5, 2), (5, 10), (10, 10)]  # (K, d), increasing #parameters
OVERLAP = 0.3
M = 100
N_LOCAL = 10000
ALPHA = 0.3
ATTACKS = {1: "mean", 2: "cov", 3: "weight"}


def load_truth(K, d, seed):
    def p(kind):
        return os.path.join(PARAM_DIR, "%s_seed_%d_ncomp_%d_d_%d_maxoverlap_%s.txt"
                            % (kind, seed, K, d, OVERLAP))
    w = np.loadtxt(p("weights"))
    mu = np.loadtxt(p("means")).reshape((-1, d))
    cov = np.loadtxt(p("covs")).T.reshape((-1, d, d))
    return mu, cov, w


def fit_locals(mu, cov, w, K, d, seed):
    total = N_LOCAL * M
    X, _ = rmixGaussian(mu, cov, w, total, seed)
    rng = np.random.RandomState(seed)
    X = X[rng.permutation(total)]
    prec = np.stack([np.linalg.inv(c) for c in cov])
    means = [None] * M; covs = [None] * M; weights = [None] * M
    for i in range(M):
        loc = X[i * N_LOCAL:(i + 1) * N_LOCAL]
        g = pMLEGMM(n_components=K, cov_reg=1.0 / np.sqrt(loc.shape[0]),
                    covariance_type="full", max_iter=10000, n_init=1, tol=1e-6,
                    weights_init=w, means_init=mu, precisions_init=prec,
                    random_state=0)
        g.fit(loc)
        means[i], covs[i], weights[i] = g.means_, g.covariances_, g.weights_
    return means, covs, weights


def run(seeds):
    rows = []
    for (K, d) in CONFIGS:
        for seed in seeds:
            mu, cov, w = load_truth(K, d, seed)
            means, covs, weights = fit_locals(mu, cov, w, K, d, seed)
            for am, aname in ATTACKS.items():
                a_means, a_covs, a_weights, byz_idx, _ = S.generate_attack(
                    means, covs, weights, ALPHA, am, K, d, M, seed,
                    failure_type="machine")
                failed = set(int(x) for x in byz_idx[0])  # machine-level: same set per comp
                which, pw = S.robustmedian(a_means, a_covs, a_weights,
                                           ground_distance="L2", coverage_ratio=0.5)
                # distance from COAT centre to each (attacked) local density
                dist = pw[which]  # robustmedian already computed pairwise L2 distances
                for i in range(M):
                    rows.append(dict(K=K, d=d, nparam=K * (1 + d + d * d),
                                     seed=seed, machine=i, attack=aname,
                                     is_failed=(i in failed), distance=float(dist[i])))
            print("done (K=%d,d=%d) seed=%d" % (K, d, seed), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        df = pd.concat([old, df]).drop_duplicates(
            subset=["K", "d", "seed", "machine", "attack"], keep="last")
    df.to_csv(OUT_CSV, index=False)
    print("wrote %d rows -> %s" % (len(df), OUT_CSV))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(OUT_CSV)
    attacks = ["mean", "cov", "weight"]
    fig, axes = plt.subplots(len(attacks), len(CONFIGS),
                             figsize=(3.2 * len(CONFIGS), 2.6 * len(attacks)),
                             squeeze=False)
    for i, atk in enumerate(attacks):
        for j, (K, d) in enumerate(CONFIGS):
            ax = axes[i][j]
            sub = df[(df.attack == atk) & (df.K == K) & (df.d == d)]
            ff = sub[~sub.is_failed]["distance"]
            fl = sub[sub.is_failed]["distance"]
            if len(sub):
                bins = np.linspace(0, np.quantile(sub["distance"], 0.99) + 1e-9, 40)
                ax.hist(ff, bins=bins, alpha=0.6, label="failure-free", color="tab:blue", density=True)
                ax.hist(fl, bins=bins, alpha=0.6, label="failed", color="tab:red", density=True)
            if i == 0:
                ax.set_title("K=%d, d=%d" % (K, d), fontsize=9)
            if j == 0:
                ax.set_ylabel("%s\ndensity" % atk, fontsize=9)
            if i == len(attacks) - 1:
                ax.set_xlabel("L2 distance to COAT", fontsize=9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Fig 7: distance-to-COAT concentration (m=100, n=1e4, alpha=0.3)",
                 y=1.06, fontsize=11)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig7_distance_concentration.png"),
                dpi=140, bbox_inches="tight")
    print("wrote fig7_distance_concentration.png")


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-50")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot()
    else:
        run(parse_seeds(args.seeds))
