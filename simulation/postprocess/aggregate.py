"""Aggregate simulation stage-2 pickles into one tidy long-format table.

Scans output/save_data/**/*_failurerate_*_rhomode_*.pickle, parses the config knobs
from each filename (and overlap from the parent folder), extracts every scalar metric
(W1 to truth, ARI, log-likelihood) for every method, and writes a long CSV with one
row per (config, method, rho, metric, seed).

Methods: coat, dfmr (per rho), trim (trimmed k-barycentre), gmr (= "Vanilla" in the
paper), oracle. Run from simulation/:  python postprocess/aggregate.py
"""

import os
import re
import glob
import pickle
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # simulation/
SAVE_ROOT = os.path.join(HERE, "output", "save_data")

FNAME_RE = re.compile(
    r"case_(?P<seed>\d+)_nsplit_(?P<m>\d+)_ncomp_(?P<K>\d+)_d_(?P<d>\d+)_ss_(?P<ss>\d+)"
    r"_failurerate_(?P<fr>[\d.]+)_attackmode_(?P<am>\d+)_failuretype_(?P<ft>[a-z]+)"
    r"_rhomode_(?P<rm>[a-z0-9]+)\.pickle$")

OVERLAP_RE = re.compile(r"overlap_(?P<ov>[\d.]+)")
METHODS = ("coat", "dfmr", "trim", "gmr", "oracle")
ARI_LL_RE = re.compile(r"^(coat|dfmr|trim|gmr|oracle)_(ARI|ll)(?:_(.+))?$")
ATTACK_NAME = {1: "mean", 2: "cov", 3: "weight"}


def classify(key):
    """Return (method, metric, rho) for a scalar-metric key, else None."""
    if "2true_W1" in key:
        pre, post = key.split("2true_W1")
        method = pre.rstrip("_")
        if method not in METHODS:
            return None
        rho = post.lstrip("_") or None
        return method, "W1", rho
    m = ARI_LL_RE.match(key)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "output", "aggregated_sim.csv"))
    args = ap.parse_args()

    files = glob.glob(os.path.join(SAVE_ROOT, "**", "*_failurerate_*_rhomode_*.pickle"),
                      recursive=True)
    print("found %d stage-2 pickles" % len(files))

    rows = []
    for f in files:
        base = os.path.basename(f)
        fm = FNAME_RE.search(base)
        if not fm:
            continue
        ovm = OVERLAP_RE.search(f.replace("\\", "/"))
        overlap = float(ovm.group("ov")) if ovm else np.nan
        seed = int(fm.group("seed"))
        m = int(fm.group("m"))
        ss = int(fm.group("ss"))
        fr = float(fm.group("fr"))
        am = int(fm.group("am"))
        rm = fm.group("rm")

        try:
            d = pickle.load(open(f, "rb"))
        except Exception as e:
            print("skip unreadable", base, e)
            continue

        for key, val in d.items():
            if not isinstance(val, (int, float, np.floating, np.integer)):
                continue
            c = classify(key)
            if c is None:
                continue
            method, metric, rho = c
            rows.append(dict(
                overlap=overlap, m=m, ss=ss, N=ss, n=ss // m,
                attack_mode=am, attack=ATTACK_NAME.get(am, str(am)),
                failure_rate=fr, rho_mode=rm, method=method,
                rho=(float(rho) if rho is not None else np.nan),
                metric=metric, value=float(val), seed=seed))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("wrote %d rows -> %s" % (len(df), args.out))
    if len(df):
        print("configs: overlaps=%s | m=%s | ss=%s | attacks=%s | rho_modes=%s | seeds=%d"
              % (sorted(df.overlap.unique()), sorted(df.m.unique()),
                 sorted(df.ss.unique()), sorted(df.attack.unique()),
                 sorted(df.rho_mode.unique()), df.seed.nunique()))
    return df


if __name__ == "__main__":
    main()
