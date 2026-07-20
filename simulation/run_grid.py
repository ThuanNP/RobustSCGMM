"""Resumable, parallel driver for the paper's simulation grid (Figures 2-6).

Reproduces Zhang et al. (2026), Section 5. For every (seed, overlap, m, ss) unit it
runs stage 1 (`global_local.py`, fit m local pMLE GMMs) if its pickle is missing,
then runs stage 2 (`simulation.py`) for each required (attack_mode, rho_mode). Every
step is skipped when its output pickle already exists, so the driver is fully
resumable -- kill it and rerun and it picks up where it left off.

The two stages are invoked as subprocesses (they rely on module-level globals set in
their __main__ blocks) using the same Python interpreter that runs this driver
(point it at the project venv). Units run concurrently across a thread pool; the
threads block on subprocesses so the GIL is not a bottleneck.

Config grid (all failure_type=machine; attacks 1=mean, 2=cov, 3=weight):

  Fig 3  vary N,alpha : overlap 0.3, m=100, ss in {5e5,1e6,2e6}          rho=main
  Fig 4  vary m,n=5k  : overlap 0.3, (m,ss) {(20,1e5),(50,2.5e5),(100,5e5)} rho=main
  Fig 5  vary m,N=5e5 : overlap 0.3, ss=5e5, m in {20,50,100}            rho=main
  Fig 6  vary overlap : overlap {0.1,0.2,0.3}, m=100, ss=5e5            rho=main
  Fig 2  rho sweep    : overlap 0.3, (m,ss) {(20,2e5),(50,5e5),(100,1e6)} rho=fig2

Usage (from simulation/):
  python run_grid.py --seeds 1-5 --figures fig4 --workers 8      # small validation
  python run_grid.py                                             # full grid, seeds 1-300
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # run children with the same interpreter (the venv)

NCOMP, DIM = 5, 10  # fixed by the paper's Gaussian-mixture setting
ATTACKS = [1, 2, 3]  # mean, cov, weight
FAILURE_RATES = [0.1, 0.2, 0.3, 0.4]  # looped inside simulation.py

# --- figure -> list of (overlap, m, ss, [attacks], rho_mode) -----------------
FIGURE_CONFIGS = {
    "fig3": [(0.3, 100, ss, ATTACKS, "main") for ss in (500000, 1000000, 2000000)],
    "fig4": [(0.3, m, ss, ATTACKS, "main")
             for (m, ss) in ((20, 100000), (50, 250000), (100, 500000))],
    "fig5": [(0.3, m, 500000, ATTACKS, "main") for m in (20, 50, 100)],
    "fig6": [(ov, 100, 500000, ATTACKS, "main") for ov in (0.1, 0.2, 0.3)],
    # Fig 2 rho-sensitivity uses n=1e4 so ss=m*1e4; only alpha=0.1 is plotted but
    # simulation.py computes all failure rates anyway.
    "fig2": [(0.3, m, m * 10000, ATTACKS, "fig2") for m in (20, 50, 100)],
}


def stage1_pickle(overlap, m, ss, seed):
    folder = os.path.join(HERE, "output", "save_data", "ss_%d" % ss,
                          "overlap_%s" % overlap)
    fname = "case_%d_nsplit_%d_ncomp_%d_d_%d_ss_%d.pickle" % (seed, m, NCOMP, DIM, ss)
    return os.path.join(folder, fname)


def stage2_pickles(overlap, m, ss, seed, attack, rho_mode):
    folder = os.path.join(HERE, "output", "save_data", "ss_%d" % ss,
                          "overlap_%s" % overlap)
    out = []
    for fr in FAILURE_RATES:
        fname = ("case_%d_nsplit_%d_ncomp_%d_d_%d_ss_%d_failurerate_%s"
                 "_attackmode_%d_failuretype_machine_rhomode_%s.pickle"
                 % (seed, m, NCOMP, DIM, ss, fr, attack, rho_mode))
        out.append(os.path.join(folder, fname))
    return out


def population_exists(overlap, seed):
    base = os.path.join(HERE, "generated_pop", "true_param")
    suffix = "_seed_%d_ncomp_%d_d_%d_maxoverlap_%s.txt" % (seed, NCOMP, DIM, overlap)
    return all(os.path.exists(os.path.join(base, k + suffix)) for k in
               ("weights", "means", "covs"))


def run_child(script, args):
    cmd = [PY, os.path.join(HERE, script)] + [str(a) for a in args]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stdout[-2000:] + "\n" + r.stderr[-2000:])
    return True, ""


def do_unit(unit):
    """unit = (seed, overlap, m, ss, subtasks) where subtasks=[(attack,rho_mode),...]."""
    seed, overlap, m, ss, subtasks = unit
    log = []

    if not population_exists(overlap, seed):
        return "MISSING_POP", "no population for seed=%d overlap=%s (run generate_all.R)" % (seed, overlap), unit

    # stage 1 (shared by all subtasks of this unit)
    s1 = stage1_pickle(overlap, m, ss, seed)
    if not os.path.exists(s1):
        ok, err = run_child("global_local.py",
                            ["--ss", ss, "--seed", seed, "--overlap", overlap,
                             "--n_split", m])
        if not ok:
            return "FAIL_STAGE1", err, unit
        log.append("stage1")

    # stage 2 per (attack, rho_mode)
    for attack, rho_mode in subtasks:
        outs = stage2_pickles(overlap, m, ss, seed, attack, rho_mode)
        if all(os.path.exists(o) for o in outs):
            continue
        ok, err = run_child("simulation.py",
                            ["--ss", ss, "--seed", seed, "--overlap", overlap,
                             "--n_split", m, "--attack_mode", attack,
                             "--failure_type", "machine", "--rho_mode", rho_mode])
        if not ok:
            return "FAIL_STAGE2", "attack=%d rho=%s\n%s" % (attack, rho_mode, err), unit
        log.append("a%d/%s" % (attack, rho_mode))

    return ("DONE" if log else "SKIP"), ",".join(log), unit


def build_units(figures, seeds):
    """Collapse the selected figures into per-(seed,overlap,m,ss) units, unioning
    the (attack, rho_mode) subtasks so shared stage-1 fits run only once."""
    # (overlap,m,ss) -> set of (attack,rho_mode)
    base = {}
    for fig in figures:
        for (overlap, m, ss, attacks, rho_mode) in FIGURE_CONFIGS[fig]:
            key = (overlap, m, ss)
            sub = base.setdefault(key, set())
            for a in attacks:
                sub.add((a, rho_mode))
    units = []
    for seed in seeds:
        for (overlap, m, ss), sub in base.items():
            units.append((seed, overlap, m, ss, sorted(sub)))
    return units


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description="Simulation grid driver (Figs 2-6)")
    ap.add_argument("--seeds", default="1-300", help="e.g. 1-300 or 1,2,5 or 1-5,10")
    ap.add_argument("--figures", default="fig2,fig3,fig4,fig5,fig6",
                    help="comma list from fig2,fig3,fig4,fig5,fig6")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    figures = [f.strip() for f in args.figures.split(",") if f.strip()]
    seeds = parse_seeds(args.seeds)
    units = build_units(figures, seeds)

    print("figures=%s | seeds=%d (%s..%s) | units=%d | workers=%d"
          % (figures, len(seeds), seeds[0], seeds[-1], len(units), args.workers),
          flush=True)

    t0 = time.time()
    counts = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_unit, u): u for u in units}
        for fut in as_completed(futs):
            status, detail, unit = fut.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if status.startswith("FAIL") or status == "MISSING_POP":
                seed, overlap, m, ss, _ = unit
                print("[%s] seed=%d overlap=%s m=%d ss=%d :: %s"
                      % (status, seed, overlap, m, ss, detail[:400]), flush=True)
            if done % 20 == 0 or done == len(units):
                el = time.time() - t0
                print("progress %d/%d | %.0fs | %s"
                      % (done, len(units), el, counts), flush=True)

    print("FINISHED in %.0fs | %s" % (time.time() - t0, counts), flush=True)


if __name__ == "__main__":
    main()
