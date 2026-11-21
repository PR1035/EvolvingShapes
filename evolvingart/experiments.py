import sys
import csv
import random
import numpy as np
import core
import ga
import hc

BUDGET = 20000


def seed_all(s):
    random.seed(s)
    np.random.seed(s)


def run_hc(seed):
    seed_all(seed)
    _, hist = hc.hill_climb(generations=BUDGET, save_every=10**9)
    return hist[-1]


def run_ga(seed, pop_size=40, elite=2):
    seed_all(seed)
    gens = (BUDGET - pop_size) // (pop_size - elite)
    _, hist = ga.run_ga(pop_size=pop_size, generations=gens,
                        elite=elite, save_every=10**9)
    return hist[-1][1]


def experiment(name, fn, seeds=(1, 2, 3), **kw):
    errs = [fn(s, **kw) for s in seeds]
    print(f"{name:35s} mean {np.mean(errs):8.1f}  std {np.std(errs):7.1f}")
    with open("results/experiments.csv", "a", newline="") as f:
        csv.writer(f).writerow([name, np.mean(errs), np.std(errs)] + errs)


def group_A():
    experiment("Hill climbing", run_hc)
    ga.CROSSOVER = "single"
    experiment("GA single-point", run_ga)
    ga.CROSSOVER = "uniform"
    experiment("GA uniform", run_ga)
    ga.CROSSOVER = "none"
    experiment("GA no crossover", run_ga)
    ga.CROSSOVER = "uniform"


def group_B():
    for n in (50, 100, 200):
        core.N_POLY = n
        experiment(f"HC polygons={n}", run_hc)
    core.N_POLY = 100


def group_C():
    for k in (1, 2, 5):
        core.N_CHANGES = k
        experiment(f"HC n_changes={k}", run_hc)
    core.N_CHANGES = 2


def group_D():
    ga.CROSSOVER = "uniform"
    for ps in (10, 40, 100):
        experiment(f"GA uniform pop={ps}", run_ga, pop_size=ps)


if __name__ == "__main__":
    {"A": group_A, "B": group_B, "C": group_C, "D": group_D}[sys.argv[1]]()