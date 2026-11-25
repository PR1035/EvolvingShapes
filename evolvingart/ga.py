import random
import copy
import numpy as np
import cv2
import core

CROSSOVER = "uniform"


def crossover(a, b):
    if CROSSOVER == "none":
        return copy.deepcopy(a)
    if CROSSOVER == "uniform":
        return [copy.deepcopy(a[i] if random.random() < 0.5 else b[i])
                for i in range(core.N_POLY)]
    cut = random.randint(1, core.N_POLY - 1)
    return copy.deepcopy(a[:cut]) + copy.deepcopy(b[cut:])


def tournament(scored, k=3):
    contestants = random.sample(scored, k)
    return min(contestants, key=lambda s: s[1])[0]


def run_ga(pop_size=40, generations=500, elite=2, save_every=50, callback=None):
    scored = []
    for _ in range(pop_size):
        ind = core.random_individual()
        scored.append((ind, core.fitness(ind)))
    evals = pop_size
    history = []

    for g in range(1, generations + 1):
        scored.sort(key=lambda s: s[1])
        new = scored[:elite]
        while len(new) < pop_size:
            p1 = tournament(scored)
            p2 = tournament(scored)
            child = core.mutate(crossover(p1, p2))
            new.append((child, core.fitness(child)))
            evals += 1
        scored = new
        best_ind, best_fit = min(scored, key=lambda s: s[1])
        history.append((evals, best_fit))

        if callback is not None:
            callback(g, evals, best_ind, best_fit)

        if g % save_every == 0:
            cv2.imwrite(f"results/ga_{g:05d}.png", core.render(best_ind))
            print(f"gen {g:4d}  evals {evals:6d}  error {best_fit:.1f}")

    return best_ind, history
