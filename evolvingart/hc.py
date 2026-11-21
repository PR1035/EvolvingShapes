import cv2
import core


def hill_climb(generations=20000, save_every=1000):
    best = core.random_individual()
    best_fit = core.fitness(best)
    history = [best_fit]

    for g in range(1, generations + 1):
        child = core.mutate(best)
        f = core.fitness(child)
        if f <= best_fit:
            best, best_fit = child, f
        history.append(best_fit)

        if g % save_every == 0:
            cv2.imwrite(f"results/hc_{g:06d}.png", core.render(best))
            print(f"gen {g:6d}  error {best_fit:.1f}")

    return best, history