import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import core
import ga

core.N_POLY = 200
core.N_CHANGES = 2
ga.CROSSOVER = "uniform"
POP_SIZE = 10
ELITE = 2
EVALS = 50_000
GENERATIONS = (EVALS - POP_SIZE) // (POP_SIZE - ELITE)

random.seed(7)
np.random.seed(7)
os.makedirs("results/final_ga", exist_ok=True)

snap = set(np.unique(np.logspace(0, np.log10(GENERATIONS), 60).astype(int)))


def to_frame(ind):
    img = core.render(ind).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


scored = []
for _ in range(POP_SIZE):
    ind = core.random_individual()
    scored.append((ind, core.fitness(ind)))
evals = POP_SIZE

history = []
frames = []

for g in range(1, GENERATIONS + 1):
    scored.sort(key=lambda s: s[1])
    new = scored[:ELITE]
    while len(new) < POP_SIZE:
        p1 = ga.tournament(scored)
        p2 = ga.tournament(scored)
        child = core.mutate(ga.crossover(p1, p2))
        new.append((child, core.fitness(child)))
        evals += 1
    scored = new

    best_ind, best_fit = min(scored, key=lambda s: s[1])
    history.append((evals, best_fit))

    if g in snap:
        frames.append(to_frame(best_ind))
    if g % 1000 == 0:
        print(f"gen {g:6d}  evals {evals:7d}  error {best_fit:.1f}")

frames += [frames[-1]] * 8
frames[0].save("results/final_ga/evolution_ga.gif", save_all=True,
               append_images=frames[1:], duration=120, loop=0)

target = cv2.resize(core.TARGET.clip(0, 255).astype(np.uint8), (640, 360),
                    interpolation=cv2.INTER_LINEAR)
final = cv2.cvtColor(np.array(to_frame(best_ind)), cv2.COLOR_RGB2BGR)
cv2.imwrite("results/final_ga/side_by_side_ga.png", np.hstack([target, final]))

xs = [e for e, _ in history]
ys = [f for _, f in history]
plt.plot(xs, ys)
plt.xlabel("Fitness evaluations")
plt.ylabel("Best error (MSE)")
plt.yscale("log")
plt.title(f"GA (uniform crossover, population {POP_SIZE}, 200 triangles)")
plt.savefig("results/final_ga/final_curve_ga.png")
np.save("results/final_ga/ga_history.npy", np.array(history))
print("Final error:", best_fit)