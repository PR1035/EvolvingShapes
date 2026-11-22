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
EVALS = 200_000
GENERATIONS = (EVALS - POP_SIZE) // (POP_SIZE - ELITE)
MILESTONES = [1000, 5000, 10000, 25000, 50000, 100000]

random.seed(7)
np.random.seed(7)
os.makedirs("results/final_ga/progress", exist_ok=True)

snap = set(np.unique(np.logspace(0, np.log10(GENERATIONS), 60).astype(int)))
milestones_left = [m for m in MILESTONES if m < EVALS]


def to_frame(ind):
    img = core.render(ind).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def labeled(img, text):
    out = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


scored = []
for _ in range(POP_SIZE):
    ind = core.random_individual()
    scored.append((ind, core.fitness(ind)))
evals = POP_SIZE

history = []
frames = []
panels = []

target = cv2.resize(core.TARGET.clip(0, 255).astype(np.uint8), (640, 360),
                    interpolation=cv2.INTER_LINEAR)
panels.append(labeled(target, "Target"))

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

    if milestones_left and evals >= milestones_left[0]:
        m = milestones_left.pop(0)
        frame = to_frame(best_ind)
        frame.save(f"results/final_ga/progress/ga_{m:07d}_evals.png")
        bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        panels.append(labeled(bgr, f"{m} evals, err {best_fit:.0f}"))

    if g % 500 == 0:
        print(f"gen {g:6d}  evals {evals:7d}  error {best_fit:.1f}")

final_frame = to_frame(best_ind)
final_frame.save("results/final_ga/progress/ga_final.png")
final_bgr = cv2.cvtColor(np.array(final_frame), cv2.COLOR_RGB2BGR)
panels.append(labeled(final_bgr, f"final, err {best_fit:.0f}"))

cols = 4
while len(panels) % cols:
    panels.append(np.zeros_like(panels[0]))
rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
cv2.imwrite("results/final_ga/progress_grid.png", np.vstack(rows))

frames += [frames[-1]] * 8
frames[0].save("results/final_ga/evolution_ga.gif", save_all=True,
               append_images=frames[1:], duration=120, loop=0)

cv2.imwrite("results/final_ga/side_by_side_ga.png", np.hstack([target, final_bgr]))

xs = [e for e, _ in history]
ys = [f for _, f in history]
plt.plot(xs, ys)
plt.xlabel("Fitness evaluations")
plt.ylabel("Best error (MSE)")
plt.yscale("log")
plt.title(f"GA (uniform crossover, population {POP_SIZE}, 200 triangles)")
plt.savefig("results/final_ga/final_curve_ga.png")
print("Final error:", best_fit)
