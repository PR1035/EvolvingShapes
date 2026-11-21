import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import core

core.N_POLY = 200
core.N_CHANGES = 2
random.seed(7)
np.random.seed(7)

ITERS = 200_000
os.makedirs("results/final", exist_ok=True)

snap = set(np.unique(np.logspace(1, np.log10(ITERS), 60).astype(int)))


def to_frame(ind):
    img = core.render(ind).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


best = core.random_individual()
best_fit = core.fitness(best)
history = [best_fit]
frames = []

for g in range(1, ITERS + 1):
    child = core.mutate(best)
    f = core.fitness(child)
    if f <= best_fit:
        best, best_fit = child, f
    history.append(best_fit)
    if g in snap:
        frames.append(to_frame(best))
    if g % 10000 == 0:
        print(f"iter {g:7d}  error {best_fit:.1f}")

frames += [frames[-1]] * 8
frames[0].save("results/final/evolution.gif", save_all=True,
               append_images=frames[1:], duration=120, loop=0)

target = cv2.resize(core.TARGET.clip(0, 255).astype(np.uint8), (640, 360),
                    interpolation=cv2.INTER_LINEAR)
final = cv2.cvtColor(np.array(to_frame(best)), cv2.COLOR_RGB2BGR)
cv2.imwrite("results/final/side_by_side.png", np.hstack([target, final]))

plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Error (MSE)")
plt.yscale("log")
plt.title("Hill climbing, 200 triangles")
plt.savefig("results/final/final_curve.png")
print("Final error:", best_fit)