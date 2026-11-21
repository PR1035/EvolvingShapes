import numpy as np
import cv2
import random
import copy

W, H = 160, 90
TARGET = cv2.imread("target.png")
TARGET = cv2.resize(TARGET, (W, H)).astype(np.float32)

N_POLY = 100
N_VERTS = 3
N_CHANGES = 2


def random_polygon():
    return {
        "pts": np.random.randint(0, [W, H], size=(N_VERTS, 2)),
        "color": np.random.randint(0, 256, 3).astype(np.float32),
        "alpha": random.uniform(0.1, 0.5),
    }


def random_individual():
    return [random_polygon() for _ in range(N_POLY)]


def render(ind):
    canvas = np.zeros((H, W, 3), np.float32)
    for p in ind:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [p["pts"].astype(np.int32)], p["color"].tolist())
        canvas = cv2.addWeighted(overlay, p["alpha"], canvas, 1 - p["alpha"], 0)
    return canvas


def fitness(ind):
    diff = render(ind) - TARGET
    return np.mean(diff ** 2)


def mutate(ind, n_changes=None):
    if n_changes is None:
        n_changes = N_CHANGES
    child = copy.deepcopy(ind)
    for _ in range(n_changes):
        p = random.choice(child)
        kind = random.choice(["move", "color", "alpha"])
        if kind == "move":
            p["pts"] = np.clip(p["pts"] + np.random.randint(-10, 11, p["pts"].shape),
                               0, [W - 1, H - 1])
        elif kind == "color":
            p["color"] = np.clip(p["color"] + np.random.randint(-30, 31, 3), 0, 255).astype(np.float32)
        else:
            p["alpha"] = float(np.clip(p["alpha"] + random.uniform(-0.1, 0.1), 0.05, 0.9))
    return child