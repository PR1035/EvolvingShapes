import numpy as np
import cv2
import random
import copy
import matplotlib.pyplot as plt


TARGET = cv2.imread("target.png")            
W, H = 160, 90                               
TARGET = cv2.resize(TARGET, (W, H)).astype(np.float32)

cv2.imwrite("results/target_small.png", TARGET)
print("Target shape:", TARGET.shape)       


N_POLY = 100       
N_VERTS = 3        

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

def mutate(ind, n_changes=2):
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

def hill_climb(generations=20000, save_every=1000):
    best = random_individual()
    best_fit = fitness(best)
    history = [best_fit]

    for g in range(1, generations + 1):
        child = mutate(best)
        f = fitness(child)
        if f <= best_fit:                  
            best, best_fit = child, f
        history.append(best_fit)

        if g % save_every == 0:
            cv2.imwrite(f"results/hc_{g:06d}.png", render(best))
            print(f"gen {g:6d}  error {best_fit:.1f}")

    return best, history

if __name__ == "__main__":
    best, history = hill_climb()
    cv2.imwrite("results/hc_final.png", render(best))

    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Error (MSE)")
    plt.yscale("log")
    plt.savefig("results/hc_curve.png")
    print("Done. Final error:", history[-1])