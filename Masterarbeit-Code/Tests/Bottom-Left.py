import random
import matplotlib
matplotlib.use('TkAgg')  # interaktives GUI-Backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------
# Parameter
# -----------------------------
W, H = 10, 10       # Containerbreite und -höhe
num_rects = 15      # Anzahl Rechtecke
max_w, max_h = 6, 6 # maximale Breite/Höhe der Rechtecke

# Zufällige Rechtecke (Breite, Höhe)
rectangles = [(random.randint(1, max_w), random.randint(1, max_h))
              for _ in range(num_rects)]

# -----------------------------
# Bottom-Left-Hilfsfunktionen
# -----------------------------
def fits(container, rect, pos):
    x, y = pos
    w, h = rect
    if x + w > W or y + h > H:
        return False
    for (rx, ry, rw, rh) in container:
        if not (x + w <= rx or rx + rw <= x or
                y + h <= ry or ry + rh <= y):
            return False
    return True

def bottom_left_place(container, rect):
    best = None
    for y in range(H - rect[1] + 1):
        for x in range(W - rect[0] + 1):
            if fits(container, rect, (x, y)):
                if best is None or y < best[1] or (y == best[1] and x < best[0]):
                    best = (x, y)
    return best

# -----------------------------
# Rechtecke platzieren
# -----------------------------
containers = []
rect_counter = 1
rect_positions = []

for rect in rectangles:
    placed = False
    for c_idx, container in enumerate(containers):
        pos = bottom_left_place(container, rect)
        if pos:
            container.append((*pos, *rect))
            rect_positions.append((c_idx, *pos, *rect, rect_counter))
            rect_counter += 1
            placed = True
            break
    if not placed:
        new_container = []
        pos = bottom_left_place(new_container, rect)
        if pos:
            new_container.append((*pos, *rect))
            containers.append(new_container)
            rect_positions.append((len(containers)-1, *pos, *rect, rect_counter))
            rect_counter += 1

# -----------------------------
# Visualisierung
# -----------------------------
fig, axes = plt.subplots(1, len(containers), figsize=(5*len(containers), 5))
if len(containers) == 1:
    axes = [axes]

for ax, c_idx in zip(axes, range(len(containers))):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_title(f'Container {c_idx+1}')

    for (cont_idx, x, y, w, h, num) in rect_positions:
        if cont_idx != c_idx:
            continue
        color = (random.random(), random.random(), random.random())
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, str(num), ha='center', va='center', color='white', fontsize=12)

plt.tight_layout()
plt.show()
