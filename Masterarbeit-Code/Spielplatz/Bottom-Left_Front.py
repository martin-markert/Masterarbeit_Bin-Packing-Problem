import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------
# Parameter
# -----------------------------
W, H, D = 100, 100, 100  # Containergröße
num_boxes = 10        # Anzahl Quader
max_w, max_h, max_d = 50, 50, 5

# Zufällige Quader (Breite, Höhe, Tiefe)
boxes = [(random.randint(1, max_w), random.randint(1, max_h), random.randint(1, max_d))
         for _ in range(num_boxes)]

# -----------------------------
# Hilfsfunktionen
# -----------------------------
def fits(container, box, pos):
    x, y, z = pos
    w, h, d = box
    if x + w > W or y + h > H or z + d > D:
        return False
    for (bx, by, bz, bw, bh, bd) in container:
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True

def bottom_left_front_place(container, box):
    best = None
    for z in range(D - box[2] + 1):
        for y in range(H - box[1] + 1):
            for x in range(W - box[0] + 1):
                if fits(container, box, (x, y, z)):
                    if (best is None or
                        z < best[2] or
                        (z == best[2] and y < best[1]) or
                        (z == best[2] and y == best[1] and x < best[0])):
                        best = (x, y, z)
    return best

# -----------------------------
# Quader platzieren
# -----------------------------
containers = []
box_counter = 1
box_positions = []

for box in boxes:
    placed = False
    for c_idx, container in enumerate(containers):
        pos = bottom_left_front_place(container, box)
        if pos:
            container.append((*pos, *box))
            box_positions.append((c_idx, *pos, *box, box_counter))
            box_counter += 1
            placed = True
            break
    if not placed:
        new_container = []
        pos = bottom_left_front_place(new_container, box)
        if pos:
            new_container.append((*pos, *box))
            containers.append(new_container)
            box_positions.append((len(containers)-1, *pos, *box, box_counter))
            box_counter += 1

# -----------------------------
# Visualisierung (korrekte Depth-Sortierung)
# -----------------------------
fig = plt.figure(figsize=(6*len(containers), 6))

for c_idx in range(len(containers)):
    ax = fig.add_subplot(1, len(containers), c_idx+1, projection='3d')
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_zlim([0, D])
    ax.set_title(f'Container {c_idx+1}')

    # Sortiere nach z-Tiefe (hinten zuerst)
    sorted_boxes = sorted(
        [(x, y, z, w, h, d, num) 
         for (cont_idx, x, y, z, w, h, d, num) in box_positions if cont_idx == c_idx],
        key=lambda b: b[2], reverse=True
    )

    for (x, y, z, w, h, d, num) in sorted_boxes:
        color = (random.random(), random.random(), random.random())
        verts = [
            [(x,y,z),(x+w,y,z),(x+w,y+h,z),(x,y+h,z)],
            [(x,y,z+d),(x+w,y,z+d),(x+w,y+h,z+d),(x,y+h,z+d)],
            [(x,y,z),(x,y+h,z),(x,y+h,z+d),(x,y,z+d)],
            [(x+w,y,z),(x+w,y+h,z),(x+w,y+h,z+d),(x+w,y,z+d)],
            [(x,y,z),(x+w,y,z),(x+w,y,z+d),(x,y,z+d)],
            [(x,y+h,z),(x+w,y+h,z),(x+w,y+h,z+d),(x,y+h,z+d)],
        ]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=color, linewidths=1, edgecolors='black', alpha=1))
        ax.text(x + w/2, y + h/2, z + d/2, str(num), color='white', ha='center', va='center')

plt.tight_layout()
plt.show()
