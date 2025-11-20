

import numpy as np
import matplotlib.pyplot as plt
import random

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

def plotBox(x, y, z, dx, dy, dz, axis, colour = None):
    verts = [(x, y, z), (x, y + dy, z), (x + dx, y + dy, z), (x + dx, y, z),                                    # Coordinates of the 8 corners of the cuboid
             (x, y, z + dz), (x, y + dy, z + dz), (x + dx, y + dy, z + dz), (x + dx, y, z + dz)]
    
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],                                                          # Defines the 6 faces of the cuboid
             [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
    
    poly3d = [[verts[vert_id] for vert_id in face] for face in faces]                                           # Lists the areas as polygons for matplotlib
    
    x, y, z = zip(*verts)
    
    axis.add_collection3d(Poly3DCollection(poly3d, facecolors = colour, linewidths = 1, edgecolors = 'black'))  # Adds the polygon to the 3D axis object


def outputResult(packing_result, bin_size = 100):
    '''
    packing_result: [box1 ,box2, box3, ...]
    box1:[l, w, h, p_x, p_y, p_z]
    '''
    # Check, whetcher boxes are overlapping
    box_num = len(packing_result)                                                                                # All the info of each box
    for i in range(box_num - 1):
        for j in range(i + 1, box_num):
            box_i = np.array(packing_result[i])                                                                  # One box has [length, width, height, x_coordinate, y_coordinate, z_coordinate]
            box_j = np.array(packing_result[j])
            box_i_centre = box_i[0:3] / 2 + box_i[3:]
            box_j_centre = box_j[0:3] / 2 + box_j[3:]
            is_overlap = (np.abs(box_j_centre - box_i_centre) < (box_i[0:3] + box_j[0:3]) / 2).all()            # box_i_centre and box_j_centre are the X/Y/Z coordinates of the box centre point. 
                                                                                                                # These are then used for the overlap check. 
                                                                                                                # Overlap = check whether the distance between the centre points is smaller the sum of half the edge lengths.
            if is_overlap:
                raise Exception(
                        f"Items are overlapping. Please check the outcome.\n"
                        f"Overlapping boxes:\n"
                        f"Box 1 coordinates({box_i[3]}, {box_i[4]}, {box_i[5]}), with size {box_i[0]}, {box_i[1]}, {box_i[2]})\n"
                        f"Box 2 coordinates({box_j[3]}, {box_j[4]}, {box_j[5]}), with size {box_j[0]}, {box_j[1]}, {box_j[2]})."
                    )

    # Calculate the use_ratio
    packing_array = np.array(packing_result)
    box_height_coordinate = packing_array[:, 2] + packing_array[:, 5]
    max_height = np.max(box_height_coordinate)                                                                  # Find the highest point of all boxes. This will is needed to calculate the bin utilisation in the z direction.
    use_ratio = packing_array[:, :3].prod(1).sum() / (bin_size * bin_size * max_height)                         # packing_array[:, :3].prod(1).sum(): Sum of products of L/W/H of each box
    print(use_ratio)

    fig = plt.figure()

    ax = Axes3D(fig)
    ax.set_xlim(0, bin_size)
    ax.set_ylim(0, bin_size)
    ax.set_zlim(0, max_height)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("uR:%s%%" % (use_ratio.item() * 100), fontproperties="SimHei")
    for box in packing_result:
        color = (random.random(), random.random(), random.random(), 1)
        length, width, height, start_l, start_w, start_h = box
        plotBox(start_l, start_w, start_h,
                length, width, height, ax, color)

    plt.show()