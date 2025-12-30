from parameters import Parameters
from environment import Environment
from explore_environment import solve_problem
from network import Actor

import numpy as np
import plotly.graph_objects as go
import random
import torch
import time

import multiprocessing as mp
from multiprocessing import Queue
from pathlib import Path

params = Parameters()

def cube_trace(x, y, z, length, width, height, scale, colour):
    edges = go.Scatter3d(                                                                                       # Some edges have to be drawn twice at a cube is no Euler circle or Euler path
                                                                                                                # (x[i]y[i]z[i]) is a coordinate, (x[i+1]y[i+1]z[i+1]) the next. A line is drawn between them and so on, making the edges of the cuboids
        x = [x, x + length, x + length, x,         x, x,          x + length, x + length, x,          x,          x + length, x + length, x + length, x + length, x,          x        ],
        y = [y, y,          y + width,  y + width, y, y,          y,          y + width,  y + width,  y,          y,          y,          y + width,  y + width,  y + width,  y + width],
        z = [z, z,          z,          z,         z, z + height, z + height, z + height, z + height, z + height, z + height, z,          z,          z + height, z + height, z        ],

        marker = dict(size = 1),
        line = dict(color = 'black', width = 3)
    )

    x      +=     scale                                                                                         # Small inward shift by scale so that the edges remain visible without overlapping the mesh Reduces the dimensions accordingly (dx -= 2*scale, etc.) Purpose: Edges are not covered by the mesh surface, resulting in a cleaner visual appearance
    y      +=     scale
    z      +=     scale
    length -= 2 * scale
    width  -= 2 * scale
    height -= 2 * scale

    colour = 'rgb(%s,%s,%s)' % colour
    
    surface = go.Mesh3d(
        x = [x, x,         x + length, x + length, x,          x,          x + length, x + length],             # Coordinates of the corner points of the cuboid
        y = [y, y + width, y + width,  y,          y,          y + width,  y + width,  y         ],
        z = [z, z,         z,          z,          z + height, z + height, z + height, z + height],

        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],                                                               # Mesh3d draws triangles. 2 triangles = 1 cuboid
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],                                                               # Those are the indices of the corner points (x[i], y[i], z[i])
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],                                                               # (i[i]j[i]k[i]) and (i[i+1]j[i+1]k[i+1]) together form one of the 6 surfaces of each cuboid. (i[i+2]j[i+2]k[i+2]) and (i[i+3]j[i+3]k[i+3]) together the next one and so on
        opacity = 1,
        color = colour,
        flatshading = True
    )

    return edges, surface


def plot_results(packing_result, bin_size_x, bin_size_y, parameters, file_name):
    '''
    packing_result: [box1, box2, box3, ...]
    box1:[length, width, height, x, y, z]
    '''

    # Check, whetcher boxes are overlapping
    box_num = len(packing_result)                                                                               # All the info of each box
    for i in range(box_num - 1):
        for j in range(i + 1, box_num):
            box_i = np.array(packing_result[i])                                                                 # One box has [length, width, height, x_coordinate, y_coordinate, z_coordinate]
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
                        f"Box with coordinates({box_i[3]}, {box_i[4]}, {box_i[5]}), with size ({box_i[0]}, {box_i[1]}, {box_i[2]}) and\n"
                        f"Box with coordinates({box_j[3]}, {box_j[4]}, {box_j[5]}), with size ({box_j[0]}, {box_j[1]}, {box_j[2]})"
                    )

    # Calculate the use_ratio
    packing_array = np.array(packing_result)
    box_height_coordinate = packing_array[:, 2] + packing_array[:, 5]
    max_height = np.max(box_height_coordinate)                                                                  # Find the highest point of all boxes. This will is needed to calculate the bin utilisation in the z direction.
    use_ratio = packing_array[:, :3].prod(1).sum() / (bin_size_x * bin_size_y * max_height)                     # packing_array[:, :3].prod(1).sum(): Sum of products of L/W/H of each box
    # print(f"\nThe use ratio is: {use_ratio * 100:.2f} %")

    # Plot result
    traces = []                                                                                                 # List of Plotly objects
    colour_limit_min = 0
    colour_limit_max = 255
    for box in packing_result:
        colour = (random.randint(colour_limit_min, colour_limit_max),                                           # RGB values
                  random.randint(colour_limit_min, colour_limit_max),
                  random.randint(colour_limit_min, colour_limit_max)
                )
        length, width, height, x, y, z = box
        scale = 0.0007 * max(bin_size_x, bin_size_y)                                                            # Plotly does not automatically scale 3D lines appropriately. Therefore, relative scaling is used here: the larger the bin area, the larger the line width.
        box_edges, box_surfaces = cube_trace(x, y, z, length, width, height, scale, colour)
        traces.append(box_surfaces)
        traces.append(box_edges)

    container_edges, _  = cube_trace(0, 0, 0, bin_size_x, bin_size_y, max_height, scale, (0, 0, 0))              # RGB black
    traces.append(container_edges)

    figure = go.Figure(data = traces)

    scale_factor = max(bin_size_x, bin_size_y, max_height)
    figure.update_layout(scene = dict(
        xaxis = dict(showbackground = False, color = 'white'),                                                  # Axis lines and ticks turn white. Trying to hide them if not needed
        yaxis = dict(showbackground = False, color = 'white'),
        zaxis = dict(showbackground = False, color = 'white'),
        aspectmode = 'manual',
        aspectratio = dict(x = bin_size_x / scale_factor,
                           y = bin_size_y / scale_factor,
                           z = max_height / scale_factor
                        )     
                    ),
                    showlegend = False
                )
    
    figure.add_annotation(text = f"{file_name} with use ratio of {use_ratio * 100:.2f} %.",
                          xref      = "paper",
                          yref      = "paper",
                          x         = 0.05,
                          y         = 0.95,
                          showarrow = False,
                          font      = dict(size = 16,
                                           color = "black"
                                        )
                        )


    plots_directory = Path(params.cwd) / "plots"
    plots_directory.mkdir(parents = True, exist_ok = True)

    file = plots_directory / f"{parameters} {file_name}.html"
    figure.write_html(file)
    
    # figure.write_html("plot.html")                                                                            # If one works without a GUI that helps seeing the result
    figure.show()


def plot_results_animation(packing_result, bin_size_x, bin_size_y, parameters, file_name):                      # TODO: Write code
    pass


def inject_fixed_pallet_boxes(boxes,
                              rotation_constraints,
                              euro_pallet_ratio            = 0.10,
                              half_pallet_ratio            = 0.10,
                              fixed_height                 = False,
                              disable_rotation             = False,
                              allow_ninety_degree_rotation = True
                            ):
    
    assert len(boxes) == len(rotation_constraints), \
        f"Amount of boxes ({len(boxes)}) and rotation_constraints ({len(rotation_constraints)}) must have equal size!"
    
    amount_boxes = len(boxes)
    n_euro       = int(euro_pallet_ratio * amount_boxes)
    n_half       = int(half_pallet_ratio * amount_boxes)
    euro_pallet  = params.euro_pallet
    half_pallet  = params.half_pallet

    indices = list(range(amount_boxes))
    random.shuffle(indices)

    euro_indices = indices[:n_euro]
    half_indices = indices[n_euro:n_euro + n_half]

    for i in euro_indices:
        boxes[i][0], boxes[i][1] = euro_pallet[0], euro_pallet[1]
        if fixed_height:
            boxes[i][3] = euro_pallet[2]
        if disable_rotation:
            rotation_constraints[i] = [0, 0, 0, 0, 0, 0]
        elif allow_ninety_degree_rotation:
            rotation_constraints[i] = [0, 0, 0, 1, 1, 1]

    for i in half_indices:
        boxes[i][0], boxes[i][1] = half_pallet[0], half_pallet[1]
        if fixed_height:
            boxes[i][3] = half_pallet[2]
        if disable_rotation:
            rotation_constraints[i] = [0, 0, 0, 0, 0, 0]
        elif allow_ninety_degree_rotation:
            rotation_constraints[i] = [0, 0, 0, 1, 1, 1]


def check_boxes_fit_in_bin(boxes, bin_size_x, bin_size_y, bin_size_z, run_index = None):
    for i, box in enumerate(boxes):
        length, width, height = box[0], box[1], box[2]
        if length > bin_size_x or width > bin_size_y or height > bin_size_z:
            run_info = f" in run {run_index}" if run_index is not None else ""
            print(f"Warning{run_info}: Box {i} with size ({length}, {width}, {height}) exceeds container size ({bin_size_x}, {bin_size_y}, {bin_size_z}) and will be ignored.")


if __name__ == '__main__':
    process_num              = params.process_num
    amount_of_test_runs      = params.amount_of_test_runs
    box_num                  = params.box_num
    bin_size_x               = params.bin_size_x
    bin_size_y               = params.bin_size_y
    bin_size_z               = params.bin_size_z
    bin_size_ds_x            = params.bin_size_ds_x
    bin_size_ds_y            = params.bin_size_ds_y
    min_factor               = params.min_factor
    max_factor               = params.max_factor
    rotation_constraints     = params.rotation_constraints
    inject_euro_pallets      = params.inject_euro_pallets

    model_version_number     = params.model_version_number
    load_file_path           = params.cwd + f"/saves/{bin_size_x}_{bin_size_y}_{bin_size_z}_{box_num}_{min_factor}_{max_factor}_{model_version_number}/actor.pth"
    device                   = params.set_device()

    action_queue_list        = [Queue(maxsize = 1) for _ in range(process_num)]
    result_queue_list        = [Queue(maxsize = 1) for _ in range(process_num)]
    process_list             = list()

    env = Environment(
        bin_size_x           = bin_size_x,
        bin_size_y           = bin_size_y,
        bin_size_z           = bin_size_z,
        bin_size_ds_x        = bin_size_ds_x,
        bin_size_ds_y        = bin_size_ds_y,
        box_num              = box_num,
        min_factor           = min_factor,
        max_factor           = max_factor,
        rotation_constraints = rotation_constraints
    )

    # box_and_rotation_constraints_array_list = []                                                              # Hard-coded boxes and rotation_constraints for constant tests
    box_and_rotation_constraints_array_list = [Environment.generate_boxes(env, bin_size_x, bin_size_y, min_factor, max_factor, box_num, rotation_constraints) for _ in range(amount_of_test_runs)]
    box_array_list = [boxes for boxes, _ in box_and_rotation_constraints_array_list]
    rotation_constraints_list = [rotation_constraints for _, rotation_constraints in box_and_rotation_constraints_array_list]

    if inject_euro_pallets:
        for run_index, (boxes, rotation_constraints) in enumerate(zip(box_array_list, rotation_constraints_list)):# This can be used, if euro pallets and half pallets shall be used
            inject_fixed_pallet_boxes(boxes, rotation_constraints)
            check_boxes_fit_in_bin(boxes, bin_size_x, bin_size_y, bin_size_z, run_index)




    for process_index in range(process_num):
        process_object = mp.Process(target = solve_problem,
                                    args = (action_queue_list[process_index],
                                            result_queue_list[process_index],
                                            box_array_list,
                                            rotation_constraints_list,
                                            env
                                        )
                                )
        process_list.append(process_object)
    [process_object.start() for process_object in process_list]

    actor = Actor(bin_size_ds_x,
                  bin_size_ds_y,
                  dim_model = params.dim_model,
                  binary_dim = params.binary_dim,
                  plane_feature_dim = params.plane_feature_dim,
                ).to(device)
    actor.load_state_dict(torch.load(load_file_path, map_location = device))

    [action_queue_list[process_index].put(False) for process_index in range(process_num)]
    result_list = [result_queue.get() for result_queue in result_queue_list]
    state_list = [result[0] for result in result_list]

    total_time = 0
    use_ratio_list = []
    packing_result_list = []

    for i in range (amount_of_test_runs):
        starting_time = time.time()

        for j in range(box_num):
            state = list(map(list, zip(*state_list)))
            state = [torch.as_tensor(np.array(s), dtype = torch.float32, device = device) for s in state]
            # state[2] = state[2].squeeze(0)                                                                    # Use if [[[...]]] instead of [[...]]

            ''' See explore_environment_multiprocessing() in agent.py for comments on the following code '''
            action, _ = actor.get_action_and_probabilities(state)
            action_list = np.array([action.detach().cpu().numpy() for action in action]).transpose()
            action_int_list = action_list.tolist()
            [action_queue_list[process_index].put(action_int_list[process_index]) for process_index in range(process_num)]  # Queue.put() needs CPU
            result_list = [result_queue.get() for result_queue in result_queue_list]
            result_list = list(map(list, zip(*result_list)))
            state_list = result_list[0]

            if result_list[2][0]:                                                                               # result_x = (state, reward, done, use_ratio)
                time_elapsed = time.time() - starting_time
                # avg_use_ratio = sum(result_list[3]) / process_num
                use_ratio_list.append(max(result_list[3]))
                packing_result_list.append(result_list[4][np.array(result_list[3]).argmax()])
                total_time += time_elapsed
                break

    use_ratio = sum(use_ratio_list) / amount_of_test_runs
    use_time = total_time / amount_of_test_runs
    print(f"\nAverage use ratio: {use_ratio:.2f} %")
    print(f"Average time: {use_time:.4f} s.")

    [process_object.join() for process_object in process_list]

    best_result = packing_result_list[np.array(use_ratio_list).argmax()]
    worst_result = packing_result_list[np.array(use_ratio_list).argmin()]

    plot_results(best_result, bin_size_x, bin_size_y, f"{bin_size_x}_{bin_size_y}_{bin_size_z}_{box_num}_{min_factor}_{max_factor}_{model_version_number}", "Best result")
    print(f"The best use ratio of instances was: {max(use_ratio_list):.2f} %")

    plot_results(worst_result, bin_size_x, bin_size_y, f"{bin_size_x}_{bin_size_y}_{bin_size_z}_{box_num}_{min_factor}_{max_factor}_{model_version_number}", "Worst result")
    print(f"The worst use ratio of instances was: {min(use_ratio_list):.2f} %")

    plot_results_animation(best_result, bin_size_x, bin_size_y, f"{bin_size_x}_{bin_size_y}_{bin_size_z}_{box_num}_{min_factor}_{max_factor}_{model_version_number}", "Animation of best result")
    plot_results_animation(worst_result, bin_size_x, bin_size_y, f"{bin_size_x}_{bin_size_y}_{bin_size_z}_{box_num}_{min_factor}_{max_factor}_{model_version_number}", "Animation of worst result")