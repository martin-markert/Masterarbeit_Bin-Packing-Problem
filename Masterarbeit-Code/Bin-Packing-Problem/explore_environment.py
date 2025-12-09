from environment import Environment

def explore_environment(action_queue,                                   # This is basically "just" there to collect the data. No training, no actor/critic. Does the step(action) and returns its results
                        result_queue,
                        bin_size_x,
                        bin_size_y,
                        bin_size_z,
                        bin_size_ds_x,
                        bin_size_ds_y,
                        box_num,
                        min_factor,
                        max_factor,
                        rotation_constraints = None,
                        number_of_iterations = 10_000_000_000
                    ):
    
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

    number_of_iterations = number_of_iterations * box_num               # Basically an artificially inflated number so that this limit is never reached in normal training.

    for _ in range(number_of_iterations):
        print(f"explore_environment: {_} out of {number_of_iterations} done")
        action = action_queue.get()                                     # Get the action (FIFO)
        if isinstance(action, list):
            action = tuple(action)
        if action is False:                                             # Start new episode
            state = env.reset()
            result_queue.put((state, 0, 0))
            action = action_queue.get()                                 # action = (box_index, position_index, rotation_index)  
            if isinstance(action, list):
                action = tuple(action)
        next_state, reward, done = env.step(action)
        if done:                                                        # If the episode is finished
            use_ratio = env.use_ratio
            next_state = env.reset()
        else:
            use_ratio = 0
        result_queue.put((next_state, reward, done, use_ratio))         # Put the result in the result queue

def solve_problem(action_queue, result_queue, box_array_list, rotation_constraints_list, env):
    # for box_array in box_array_list:
    for box_array, rotation_constraints in zip(box_array_list, rotation_constraints_list):
        done = False
        while not done:
            action = action_queue.get()
            if isinstance(action, list):
                action = tuple(action)
            if action is False:
                state = env.reset(box_array, rotation_constraints)
                result_queue.put((state, 0, 0))
                action = action_queue.get()
                if isinstance(action, list):
                    action = tuple(action)
            next_state, reward, done = env.step(action)
            if done:
                use_ratio = env.use_ratio
                packing_result = env.packing_result
                next_state = env.reset(box_array, rotation_constraints)
            else:
                use_ratio = 0
                packing_result = 0
            result_queue.put((next_state, reward, done, use_ratio, packing_result))