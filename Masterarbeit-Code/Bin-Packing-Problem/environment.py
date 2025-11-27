import numpy as np
import math
import copy

import warnings

class Environment:                                              # See chapter 4.1
    def __init__(self,
                 bin_size_x,                                    # 100 in the paper
                 bin_size_y,                                    # 100 in the paper
                 bin_size_z,                                    # Now set to high number. Later for multi-bin use it will be smaller
                 bin_size_ds_x,                                 # 10 in the paper
                 bin_size_ds_y,                                 # 10 in the paper
                 box_num,                                       # Whatever your heart desires
                 min_factor = 0.1,                              # Chosen by the authors
                 max_factor = 0.5,                              # Chosen by the authors
                 rotation_constraints = None,
                 # bin_height_if_not_start_with_all_zeros = None  # For debugging only, remove later
                ):
        
        if (bin_size_x < 1 or bin_size_y < 1 or bin_size_z < 1 or bin_size_ds_x < 1 or bin_size_ds_y < 1):
            raise ValueError(
                    f"Bin sizes must not be less than 1. "
                    f"You entered "
                    f"bin_size_x = {bin_size_x}, bin_size_y = {bin_size_y}, bin_size_z = {bin_size_z}, bin_size_ds_x = {bin_size_ds_x}, bin_size_ds_y = {bin_size_ds_y}."
                )
        if (box_num < 1):
            raise ValueError(
                    f"Box amount must not be less than 1. "
                    f"You entered "
                    f"box_num = {box_num}."
                )
        if (min_factor <= 0 or max_factor > 1 or min_factor > max_factor):
            raise ValueError(
                    f"min_factor must be smaller than max_factor. "
                    f"Also, min_factor must be more than 0 and max_factor must be less or equal to 1. "
                    f"You entered "
                    f"min_factor = {min_factor}, max_factor = {max_factor}."
                )
        
        # max_steps = 99999

    # Environment Parameters
        self.bin_size_x = bin_size_x                    # Original bin size x
        self.bin_size_y = bin_size_y                    # Original bin size y
        self.bin_size_z = bin_size_z                    # Original bin size z
        self.bin_size_ds_x = bin_size_ds_x              # Size after downsampling of the bin
        self.bin_size_ds_y = bin_size_ds_y              # Size after downsampling of the bin
        self.box_num = box_num                          # Amount of boxes started with (fixed value)
        self.min_factor = min_factor                    # Minimal size of the boxes to be packed relative to the container size
        self.max_factor = max_factor                    # Maximum size of the boxes to be packed relative to the container size
        self.rotation_constraints = rotation_constraints# If None, all rotations are allowed


    # Environment Variables
        self.gap = 0                                    # Difference between the total volume of the bin (L×W×H) and the total volume of all boxes placed in the bin    <-- See chapter 3.2.1
        self.total_bin_volume = 0                       # Total volume of the bin                                                                                       <-- See chapter 3.2.1
        self.total_box_volume = 0                       # Volume of all boxes placed in the bin                                                                         <-- See chapter 3.2.1
        self.max_indices = None                         # Indices of the "best" spot with the most space per downsampling block
        self.packing_result = []                        # List of all packed boxes [l, w, h, x, y, z]. Will be used for visualisation
        self.residual_box_num = box_num                 # Amount of boxes that have not been packed (variable value)

    # Environment constraints   
        self.original_bin_heights = np.zeros((self.bin_size_x, self.bin_size_y))                                  # Creates the empty 2D Matrix of the bin with all heights = 0
        # if bin_height_if_not_start_with_all_zeros is None:
        #     bin_height_if_not_start_with_all_zeros = np.zeros((self.bin_size_x, self.bin_size_y))
        # self.original_bin_heights = bin_height_if_not_start_with_all_zeros                                          # TODO: This line is for debugging. Remove those 3 lines later and replace by line above
        
        self.original_plane_features = self.get_bin_features(self.original_bin_heights)                             # Calculates the plane-feature matrix
        self.block_size_x = self.bin_size_x // self.bin_size_ds_x                                                   # Size of one downsampling block (cells per block in the x direction) --> Technically “//” is not needed as for the downsampling only divisions with no remainder are allowed
        self.block_size_y = self.bin_size_y // self.bin_size_ds_y                                                   # Size of one downsampling block (cells per block in the y direction)
        self.downsampling_is_needed = self.bin_size_ds_x < self.bin_size_x or self.bin_size_ds_y < self.bin_size_y  # Is downsampling deeded or is the original bin size used?

        if self.bin_size_ds_x > self.bin_size_x or self.bin_size_ds_y > self.bin_size_y:                            # Do the downsampling only, if the end sizes ar snaller than the original ones
            raise ValueError(
                    f"Downsampling sizes must be smaller or equal to the original sizes "
                    f"({bin_size_x}, {bin_size_y}). You entered "
                    f"({bin_size_ds_x}, {bin_size_ds_y})."
                )
        elif self.downsampling_is_needed:
            self.original_downsampled_plane_features, self.original_max_incides = self.downsampling(self.original_plane_features)

        # TODO Test behaviour with empty list. Maybe it works,although it is very useless
        if contains_empty_list(self.rotation_constraints):
            warnings.warn(
                f"Found empty values --> [] <-- in rotation_constraints: {self.rotation_constraints}.\n"
                f"This can lead to errors. Avoid it. It does make zero sense to allow not a single rotation.\n"
                f"If you want to allow all rotations, set rotation_constraints to None.", 
                UserWarning
            )



    def get_distances(self, heights_of_cells, move_along_x_axis, equal_height = True, negative_axis_direction = False):
        '''
        get_distances and get_bin_features together are a vectorised approach to get the plane features: 
        harder to understand than loops, but faster for large bin size. As a matrix is used, so (0, 0) is on the top left
        '''
        
        plane_feature = np.ones_like(heights_of_cells)                                                          # Create Matrix with all ones with (bin_size_x, bin_size_y)
        
        if move_along_x_axis:                                                                                   # Looking along the x-axis for eˡ e⁻ˡ, fˡ (see Figure 2 in Paper)
            if negative_axis_direction:                                                                         # e⁻ˡ
                count = np.zeros((1, self.bin_size_x))
                for y_index in range(1, self.bin_size_y):                                                       # Column-by-column examination of the heights_of_cells matrix (going forwards, comparing backwards)
                    comparison = abs(heights_of_cells[:, y_index] - heights_of_cells[:, y_index - 1]) == 0      # Compare height difference to previous column
                    count = np.where(comparison, count + 1, 0)                                                  # Reset to 0, as going back the cell itself does not count
                    plane_feature[:, y_index] = count
                    plane_feature[:, 0] = 0                                                                     # As index 0 is not looked at the first column has to be manually set to 0 as the matrix is initialised with ones    
            else: 
                count = np.ones((1, self.bin_size_x))
                for y_index in range(self.bin_size_y - 2, -1, -1):
                    if equal_height:                                                                            # eˡ
                        comparison = abs(heights_of_cells[:, y_index] - heights_of_cells[:, y_index + 1]) == 0
                    else:                                                                                       # fˡ
                        comparison = heights_of_cells[:, y_index] >= heights_of_cells[:, y_index + 1]
                    count = np.where(comparison, count + 1, 1)
                    plane_feature[:, y_index] = count    
        else:                                                                                                   # Looking along the y-axis for eʷ e⁻ʷ, fʷ (see Figure 2 in Paper)
            if negative_axis_direction:                                                                         # e⁻ʷ
                count = np.zeros((1, self.bin_size_y))
                for x_index in range(1, self.bin_size_x):
                    comparison = abs(heights_of_cells[x_index, :] - heights_of_cells[x_index - 1, :]) == 0
                    count = np.where(comparison, count + 1, 0)
                    plane_feature[x_index, :] = count
                    plane_feature[0, :] = 0
            else:
                count = np.ones((1, self.bin_size_y))
                for x_index in range(self.bin_size_x - 2, -1, -1):
                    if equal_height:                                                                            # eʷ
                        comparison = abs(heights_of_cells[x_index, :] - heights_of_cells[x_index + 1, :]) == 0
                    else:                                                                                       # fʷ
                        comparison = heights_of_cells[x_index, :] >= heights_of_cells[x_index + 1, :]
                    count = np.where(comparison, count + 1, 1)
                    plane_feature[x_index, :] = count
                                                                                                        
        return plane_feature


    def get_bin_features(self, heights_of_cells):   # heights_of_cells = container_matrix = L×W Matrix, that has the heights of the cells as its values
        height_feature = heights_of_cells
        e_l_feature = self.get_distances(heights_of_cells, move_along_x_axis = True)                                            # eˡ  plane features (see Figure 2 in Paper)
        e_w_feature = self.get_distances(heights_of_cells, move_along_x_axis = False)                                           # eʷ  plane features
        e_negative_l_feature = self.get_distances(heights_of_cells, move_along_x_axis = True, negative_axis_direction = True)   # e⁻ˡ plane features
        e_negative_w_feature = self.get_distances(heights_of_cells, move_along_x_axis = False, negative_axis_direction = True)  # e⁻ʷ plane features
        f_l_feature = self.get_distances(heights_of_cells, move_along_x_axis = True, equal_height = False)                      # fˡ  plane features
        f_w_feature = self.get_distances(heights_of_cells, move_along_x_axis = False, equal_height = False)                     # fʷ  plane features
        
        bin_features = np.stack([height_feature, e_l_feature, e_w_feature, e_negative_l_feature, e_negative_w_feature, f_l_feature, f_w_feature], -1)
        
        return bin_features
    
                                                                                                                                # For parameters see chapter 4.1
    def generate_boxes(self, bin_size_x, bin_size_y, min_factor, max_factor, box_num, rotation_constraints = None):             # The paper allows all 6 rotations, but in the "logistics reality" not all might be allowed
        box_sizes_x_axis = np.random.randint(math.ceil(bin_size_x * min_factor), int(bin_size_x * max_factor + 1), box_num)     # ceil() ensures that the value never becomes smaller than bin_size_x * min_factor, which is truncated. Furthermore, it cannot be 0 either.
        box_sizes_y_axis = np.random.randint(math.ceil(bin_size_y * min_factor), int(bin_size_y * max_factor + 1), box_num)
        box_sizes_z_axis = np.random.randint(math.ceil(min(bin_size_x, bin_size_y) * min_factor), int(max(bin_size_x, bin_size_x) * max_factor + 1), box_num)
        boxes = np.stack([box_sizes_x_axis, box_sizes_y_axis, box_sizes_z_axis], -1)

        if rotation_constraints is None:                                                                                        # Every rotation is allowed for each box
            rotation_constraints = [list(range(6)) for _ in range(box_num)]
        elif len(rotation_constraints) == 1:                                                                                    # Same rotation constraint for all boxes
                rotation_constraints = rotation_constraints * box_num
        elif len(rotation_constraints) != box_num:                                                                              # Individial rotation constraints
                raise ValueError(
                        f"Length of rotation_constraints ({len(rotation_constraints)}) must match box_num ({box_num})."
                    )
      
        # boxes_with_rotation_constraints = np.array([[*boxes[i], rotation_constraints[i]] for i in range(box_num)], dtype = object)    # <-- PyTorch does not really like objects, so it is rturned separately
        
        rotation_constraints = [sorted(rc) for rc in rotation_constraints]                                                      # Make the indices sorted
        self.rotation_constraints = copy.deepcopy(rotation_constraints)                                                         # np.copy() does not work with inconsistent sizes like rotation_constraints = [[1], [2, 4]]
        return boxes, rotation_constraints
    

    def downsampling(self, plane_features):
        '''
        The authors do not specify whether they work with uniform divisions without remainder or with irregular downsampling.
        For reasons of simplicity/symmetry, only divisions without remainder are permitted here.

        Why downsampling at all?
        The Authors state: A container with a large size results in a large action space for the position action. 
        Therefore, we should downsample the container state to reduce the action space before feeding it into the container encoder.
        '''
        if (plane_features.shape[0] % self.bin_size_ds_x != 0) or (plane_features.shape[1] % self.bin_size_ds_y != 0):          # Only divisions without remainder are possible
            raise ValueError(
                    f"Downsampling not possible: "
                    f"original_bin_size_x = {plane_features.shape[0]} / downsampled_bin_size_x = {self.bin_size_ds_x} "
                    f"and/or original_bin_size_y = {plane_features.shape[1]} / downsampled_bin_size_y = {self.bin_size_ds_y} "
                    f"are/is not divisible without a remainder."
                )                                                                                                                 
        feature_num = plane_features.shape[2]                                                                                   # Downsamples as described in chapter 3.1.3, Container encoder.
        plane_features_split_x = np.stack(np.split(plane_features, self.bin_size_ds_x, 0), 0)                                   # Splits the original bin size along the x-axis
        plane_features_split_xy = np.stack(np.split(plane_features_split_x, self.bin_size_ds_y, 2), 1)                          # Splits the original bin size along the y-axis
        plane_features_split_xy = plane_features_split_xy.reshape(self.bin_size_ds_x * self.bin_size_ds_y, -1, feature_num)     # First dimension: all ds_x*ds_y blocks combined. Second dimension: all cells within a block combined flattned. Third dimension: the features remain the same.
        max_target = plane_features_split_xy[:, :, 1] * plane_features_split_xy[:, :, 2]                                        # Gives the eˡ*eʷ for each cell of a ds_block
        indices_of_largest_values = max_target.argmax(-1).reshape(-1, 1, 1)                                                     # Indices of cells with largest eˡ*eʷ in each ds_block. Each block starts with index 0, as blocks are condidered their own entity.

        bin_state_ds = np.take_along_axis(plane_features_split_xy, indices_of_largest_values, 1).reshape(self.bin_size_ds_x,
                                                                                                         self.bin_size_ds_y, -1)# Takes the values needed from plane_features_split_xy based on indices_of_largest_values 
        
        return bin_state_ds, indices_of_largest_values 
    
    
    def reset(self, boxes = None, rotation_constraints = None):                                     # Resets everything to the original states (hence the name). Used so at every episode the agent gets a fresh environment
        self.gap = 0
        self.total_bin_volume = 0
        self.total_box_volume = 0
        self.packing_result = []
        self.residual_box_num = self.box_num

        if boxes is None:
            if rotation_constraints is None:
                boxes, rotation_constraints = self.generate_boxes(
                    self.bin_size_x, self.bin_size_y,
                    self.min_factor, self.max_factor,
                    self.box_num, self.rotation_constraints
                    )  
            else:
                raise ValueError(
                        "You passed no boxes but their rotation constraints. "
                        "Are you sure you wanted this?"
                    )
        else:
            if rotation_constraints is None:                                                        # If there are boxes but no rotation_constraints were passed
                _, rotation_constraints = self.generate_boxes(                                      # Then allow all rotations
                self.bin_size_x, self.bin_size_y,
                self.min_factor, self.max_factor,
                self.box_num, self.rotation_constraints
        )

        self.boxes = np.copy(boxes)                                                                 # Not sure, whether the copy() is needed, but just in case...
        self.rotation_constraints = copy.deepcopy(rotation_constraints)                             # np.copy() does not work with inconsistent sizes like rotation_constraints = [[1], [2, 4]]

        self.bin_height = np.copy(self.original_bin_heights)                                        # Not sure, whether the copy() is needed, but just in case...
        plane_features = np.copy(self.original_plane_features)                                      # Not sure, whether the copy() is needed, but just in case...

        if self.downsampling_is_needed:            
            plane_features = np.copy(self.original_downsampled_plane_features)                      # Not sure, whether the copy() is needed, but just in case...
            self.max_indices = np.copy(self.original_max_incides)                                   # Not sure, whether the copy() is needed, but just in case...
            packing_mask = self.get_packing_mask(self.boxes, rotation_constraints, self.max_indices)
        else:
            packing_mask = self.get_packing_mask(self.boxes, rotation_constraints)                  # If no downsampling is needed, no index list is needed, too
        
        self.state = (plane_features, self.boxes, self.rotation_constraints, packing_mask)

        return self.state


    def get_packing_mask(self, boxes, rotation_constraints, indices_of_largest_values = None):
        x_residual_size = np.zeros((self.bin_size_ds_x, self.bin_size_ds_y)) + np.arange(self.bin_size_ds_x, 0, -1).reshape(-1, 1) * self.block_size_x  # Remaining length in x-direction per ds_block. reshape(-1, 1) ensures that x values vary row by row
        y_residual_size = np.zeros((self.bin_size_ds_x, self.bin_size_ds_y)) + np.arange(self.bin_size_ds_y, 0, -1) * self.block_size_y                 # Remaining length in y-direction per ds_block
        x_y_residual_sizes = np.stack([x_residual_size, y_residual_size], 2)                                                                            # Combining both values to see, how much space is left per cell x-wise and y-wise.          --> Shape (bin_size_ds_x,bin_size_ds_y,2)
        available_box_num = self.residual_box_num
        box_array = boxes[:self.residual_box_num]                                                                                                       # Take the amount of unpacked boxes into box_array
        box_rotation_array = generate_box_rotations(box_array)                                                                                          # Calculates all permutations: Shape (-1, 6, 3)
        box_rotation_array = box_rotation_array[:, :, :2]                                                                                               # Only considering L and W, ignoring the height, as it only matters whether the box fits, 
                                                                                                                                                        # not how high it is (since there is only one container with infinite height in the paper).

        if indices_of_largest_values is not None:                                                                                                       # Only needed, if downsampling has been done (theoretically one could do bin_size_x = bin_size_ds_x or bin_size_y = bin_size_ds_y, 
                                                                                                                                                        # then you don't need downsampling(), as used in step())                                    --> Oringinal shape: (9,1,1)
            coordinates_of_max_indices = np.stack([indices_of_largest_values // (self.block_size_y), indices_of_largest_values % (self.block_size_y)], -1) # Converts the flat indices into an (x, y) coordinate: “//” gives the x, “%” gives the y --> Now shape (9,1,2)
            x_y_residual_sizes = x_y_residual_sizes - coordinates_of_max_indices.reshape(self.bin_size_ds_x, self.bin_size_ds_y, 2)                     # When downsampling the positional information gets lost. Here it is retreived again        --> Now shape (bin_size_ds_x,bin_size_ds_y,2)
            '''
            Why are the two lines above there?

            x_y_residual_sizes describes (at downsampling  block level) the available residual/space size (in block units) in the x and y directions.

            indices_of_largest_values gives the index of the representative point (the subcell with the largest eˡ*eʷ) within each block.

            By converting indices to (row, col) in coordinates_of_max_indices = ... 
            and then subtracting it from x_y_residual_sizes, 
            x_y_residual_sizes is shifted by this offset so that the comparisons take into account the actual, sub-block-accurate available area. 
            This is necessary because position information is usually lost during downsampling. 
            This step returns the location of the best point in the block as an offset back to the block grid.

            Without this correction, the test “does the (L×W) projection of a box fit into this block?” 
            might incorrectly allow/reject the box because the representative subfield within the block is not located at the block origin.

            If there is no downsampling, x_y_residual_sizes is already in the original coordinates, so this is not needed.

            Example:
            coordinates_of_max_indices.reshape(3,3,2) =     x_y_residual_sizes =            # Shape also (3,3,2)
                [                                               [
                 [ [0,0], [0,1], [0,0] ],                        [ [9,9], [9,6], [9,3] ],
                 [ [0,0], [0,0], [0,1] ],                        [ [6,9], [6,6], [6,3] ],
                 [ [0,0], [0,0], [0,1] ]                         [ [3,9], [3,6], [3,3] ]   
                ]                                               ]

            x_y_residual_sizes =
                [                                                     [
                 [ [9,9] - [0,0],  [9,6] - [0,1],  [9,3] - [0,0] ],    [ [9,9], [9,5], [9,3] ],
                 [ [6,9] - [0,0],  [6,6] - [0,0],  [6,3] - [0,1] ], =  [ [6,9], [6,6], [6,2] ],
                 [ [3,9] - [0,0],  [3,6] - [0,0],  [3,3] - [0,1] ]     [ [3,9], [3,6], [3,2] ]
                ]                                                     [
            '''
        packing_available = box_rotation_array.reshape(
            -1, 6, 1, 1, 2) <= x_y_residual_sizes.reshape(1, 1, self.bin_size_ds_x, self.bin_size_ds_y, 2)  # Mask that shows which box rotation combinations are possible at which grid positions.
                                                                                                            # reshape(-1, 6, 1, 1, 2) --> (residual_box_num, 6, 2) extended to: -1 = residual_box_num 6 = rotations 1, 1 = placeholder dimensions for the downsampling blocks (ds_x, ds_y) 2 = x and y dimensions
                                                                                                            # reshape(1, 1, self.bin_size_ds_x, self.bin_size_ds_y, 2) --> (bin_size_ds_x, bin_size_ds_y, 2) extended 
        packing_available = packing_available.all(-1)                                                       # Box must fit in x and y direction. So this asks, for example: Box 3, rotation 5, block (7,2) --> possible?
        
        allowed_rotations_mask = np.zeros((available_box_num, 6), dtype = bool)                             # Use the Rotation constraints:
        for i in range(available_box_num):                                                                  # This is done in this way so that the number of rotations is always 6
            allowed_rotations_mask[i, rotation_constraints[i]] = True                                       # Auto checks for out-of-bound indices

        allowed_rotations_mask_broadcasted = allowed_rotations_mask[:, :, None, None]                       # Broadcasting on position dimensions
        
        packing_available = packing_available & allowed_rotations_mask_broadcasted                          # Only keep allowed rotations

        packing_available = np.pad(packing_available,
                                   ((0, self.box_num - available_box_num), (0, 0), (0, 0), (0, 0)),         # The network (e.g. PyTorch) would always expect the same input shape. If there are fewer boxes than expected, errors occur. Therefore, theoretical boxes are created using padding, which can't be placed, though.
                                   constant_values = False)         
        
        return ~packing_available                                                                           # Why inverted? In the masks used in PyTorch and most Transformer implementations, the following applies:
                                                                                                            # True = this position is masked/ignored
                                                                                                            # False = this position is considered/may be included
    

    def step(self, action:tuple):
        '''
        Several steps are being done here:
        Step gets the action by the transformer, telling it where which box in what rotation shall be placed where.
        The box is placed in the specified position with the specified rotation.
        Then the reward is calculated as in chapter 3.2.1
        Then there is a final check, wheter everything is done or another step will be needed
        '''
        if not isinstance(action, tuple) or len(action) != 3:
            raise ValueError(
                    f"action must be a tuple with exactly 3 values, "
                    f"but got type {type(action).__name__} with value {action}, "
                    f"which has {len(action) if hasattr(action, '__len__') else 'N/A'} values."
                )

        
        for i, v in enumerate(action):
            # if isinstance(v, (list, tuple, set, dict)):
            if not isinstance(v, int):
                raise TypeError(f"action[{i}] must be an integer (found: {type(v).__name__} with value {v})")

        # position, box_index, rotation = action                                            # "action" contains the three sub-actions by the 3 transformers: position in container (index of the downsampling_block), box to place and its orientation (as in chapter 3.1.2)
        box_index, position, rotation = action                                              # Technically that is the wrong order if looked at it from the order of the transformers
        
        if self.max_indices is not None:                                                    # If downsampling took place (standard use case)
            val_array = self.max_indices[position]                                          # How to get the real coordinates of the non-downsampled container? The transformer decision of the position is based on the downsampling_block.
                                                                                            # So “position” cannot be the absolute coordinates, as the transformer does not know the non-downsampled container size. So “position” must be an index. The index is the downsampling_block, where the box shall be placed.
                                                                                            # With self.max_indices[position] one gets the index of the best placment position of that selected downsampling_block
            if val_array.size != 1:
                raise ValueError(
                        f"Expected exactly one element, got "
                        f"{val_array} with size {val_array.size}"
                    )  
                                 
            box_placement_position_in_downsampling_block = int(val_array.item())            # With that information together with self.block_size_x and self.block_size_y and self.bin_size_ds_x and self.bin_size_ds_y one can reconstruct the absolute position inside the non-downsampled bin.
            
            position_x = ((position // self.bin_size_ds_y) * self.block_size_x) + (box_placement_position_in_downsampling_block // self.block_size_y)   # Convert the linear index from the downsampled grid to real container coordinates:
                                                                                                                                                        # 1. position // self.bin_size_ds_y --> x-coordinate in the downsampled grid
                                                                                                                                                        # 2. * self.block_size_x --> x-coordinate of the (0, 0) position of the downsampling_block in the non-downsampled container
                                                                                                                                                        # 3. + (box_placement_position_in_downsampling_block // self.block_size_y) --> exact x-coordinate inside the downsampling_block
            position_y = ((position % self.bin_size_ds_y) * self.block_size_y) + (box_placement_position_in_downsampling_block % self.block_size_y)     # Same logic for position_y, but for columns, so its * self.block_size_y                               
        else:
            position_x = position // self.bin_size_y
            position_y = position % self.bin_size_y
        
        chosen_box = self.boxes[box_index]                                                                                  # Selects the chosen box from the box_list.
        chosen_box_with_chosen_rotation = generate_box_rotations(np.atleast_2d(chosen_box), rotation_indices = rotation)    # Get the box's chosen rotation.

        box_length, box_width, box_height = map(int, chosen_box_with_chosen_rotation[0])                                    # Because the Array had to be made 2D, now it has to be inexed for map() to work
        if box_length < 1 or box_width < 1 or box_height < 1:                                                               # Should never be called as generate_boxes() does not create illegal boxes. But at constant_values = -1e9 something could happen
            raise ValueError(
                    f"Box dimension must not be smaller than (1, 1, 1). "
                    f"This box has ({box_length}, {box_width}, {box_height})"
                )


    # --- Placing the box ---

        place_area = self.bin_height[position_x:position_x + box_length, position_y:position_y + box_width]                 # Check the heights at the area, where the box shall be placed
        position_z = int(np.max(place_area))                                                                                # Highest value of the place_area To know at which height the box shall be placed
        
        if position_x + box_length > self.bin_size_x or position_y + box_width > self.bin_size_y:
        # if position_x + box_length > self.bin_size_x or position_y + box_width > self.bin_size_y or position_z + box_height > self.bin_size_z:
            # TODO: Later add height check, when this is also capped
            raise ValueError(                                                                                               # Should never be called as there are checks to prevent this
                    f"The box with dimensons "
                    f"({box_length}, {box_width}, {box_height}) "
                    f"does not fit into the container at position "
                    f"({position_x}, {position_y}, {position_z}). "
                )
        place_area[:, :] = int(np.max(place_area) + box_height)                                                             # place_area is a view of self.bin_height, so self.bin_height is updated automatically
        self.packing_result.append([box_length, box_width, box_height, position_x, position_y, position_z])                 # Save the position and dimension of the places box

        self.boxes = np.delete(self.boxes, box_index, 0)                                                                    # Removes the placed box from the list
        self.boxes = np.pad(
            self.boxes, ((0, self.box_num - self.boxes.shape[0]), (0, 0)), constant_values = -1e9)                          # The array is then filled back to its original length self.box_num so that the dimension remains the same (this is important because the neural network expects a fixed input size). 
                                                                                                                            # The missing rows are filled with a very small value (−1e9), which effectively serves as “not present”.
                                                                                                                            # -np.inf could lead to problems like NaN or with softmax
        used_rotation_constraint = self.rotation_constraints[box_index]                                                     # Take the rotation_constraints of box [box_inedx] and put it to the end of the rotation_constraints
        self.rotation_constraints = self.rotation_constraints[:box_index] + self.rotation_constraints[box_index + 1:]
        self.rotation_constraints.append(used_rotation_constraint)
        
        self.residual_box_num -= 1

        plane_features_after_placing_box = self.get_bin_features(self.bin_height)                                           # Calculate the new not-downsampled bin state (residual features)
        if self.downsampling_is_needed:                                                                                     # Why not global downsampling just once at the very beginning?
                                                                                                                            # Because downsampling() calculates the best placing spot per downsampling_block.
                                                                                                                            # This spot, based on the new bin state after placing a box, can change.
            
            if self.residual_box_num != 0:                                                                                  # There are unpacked boxes left                                                                          
                plane_features_after_placing_box, self.max_indices = self.downsampling(plane_features_after_placing_box)    # Calculate the new bin state (residual features) --> Downsampled plane features
                packing_mask = self.get_packing_mask(self.boxes, self.rotation_constraints, self.max_indices)               # Get the packing mask with the max_indices
            else:                                                                                                           # There are no unpacked boxes left
                plane_features_after_placing_box, self.max_indices = self.downsampling(plane_features_after_placing_box)    # Calculate the new bin state (residual features) --> Downsampled plane features
                packing_mask = np.ones((self.box_num, 6, self.bin_size_ds_x, self.bin_size_ds_y))                           # Packing mask with no allowed spaces
        else:
            if self.residual_box_num != 0:                                                                                  # There are unpacked boxes left
                packing_mask = self.get_packing_mask(self.boxes, self.rotation_constraints)                                 # Get the packing mask without the max_indices
            else:                                                                                                           # There are no unpacked boxes left)
                packing_mask = np.ones((self.box_num, 6, self.bin_size_ds_x, self.bin_size_ds_y))                           # Packing mask with no allowed spaces

        self.packing_mask = packing_mask
        self.state = (plane_features_after_placing_box, self.boxes, self.rotation_constraints, packing_mask)


    # --- Calculating the reward (see chapter 3.2.1) ---

        self.total_bin_volume = self.bin_size_x *  self.bin_size_y * int(np.max(self.bin_height))                           # Total used volume of the box including gaps    
        self.total_box_volume += box_length * box_width * box_height                                                        # Add the volume of the just places box

        new_gap = self.total_bin_volume - self.total_box_volume                                                             # Difference between total used volume of container and the sum of all places boxes
        reward = self.gap - new_gap                                                                                         # Reward is the difference between the gap and the gap in the step before (self.gap = 0 in the very first step)
        reward_normalised = reward / (self.bin_size_x * self.bin_size_y)                                                    # Normalise reward so it is not dependant on the the bin's dimensions
        self.gap = new_gap
        self.use_ratio = (self.total_box_volume / self.total_bin_volume) * 100

    
    # --- Done? ---    

        if self.residual_box_num == 0 or packing_mask.all():                                                                # No boxes left or nothing can be placed anymore (box full) (as of now the height is unlimited anyways)
            done = True
        else:
            done = False
        
        return self.state, reward_normalised, done
    

# TODO: Check, whether the dimensions are returned correctly as needed [] or [[]] or [[[]]] ...
def generate_box_rotations(boxes, rotation_indices = None):                                     # Based on this coordinate system:
    rotations = np.array([                                                                      # z
        [0, 1, 2],  # 0: (x, y, z) --> Original State                                           # ^
        [1, 0, 2],  # 1: (y, x, z) --> Box rotated 90° around the height axis (z)               # |
        [2, 1, 0],  # 2: (z, y, x) --> Box tipped forward/backward                              # |_____> y
        [1, 2, 0],  # 3: (y, z, x) --> Box tipped forward/backward and then rotated 90°         #  \
        [0, 2, 1],  # 4: (x, z, y) --> Box tipped to the left or right                          #   \
        [2, 0, 1]   # 5: (z, x, y) --> Box tipped to the left or right and then rotated 90°     #    _|
    ])                                                                                          #      x
                                                                                                #
    if rotation_indices is not None:                                                            #       ^
        rotations = rotations[rotation_indices]                                                 #       |
                                                                                                #     Viewer
    return boxes[:, rotations]


'''
    Auxiliary function
'''
def contains_empty_list(lst):
    if not isinstance(lst, list):
        return False
    if lst == []:
        return True
    return any(contains_empty_list(i) for i in lst)