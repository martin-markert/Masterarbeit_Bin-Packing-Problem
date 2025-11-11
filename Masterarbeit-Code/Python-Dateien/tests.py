import parameters as p

from environment import *
from network import *

import torch
import torch.nn as nn
import numpy as np


'''
    --- Initialise the environment ---
'''


def make_empty_environment():
    bin_size_x, bin_size_y, bin_size_z = 9, 9, 100
    env = Environment(
        bin_size_x           = bin_size_x,
        bin_size_y           = bin_size_y,
        bin_size_z           = bin_size_z,
        bin_size_ds_x        = 3,
        bin_size_ds_y        = 3,
        box_num              = 3,
        min_factor           = 0.1,
        max_factor           = 0.5,
        rotation_constraints = None,
        bin_height_if_not_start_with_all_zeros = np.zeros((bin_size_x, bin_size_y), dtype = int)
    )
    state = env.reset()
    # state = (plane_features, boxes, rotation_constraints, packing_mask)
    # plane_features = torch.tensor(state[0], dtype = torch.float32).unsqueeze(0)
    # boxes = torch.tensor(state[1], dtype = torch.long).unsqueeze(0)
    # rotation_constraints = state[2]
    # packing_mask = torch.tensor(state[3], dtype = torch.bool).unsqueeze(0)
    
    # return (plane_features, boxes, rotation_constraints, packing_mask), env                               # env = (plane_features, self.boxes, self.rotation_constraints, packing_mask)
    return state, env


def make_environment_with_prefilled_container(container_matrix):
    bin_size_x, bin_size_y = container_matrix.shape
    env = Environment(
        bin_size_x           = bin_size_x,
        bin_size_y           = bin_size_y,
        bin_size_z           = 100,
        bin_size_ds_x        = 3,
        bin_size_ds_y        = 3,
        box_num              = 2,
        min_factor           = 0.1,
        max_factor           = 0.5,
        rotation_constraints = None,                                # Or Use [[1,2,3]] or [[1], [2, 4]]. TODO: [[]] or [[], []] has not been tested, yet
        bin_height_if_not_start_with_all_zeros = container_matrix
    )



'''
    --- Testing the file environment.py ---
'''

def test_empty_environment_initialisation():
    env = Environment(
        bin_size_x    = 99,
        bin_size_y    =  4,
        bin_size_z    = 10,
        bin_size_ds_x = 33,
        bin_size_ds_y =  2,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = None
    )

# Environment Parameters
    assert env.bin_size_x             == 99
    assert env.bin_size_y             ==  4
    assert env.bin_size_z             == 10
    assert env.bin_size_ds_x          == 33
    assert env.bin_size_ds_y          ==  2
    assert env.box_num                ==  2
    assert env.min_factor             ==  0.1
    assert env.max_factor             ==  0.5
    assert env.rotation_constraints   == None
    
    # Environment Variables
    assert env.gap == 0
    assert env. total_bin_volume      ==  0
    assert env.total_box_volume       ==  0
    assert env.max_indices            == None
    assert env.packing_result         == []
    assert env.residual_box_num       ==  2
    
# Environment constraints
    assert np.all(env.original_bin_heights == 0)
    # assert env.original_plane_features ==                 # Tested seperately, when get_bin_features() is tested
    assert env.block_size_x           == 99 // 33
    assert env.block_size_y           ==  4 //  2
    assert env.downsampling_is_needed == env.bin_size_ds_x < env.bin_size_x or env.bin_size_ds_y < env.bin_size_y

    print("test_empty_environment_initialisation passed.")


def test_prefilled_environment_initialisation(container_matrix):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

# Environment Parameters
    assert env.bin_size_x             ==  9
    assert env.bin_size_y             ==  9
    assert env.bin_size_z             == 10
    assert env.bin_size_ds_x          ==  3
    assert env.bin_size_ds_y          ==  3
    assert env.box_num                ==  2
    assert env.min_factor             ==  0.1
    assert env.max_factor             ==  0.5
    assert env.rotation_constraints   == None
    
    # Environment Variables
    assert env.gap == 0
    assert env. total_bin_volume      ==  0
    assert env.total_box_volume       ==  0
    assert env.max_indices            == None
    assert env.packing_result         == []
    assert env.residual_box_num       ==  2
    
# Environment constraints
    assert np.array_equal(env.original_bin_heights, container_matrix)
    # assert env.original_plane_features ==                 # Tested seperately, when get_bin_features() is tested
    assert env.block_size_x           ==  9 //  3
    assert env.block_size_y           ==  9 //  3
    assert env.downsampling_is_needed == env.bin_size_ds_x < env.bin_size_x or env.bin_size_ds_y < env.bin_size_y

    print("test_prefilled_environment_initialisation passed.")


def test_get_distances_and_bin_features(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    if container_matrix is None:
        container_matrix = np.zeros((env.bin_size_x, env.bin_size_y), dtype = int)
    
    distances = env.get_distances(container_matrix, move_along_x_axis = True)
    assert distances.shape == container_matrix.shape

    features = env.get_bin_features(container_matrix)
    assert features.shape == (9, 9, 7)

    print("test_get_distances_and_bin_features passed.")


def test_generate_boxes(container_matrix = None, rotation_constraints = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    boxes, rotations = env.generate_boxes(env.bin_size_x, env.bin_size_y, env.min_factor, env.max_factor, env.box_num, rotation_constraints)
    assert boxes.shape == (env.box_num, 3)
    assert len(rotations) == env.box_num
    print("test_generate_boxes passed.")


def test_downsampling(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    if container_matrix is None:
        container_matrix = np.zeros((env.bin_size_x, env.bin_size_y), dtype = int)

    bin_features = env.get_bin_features(container_matrix)
    downsampling_features, indices = env.downsampling(bin_features)
    assert downsampling_features.shape == (env.bin_size_ds_x, env.bin_size_ds_y, 7)                     # 7 for the 7 plane_festures
    assert indices.shape == (env.bin_size_ds_x* env.bin_size_ds_y, 1, 1)
    print("test_downsampling passed.")


def test_reset(container_matrix = None, boxes = None, rotation_constraints = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    if boxes is not None:
        if rotation_constraints is None:
            rotation_constraints = np.tile(np.arange(6), (boxes.shape[0], 1))
        elif len(rotation_constraints) == 1:
                rotation_constraints = rotation_constraints * env.box_num
        packing_mask = env.get_packing_mask(boxes, rotation_constraints)
    # print(
    #     f"Environment before resetting:\n"
    #     f"plane_features:\n{env.original_downsampled_plane_features}\n\n"
    #     f"boxes:\n{boxes}\n\n"
    #     f"rotation_constraints:\n{env.rotation_constraints}\n\n"
    #     f"packing_mask:\nNot available\n\n"    
    # )


    env_reset = env.reset(boxes = boxes, rotation_constraints = rotation_constraints)

    plane_features_after_reset = env_reset[0]
    boxes_after_reset = env_reset[1]
    rotation_constraints_after_reset = env_reset[2]
    packing_mask = env_reset[3]

    # print(
    #     f"Environment after resetting:\n"
    #     f"plane_features:\n{plane_features_after_reset}\n\n"
    #     f"boxes:\n{boxes_after_reset}\n\n"
    #     f"rotation_constraints:\n{rotation_constraints}\n\n"
    #     f"packing_mask:\n{packing_mask}\n\n"
    # )
    assert np.array_equal(env.original_downsampled_plane_features, plane_features_after_reset)
    if boxes is not None:
        assert np.array_equal(boxes, boxes_after_reset)
    assert np.array_equal(env.rotation_constraints, rotation_constraints_after_reset)

    assert len(env_reset) == 4
    print("test_reset passed.")


def test_get_packing_mask(container_matrix = None, rotation_constraints = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    boxes = np.array([
        [1, 2, 3],
        [9, 9, 10]
    ])

    if rotation_constraints is None:                # In real life this would have been done in generate_boxes()
        rotation_constraints = [list(range(6)) for _ in range(len(boxes))]
    elif len(rotation_constraints) == 1:                                                                                    # Same rotation constraint for all boxes
                rotation_constraints = rotation_constraints * len(boxes)

    if env.downsampling_is_needed:
        _ = env.reset()

        max_indices = env.max_indices
        packing_mask = env.get_packing_mask(boxes, rotation_constraints, max_indices)
    else:
        packing_mask = env.get_packing_mask(boxes, rotation_constraints)

    assert isinstance(packing_mask, np.ndarray)
    assert packing_mask.shape[0] == len(boxes)
    # assert packing_mask.shape[1] == rotation_constraints.shape[1]         # The shape is not clear, as each box can have individual rotation constraints
    assert packing_mask.shape[2] == env.bin_size_ds_x
    assert packing_mask.shape[3] == env.bin_size_ds_y
    assert packing_mask.ndim in [4]
    assert packing_mask.dtype == bool or packing_mask.dtype == np.bool_
    
    # print(packing_mask)
    print("test_get_packing_mask passed")


def test_step(box_num = 1, container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  box_num,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    env.reset()
    action = (0, 0, 0)                                  # action = (position_indes, box_index, rotation_index)
    state, reward, done = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    # print(
    #     f"State after step:\n{state}\n\n"
    #     f"Reward after step:\n{reward}\n\n"
    #     f"Done after step?:\n{'Yes' if done else 'No'}\n\n"
    # )
    print("test_step passed.")


def test_generate_box_rotations():                      # Test for rotation_indices = [[]] is not needed as this is converted to [[0, 1, 2, 3, 4, 5]] in generate_boxes()
    boxes = np.array([[0, 1, 2],
                      [1, 2, 3]])
    num_boxes, num_dims = boxes.shape
    
    generated_rotations = generate_box_rotations(boxes)
    expected_rotations = np.array([[[0, 1, 2],
                                    [1, 0, 2],
                                    [2, 1, 0],
                                    [1, 2, 0],
                                    [0, 2, 1],
                                    [2, 0, 1]],

                                   [[1, 2, 3],
                                    [2, 1, 3],
                                    [3, 2, 1],
                                    [2, 3, 1],
                                    [1, 3, 2],
                                    [3, 1, 2]]])
    
    assert generated_rotations.shape == (num_boxes, 6, num_dims)
    assert np.array_equal(generated_rotations, expected_rotations)
    
    rotation_indices = [0, 2]
    generated_rotations_subset = generate_box_rotations(boxes, rotation_indices)
    expected_rotations_subset = np.array([[[0, 1, 2],
                                           [2, 1, 0]],

                                          [[1, 2, 3],
                                           [3, 2, 1]]])

    assert generated_rotations_subset.shape == (num_boxes, len(rotation_indices), num_dims)
    assert np.array_equal(generated_rotations_subset, expected_rotations_subset)

    print("test_generate_box_rotations passed.")



def test_contains_empty_list():
    assert contains_empty_list([]) == True
    assert contains_empty_list([[], 1]) == True
    assert contains_empty_list([1, 2, 3]) == False
    assert contains_empty_list([[1], [2]]) == False
    print("test_contains_empty_list passed.")




'''
    --- Testing the file network.py ---
'''


def test_box_embed(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    box_state = state[1]
    box_state = np.expand_dims(box_state, axis = 0)                                                         # Artificially adds extra dimension batch_size. For testing purposes it is 1
    model = Box_Embed(dim_model = params.dim_model,
                      dim_hidden_1 = params.box_embed_dim_hidden_1,
                      binary_dim = params.binary_dim
                    )
    box_embedding = model(box_state)
    # print(f"Box_Ebbed output:\n{box_embedding}\n")
    
    assert box_embedding.shape == (1, env.box_num, params.dim_model)                                        # Shape: [samples, num_boxes, dim_model]

    print("test_box_embed passed.")


def test_box_encoder(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    box_state = state[1]
    box_state = np.expand_dims(box_state, axis = 0)                                                         # Artificially adds extra dimension batch_size. For testing purposes it is 1
    encoder = Box_Encoder(dim_model = params.dim_model,
                          binary_dim = params.binary_dim
                        )
    box_encoding = encoder(box_state)
    # print(f"Box_Encoder output:\n{box_encoding}\n")

    assert box_encoding.shape == (1, env.box_num, params.dim_model)                                         # Shape: [samples, num_boxes, dim_model]

    print("test_box_encoder passed.")


def test_spatial_positional_encoding(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    dim_model = params.dim_model
    bin_size_x = env.bin_size_ds_x
    bin_size_y = env.bin_size_ds_y
    seq_len = bin_size_x * bin_size_y
    bin_embedding = torch.randn(params.batch_size, seq_len, dim_model)                                      # TODO: Better have real values, not randomly generated ones
    encoder = Spatial_Positional_Encoding(dim_model,
                                          bin_size_x,
                                          bin_size_y
                                        )
    out = encoder(bin_embedding)
    # print(f"Spatial_Positional_Encoding output:\n{out}\n")

    assert out.shape == (1, seq_len, params.dim_model)                                                      # Shape: [samples, num_boxes, dim_model]

    print("test_spatial_positional_encoding passed.")

    
def test_bin_encoder(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    plane_features = state[0]
    encoder = Bin_Encoder(dim_model = params.dim_model,
                          bin_size_x = env.bin_size_ds_x,                                                   # TODO: Shall that be the downsampled sizes?. Yes, if the ds_size is smaller than size 
                          bin_size_y = env.bin_size_ds_y,
                          plane_feature_dim = params.plane_feature_dim,
                          binary_dim = params.binary_dim
                        )
    out = encoder(plane_features)

    seq_len = env.bin_size_ds_x * env.bin_size_ds_y                                                         # Bin_encoder encodes the downsampled bis sizes
    # print(out)

    assert out.shape == (1, seq_len, params.dim_model)                                                      # Shape: [samples, seq_len, dim_model]

    print("test_spatial_positional_encoding passed.")


def test_transformer_encoder(container_matrix = None):                                                      # TODO: Check, whether all the shapes are chosen correctly
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    plane_features = state[0]                                                                               # Shape: [bin_x, bin_y, plane_feature_dim]
    _, _, plane_feature_dim = plane_features.shape
    seq_len = env.bin_size_ds_x * env.bin_size_ds_x                                                         # TODO: Really the downsampled stuff? Yes, if the ds_size is smaller than size 

    x = torch.tensor(plane_features, dtype=torch.float32).unsqueeze(0)                                      # Shape: [batch_size, bin_x, bin_y, plane_feature_dim]
    x = x.view(params.batch_size, seq_len, plane_feature_dim)
    embedding_layer = nn.Linear(plane_feature_dim, params.dim_model)
    x = embedding_layer(x)
    pos_encoder = Spatial_Positional_Encoding(params.dim_model, env.bin_size_ds_x, env.bin_size_ds_x)
    x = pos_encoder(x)                  
    
    model = Transformer_Encoder(dim_model = params.dim_model,
                                num_heads = params.transformer_encoder_num_head,
                                dim_hidden_1 = params.transformer_encoder_dim_hidden_1,
                                num_layers = params.transformer_encoder_num_layers
                            )
    out = model(x)
    # print(f"Transformer_Encoder output:\n{out}\n")
    
    assert out.shape == (1, seq_len, params.dim_model)                                                      # Shape: [batch_size, seq_len, dim_model]

    print("test_spatial_positional_encoding passed.") 


def test_transformer_decoder(container_matrix = None):                                                                         
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    num_boxes = env.box_num
    dim_model = params.dim_model
    
    query = torch.randn(params.batch_size, num_boxes, dim_model)                                            # Shape: [batch_size, seq_len, dim_model] --> seq_len: e.g. 10 positions or 10 boxes
    key = torch.randn(params.batch_size, num_boxes, dim_model)                                              # Shape: [batch_size, seq_len, dim_model] --> seq_len: can differ from query
    model = Transformer_Decoder(dim_model = params.dim_model,
                                num_heads = params.transformer_encoder_num_head,
                                dim_hidden_1 = params.transformer_decoder_dim_hidden_1,
                                num_layers = params.transformer_decoder_num_layers
                            )
    out = model(query, key)
    # print(f"Transformer_Decoder output:\n{out}\n")
    
    assert out.shape == (1, env.box_num, params.dim_model)                                                  # Shape: [batch_size, box_num, dim_model]

    print("test_transformer_decoder passed.") 

    
def test_position_selection(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    seq_len = env.bin_size_x * env.bin_size_y
    box_num = env.box_num 
    dim_model = params.dim_model   
    
    bin_encoder_out = torch.randn(params.batch_size, seq_len, dim_model)                                    # Shape: [batch_size, seq_len, dim_model]
    box_encoder_out = torch.randn(params.batch_size, box_num, dim_model)                                    # Shape: [batch_size, box_num, dim_model]
    decoder = Position_Selection(dim_model = params.dim_model)
    position_logits  = decoder(bin_encoder_out, box_encoder_out)
    # print(f"Position_Selection output:\n{position_logits}\n")

    assert bin_encoder_out.shape == (1, seq_len, params.dim_model)                                          # Shape: [batch_size, seq_len, dim_model]
    assert box_encoder_out.shape == (1, env.box_num, params.dim_model)                                      # Shape: [batch_size, box_num, dim_model]
    assert position_logits .shape == (1, seq_len)                                                           # Shape: [batch_size, seq_len] --> Position_Selection does squeeze(-1), so dim_model is lost

    print("test_position_selection passed.") 


def test_box_selection(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    seq_len = env.bin_size_x * env.bin_size_y
    box_num = env.box_num 
    dim_model = params.dim_model
    plane_feature_dim = params.plane_feature_dim   
    boxes = state[1]                                                                                        # Shape: [num_boxes, 3]
    boxes = np.expand_dims(boxes, axis = 0)                                                                 # add dimension: batch_size = 1
    bin_embedding = torch.randn(params.batch_size, seq_len, dim_model)                                      # TODO: Better have real values, not randomly generated ones
    
    box_encoder = Box_Encoder(dim_model = params.dim_model,
                              binary_dim = params.binary_dim
                            )
    box_encoding = box_encoder(boxes)                                                                       # Shape: [batch_size, box_num, dim_model]

    pos_encoder = Spatial_Positional_Encoding(dim_model,
                                              env.bin_size_x,
                                              env.bin_size_y
                                            )
    position_features = pos_encoder(bin_embedding)                                                          # Shape: [batch_size, seq_len, plane_feature_dim]
    
    # box_encoding_old = torch.randn(params.batch_size, box_num, dim_model)
    # position_features_old = torch.randn(params.batch_size, seq_len, plane_feature_dim)
    position_encoding = torch.randn(params.batch_size, seq_len, dim_model)                                  # Shape: [batch_size, seq_len, dim_model]
    decoder = Box_Selection(dim_model = params.dim_model, 
                            plane_feature_dim = params.plane_feature_dim,
                            binary_dim = params.binary_dim
                        )
    out = decoder(box_encoding, position_features, position_encoding)
    print(f"Box_Selection output:\n{out}\n")
    print(f"Box_Selection output shape:\n{out.shape}")


def test_rotation_selection(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    batch_size = params.batch_size
    dim_model = params.dim_model
    binary_dim = params.binary_dim
    seq_len = env.bin_size_ds_x * env.bin_size_ds_y
    plane_features = state[0]
    plane_feature_dim = 7                                                                                   # or params.plane_feature_dim? But then shape mismatch
    
    selected_box = state[1][0]                                                                              # Simulate the one selected box
    selected_box = torch.tensor(selected_box, dtype = torch.float32).unsqueeze(0)
    position_features = torch.tensor(plane_features, dtype = torch.float32).unsqueeze(0)
    position_features = position_features.view(batch_size, seq_len, plane_feature_dim)
    
    embedding_layer = nn.Linear(plane_feature_dim, dim_model)
    position_features = embedding_layer(position_features)
    
    pos_encoder = Spatial_Positional_Encoding(dim_model = dim_model,
                                              bin_size_x = env.bin_size_ds_x,
                                              bin_size_y = env.bin_size_ds_y
                                            )
    position_encoding = pos_encoder(position_features)

    # position_features_old = torch.randn(1, 81, 7)
    # position_encoding_old = torch.randn(batch_size, seq_len, dim_model)                                  # Shape: [batch_size, seq_len, dim_model]
    
    rotation_decoder = Rotation_Selection(dim_model = dim_model,
                                          plane_feature_dim = plane_feature_dim,
                                          binary_dim = binary_dim
                                        )
    out = rotation_decoder(selected_box, position_features, position_encoding)
    print(f"Rotation_Selection output:\n{out}\n")
    print(f"Rotation_Selection output shape:\n{out.shape}")


def test_actor():
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = tuple(
        torch.as_tensor(s, dtype = torch.float32, device = device) if i < 3 else torch.as_tensor(s, dtype = torch.bool, device = device)
        for i, s in enumerate(state)
    )


    actor = Actor(
        bin_size_x = env.bin_size_x,
        bin_size_y = env.bin_size_y,
        dim_model = params.dim_model,
        binary_dim = params.binary_dim,
        plane_feature_dim = state[0].shape[-1]
    )
    
    actor.to(device)

    bin_state = state[0].unsqueeze(0)                                                               # In prducitve use, batch_size would be already there (I hope)
    box_state = state[1].unsqueeze(0)
    rotation_constraints = state[2].unsqueeze(0)
    packing_mask = state[3].unsqueeze(0)

    state = (bin_state, box_state, rotation_constraints, packing_mask)
    
    with torch.no_grad():
        probabilities, action = actor(state)

    print("Actor output:")
    print("Probabilities (tuple of 3):", [p.shape for p in probabilities])
    print("Action (tuple):", [a.shape for a in action])


def test_critic(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = tuple(
        torch.as_tensor(s, dtype = torch.float32, device = device) if i < 3 else torch.as_tensor(s, dtype = torch.bool, device = device)
        for i, s in enumerate(state)
    )


    bin_state = state[0].unsqueeze(0)                                                               # In prducitve use, batch_size would be already there (I hope)
    box_state = state[1].unsqueeze(0)
    rotation_constraints = state[2].unsqueeze(0)
    packing_mask = state[3].unsqueeze(0)

    state = (bin_state, box_state, rotation_constraints, packing_mask)

    critic = Critic(bin_size_x = env.bin_size_x,
                    bin_size_y = env.bin_size_y,
                    dim_model = params.dim_model,
                    binary_dim = params.binary_dim,
                    plane_feature_dim = state[0].shape[-1]
                )
    
    critic.to(device)

    with torch.no_grad():
        value = critic(state)
    print(f"Critic value:\n{value}\n")
    print(f"Critic value shape:\n{value.shape}")


def test_box_selection_without_plane_features(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset()
    dim_model = params.dim_model
    seq_len = env.bin_size_x * env.bin_size_y
    boxes = state[1]                                                                                        # Shape: [num_boxes, 3]
    boxes = np.expand_dims(boxes, axis = 0)                                                                 # add dimension: batch_size = 1
    
    encoder = Box_Encoder(dim_model = params.dim_model,
                          binary_dim = params.binary_dim
                        )
    box_encoding = encoder(boxes)  
    
    # box_encoding_old = torch.randn(1, 3, 32)
    
    bin_embedding = torch.randn(params.batch_size, seq_len, dim_model)
    
    model = Box_Selection_Without_Plane_Features(dim_model = params.dim_model)
    out = model(box_encoding, bin_embedding)
    # print(f"Box_Selection_Without_Plane_Features output:\n{out}\n")
    
    assert out.shape == (1, env.box_num)                                                      # Shape: [batch_size, num_boxes]

    print("test_spatial_positional_encoding passed.") 










if __name__ == "__main__":
    
    container_matrix = np.array([
        [5,5,5,5,4,4,4,4,4],
        [5,5,5,5,4,4,4,4,4],
        [5,5,5,5,4,4,4,4,4],
        [2,2,2,2,2,2,0,3,3],
        [2,2,2,2,2,2,0,3,3],
        [2,2,2,2,2,2,0,3,3],
        [2,2,2,2,2,2,0,3,3],
        [2,2,2,2,2,2,0,3,3],
        [0,0,0,0,0,0,0,3,3]
    ], dtype = int)
    
    boxes = np.array([[1, 3, 2],
                      [2, 1, 2]])



    ''' environment.py '''

    # test_empty_environment_initialisation()                                                           # Works
    # test_prefilled_environment_initialisation(container_matrix)                                       # Works


    # test_get_distances_and_bin_features()                                                             # Works
    # test_get_distances_and_bin_features(container_matrix)                                             # Works
    
    # test_generate_boxes()                                                                             # Works
    # test_generate_boxes(container_matrix)                                                             # Works
    # test_generate_boxes(rotation_constraints = [[1]])                                                 # Works
    # test_generate_boxes(container_matrix, rotation_constraints = [[1]])                               # Works
    
    # test_downsampling()                                                                               # Works
    # test_downsampling(container_matrix)                                                               # Works
    
    # test_reset()                                                                                      # Works 
    # test_reset(container_matrix = container_matrix)                                                   # Works
    # test_reset(boxes = boxes)                                                                         # Works
    # test_reset(rotation_constraints = [[1]])                                                          # Does not work, which is good. No boxes but their constrints is kind of useless
    # test_reset(container_matrix = container_matrix, boxes = boxes)                                    # Works
    # test_reset(container_matrix = container_matrix, rotation_constraints = [[1]])                     # Does not work, which is good. No boxes but their constrints is kind of useless
    # test_reset(boxes = boxes, rotation_constraints = [[1]])                                           # Works
    # test_reset(container_matrix = container_matrix, boxes = boxes, rotation_constraints = [[1]])      # Works
    
    # test_get_packing_mask()                                                                           # Works
    # test_get_packing_mask(container_matrix = container_matrix)                                        # Works
    # test_get_packing_mask(rotation_constraints = [[0, 2]])                                            # Works
    # test_get_packing_mask(container_matrix = container_matrix, rotation_constraints = [[0, 2]])       # Works
    # test_get_packing_mask(rotation_constraints = [[0, 2], [0]])                                       # Works
    # test_get_packing_mask(container_matrix = container_matrix, rotation_constraints = [[0, 2], [0]])  # Works
    ''' Manually tested: No downsampling needed (same principle, but without the max_indices) '''
    
    # test_step(box_num = 1)                                                                            # Works
    # test_step(box_num = 1, container_matrix = container_matrix)                                       # Works
    # test_step(box_num = 2)                                                                            # Works
    # test_step(box_num = 2, container_matrix = container_matrix)                                       # Works
    
    # test_generate_box_rotations()                                                                     # Works
    
    # test_contains_empty_list()                                                                        # Works



    ''' network.py '''
    
    # test_box_embed()                                                                                  # Works
    # test_box_embed(container_matrix)                                                                  # Works
    
    # test_box_encoder()                                                                                # Works
    # test_box_encoder(container_matrix)                                                                # Works
    
    # test_spatial_positional_encoding()                                                                # Works
    # test_spatial_positional_encoding(container_matrix)                                                # Works
    
    # test_bin_encoder()                                                                                # Works
    # test_bin_encoder(container_matrix)                                                                # Works
   
    # test_transformer_encoder()                                                                        # Works
    # test_transformer_encoder(container_matrix)                                                        # Works
    
    # test_transformer_decoder()                                                                        # Works
    # test_transformer_decoder(container_matrix)                                                        # Works
    
    # test_position_selection()                                                                         # Works
    # test_position_selection(container_matrix)                                                         # Works
    
    # test_box_selection()
    
    # test_rotation_selection()
    
    # test_actor()
    
    test_critic()
    
    # test_box_selection_without_plane_features()                                                       # Works
    # test_box_selection_without_plane_features(container_matrix)                                       # Works
    


    ''' agent.py '''
    
    #