import parameters as p

from environment import *
from network import *
from agent import *
from explore_environment import *

import torch
import torch.nn as nn
import numpy as np
import queue


'''
    --- Initialise the environment ---
'''

params = p.Parameters() 

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

    print("test_bin_encoder passed.")


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

    print("test_transformer_encoder passed.") 


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

# Boxes stuff
    boxes = state[1]                                                                                        # Shape: [num_boxes, 3]
    boxes = np.expand_dims(boxes, axis = 0)                                                                 # add dimension: batch_size = 1 --> [batch_size, num_boxes, 3]
    box_mask = boxes[:, :, 0] < 0
    box_mask = torch.tensor(box_mask, dtype = torch.float32)  
    
    box_encoder = Box_Encoder(dim_model = params.dim_model,
                              binary_dim = params.binary_dim
                            )
    box_encoding = box_encoder(boxes, box_mask)

# Plane-feature stuff
    bin_state = torch.tensor(state[0], dtype = torch.float32)                                               # In real life this should already be a PyTorch tensor
    bin_state = bin_state.unsqueeze(0)                                                                      # Shape: [1, 3, 3, 7]
    bin_state_flat = bin_state.flatten(1, 2)                                                                # Shape: [1, 3 * 3, 7]

    softmax_position_index = nn.Softmax(-1)
    position_action = Position_Selection(params.dim_model)

    bin_encoder = Bin_Encoder(params.dim_model, env.bin_size_ds_x, env.bin_size_ds_y, params.plane_feature_dim, params.binary_dim)
    bin_encoding = bin_encoder(bin_state_flat)

    position_logits = position_action(bin_encoding, box_encoding)

    packing_mask = torch.tensor(state[3], dtype = torch.float32)                                            # In real life this should already be a PyTorch tensor
    packing_mask = packing_mask.unsqueeze(0)  
    packing_mask = packing_mask.flatten(3, 4)

    position_mask = packing_mask.all(1).all(1)
    pos_mask_softmax = torch.where(position_mask, -1e9, 0.0)

    position_index_prob = softmax_position_index(position_logits + pos_mask_softmax)
    position_index = torch.multinomial(position_index_prob, num_samples = 1, replacement = True).squeeze(-1)

    plane_features = select_values_by_indices(bin_state_flat, position_index).unsqueeze(-2)

# Positional-encoding stuff
    position_encoding = select_values_by_indices(bin_encoding, position_index).unsqueeze(-2)

    decoder = Box_Selection(dim_model = params.dim_model,
                            plane_feature_dim = params.plane_feature_dim,
                            binary_dim = params.binary_dim
                        )
    
    box_selection_logits = decoder(box_encoding,                                                                       # Expected shape: (batch, seq_len)
                                   plane_features,
                                   position_encoding
                                )
    
    # print(box_encoding.shape)
    # print(plane_features.shape)
    # print(position_encoding.shape)
    # print(box_selection_logits)
    
    assert box_selection_logits.shape == (1, env.box_num)
    assert not torch.isnan(box_selection_logits).any()
    assert box_selection_logits.dtype == torch.float32

    print ("test_box_selection passed")


def test_rotation_selection(container_matrix = None, boxes = None, rotation_constraints = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset(boxes = boxes, rotation_constraints = rotation_constraints)

    rotation_select = Rotation_Selection(dim_model = params.dim_model,
                                         dim_hidden_1 = params.rotation_selection_dim_hidden_1,
                                         plane_feature_dim = params.plane_feature_dim,
                                         binary_dim = params.binary_dim
                                        )

# Box-encoder stuff
    boxes = state[1]                                                                                        # Shape: [num_boxes, 3]
    boxes = np.expand_dims(boxes, axis = 0)                                                                 # add dimension: batch_size = 1 --> [batch_size, num_boxes, 3]
    box_mask = boxes[:, :, 0] < 0
    box_mask = torch.tensor(box_mask, dtype = torch.float32) 

    rotation_constraints = state[2]
    box_rotation_shape_all = generate_box_rotations_torch(boxes, rotation_constraints = rotation_constraints)

    softmax_box_index = nn.Softmax(-1)

    select_box_action = Box_Selection(dim_model = params.dim_model,
                                      plane_feature_dim = params.plane_feature_dim,
                                      binary_dim = params.binary_dim
                                    )
    
    box_encoder = Box_Encoder(dim_model = params.dim_model, binary_dim = params.binary_dim)
    box_encoding = box_encoder(boxes, box_mask)

# Position-feature stuff
    bin_state = torch.tensor(state[0], dtype = torch.float32)                                               # In real life this should already be a PyTorch tensor
    bin_state = bin_state.unsqueeze(0)                                                                      # Shape: [1, 3, 3, 7]
    bin_state_flat = bin_state.flatten(1, 2)                                                                # Shape: [1, 3 * 3, 7]
    
    softmax_position_index = nn.Softmax(-1)

    position_action = Position_Selection(dim_model = params.dim_model)
    bin_encoder = Bin_Encoder(dim_model = params.dim_model,
                              bin_size_x = env.bin_size_ds_x,
                              bin_size_y = env.bin_size_ds_x,
                              plane_feature_dim = params.plane_feature_dim,
                            binary_dim = params.binary_dim
                            )

    bin_encoding = bin_encoder(bin_state_flat)
    position_logits = position_action(bin_encoding, box_encoding)

    packing_mask = torch.tensor(state[3], dtype = torch.float32)                                            # In real life this should already be a PyTorch tensor
    packing_mask = packing_mask.unsqueeze(0)  
    packing_mask = packing_mask.flatten(3, 4)

    position_mask = packing_mask.all(1).all(1)
    pos_mask_softmax = torch.where(position_mask, -1e9, 0.0)

    position_index_prob = softmax_position_index(position_logits + pos_mask_softmax)
    position_index = torch.multinomial(position_index_prob, num_samples = 1, replacement = True).squeeze(-1)
    
    position_feature = select_values_by_indices(bin_state_flat, position_index).unsqueeze(-2)

# Position-encoder stuff
    position_encoder = select_values_by_indices(bin_encoding, position_index).unsqueeze(-2)
    
    
    
    box_index_logits = select_box_action(box_encoding, position_feature, position_encoder)

    box_select_mask_all = packing_mask.all(2).transpose(1, 2)
    box_select_mask = select_values_by_indices(box_select_mask_all, position_index)
    box_softmax_mask = torch.where(box_select_mask, -1e9, 0.0)

    box_index_prob = softmax_box_index(box_index_logits + box_softmax_mask)
    box_index = torch.multinomial(box_index_prob, num_samples = 1, replacement = True).squeeze(-1)
    
    rotation_index = select_values_by_indices(box_rotation_shape_all, box_index)
    plane_features = select_values_by_indices(bin_state_flat, position_index).unsqueeze(-2)
    position_encoding = select_values_by_indices(bin_encoding, position_index).unsqueeze(-2)

    rotation_logits = rotation_select(rotation_index,
                                      plane_features,
                                      position_encoding
                                    )
    
    # print(rotation_logits)

    assert rotation_logits.ndim == 2
    assert rotation_logits.shape[0] == 1
    assert not torch.isnan(rotation_logits).any()

    print ("test_rotation_selection passed")


def test_actor(container_matrix = None, boxes = None, rotation_constraints = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset(boxes, rotation_constraints)

    bin_state = torch.tensor(state[0], dtype = torch.float32)
    bin_state = bin_state.unsqueeze(0)
    box_state = torch.tensor(state[1], dtype = torch.float32)
    box_state = box_state.unsqueeze(0)
    rotation_constraints = state[2]
    # rotation_constraints = torch.tensor(state[2], dtype = torch.float32)
    # rotation_constraints = rotation_constraints.unsqueeze(0)
    packing_mask = torch.tensor(state[3], dtype = torch.float32)
    packing_mask = packing_mask.unsqueeze(0)

    state = (bin_state, box_state, rotation_constraints, packing_mask)

    actor = Actor(bin_size_x = env.bin_size_ds_x,
                  bin_size_y = env.bin_size_ds_y,
                  dim_model = params.dim_model,
                  binary_dim = params.binary_dim,
                  plane_feature_dim = params.plane_feature_dim
                )

    probabilities, action = actor(state)

    print(
            f"Box probabilities:\t{probabilities[0]}\n"
            f"Position probabilities:\t{probabilities[1]}\n"
            f"Rotation probabilities:\t{probabilities[2]}\n\n"
        )
    print(
            f"Box index:\t{action[0]}\n"
            f"Position index:\t{action[1]}\n"
            f"Rotation index:\t{action[2]}"
        )

    print("test_actor passed")

    assert isinstance(probabilities, tuple) and len(probabilities) == 3
    assert all(torch.is_tensor(p) for p in probabilities)
    assert all((p >= 0).all() for p in probabilities)
    assert isinstance(action, tuple) and len(action) == 3
    for a in action:
        assert torch.is_tensor(a)


# def test_critic(container_matrix = None):
#     env = Environment(
#         bin_size_x    =  9,
#         bin_size_y    =  9,
#         bin_size_z    = 10,
#         bin_size_ds_x =  3,
#         bin_size_ds_y =  3,
#         box_num       =  2,
#         bin_height_if_not_start_with_all_zeros = container_matrix
#     )

#     state = env.reset()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     state = tuple(
#         torch.as_tensor(s, dtype = torch.float32, device = device) if i < 3 else torch.as_tensor(s, dtype = torch.bool, device = device)
#         for i, s in enumerate(state)
#     )


#     bin_state = state[0].unsqueeze(0)                                                               # In prducitve use, batch_size would be already there (I hope)
#     box_state = state[1].unsqueeze(0)
#     rotation_constraints = state[2].unsqueeze(0)
#     packing_mask = state[3].unsqueeze(0)

#     state = (bin_state, box_state, rotation_constraints, packing_mask)

#     critic = Critic(bin_size_x = env.bin_size_x,
#                     bin_size_y = env.bin_size_y,
#                     dim_model = params.dim_model,
#                     binary_dim = params.binary_dim,
#                     plane_feature_dim = state[0].shape[-1]
#                 )
    
#     critic.to(device)

#     with torch.no_grad():
#         value = critic(state)
#     print(f"Critic value:\n{value}\n")
#     print(f"Critic value shape:\n{value.shape}")


def test_critic(container_matrix = None, boxes = None, rotation_constraints = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    state = env.reset(boxes, rotation_constraints)

    bin_state = torch.tensor(state[0], dtype = torch.float32)
    bin_state = bin_state.unsqueeze(0)
    box_state = torch.tensor(state[1], dtype = torch.float32)
    box_state = box_state.unsqueeze(0)
    rotation_constraints = state[2]
    # rotation_constraints = torch.tensor(state[2], dtype = torch.float32)
    # rotation_constraints = rotation_constraints.unsqueeze(0)
    packing_mask = torch.tensor(state[3], dtype = torch.float32)
    packing_mask = packing_mask.unsqueeze(0)

    state = (bin_state, box_state, rotation_constraints, packing_mask)
    
    critic = Critic(bin_size_x = env.bin_size_ds_x,
                    bin_size_y = env.bin_size_ds_y,
                    dim_model = params.dim_model,
                    binary_dim = params.binary_dim,
                    plane_feature_dim = state[0].shape[-1]
                    # plane_feature_dim = params.plane_feature_dim
                )
    
    value = critic(state)

    print(value)
    
    assert value.shape == (1, 1)
    assert not torch.isnan(value).any()

    print("test_critic passed")


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
    
    
    bin_embedding = torch.randn(params.batch_size, seq_len, dim_model)
    
    model = Box_Selection_Without_Plane_Features(dim_model = params.dim_model)
    out = model(box_encoding, bin_embedding)
    # print(f"Box_Selection_Without_Plane_Features output:\n{out}\n")
    
    assert out.shape == (1, env.box_num)                                                      # Shape: [batch_size, num_boxes]

    print("test_box_selection_without_plane_features passed.") 



'''
    --- Testing the file agent.py ---
'''


def test_agent_init(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    agent = Agent(bin_size_x = env.bin_size_ds_x,
                  bin_size_y = env.bin_size_ds_y,
                  learning_rate_actor = params.learning_rate_actor,
                  learning_rate_critic = params.learning_rate_critic
                )

    assert isinstance(agent.actor, torch.nn.Module)
    assert isinstance(agent.critic, torch.nn.Module)

    print("test_agent_init passed")


def test_update_net(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    agent = Agent(bin_size_x = env.bin_size_ds_x,
                  bin_size_y = env.bin_size_ds_y,
                  learning_rate_actor = params.learning_rate_actor,
                  learning_rate_critic = params.learning_rate_critic
                )
    # Dummy buffer
    buffer_len = 4
    state_dim = (9*9, params.plane_feature_dim)

    state = env.reset()
    action = (0, 0, 0)
    
    dummy_state = np.random.randn(buffer_len, *state_dim).astype(np.float32)
    dummy_action = np.random.randn(buffer_len, 3).astype(np.float32)
    dummy_probs = np.random.rand(buffer_len, 3).astype(np.float32)
    dummy_rewards = np.random.randn(buffer_len).astype(np.float32)
    dummy_masks = np.random.rand(buffer_len).astype(np.float32)
    
    buffer = [np.stack([dummy_state]*3, axis = 1),
              np.stack([dummy_action]*3, axis = 1),
              np.stack([dummy_probs]*3, axis = 1),
              dummy_rewards,
              dummy_masks]
    
    loss_critic, loss_actor, logprob_mean = agent.update_net(buffer, batch_size = params.batch_size, repeat_times = 1)
    assert isinstance(loss_critic, float)
    assert isinstance(loss_actor, float)
    assert isinstance(logprob_mean, float)


def test_get_reward_sum(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )
    
    agent = Agent(
        bin_size_x = env.bin_size_x,
        bin_size_y = env.bin_size_y,
        learning_rate_actor = params.learning_rate_actor,
        learning_rate_critic = params.learning_rate_critic
    )

    buffer_len = 5
    rewards = np.ones(buffer_len, dtype = np.float32)
    masks = np.ones(buffer_len, dtype = np.float32) * params.discount_factor
    values = np.zeros(buffer_len, dtype = np.float32)
    
    sum_rewards, advantages = agent.get_reward_sum(buffer_len, masks, rewards, values)

    print(sum_rewards)
    print(advantages)

    assert sum_rewards.shape[0] == buffer_len
    assert advantages.shape[0] == buffer_len
    assert isinstance(sum_rewards[0], np.float32)

    print("test_get_reward_sum passed")


def test_optimiser_update(container_matrix = None):
    env = Environment(
        bin_size_x    =  9,
        bin_size_y    =  9,
        bin_size_z    = 10,
        bin_size_ds_x =  3,
        bin_size_ds_y =  3,
        box_num       =  2,
        bin_height_if_not_start_with_all_zeros = container_matrix
    )

    actor = Actor(bin_size_x = env.bin_size_ds_x,
                  bin_size_y = env.bin_size_ds_y,
                  dim_model = params.dim_model,
                  binary_dim = params.binary_dim,
                  plane_feature_dim = params.plane_feature_dim
                )
    
    critic = Critic(bin_size_x = env.bin_size_ds_x,
                    bin_size_y = env.bin_size_ds_y,
                    dim_model = params.dim_model,
                    binary_dim = params.binary_dim,
                    plane_feature_dim = params.plane_feature_dim
                )
    
    model = torch.nn.Linear(5, 2)
    actor_optimiser = torch.optim.Adam(actor.parameters(), params.learning_rate_actor)
    critic_optimiser = torch.optim.Adam(critic.parameters(), params.learning_rate_actor)
    x = torch.randn(3, 5)
    y = torch.randn(3, 2)
    criterion = torch.nn.MSELoss()
    loss = criterion(model(x), y)
    
    # Agent.optimiser_update(actor_optimiser, loss)                                         # Only one of both work at the same time
    Agent.optimiser_update(critic_optimiser, loss)

    print("test_optimiser_update passed")



'''
    --- Testing the file explore_environment.py ---
'''

def test_explore_environment():
    action_queue = queue.Queue()
    result_queue = queue.Queue()

    action_queue.put(False)        # Reset signal                                           #  TODO: Use real values
    result_queue.put([0, 0, 0])    # First "action" (Dummy)                                 #  TODO: Use real values

    explore_environment(action_queue,
                        result_queue,
                        bin_size_x           =  9,
                        bin_size_y           =  9,
                        bin_size_z           = 10,
                        bin_size_ds_x        =  3,
                        bin_size_ds_y        =  3,
                        box_num              =  2,
                        min_factor           =  0.1,
                        max_factor           =  0.5,
                        rotation_constraints = None,
                        number_of_iterations = 10
                    )
    
    state, reward, done, use_ratio = result_queue.get()
    
    print(f"Entries in state: {len(state)}")
    print("Reward:", reward)
    print("Done:", done)
    print("Use ratio:", use_ratio)


def test_solve_problem():
    action_queue = queue.Queue()
    result_queue = queue.Queue()

    action_queue.put(False)        # Reset signal                                           #  TODO: Use real values
    result_queue.put([0, 0, 0])    # First "action" (Dummy)                                 #  TODO: Use real values

    env = Environment(
        bin_size_x           =  9,
        bin_size_y           =  9,
        bin_size_z           = 10,
        bin_size_ds_x        = 3,
        bin_size_ds_y        = 3,
        box_num              = 2,
        min_factor           = 0.1,
        max_factor           = 0.5,
        rotation_constraints = None
    )

    # Two part instances
    box_list = [
        boxes,
        boxes,
    ]

    solve_problem(action_queue, result_queue, box_list, env)

    next_state, reward, done, use_ratio, packing_result = result_queue.get()

    print(f"Entries is state: {len(next_state)}")
    print("Reward:", reward)
    print("Done:", done)
    print("Use ration:", use_ratio)
    print("Packing result:", packing_result)



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
    
    # test_box_selection()                                                                              # Works
    # test_box_selection(container_matrix)                                                              # Works
    
    # test_rotation_selection()                                                                         # Works
    # test_rotation_selection(container_matrix)                                                         # Works
    # test_rotation_selection(boxes = boxes, rotation_constraints = [[1], [0, 1]])                      # Works
    # test_rotation_selection(container_matrix, boxes, [[1], [0, 1]])                                   # Works  
    
    # test_actor()                                                                                      # Works
    # test_actor(container_matrix)                                                                      # Works
    # test_actor(boxes = boxes, rotation_constraints = [[1], [0, 1]])
    # test_actor(container_matrix, boxes, [[1], [0, 1]]) 
    
    # test_critic()
    # test_critic(container_matrix)
    # test_critic(boxes = boxes, rotation_constraints = [[1], [0, 1]])
    # test_critic(container_matrix, boxes, [[1], [0, 1]]) 

    
    # test_box_selection_without_plane_features()                                                       # Works
    # test_box_selection_without_plane_features(container_matrix)                                       # Works
    


    ''' agent.py '''
    
    # test_agent_init()                                                                                 # Works
    # test_agent_init(container_matrix)                                                                 # Works

    # TODO test_explore_environment_multiprocessing()

    # test_update_net()

    # test_get_reward_sum()                                                                             # Works. TODO: Test with real inputs, not dummy ones
    # test_get_reward_sum(container_matrix)                                                             # Works. TODO: Test with real inputs, not dummy ones

    # test_optimiser_update()                                                                           # Works
    # test_optimiser_update(container_matrix)                                                           # Works



    ''' explore_environment.py '''

    # test_explore_environment()                                                                        # TODO: Test with real inputs, not dummy ones

    # test_solve_problem()                                                                              #TODO: Test with real inputs, not dummy ones