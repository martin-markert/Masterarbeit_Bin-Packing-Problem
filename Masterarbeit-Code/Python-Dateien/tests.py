import parameters as p

from environment import *
from network import *

import torch
import torch.nn as nn
import numpy as np

'''
    --- Testing the file environment.py
'''

def test_env_with_matrix(container_matrix):
    bin_size_x, bin_size_y = container_matrix.shape
    bin_size_z = np.max(container_matrix) + 1
    env = Environment(
        bin_size_x = bin_size_x,
        bin_size_y = bin_size_y,
        bin_size_z = bin_size_z,
        bin_size_ds_x = 3,
        bin_size_ds_y = 3,
        box_num = 2,
        # min_factor = 0.1,
        # max_factor = 0.5,
        rotation_constraints = None,                                # Or Use [[1,2,3]] or [[1], [2, 4]]. TODO: [[]] or [[], []] has not been tested, yet
        bin_height_if_not_start_with_all_zeros = container_matrix   # for debugging only, in real life one starts with an empty box
    )

    bin_features = env.get_bin_features(container_matrix)


    # print("\n=== Original bin features ===")
    # print(bin_features)
   

    # Ausgabe formatieren:
    print("\n=== Original bin features ===")
    for x in range(bin_size_x):
        row = " ".join(
            f"{tuple(int(f) for f in bin_features[x, y, :])}" 
            for y in range(bin_size_y)
        )
        print(row)
    

    
    
    bin_state_ds, indices_of_largest_values = env.downsampling(bin_features)

    print("\n=== Downsampled bin_state ===")
    print(bin_state_ds)

    print("\n=== Max indices ===")
    print(indices_of_largest_values)


    print("\n=== Downsampled Bin State (bin_state_ds) ===")
    for x in range(bin_state_ds.shape[0]):
        row = " ".join(
            f"{tuple(int(f) for f in bin_state_ds[x, y, :])}"
            for y in range(bin_state_ds.shape[1])
        )
        print(row)

    print("\n=== Max indices ===")
    print(indices_of_largest_values.reshape(bin_state_ds.shape[0], bin_state_ds.shape[1]))



    boxes, rotation_constraints = env.generate_boxes(env.bin_size_x, env.bin_size_x, env.min_factor, env.max_factor, env.box_num, env.rotation_constraints)
    
    print("\n=== Box array ===")
    print(boxes)



    mask = env.get_packing_mask(boxes, rotation_constraints, indices_of_largest_values)
    
    print("\n=== Mask ===")
    print(mask)   



    reset_plane_features, reset_boxes, reset_rotation_constraints, reset_packing_mask = env.reset(boxes)
    print("\n=== Reset ===")
    print("\nReset Plane features:")
    for plane in reset_plane_features:
        row_strings = []
        for cell in plane:
            row_strings.append(f"({', '.join(str(int(x)) for x in cell)})")
        print(" ".join(row_strings))

    print(f"\n\nReset boxes:\n{reset_boxes} \n\nReset rotation constraints:\n{reset_rotation_constraints} \n\nReset packing mask:\n{reset_packing_mask}")


    for _ in range(env.box_num):
        dummy_action = (0, 0, 0)

        state, reward, done = env.step(dummy_action)
        print("\n=== Step ===")
        print("State:", state)
        print("Reward:", reward)
        print("Done:", done)




'''
    --- Testing the file network.py
'''


# --- Environment Setup ---
def make_env_state():
    bin_size_x, bin_size_y, bin_size_z = 9, 9, 10
    env = Environment(
        bin_size_x = bin_size_x,
        bin_size_y = bin_size_y,
        bin_size_z = bin_size_z,
        bin_size_ds_x = 3,
        bin_size_ds_y = 3,
        box_num = 3,
        min_factor = 0.1,
        max_factor = 0.5,
        rotation_constraints = None,
        bin_height_if_not_start_with_all_zeros = np.zeros((bin_size_x, bin_size_y), dtype = int)
    )
    state = env.reset()
    # state = (plane_features, boxes, rotation_constraints, packing_mask)
    # plane_features = torch.tensor(state[0], dtype = torch.float32).unsqueeze(0)
    # boxes = torch.tensor(state[1], dtype = torch.long).unsqueeze(0)
    # rotation_constraints = state[2]
    # packing_mask = torch.tensor(state[3], dtype = torch.bool).unsqueeze(0)
    
    # return (plane_features, boxes, rotation_constraints, packing_mask), env       # env = (plane_features, self.boxes, self.rotation_constraints, packing_mask)
    return state, env


def test_box_embed():
    state, _ = make_env_state()
    box_state = state[1]
    box_state = np.expand_dims(box_state, axis = 0)                                 # Artificially adds extra dimension batch_size. For testing purposes it is 1
    model = Box_Embed(dim_model = params.dim_model,
                      dim_hidden_1 = params.box_embed_dim_hidden_1,
                      binary_dim = params.binary_dim
                    )
    out = model(box_state)
    print(f"Box_Embed output:\n{out}\n")
    print(f"Box_Embed output shape:\n{out.shape}")                                  # Shape: [samples, num_boxes, dim_model]


def test_box_encoder():
    state, _ = make_env_state()
    box_state = state[1]
    box_state = np.expand_dims(box_state, axis = 0)                                 # Artificially adds extra dimension batch_size. For testing purposes it is 1
    encoder = Box_Encoder(dim_model = params.dim_model,
                          binary_dim = params.binary_dim
                        )
    out = encoder(box_state)
    print(f"Box_Encoder output:\n{out}\n")
    print(f"Box_Encoder output shape:\n{out.shape}")                                # Shape: [samples, num_boxes, dim_model]


def test_spatial_positional_encoding():
    _, env = make_env_state()
    dim_model = params.dim_model
    bin_size_x = env.bin_size_x
    bin_size_y = env.bin_size_y
    seq_len = bin_size_x * bin_size_y
    input_embeddings = torch.randn(1, seq_len, dim_model)                           # TODO: Better have real ones, not randomly generated ones
    encoder = Spatial_Positional_Encoding(dim_model,
                                          bin_size_x,
                                          bin_size_y
                                        )
    out = encoder(input_embeddings)
    print(f"Spatial_Positional_Encoding output:\n{out}\n")
    print(f"Spatial_Positional_Encoding output shape:\n{out.shape}")                # Shape: [batch_size, seq_len, dim_model]

    
def test_bin_encoder():
    state, env = make_env_state()
    plane_features = state[0]
    encoder = Bin_Encoder(dim_model = params.dim_model,
                          bin_size_x = env.bin_size_x,
                          bin_size_y = env.bin_size_y,
                          plane_feature_dim = plane_features.shape[-1],
                          binary_dim = params.binary_dim
                        )
    out = encoder(plane_features)
    print(f"Bin_Encoder output:\n{out}\n")
    print(f"Bin_Encoder output shape:\n{out.shape}")    


def test_transformer_encoder():
    x = torch.randn(2, 10, 32)                                                      # Shape: [batch_size, seq_len, dim_model]          TODO :Update to real values                                    
    model = Transformer_Encoder(dim_model = params.dim_model,
                                num_heads = params.transformer_encoder_num_head,
                                dim_hidden_1 = params.transformer_encoder_dim_hidden_1,
                                num_layers = params.transformer_encoder_num_layers
                            )
    out = model(x)
    print(f"Transformer_Encoder output:\n{out}\n")
    print(f"Transformer_Encoder output shape:\n{out.shape}") 


def test_transformer_decoder():
    query = torch.randn(2, 10, 32)
    key = torch.randn(2, 15, 32)
    model = Transformer_Decoder(dim_model = params.dim_model,
                                num_heads = params.transformer_encoder_num_head,
                                dim_hidden_1 = params.transformer_decoder_dim_hidden_1,
                                num_layers = params.transformer_decoder_num_layers
                            )
    out = model(query, key)
    print(f"Transformer_Decoder output:\n{out}\n")
    print(f"Transformer_Decoder output shape:\n{out.shape}")

    
def test_position_selection():
    bin_encoder_out = torch.randn(1, 81, 32)
    box_encoder_out = torch.randn(1, 3, 32)
    decoder = Position_Selection(dim_model = params.dim_model
                                )
    out = decoder(bin_encoder_out, box_encoder_out)
    print(f"Position_Selection output:\n{out}\n")
    print(f"Position_Selection output shape:\n{out.shape}")


def test_box_selection():
    box_encoding = torch.randn(1, 3, 32)
    position_features = torch.randn(1, 81, 7)
    position_encoding = torch.randn(1, 81, 32)
    decoder = Box_Selection(dim_model = params.dim_model, 
                            plane_feature_dim = params.plane_feature_dim,
                            binary_dim = params.binary_dim
                        )
    out = decoder(box_encoding, position_features, position_encoding)
    print(f"Box_Selection output:\n{out}\n")
    print(f"Box_Selection output shape:\n{out.shape}")


def test_rotation_selection():
    box_rot_state = torch.randint(1, 10, (1, 6, 3))
    position_features = torch.randn(1, 81, 7)
    position_encoding = torch.randn(1, 81, 32)
    decoder = Rotation_Selection(dim_model = params.dim_model,
                                 dim_hidden_1 = params.rotation_selection_dim_hidden_1,
                                 plane_feature_dim = params.plane_feature_dim,
                                 binary_dim = params.binary_dim
                                )
    out = decoder(box_rot_state, position_features, position_encoding)
    print(f"Rotation_Selection output:\n{out}\n")
    print(f"Rotation_Selection output shape:\n{out.shape}")


def test_actor():
    state, env = make_env_state()
    actor = Actor(
        bin_size_x = env.bin_size_x,
        bin_size_y = env.bin_size_y,
        dim_model = params.dim_model,
        binary_dim = params.binary_dim,
        plane_feature_dim = state[0].shape[-1]
    )
    with torch.no_grad():
        probabilities, action = actor(state)
    print("Actor output:")
    print("Probabilities (tuple of 3):", [p.shape for p in probabilities])
    print("Action (tuple):", [a.shape for a in action])


def test_critic():
    state, env = make_env_state()
    critic = Critic(bin_size_x = env.bin_size_x,
                    bin_size_y = env.bin_size_y,
                    dim_model = params.dim_model,
                    binary_dim = params.binary_dim,
                    plane_feature_dim = state[0].shape[-1]
                )
    with torch.no_grad():
        value = critic(state)
    print(f"Critic value:\n{value}\n")
    print(f"Critic value shape:\n{value.shape}")


def test_box_selection_without_plane_features():
    box_encoding = torch.randn(1, 3, 32)
    bin_embedding = torch.randn(1, 81, 32)
    model = Box_Selection_Without_Plane_Features(dim_model = params.dim_model)
    out = model(box_encoding, bin_embedding)
    print(f"Box_Selection_Without_Plane_Features output:\n{out}\n")
    print(f"Box_Selection_Without_Plane_Features output shape:\n{out.shape}")










if __name__ == "__main__":
    
    ''' environment.py '''
    container_matrix = np.array([   # Add a check, whether height values are integers? Kind of useless, as it later will be generated anyway
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
    
    # test_env_with_matrix(container_matrix)


    ''' network.py '''
    
    # test_box_embed()                                                              # Works
    # test_box_encoder()                                                            # Works
    # test_spatial_positional_encoding()                                            # Works
    test_bin_encoder()
    # test_transformer_encoder()
    # test_transformer_decoder()
    # test_position_selection()
    # test_box_selection()
    # test_rotation_selection()
    # test_actor()
    # test_critic()
    # test_box_selection_without_plane_features()