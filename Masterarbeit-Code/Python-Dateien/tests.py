from environment import *
from network import *

import torch
import torch.nn as nn
import numpy as np
import math
import copy
import warnings

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
        rotation_constraints = [[1], []],                                # Or Use [[1,2,3]] or [[1], [2, 4]]. TODO: [[]] or [[], []] has not been tested, yet
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



container_matrix = np.array([   # Add a check, whether height values are integers?
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





'''
    --- Testing the file network.py
'''

# -----------------------------
# Hyperparameter / Dummy Inputs
# -----------------------------
batch_size = 2
box_num = 3
dim_model = 8
binary_dim = 4
plane_feature_dim = 5
bin_size_x = 3
bin_size_y = 3
seq_len = bin_size_x * bin_size_y

# Dummy Inputs
boxes = torch.randint(1, 16, (batch_size, box_num, 3))
plane_features = torch.rand(batch_size, bin_size_x, bin_size_y, plane_feature_dim)
rotation_constraints = torch.zeros(batch_size, 6)
packing_mask = torch.zeros(batch_size, box_num, 6, bin_size_x, bin_size_y)

state = (plane_features, boxes, rotation_constraints, packing_mask)

print("=== Testing Helper Functions ===")
bin_tensor = convert_decimal_tensor_to_binary(boxes, binary_dim)
print("convert_decimal_tensor_to_binary output shape:", bin_tensor.shape)

indices = torch.tensor([0,1])
tensor_batch = torch.rand(2,3,4)
selected = select_values_by_indices(tensor_batch, indices)
print("select_values_by_indices output shape:", selected.shape)

rots = generate_box_rotations_torch(boxes)
print("generate_box_rotations_torch output shape:", rots.shape)

# -----------------------------
# Encoder Tests
# -----------------------------
print("\n=== Testing Encoders ===")
box_embed = Box_Embed(dim_model, dim_hidden_1=dim_model*2, binary_dim=binary_dim)
out = box_embed(boxes)
print("Box_Embed output shape:", out.shape)

box_encoder = Box_Encoder(dim_model, binary_dim=binary_dim)
out = box_encoder(boxes)
print("Box_Encoder output shape:", out.shape)

pos_enc = Spatial_Positional_Encoding(dim_model, bin_size_x, bin_size_y)
x = torch.rand(batch_size, seq_len, dim_model)
out = pos_enc(x)
print("Spatial_Positional_Encoding output shape:", out.shape)

bin_state_flat = plane_features.flatten(1,2)
bin_encoder = Bin_Encoder(dim_model, bin_size_x, bin_size_y, plane_feature_dim, binary_dim)
out = bin_encoder(bin_state_flat)
print("Bin_Encoder output shape:", out.shape)

transformer_enc = Transformer_Encoder(dim_model)
x = torch.rand(batch_size, 4, dim_model)
out = transformer_enc(x)
print("Transformer_Encoder output shape:", out.shape)

# -----------------------------
# Decoder Tests
# -----------------------------
print("\n=== Testing Decoders ===")
transformer_dec = Transformer_Decoder(dim_model)
query = torch.rand(batch_size, 2, dim_model)
key = torch.rand(batch_size, 4, dim_model)
out = transformer_dec(query, key)
print("Transformer_Decoder output shape:", out.shape)

position_sel = Position_Selection(dim_model)
position_logits = position_sel(bin_encoder(bin_state_flat), box_encoder(boxes))
print("Position_Selection logits shape:", position_logits.shape)

box_sel = Box_Selection(dim_model, plane_feature_dim=plane_feature_dim, binary_dim=binary_dim)
rotation_sel = Rotation_Selection(dim_model, dim_model*2, plane_feature_dim=plane_feature_dim, binary_dim=binary_dim)
pos_feat = torch.rand(batch_size, 1, dim_model)
pos_enc_tensor = torch.rand(batch_size, 1, dim_model)
box_logits = box_sel(box_encoder(boxes), pos_feat, pos_enc_tensor)
rotation_logits = rotation_sel(boxes, pos_feat, pos_enc_tensor)
print("Box_Selection logits shape:", box_logits.shape)
print("Rotation_Selection logits shape:", rotation_logits.shape)

# -----------------------------
# Actor & Critic Tests
# -----------------------------
print("\n=== Testing Actor & Critic ===")
actor = Actor(bin_size_x, bin_size_y)
critic = Critic(bin_size_x, bin_size_y)

probabilities, actions = actor(state)
value = critic(state)

print("Actor probabilities shapes:", [p.shape for p in probabilities])
print("Actor actions shapes:", [a.shape for a in actions])
print("Critic value shape:", value.shape)

# -----------------------------
# Softmax & Masking Test
# -----------------------------
print("\n=== Softmax & Masking Test ===")
mask = torch.tensor([[True, False, True]])
logits = torch.rand_like(mask, dtype=torch.float32)
masked_logits = torch.where(mask, -1e9, logits.float())
probs = torch.softmax(masked_logits, dim=-1)
print("Masked softmax probabilities:", probs)
print("Sum along last dim (should be 1):", probs.sum(dim=-1))


