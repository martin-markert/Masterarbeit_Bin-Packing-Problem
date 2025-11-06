import parameters as p

import warnings
import numpy as np
import torch
import torch.nn as nn 

# TODO: Rotations
# TODO: Multiple containers with their boxes to choose from --> New container-decision decoder or part of the exisiting container decoder?

params = p.Parameters()                                                                                         # Dimension parameters



'''
    --- All the stuff needed for the Encoders ---
'''

class Box_Embed(nn.Module):                                                                                     # See chapter 3.1.3 --> Box encoder
    def __init__(self,
                 dim_model = params.dim_model,
                 dim_hidden_1 = params.box_embed_dim_hidden_1,                                                  # dim_hidden_x is the dimensionality of the hidden layer(s) of the feedforward neural network.                                                     
                                                                                                                # Add more hidden layers here, if needed 
                 binary_dim = params.binary_dim
                ):                                                                                                      
        super().__init__()

        self.binary_dim = binary_dim
        self.encoder = nn.Sequential(nn.Linear(binary_dim, dim_model), nn.Tanh(),                               # First layer: Makes the input that is [binary_dim] bits long [dim_model] bits long. Tanh() makes the input non-linear (between -1 and 1)
                                     nn.Linear(dim_model, dim_hidden_1), nn.Tanh(),                             # Second layer: [dim_model] --> [dim_hidden_1] --> The network learns complex non-linear relationships between bits.
        #                            nn.Linear(dim_hidden_1, dim_hidden_2), nn.Tanh(),                          # Extra layers if needed
        #                            ...
                                     nn.Linear(dim_hidden_1, dim_model))                                        # Last layer: [dim_hidden_1] --> [dim_model] --> Back to [dim_model], suitable for the transformer. Final embedddings for the transformer
                                                                                                                # TODO: Is Tanh() good to add non-linearity or should something else be used?
    ''' Why three layers?

        | Layers          | Advantage                                                    | Disadvantage                                                               |
        | --------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------- |
        | 1 layer         | Simple, fast                                                 | No non-linear relationships between L/W/H detectable                       |
        | 2 layers        | Acceptable, fewer parameters                                 | Possibly insufficient capacity for complex patterns                        |
        | ~3 layers       | Hopefully a good balance                                     | We will see ...                                                            |
        | 10+ layers      | Very high capacity                                           | Training problems, overfitting, unnecessary for small input size (~8 bits) |
        | Heaps of layers | Theoretically universal approximator, practically impossible | Memory, training, instability                                              |

    '''

    def forward(self, box_state):
        if not isinstance(box_state, torch.Tensor):
            box_state = torch.tensor(box_state, dtype = torch.float32)
        
        batch_size, box_number, _ = box_state.size()                                                            # _ would be L/B/H

        box_state = convert_decimal_tensor_to_binary(
             box_state, self.binary_dim).view(batch_size, box_number, 3, self.binary_dim)                       # .view() makes sure that the three box axes are kept apart
                                                                                                                # ([0,0,1,1,     ([[0,0,1,1], 
                                                                                                                #   0,0,0,1,  -->  [0,0,0,1],
                                                                                                                #   0,0,1,0])      [0,0,1,0]])

                                                                        
        box_embed = self.encoder(box_state)
        box_embed = torch.mean(box_embed, -2, keepdim = False)                                                  # Paper says: Since in our problem each box can be rotated, we expect that the boxes with the same dimension but different orientation will provide the same information to the network. 
                                                                                                                # Therefore, we perform the average operation on the second dimension of the sequence, resulting in a new N×dim_model sequence.
                                                                                                                # batch_size x box_number x dim_model
        return box_embed
    


class Box_Encoder(nn.Module):
    def __init__(self,
                 dim_model,
                 binary_dim = params.binary_dim
                ):
        super().__init__()

        self.embedding = Box_Embed(dim_model = dim_model, dim_hidden_1 = dim_model * 2, binary_dim = binary_dim)# Do the embedding
        self.transformer = Transformer_Encoder(dim_model)                                                       # Do the magic transformer stuff

    def forward(self, box_state, box_mask = None):
        box_embed = self.embedding(box_state)                                                                   # Take the boxes and do the embedding
        box_encoding = self.transformer(box_embed, box_mask)                                                    # Now each box has a contextualised representation that contains information about the other boxes.

        return box_encoding
    


class Spatial_Positional_Encoding(nn.Module):                                                                   # Calculates the positional encodings for the ontainer encoder (see Figure 3b)
        def __init__(self,
                     dim_model,
                     bin_size_x,
                     bin_size_y
                    ):
            super().__init__()

            if dim_model < 4 or dim_model % 4 != 0:
                raise ValueError(f"dim_model must be >= 4 and divisible by 4, got {dim_model}")                 # Later “pos_features = dim_model // 2” is called: pos_features needs to be divisible by 2, so pos[:, :, :, 0::2] and pos[:, :, :, 1::2] have the same avount of values

            dummy_tensor = torch.ones((1, bin_size_x, bin_size_y))                                              # Dummy tensor with all ones, e.g., dummy_tensor = [[1, 1, 1],   
                                                                                                                                                                  # [1, 1, 1],
                                                                                                                                                                  # [1, 1, 1]]

            x_embed = dummy_tensor.cumsum(2, dtype=torch.float32)                                               # Calculates cumulative indices for each column (x_embed) and row (y_embed) --> position values.
                                                                                                                # x_embed = [[1, 2, 3],                                                                                                                            
                                                                                                                           # [1, 2, 3],
                                                                                                                           # [1, 2, 3]]

            y_embed = dummy_tensor.cumsum(1, dtype=torch.float32)                                               # y_embed = [[1, 1, 1]                                                                                                                                                                                                                       # x_embed[0] = [[1, 2, 3],                                                                                                                            
                                                                                                                           # [2, 2, 2],
                                                                                                                           # [3, 3, 3]]
            
            pos_features = dim_model // 2                                                                       # Fixed positional encodings are used as in Parmar et al., 2018:
                                                                                                                # Since two coordinates need to be represented, d/2 of the dimensions is used to encode the row number and the other d/2 of the dimensions to encode the the column number
            
            dimension_tensor = torch.arange(pos_features, dtype = torch.float32)                                # Creates a 1D tensor [0, 1, 2, …, pos_features-1] as frequency scale for sine/cosine positional encoding from Vaswani et al., 2017
                                                                                                                # dimension_tensor is the divisor/frequencies for the sine/cosine functions (as in Vaswani et al., 2017).
            dimension_tensor = 10000 ** (2 * (torch.div(dimension_tensor, 2, rounding_mode='trunc') / pos_features))    # <-- See chapter 3.5 in Vaswani et al., 2017
            pos_x = x_embed[:, :, :, None] / dimension_tensor                                                   # Adds a new dimension to enable broadcasting across embedding dimensions, and divides positional indices by dimension_tensor to apply frequency scaling. This prepares the values for multi-frequency Sin/Cos positional encoding.
            pos_y = y_embed[:, :, :, None] / dimension_tensor
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim = 4).flatten(3)   # Indices [0, 2, 4, ...] --> as in Vaswani et al., 2017
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim = 4).flatten(3)   # Indices [1, 3, 5, ...]
                                                                                                                # torch.stack((…)) --> combines Sin and Cos results in a new dimension
                                                                                                                # dim = 4 --> additional dimension for pairing
                                                                                                                # flatten(3) --> combines Sin/Cos back into one dimension per position encoding
                                                                                                                # Result: Tensor (1, bin_x, bin_y, pos_feature) --> each position has unique encoding dimensions combined with Sin/Cos
            
            pos = torch.cat((pos_y, pos_x), dim = 3).flatten(1, 2)                                              # Combined y- and x-encodings --> dim_model dimensional per position.
                                                                                                                # Flatten: Sequence for transformer
                                                                                                                # Example: (1, 3, 3, 8) --> (1, 3*3, 8) --> (1, 9, 8)
            self.register_buffer('pos', pos)                                                                    # As the positional encodings are fixed (not learnable), they need to be saved as a fixed tensor. They are the same during the entire training
        
        def forward(self, bin_embedding):
            return bin_embedding + self.pos                                                                  # Add the corresponding position vector to each input embedding



class Bin_Encoder(nn.Module):                                                                                   # This class represents both: Bin_Embed and Bin_Encoder: 
                                                                                                                # The embedding of the bin is just linear, so it is easily implemented in this class, not a separate one 
    def __init__(self,
                 dim_model,
                 bin_size_x,
                 bin_size_y,
                 plane_feature_dim,
                 binary_dim                                                                                     # Why not binary_dim = params.binary_dim? In Box_Encoder: Boxes --> fixed discretisation (e.g., always 8 bits) --> set to default. In BinEncoder: Bin --> features variable, different numbers (I guess)
                ):    
        super().__init__()

        self.binary_dim = binary_dim
        self.positional_encoding_of_bin = Spatial_Positional_Encoding(dim_model, bin_size_x, bin_size_y)        # Fixed positional encoding needed for the bin coordinates
        self.embedding = nn.Linear(plane_feature_dim, dim_model)                                                # Linear, because no complicated internal representation is necessary. The features should only be projected into the dim_model space.
        self.transformer = Transformer_Encoder(dim_model)

    def forward(self, bin_state):
        bin_embedding = convert_decimal_tensor_to_binary(bin_state, self.binary_dim)
        bin_embedding = self.embedding(bin_embedding)
        bin_embedding = self.positional_encoding_of_bin(bin_embedding)
        bin_encoding = self.transformer(bin_embedding)

        return bin_encoding



class Transformer_Encoder(nn.Module):
    def __init__(self,
                 dim_model = params.dim_model,
                 num_heads = params.transformer_encoder_num_head,                                               # Amount of attention heads
                 dim_hidden_1 = params.transformer_encoder_dim_hidden_1,                                        # Hidden layers/feed-forward layers
                 dropout = params.transformer_encoder_dropout,
                 num_layers = params.transformer_encoder_num_layers):                                           # Amount of encoder layers (see “Nx” in Figure 1 in Vaswani et al., 2017)
        super().__init__()

        layers = nn.TransformerEncoderLayer(dim_model, num_heads, dim_hidden_1, dropout, batch_first = True)    # A single transformer layer (Multi-Head Self-Attention + Feedforward network (MLP) + Residual Connections + LayerNorm)
        self.transformer = nn.TransformerEncoder(layers, num_layers)                                            # Stack of num_layers encoder layers:
                                                                                                                # Executes the layers one after the other. Each layer receives the output of the previous layer as input.                                                                                                                 

    def forward(self, input_features, mask = None):                                                             # Optional: src_key_padding_mask to mask certain tokens (boxes or bins)
                                                                                                                # What happens here?
                                                                                                                # Self-attention per layer, Feedforward per layer, Residual + Norm, num-layers stack
                                                                                                                # Does this self.transformer_encoder_num_layers times
        
        contextualised_feature_vectors = self.transformer(input_features, src_key_padding_mask = mask)          # Contextualised feature vectors, i.e.:
                                                                                                                # Box encoder: Each box represents not only itself, but also its relationship to all other boxes
                                                                                                                # Bin encoder: Each bin cell represents not only its features, but also the “topology”

        return contextualised_feature_vectors



'''
    --- All the stuff needed for the Decoders ---
'''

class Transformer_Decoder(nn.Module):

    def __init__(self,        
                 dim_model = params.dim_model,
                 num_heads = params.transformer_decoder_num_head,                                               # Amount of attention heads
                 dim_hidden_1 = params.transformer_decoder_dim_hidden_1,                                        # Hidden layers/feed-forward layers
                 dropout = params.transformer_decoder_dropout,
                 num_layers = params.transformer_decoder_num_layers                                             # Amount of decoder layers (see “Nx” in Figure 1 in Vaswani et al., 2017)
                ):
        super().__init__()

        layers = nn.TransformerDecoderLayer(dim_model, num_heads, dim_hidden_1, dropout, batch_first = True)
        self.transformer = nn.TransformerDecoder(layers, num_layers)

    def forward(self, query, key, query_mask = None, key_mask = None):
        contextualised_feature_vectors = self.transformer(query, key, tgt_key_padding_mask = query_mask, memory_key_padding_mask = key_mask)

        return contextualised_feature_vectors



'''
    --- Selections of position, box and its rotation ---
    
    TODO: Choose from multiple containers
'''


class Position_Selection(nn.Module):
    def __init__(self,
                 dim_model
                ):
        super().__init__()

        self.position_decoder = Transformer_Decoder(dim_model)
        self.fully_connected = nn.Linear(dim_model, 1)

    def forward(self, bin_encoder, box_encoder):
        position_decoding = self.position_decoder(bin_encoder, box_encoder)                                     # Decoder(Query, Key, Value) = Decoder(bin_encoder, box_encoder, box_encoder) --> See chapter 3.1.3, Position decoder
                                                                                                                # Query: Represents the possible positions in the bin, i.e. the question: How well does a box fit here?
                                                                                                                # Key: What is important about me (my state) for a box to fit here?
                                                                                                                # Value: Here are my details, use them to evaluate the position.
        position_logits = self.fully_connected(position_decoding).squeeze(-1)

        return position_logits



class Box_Selection(nn.Module):                                                                                 # For the selection of the box the plane features are needed
                                                                                                                # The critic might get Box_Select_Action_Without_Plane_Features. Why?
                                                                                                                # Critic does not need the exact position features, only the box and bin embeddings, to estimate how “good” a state is.
    def __init__(self,
                 dim_model = params.dim_model,
                 dim_hidden_1 = params.box_selection_dim_hidden_1,
                                                                                                                # Add more hidden layers here, if needed
                 plane_feature_dim = params.plane_feature_dim,
                 binary_dim = params.binary_dim
                ):
        super().__init__()

        self.binary_dim = binary_dim
        self.position_embedding = nn.Sequential(nn.Linear(plane_feature_dim, dim_model), nn.Tanh(),             # Why this? Because the plane features are “raw” data they are embedded here in the dim_model space so that boxes can get attention to position information.
                                                nn.Linear(dim_model, dim_hidden_1), nn.Tanh(),
        #                                       nn.Linear(dim_hidden_1, dim_hidden_2), nn.Tanh(),
        #                                       ...
                                                nn.Linear(dim_hidden_1, dim_model))
        self.box_decoder = Transformer_Decoder(dim_model)
        self.fully_connected = nn.Linear(dim_model, 1)                                                          # Linear layer that generates a single score per position from the decoder output.

    def forward(self, box_encoding, position_features, position_encoding):  # <-- TODO Rename to position_embedding???
        position_features = convert_decimal_tensor_to_binary(position_features, self.binary_dim)                # Why this? See comment above
        position_embedding = self.position_embedding(position_features)
        position_embedding = position_embedding + position_encoding
        box_decoding = self.box_decoder(box_encoding, position_embedding)                                       # Decoder(Query, Key, Value) = Decoder(box_encoding, position_embedding, position_embedding) --> See chapter 3.1.3, Selection decoder
                                                                                                                # Query: The box asks: How well do I fit into the current bin situation?
                                                                                                                # Key: Which bin characteristics are relevant for selecting the box?
                                                                                                                # Value: The bin details used to evaluate the selection of the box

        box_logits = self.fully_connected(box_decoding).squeeze(-1)                                             # fully_connected() reduces the transformer output to one score per position (box_logits).

        return box_logits



class Rotation_Selection(nn.Module):
    def __init__(self,
                dim_model = params.dim_model,
                dim_hidden_1 = params.rotation_selection_dim_hidden_1,
                                                                                                                # Add more hidden layers here, if needed
                plane_feature_dim = params.plane_feature_dim,
                binary_dim = params.binary_dim
            ):
        super().__init__()


        self.binary_dim = binary_dim
        self.box_rotation_embedding = nn.Linear(3 * binary_dim, dim_model)                                      # 3 * binary_dim for L, W & H
        self.position_embedding = nn.Sequential(nn.Linear(plane_feature_dim, dim_model), nn.Tanh(),
                                                nn.Linear(dim_model, dim_hidden_1), nn.Tanh(),
        #                                       nn.Linear(dim_hidden_1, dim_hidden_2), nn.Tanh(),                  # Extra layers if needed
        #                                       ...
                                                nn.Linear(dim_hidden_1, dim_model))
        self.rotation_decoder = Transformer_Decoder(dim_model)
        self.fully_connected = nn.Linear(dim_model, 1)

    def forward(self, box_rot_state, position_features, position_encoder):
        position_features = convert_decimal_tensor_to_binary(position_features, self.binary_dim)
        box_rotation_features = convert_decimal_tensor_to_binary(box_rot_state, self.binary_dim)
        position_embedding = self.position_embedding(position_features)
        position_embedding = position_embedding + position_encoder
        box_rotation_embedding = self.box_rotation_embedding(box_rotation_features)
        rotation_decoding = self.rotation_decoder(box_rotation_embedding, position_embedding)                   # Decoder(Query, Key, Value) = Decoder(box_rotation_embedding, position_embedding, position_embedding) --> See chapter 3.1.3, Orientation decoder
        rotation_logits = self.fully_connected(rotation_decoding).squeeze(-1)

        return rotation_logits



'''
    --- Actor-critic network ---
'''

class Actor(nn.Module):
    def __init__(self,
                 bin_size_x,
                 bin_size_y,
                 dim_model = params.dim_model,
                 binary_dim = params.binary_dim,
                 plane_feature_dim = params.plane_feature_dim
                ):
        super().__init__()

        self.binary_dim = binary_dim
        self.plane_feature_dim = plane_feature_dim
        
        self.box_encoder = Box_Encoder(dim_model, binary_dim = self.binary_dim)
        self.bin_encoder = Bin_Encoder(dim_model, bin_size_x, bin_size_y, self.plane_feature_dim, self.binary_dim)
        
        self.softmax_position_index = nn.Softmax(-1)
        self.softmax_box_index = nn.Softmax(-1)
        self.softmax_rotation = nn.Softmax(-1)

        self.position_action = Position_Selection(dim_model)                                                    # Selection only dependant on bin_state, so everything else is not needed                                                           
        self.select_box_action = Box_Selection(dim_model, plane_feature_dim = self.plane_feature_dim, binary_dim = self.binary_dim)
        self.rotation_action = Rotation_Selection(dim_model, dim_model * 2, self.plane_feature_dim, self.binary_dim)

    '''
    Takes the state
    returns Positition, Box and its rotation

    The selection is made via Transformers + Linear Layer --> Logits --> Softmax --> Sampling.
    '''
    def forward(self, state, action_old = None):                                                                # state = (plane_features, boxes, rotation_constraints, packing_mask) as in file environment.py
                                                                                                                # Everything in forward() shall be torch, not np. array or whatever
                                                                                                                # action_old can be from calculating log probability in PPO
        '''
        Important: Everything in state geth an extra dimension "batch_number"
        .
        bin_state = (bin_size_x, bin_size_y, feature_num) --> (batch_size, bin_size_x, bin_size_y, feature_num)
        box_state = (box_num, 3) --> (batch_size, box_num, 3)
        rotation_constraints = (num_allowed_rotations) --> (batch_size, num_allowed_rotations)
        packing_mask = (unpacked_boxes_num, 6, bin_size_ds_x, bin_size_ds_y) --> (batch_size, unpacked_boxes_num, 6, bin_size_ds_x, bin_size_ds_y)

        Why? To allow starting the environment multiple times in parallel during training — i.e. several environments running simultaneously.



        Where will this be started?

        probably in the train.py there will be something like this:
        process_num = 1234567 
        action_queue_list = [Queue(maxsize=1) for _ in range(process_num)]
        result_queue_list = [Queue(maxsize=1) for _ in range(process_num)]
        
        state = list(map(list, zip(*state_list)))

        And the Agent will use something like this:
        result_list = [result_queue.get() for result_queue in result_queue_list]
        state_list = [result[0] for result in result_list]

        '''

        # Shall be torch.tensors (at least for flatten() to work)
        bin_state = state[0]
        box_state = state[1]
        rotation_constraints = state[2]                                                                         # TODO: Currently unused
        packing_mask = state[3].flatten(3, 4)                                                                   #  x and y put into one dimension    
        device = bin_state.device
        # batch_size = bin_state.size()[0]
                                                                                                                # Theoretically one might have multiple environment processes --> multiple trajectories simultaneously
                                                                                                                # Then bin_state contains multiple batch elements.
        box_mask = box_state[:, :, 0] < 0                                                                       # Placed boxes have constant_values = -1e9 (see step() in environment.py)
        rotation_mask = rotation_constraints[:, :, 0] < 0                                                       # TODO: Currently unused
        box_encoder = self.box_encoder(box_state, box_mask)
        bin_state_flat = bin_state.flatten(1, 2)
        bin_encoder = self.bin_encoder(bin_state_flat)

        ''' --- Position --- '''
        position_logits = self.position_action(bin_encoder, box_encoder)                                        # Returns the raw values for softmax for positions
        box_rotation_shape_all = generate_box_rotations_torch(box_state,                                        # Shape should be (batch_size, -1, 6, 3)
                                                              rotation_indices = rotation_constraints)          # Why torch and not np.array?
                                                                                                                # One cannot apply Torch operations to it. If one wants to use it in the Actor-forward() function, one has to copy tensors to the GPU (torch.from_numpy(...)) --> unnecessarily slower.
        position_mask = packing_mask.all(1).all(1)                                                              # packing_mask contains True for non-packable positions (see ~packing_available in environment.py).
                                                                                                                # This mask array is later used for softmax masking of the positions:
                                                                                                                # Start with (batch_size, unpacked_boxes_num, 6, bin_size_ds_x * bin_size_ds_y) --> .all() --> (batch_size, 6, bin_size_ds_x * bin_size_ds_y) --> .all() --> (batch_size, bin_size_ds_x * bin_size_ds_y)
                                                                                                                # first .all(1): What is done? position_mask[b, r, p] = True if all boxes b are invalid for this rotation r and position p.
                                                                                                                # second .all(1): What is done? position_mask[b, p] = True if no box b in any rotation can be packed at position p
                                                                                                                # So the end result can be interpreted as: Can any box in any rotation be packed at a certain position?

        position_mask_softmax = torch.where(position_mask, -1e9, 0.0)                                           # Masking before softmax. If a position is invalid (position_mask == True), then add -1e9 to the logits. If a position is valid (position_mask == False), then add nothing, i.e. 0.0.
        position_index_probabilities = self.softmax_position_index(position_logits + position_mask_softmax)     # Do the softmax to get the probabilities

        box_select_mask_all = packing_mask.all(2).transpose(1, 2)                                               # Maks for all boxes. True, when no rotation of a box is possible at this position
        box_rotation_mask_all = packing_mask.permute(0, 3, 1, 2).to(dtype = torch.bool)                         # Mask for all rotations. Changes the order of the dimensions so that rotation masks fit correctly to the boxes and positions.
                                                                                                                # Shows for each position if one box can be places in a cretain rotation

        if action_old != None:                                                                                  # Training (Exploitation) --> action_old is a tuple: (box_index, position_index, rotation_index)
            position_index = torch.as_tensor(action_old[1], dtype = torch.int64).to(device)
        else:                                                                                                   # Sampling (Exploration)
            position_index = torch.multinomial(position_index_probabilities, num_samples = 1, replacement = True).squeeze(-1)  

        position_feature = select_values_by_indices(bin_state_flat, position_index).unsqueeze(-2)               # Unsqueeze adds an extra dimension ()1. Now: shape: batch_size x 1 x feature_dim. Why? The transformer expects a sequence dimension (Batch × Seq × Features).
                                                                                                                # Basically: Selects the features of the selected position from each container in the batch. Formats the result so that it can be entered into the transformer as a sequence.

        position_encoder = select_values_by_indices(bin_encoder, position_index).unsqueeze(-2)                  # Returns the context-aware transformer embedding of the selected position. Used later in Box_Selection to condition the box decoding on this position.                                                                                                   

        ''' --- Box --- '''
        box_select_mask = select_values_by_indices(box_select_mask_all, position_index)                         # Shows which boxes are unavailable at this position for each batch.
        box_logits = self.select_box_action(box_encoder, position_feature, position_encoder)                    # Returns the raw values for softmax for boxes
        box_mask_softmax = torch.where(box_select_mask, -1e9, 0.0)                                              # Masking before softmax. If a box is invalid (box_select_mask == True), then add -1e9 to the logits. If a position is valid (box_select_mask == False), then add nothing, i.e. 0.0.
        box_index_probabilities = self.softmax_box_index(box_logits + box_mask_softmax)                         # Do the softmax to get the probabilities

        if action_old != None:
            box_index = torch.as_tensor(action_old[0], dtype = torch.int64).to(device)                          # action_old is a tuple: (box_index, position_index, rotation_index)
        else:
            box_index = torch.multinomial(box_index_probabilities, num_samples = 1, replacement = True).squeeze(-1)
        
        box_rotation_shape = select_values_by_indices(box_rotation_shape_all, box_index)                        # Selects rotations for selected box
        
        ''' --- Rotation --- '''
        rotation_logits = self.rotation_action(box_rotation_shape, position_feature, position_encoder)          # Returns the raw values for softmax for rotations

        rotation_mask_all = select_values_by_indices(box_rotation_mask_all, position_index)                     # Returns invalid rotations
        rotation_mask = select_values_by_indices(rotation_mask_all, box_index)                                  # Returns rotations of selected box
        rot_mask_softmax = torch.where(rotation_mask, -1e9, 0.0)                                                # Masking before softmax. If a rotation is invalid (rotation_mask == True), then add -1e9 to the logits. If a position is valid (rotation_mask == False), then add nothing, i.e. 0.0.
        rotation_probabilities = self.softmax_rotation(rotation_logits + rot_mask_softmax)                      # Do the softmax to get the probabilities

        if action_old != None:
            rotation_index = torch.as_tensor(action_old[2], dtype = torch.int64).to(device)                     # action_old is a tuple: (box_index, position_index, rotation_index)
        else:
            rotation_index = torch.multinomial(rotation_probabilities, num_samples = 1, replacement = True).squeeze(-1)
        
        # rotation_index = torch.multinomial(rotation_probabilities, num_samples = 1, replacement = True).squeeze(-1) # <-- Or maybe always choose the rotation deterministically for simplicity reasons? position_index and box_index --> important policy decisions, deterministic adoption for log probability. rotation_index --> relatively “local” decision, stochastic, covered later by log probability.

        probabilities = (box_index_probabilities, position_index_probabilities, rotation_probabilities)         # Softmax probabilities for positions, boxes, and rotations
        action = (box_index, position_index, rotation_index)                                                    # Indices of the selected position, box and its rotation

        return probabilities, action                                                                            # Probabilities are only needed for the PPO updates (Log-Prob + Entropy) --> Only for training/optimisation


    def get_action_and_probabilities(self, state):                                                              # state = (plane_features, boxes, rotation_constraints, packing_mask) as in file environment.py
        probabilities_of_actions, action = self.forward(state)                                                  # Simply calls forward() and returns the stuff in reverse order
        return action, probabilities_of_actions                                                                 # TODO: Needed?
                                                                                                              

class Critic(nn.Module):
    def __init__(self,
                 bin_size_x,
                 bin_size_y,
                 dim_model = params.dim_model,
                 binary_dim = params.binary_dim,
                 plane_feature_dim = params.plane_feature_dim
                ):
        super().__init__()

        self.binary_dim = binary_dim
        self.plane_feature_dim = plane_feature_dim
        
        self.box_encoder = Box_Encoder(dim_model, binary_dim = self.binary_dim)                                 # Chapter 3.1.3, Value network: The value network uses Transformer encoder–decoder architecture like the policy network [...]
        self.bin_encoder = Bin_Encoder(dim_model, bin_size_x, bin_size_y, self.plane_feature_dim, self.binary_dim)
        self.select_box_action = Box_Selection_Without_Plane_Features(dim_model)                                # BoxSelectAction Is “misused” here: In the actor, it is used to select actions. In the critic, it is used as a decoder between bin and box encodings to form a common representation. The decoder output delivers a tensor of the form (batch, bin_size_x*bin_size_y, dim_model) — i.e. a spatial value map. This means that the network learns: How good would it be to place any box at any position in the container?
                                                                                                                # TODO: Check real shape

        self.fully_connected = nn.Sequential(nn.Linear(bin_size_x * bin_size_y, bin_size_x),                    # Chapter 3.1.3, Value network: [...] to output a numerical value 𝑉().
                                             nn.ReLU(),                                                         # Why not nn.Tanh()? Because the critic needs to output any real number, not just between -1 and 1. Tanh would clip big or small values and mess up how the critic learns reward scales.
                                             nn.Linear(bin_size_x, 1))                                          # Decoder, --> value

    def forward(self, state):                                                                                   # state = (plane_features, boxes, rotation_constraints, packing_mask) as in file environment.py
        bin_state = state[0]
        box_state = state[1]
        # rotation_constraints = state[2]                                                                       # Chapter 3.1: The value network outputs a numerical value that represents the state value 𝑉𝜋(𝑠ₜ).
        # packing_mask = state[3]                                                                               # rotation_constraints = state[2] and
                                                                                                                # packing_mask = state[3] are not used. Just the current state. The rotation and the packing maks are not part of the value-calculation. They are just important for the packing choice
                                                                                                                # Advantages of using that: The Critic could assess more realistically how much space is still available and which boxes can still be placed sensibly. V(s) would reflect more precisely on which actions are still possible in this state.
                                                                                                                # Disadvantages of using it: 1. Potential training complexity
                                                                                                                #                               More input means larger networks or more complex transformer layers to integrate the masks meaningfully. There is a risk that the critic will become too tied to specific pack details and will generalise less well.
                                                                                                                #                            2. Change in the role of the critic:
                                                                                                                #                               Currently, the critic abstracts across boxes and bin states. With packing_mask, it partially approaches action-dependent information, which theoretically makes it ‘action-aware’ – similar to a Q-function Q(s,a), although it still evaluates a state.

        box_mask = box_state[:, :, 0] < 0                                                                       # Placed boxes have constant_values = -1e9 (see step() in envuronment.py)
        box_encoder = self.box_encoder(box_state, box_mask)
        bin_state_flat = bin_state.flatten(1, 2)
        bin_encoder = self.bin_encoder(bin_state_flat)

        value = self.select_box_action(bin_encoder, box_encoder)                                                # Take the current states of the bin and box and tell how good it is?
        '''
        Or just use the "normal" Box_selection with dummy plane_features?
        dummy_plane_features = torch.zeros(batch_size, num_positions, plane_feature_dim, device = bin_state.device)
        values = self.select_box_action(box_encoder, dummy_plane_features, bin_encoder)

        '''  

        value = self.fully_connected(value)

        return value
                                                                                                                



class Box_Selection_Without_Plane_Features(nn.Module):                                                          # Needed or should the critic just use Box_Selection?
    def __init__(self,
                 dim_model
                ):
        super().__init__()
        
        self.box_decoder = Transformer_Decoder(dim_model)
        self.fully_connected = nn.Linear(dim_model, 1)                                                          # Linear layer that generates a single score per position from the decoder output.

    def forward(self, box_encoding, bin_embedding):
        box_decoding = self.box_decoder(box_encoding, bin_embedding)                                            # Decoder(Query, Key, Value) = Decoder(box_encoding, bin_embedding, bin_embedding)
                                                                                                                # Query: The box asks: How well do I fit into the current bin situation?
                                                                                                                # Key: Which bin characteristics are relevant for selecting the box?
                                                                                                                # Value: The bin details used to evaluate the selection of the box
        
        box_logits = self.fully_connected(box_decoding).squeeze(-1)                                             # fully_connected() reduces the transformer output to one score per position (box_logits).

        return box_logits



'''
    --- Additional helping functions ---
'''

# def convert_decimal_tensor_to_binary_new_but_not_working(tensor, bit_length):                                   # Why can this conversion of tensors to binary be useful? It stabilises the input
#     tensor = tensor.long()                                                                                      # Safety machanism so that (tensor & mask) work for sure
#     max_val = 2 ** bit_length - 1                                                                               # Instead of specifying large values (e.g. width = 90) directly as scalars, they are passed as n-bit vectors (e.g. [0, 1, 0, 1, 1, 0, 1, 0]).
#     if (tensor > max_val).any():                                                                                # This stabilises training, prevents large number ranges and allows the transformer to recognise bitwise patterns.
#         warnings.warn(                                                                                          # TODO: Truncates from the left, if bit_length is too small. Problem?
#             f"At convert_tensor_to_binary() values in x are truncated to {bit_length} bits. "
#             f"Maximum representable decimal value: {max_val}", 
#             UserWarning
#         )
#     # mask = 2 ** torch.arange(bit_length - 1, -1, -1, device = tensor.device, dtype = tensor.dtype)
#     mask = (2 ** torch.arange(bit_length - 1, -1, -1, device=tensor.device)).long()
    
#     tensor_expanded = tensor.unsqueeze(-1)
#     tensor_bin = (tensor_expanded & mask) != 0
#     return tensor_bin.int()


def convert_decimal_tensor_to_binary(tensor, bit_length):
    
    max_val = 2 ** bit_length - 1                                                                               # Instead of specifying large values (e.g. width = 90) directly as scalars, they are passed as n-bit vectors (e.g. [0, 1, 0, 1, 1, 0, 1, 0]).
    if (tensor > max_val).any():                                                                                # This stabilises training, prevents large number ranges and allows the transformer to recognise bitwise patterns.
        warnings.warn(                                                                                          # TODO: Truncates from the left, if bit_length is too small. Problem?
            f"At convert_tensor_to_binary() values in x are truncated to {bit_length} bits.\n"
            f"The maximum representable decimal value with {bit_length} bits is {max_val}.\n"
            f"The given tensor has at least one value larger than {max_val}:\n{tensor}", 
            UserWarning
        )

    if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype = torch.float32)
    
    if bit_length == 1:
        tensor_encoding = tensor
    else:
        binary_list = []
        divide = tensor
        for _ in range(bit_length):
            binary = divide % 2
            binary_list.insert(0, binary)
            divide = torch.div(divide, 2, rounding_mode = 'trunc')

        tensor_encoding = torch.stack(binary_list, -1).flatten(-2, -1)
    return tensor_encoding


def select_values_by_indices(tensor_batch, indices):                                                            # Selects values from a tensor besed on the indices
    index = indices + torch.arange(0, tensor_batch.size(0)).to(indices.device) * tensor_batch.size(1)           # tensor_batch = torch.tensor([[[1], [2], [3]],
    tensor_select = tensor_batch.flatten(0, 1).index_select(0, index.to(torch.int))                             #                              [[4], [5], [6]]])
                                                                                                                # indices = torch.tensor([[0, 2],
                                                                                                                #                         [1, 2]])
                                                                                                                # Output: tensor([[1],
                                                                                                                #                 [3],
                                                                                                                #                 [5],
    return tensor_select                                                                                        #                 [6]])


# TODO: Check, whether the dimensions are returned correctly as needed [] or [[]] or [[[]]] ...
def generate_box_rotations_torch(boxes, rotation_constraints = None):                                           # TODO: Illegal rotations are returned as [0, 0, 0] to keep the dimension equal, when something like [[1], [2, 4]] is used. Is that a good solution?
                                                                                                                # Maybe have all rotations + an extra rotation_mask?
                                                                                                                # It should be documented that these rotations are interpreted as invalid.
    
    if isinstance(boxes, np.ndarray):
        boxes = torch.tensor(boxes)
    
    device = boxes.device
    batch_size, box_num, _ = boxes.shape
                                                                                                # Based on this coordinate system:
    base_rotations = torch.tensor([                                                             # z
        [0, 1, 2],  # 0: (x, y, z) --> Original State                                           # ^
        [1, 0, 2],  # 1: (y, x, z) --> Box rotated 90° around the height axis (z)               # |
        [2, 1, 0],  # 2: (z, y, x) --> Box tipped forward/backward                              # |_____> y
        [1, 2, 0],  # 3: (y, z, x) --> Box tipped forward/backward and then rotated 90°         #  \
        [0, 2, 1],  # 4: (x, z, y) --> Box tipped to the left or right                          #   \
        [2, 0, 1]   # 5: (z, x, y) --> Box tipped to the left or right and then rotated 90°     #    _|
    ], device = device)                                                                         #      x
                                                                                                #
                                                                                                #       ^
                                                                                                #       |
                                                                                                #     Viewer
    # Case 1: All rotations are allowed
    if rotation_constraints is None:
        allowed_rot = base_rotations
        num_rot = 6

    elif isinstance(rotation_constraints[0], list):
        # Case 2: Same rotation for all boxes
        if len(rotation_constraints) == 1:
            r_tensor = base_rotations[rotation_constraints[0]]
            allowed_rot = r_tensor.unsqueeze(0).expand(box_num, -1, -1)
            num_rot = allowed_rot.shape[1]
        else:
            # Case 3:Individual rotation constraints per box
            max_rot = max(len(r) for r in rotation_constraints)
            allowed_rot_list = []
            for r in rotation_constraints:
                r_tensor = base_rotations[r]
                if len(r) < max_rot:
                    pad = torch.zeros((max_rot - len(r), 3), device=device, dtype=torch.long)                   # Padding to [0, 0, 0], to keep the shape the same, so that torch is happy. I am not happy with that, yet
                    r_tensor = torch.cat([r_tensor, pad], dim=0)
                allowed_rot_list.append(r_tensor)
            allowed_rot = torch.stack(allowed_rot_list, dim=0)
            num_rot = allowed_rot.shape[1]

    else:
        raise ValueError(
                f"rotation_constraints must be None, list[int] --> [x, y, z] <--, or list[list[int]] --> [[v, w], [x, y, z]] <--. "
                f"Got {rotation_constraints}"
            )

    # Expand for Batch
    boxes_expand = boxes.unsqueeze(2).expand(-1, -1, num_rot, -1)
    if allowed_rot.dim() == 2:
        allowed_rot_expand = allowed_rot.unsqueeze(0).unsqueeze(0).expand(batch_size, box_num, -1, -1)
    else:
        allowed_rot_expand = allowed_rot.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return torch.gather(boxes_expand, dim = 3, index = allowed_rot_expand)                                      # Gather along the last dimension (x, y, z)



'''
    --- Deprecated functions ---
'''

class Position_Selection_Old(nn.Module):
    def __init__(self,
                 dim_model,
                 binary_dim
                ):
        super().__init__()

        '''
        Even if position is chosen first, the transformer must know the boxes in order to predict realistic, packable positions. 
        The first three lines serve this purpose: box context for position evaluation.
        '''
        self.box_embedding = Box_Embed(dim_model, dim_model * 2, binary_dim)
        self.other_boxes_encoder = Box_Encoder(dim_model, binary_dim = binary_dim)
        self.box_encoder = Transformer_Decoder(dim_model)                                                       # self.box_encoder is actually a transformer decoder, not an encoder in the traditional sense
                                                                                                                # Why is a decoder used?
                                                                                                                # - It takes the selected box as a query and the remaining boxes as memory.
                                                                                                                # - This allows it to specifically evaluate the selected box in relation to all other boxes.
                                                                                                                # - A normal transformer encoder couldn't do this because it treats the entire sequence equally.
                                                                                                                # Why the name box_encoder?
                                                                                                                # - The result is a box embedding that encodes the selected box in the context of all other boxes.
                                                                                                                # - So, in the end, the decoder produces an encoded box embedding – hence the name encoder.
        self.position_decoder = Transformer_Decoder(dim_model)
        self.fully_connected = nn.Linear(dim_model, 1)                                                          # Linear layer that generates a single score per position from the decoder output.

    def forward(self, box_shape, box_other_state, bin_encoder, box_other_mask):
        box_embedding = self.box_embedding(box_shape)                                                           # Embedding of a hypothetical selected box
        other_boxes_encoder = self.other_boxes_encoder(box_other_state, box_other_mask)                         # Encoding of the other hytpthetical boxes
        box_encoder = self.box_encoder(box_embedding, other_boxes_encoder)                                      # Combines selected hypothetical box with the other hypothetical boxes
        
        position_decoding = self.position_decoder(bin_encoder, box_encoder)                                     # Decoder(Query, Key, Value) = Decoder(bin_encoder, box_encoder, box_encoder) --> See chapter 3.1.3, Position decoder
                                                                                                                # Query: Represents the possible positions in the bin, i.e. the question: How well does a box fit here?
                                                                                                                # Key: What is important about me (my state) for a box to fit here?
                                                                                                                # Value: Here are my details, use them to evaluate the position.

        position_logits = self.fully_connected(position_decoding).squeeze(-1)                                   # fully_connected() reduces the transformer output to one score per position (position_logits).

        return position_logits
    





'''
    --- Tests ---
'''

    # Spatial_Positional_Encoding

# dim_model = 8
# bin_size_x = 3
# bin_size_y = 3
# seq_len = bin_size_x * bin_size_y

# batch_size = 1
# box_num = 3
# box_state = torch.tensor([[[1, 2, 3],
#                            [4, 5, 6],
#                            [7, 8, 9]]], dtype=torch.long)

# binary_dim = 4

# encoder = Spatial_Positional_Encoding(dim_model, bin_size_x, bin_size_y)
# x = torch.arange(batch_size * seq_len * dim_model, dtype=torch.float32).reshape(batch_size, seq_len, dim_model)
# out = encoder(x)
# print("SpatialPositionEncoding output:\n", out)