import numpy as np
import torch
import os

class Parameters():
    def __init__(self):
        
        '''
        Arguments for initialise(self)
        '''
        self.load_model    = True
        self.cwd           = "/home/markert/Masterarbeit-Code/Trainings"    # Current working directory     
        self.load_step     =          0
        self.random_seed   = np.random.randint(0, 2**32)
        self.num_threads   =          4                             # Determines how many GPU/CPU threads PyTorch uses internally



        '''
        Arguments for train.py
        '''
        self.break_step    =          2 ** 30
        self.process_num   =          4                             # Shall not be more than CPU kernels available


        '''
        Arguments for the Environment as stated in chapter 4.1
        '''
        self.bin_size_x    =       100                             # Like in the paper
        self.bin_size_y    =       100                             # Like in the paper
        self.bin_size_z    =   100_000                             # <-- As for now it can be anything
        self.bin_size_ds_x =        10                             # Like in the paper (ds = downsampled)
        self.bin_size_ds_y =        10                             # Like in the paper
        self.box_num       =       100                             # <-- Whatever your heart desires
        self.min_factor    =         0.1                           # Like in the paper (In the paper they divide by 10, I don't, I multiply by 1/10)
        self.max_factor    =         0.5                           # Like in the paper
        self.rotation_constraints = None                           # [[0, 1], [5], [0, 4, 2], [1, 2], [0, 1, 2, 3, 4, 5]]

        self.save_dir    = "{:d}_{:d}_{:d}_{:d}_{:.1f}_{:.1f}".format(self.bin_size_x, self.bin_size_y, self.bin_size_z, self.box_num, self.min_factor, self.max_factor)


        '''
        Arguments of the transformer-related things (network.py)
        Note, that mostly those values are guesses, as the authors did not state the dimensions of their best model

        TODO: Alter values when testing
        '''
        

        ''' General ''' 
        self.binary_dim                       =   16                        # Values not specified by authors --> Randon choice of dimensions (well not quite, it is an edicated guess). Will most likey be altered when testing   
        self.dim_model                        =  128          
        self.plane_feature_dim                =    7 * self.binary_dim      # 7 Plane features (6 features + height) * the binary_dim
        self.batch_size                       =   16
        self.gpu_id                           =    0                        # 0 --> first GPU, 1 second and so on. -1 --> CPU
        self.base_rotations = np.array([                                                            # Based on this coordinate system:
            [0, 1, 2],  # 0: (x, y, z) --> Original State                                           # z
            [1, 0, 2],  # 1: (y, x, z) --> Box rotated 90° around the height axis (z)               # ^
            [2, 1, 0],  # 2: (z, y, x) --> Box tipped forward/backward                              # |
            [1, 2, 0],  # 3: (y, z, x) --> Box tipped forward/backward and then rotated 90°         # |_____> y
            [0, 2, 1],  # 4: (x, z, y) --> Box tipped to the left or right                          #  \
            [2, 0, 1]   # 5: (z, x, y) --> Box tipped to the left or right and then rotated 90°     #    _|
        ])                                                                                          #      x
                                                                                                    #      ^
                                                                                                    #      |
                                                                                                    #    Viewer
        
        
        ''' Box_Embed() '''                                                 # See chapter 3.1.3 --> Box encoder
        self.box_embed_dim_hidden_1           =  256                        # Dimensionality of the hidden layer(s) of the feedforward neural network. Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        # self.box_embed_dim_hidden_2         =  256


        '''Box_Selection()'''
        self.box_selection_dim_hidden_1       =  256                        # Dimensionality of the hidden layer(s) of the feedforward neural network. Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        # self.box_selection_dim_hidden_2     =  256
        
        
        '''Rotation_Selection()'''
        self.rotation_selection_dim_hidden_1  =  256                        # Dimensionality of the hidden layer(s) of the feedforward neural network. Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        # self.rotation_selection_dim_hidden_2=  256

        
        ''' Transformer Encoder ''' 
        self.transformer_encoder_num_head     =    4                        # Number of heads in the multi-head-attention model (see Figure 2 in Vaswani et al., 2017). self.dim_model % self.num_head = 0 shall be the case
                                                                            # When one has several heads per layer the heads are independent of each other. This means that the model can learn different patterns with each head. 
                                                                            # For example, one head might pay most attention to the next word in each sentence, and another head might pay attention to how nouns and adjectives combine.
        self.transformer_encoder_dim_hidden_1 = 1024                        # Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        self.transformer_encoder_num_layers   =    2                        # Amount of encoder layers (see “Nx” in Figure 1 in Vaswani et al., 2017)
        self.transformer_encoder_dropout      =    0                        # Dropout: 
                                                                            # Usually ~ 0.1 - 0.3, but 0 could lead to better results in Reinforce Learning, especially as stable action probabilities are better/needed for PPO updates.


        ''' Transformer Decoder '''                                         # For symmety reasons the same dimensions as in the transformer encoder are taken. This can be altered, of course
                                                                            # Why? 
                                                                            # Uniform capacity: The transformer should have similar expressiveness for both boxes and position/rotation. 
                                                                            # Simplicity: Default values are the same --> easier hyperparameter tuning.
        self.transformer_decoder_num_head     = self.transformer_encoder_num_head
        self.transformer_decoder_dim_hidden_1 = self.transformer_encoder_dim_hidden_1
        self.transformer_decoder_num_layers   = self.transformer_encoder_num_layers
        self.transformer_decoder_dropout      = self.transformer_encoder_dropout


        ''' Agent '''
        self.ratio_clip                       =    0.12                     # Clip rate of ϵ as in chapter 4.1
        self.lambda_entropy                   =    0.02                     # A weight for entropy regularisation. Value not mentioned in the paper, so it is guessed.
        self.lambda_gae_adv                   =    0.95                     # λ for Generalized Advantage Estimation in PPO as in Schulman et al. - Proximal Policy Optimization Algorithms

        self.learning_rate_actor              =    1e-5                     # As in chapter 4.1
        self.learning_rate_critic             =    1e-4                     # As in chapter 4.1
        self.target_step                      = 1000                        # The agent acts target_step time steps in the environment --> The more boxes, the higher the number shall be
        self.reward_scale                     =    1 
        self.discount_factor                  =    0.99                     # As in chapter 4.1 (looking more into the future the higher this value is)
        self.repeat_times                     =    8                        # How often random batches are generated from the same buffer before new data is collected.


    def initialise(self):

        os.chdir(self.cwd)
        
        if not os.path.exists("save"):                                      # Is there a folder to safe the stuff? No? Well then let's create one
            os.makedirs("save")

        save_index = 0                                                      # Numbering of training runs
        while True:
            save_dir = self.save_dir + "_{}".format(save_index)             # Sequential numbering of file names

            if not os.path.exists("save/" + save_dir):
                if save_index == 0:                                         # No previous model to load
                    self.load_model = False
                    print("No save file available. load_model is set to False")
                if self.load_model:
                    save_dir = self.save_dir + "_{}".format(save_index - 1)
                else:
                    os.makedirs("save/" + save_dir)
                self.save_dir = save_dir
                self.cwd = "./" + "save/" + save_dir + "/"
                if self.load_model:
                    with open(self.cwd + "last_step.txt","r") as f:
                        self.load_step = int(f.read())
                        print(f"Load step: {self.load_step}")
                break
            else:
                save_index += 1

        np.random.seed(self.random_seed)                                    # Sets a fixed seed to make transfomer deterministic
        torch.manual_seed(self.random_seed)                                 # Same for PyTorch
        torch.set_num_threads(self.num_threads)                             # Determines how many CPU threads PyTorch uses internally
        torch.set_default_dtype(torch.float32)                              # Sets the default data type for newly created tensors to float32. This can be useful for controlling memory consumption and consistency in model training.


    def set_device(self):
        if torch.cuda.is_available() and self.gpu_id >= 0:
            device = torch.device(f'cuda:{self.gpu_id}')
            torch.cuda.empty_cache()
            print(f"GPU device set to: {str(torch.cuda.get_device_name(device))}")
        else:
            print("Device set to: cpu")
            device = torch.device('cpu')
        return device