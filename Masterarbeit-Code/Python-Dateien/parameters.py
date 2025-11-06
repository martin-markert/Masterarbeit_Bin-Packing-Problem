class Parameters():
    def __init__(self):

        '''
        Arguments for the Environment as stated in chapter 4.1
        '''
        self.bin_size_x    =  100
        self.bin_size_y    =  100
        self.bin_size_z    = 1000   # <-- As for now it can be anything
        self.bin_size_ds_x =   10   # ds = downsampled
        self.bin_size_ds_y =   10
        self.box_num       =   50   # <-- Or whatever your heart desires
        self.min_factor    =    0.1 # In the paper they devide by 10, I don't, I multiply by 1/10
        self.max_factor    =    0.5



        '''
        Arguments of the transformer-related things (network.py)
        Note, that mostly those values are guesses, as the authors did not state the dimensions of their best model

        TODO: Alter values when testing
        '''
        

        ''' General ''' 
        self.binary_dim                       =    8    # Values not specified by authors --> Randon choice of dimensions (well not quite, it is an edicated guess). Will most likey be altered when testing   
        self.dim_model                        =  128          
        self.plane_feature_dim                =    7 * self.binary_dim    # 7 Plane features (6 features + height) * the binary_dim
        self.batch_size                       =    1
        
        
        ''' Box_Embed() '''                             # See chapter 3.1.3 --> Box encoder
        self.box_embed_dim_hidden_1           =  256    # Dimensionality of the hidden layer(s) of the feedforward neural network. Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        # self.box_embed_dim_hidden_2         =  256


        '''Box_Selection()'''
        self.box_selection_dim_hidden_1       =  256    # Dimensionality of the hidden layer(s) of the feedforward neural network. Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        # self.box_selection_dim_hidden_2     =  256
        
        
        '''Rotation_Selection()'''
        self.rotation_selection_dim_hidden_1  =  256    # Dimensionality of the hidden layer(s) of the feedforward neural network. Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        # self.rotation_selection_dim_hidden_2=  256

        
        ''' Transformer Encoder ''' 
        self.transformer_encoder_num_head     =    4    # Number of heads in the multi-head-attention model (see Figure 2 in Vaswani et al., 2017). self.dim_model % self.num_head = 0 shall be the case
                                                        # When one has several heads per layer the heads are independent of each other. This means that the model can learn different patterns with each head. 
                                                        # For example, one head might pay most attention to the next word in each sentence, and another head might pay attention to how nouns and adjectives combine.
        self.transformer_encoder_dim_hidden_1 = 1024    # Shall be > self.dim_model then the feed-forward network can learn more complex transformations
        self.transformer_encoder_num_layers   =    2    # Amount of encoder layers (see “Nx” in Figure 1 in Vaswani et al., 2017)
        self.transformer_encoder_dropout      =    0    # Dropout: 
                                                        # Usually ~ 0.1 - 0.3, but 0 could lead to better results in Reinforce Learning, especially as stable action probabilities are better/needed for PPO updates.


        ''' Transformer Decoder '''                     # For symmety reasons the same dimensions as in the transformer encoder are taken. This can be altered, of course
                                                        # Why? 
                                                        # Uniform capacity: The transformer should have similar expressiveness for both boxes and position/rotation. 
                                                        # Simplicity: Default values are the same --> easier hyperparameter tuning.
        self.transformer_decoder_num_head     = self.transformer_encoder_num_head
        self.transformer_decoder_dim_hidden_1 = self.transformer_encoder_dim_hidden_1
        self.transformer_decoder_num_layers   = self.transformer_encoder_num_layers
        self.transformer_decoder_dropout      = self.transformer_encoder_dropout


        ''' Agent '''
        self.ratio_clip                       =    0.12     # Clip rate of ϵ as in chapter 4.1
        self.lambda_entropy                   =    0.02     # A weight for entropy regularisation. Value not mentioned in the paper, so it is guessed.
        self.lambda_gae_adv                   =    0.95     # λ for Generalized Advantage Estimation in PPO as in Schulman et al. - Proximal Policy Optimization Algorithms

        self.learning_rate_actor              =    1e-5     # As in chapter 4.1
        self.learning_rate_critic             =    1e-4     # As in chapter 4.1
        self.discount_factor                  =    0.99     # As in chapter 4.1 (looking more into the future the higher this value is)


        

                                           
                                                 