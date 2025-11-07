from network import Actor, Critic
import parameters as p

import torch
import logging
import numpy as np

params = p.Parameters()  

class Agent:
    def __init__(self,
                 bin_size_x,
                 bin_size_y,
                 learning_rate_actor,                                                           # In the paper it is 1e-5
                 learning_rate_critic,                                                          # In the paper it is 1e-4
                 use_gae = True,                                                                # Can be removed adn also self.if_per_or_gae = True in params
                 load_model = None,                                                             # Whether a pre-trained model shall be used. Useful for checkpoints
                 cwd = None,                                                                    # Current working directory. Where shall the models be saved?
                 gpu_id = 0,                                                                    # 0 --> first GPU, -1 --> CPU.
                 env_num = 1
                ):                                                                              # Number of parallel environments
        super().__init__()

        self.criterion = torch.nn.MSELoss()                                                     # Loss for the critic as in chapter 3.2.2

        self.ratio_clip = params.ratio_clip                                                     # Clip rate of ϵ as in chapter 4.1
        self.lambda_entropy = params.lambda_entropy                                             # A weight for entropy regularisation. Value not mentioned in the paper, so it is guessed.
        self.lambda_gae_adv = params.lambda_entropy                                             # λ for Generalized Advantage Estimation in PPO as in Schulman et al. - Proximal Policy Optimization Algorithms

        self.device = torch.device(f"cuda:{gpu_id}" if (                                        # Shall the training be done on the GPU or the CPU?
        torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.trajectory_list = [list() for _ in range(env_num)]                                 # Creates a list for each parallel environment to store trajectories --> sequence of states, actions, and rewards experienced by the agent. Necessary to perform PPO updates.

        self.actor = Actor(bin_size_x,
                           bin_size_y,
                           dim_model = params.dim_model,
                           binary_dim = params.binary_dim,
                           plane_feature_dim = params.plane_feature_dim
                        ).to(self.device)
        
        self.critic = Critic(bin_size_x,
                             bin_size_y,
                             dim_model = params.dim_model,
                             binary_dim = params.binary_dim,
                             plane_feature_dim = params.plane_feature_dim
                            ).to(self.device)
        
        if load_model:                                                                          # If a checkpoint is available and passed, it will be used
            try:
                logging.info(f"Load models from directory: {cwd}")
                self.actor.load_state_dict(torch.load(cwd + "actor.pth"))
                self.critic.load_state_dict(torch.load(cwd + "critic.pth"))
                logging.info(f"Models successfully loaded from {cwd}")
            except Exception as e:
                logging.warning(f"Failed to load models from {cwd}: {e}")

        self.act_optim = torch.optim.Adam(self.actor.parameters(), learning_rate_actor)         # The adam optimiser is being use, as stated by the authors
        self.cri_optim = torch.optim.Adam(self.critic.parameters(), learning_rate_critic)


    def select_action(self, state):                                                             # Selects the action based on the current state
        state = [torch.as_tensor(s, dtype=torch.float32, device = self.device).unsqueeze(0) for s in state] # Converts each element s into a PyTorch tensor so that PyTorch can calculate with it and adds a batch dimension.
        action, probabilities = self.actor.get_action_and_probabilities(state)

        action = [act[0].detach().cpu().numpy() for act in action]                              # act[0] --> because there is a batch dimension of 1, the “real action” is extracted.      
        probs = [prob[0, :].detach().cpu().numpy() for prob in probs]                           # .detach() detaches the tensor from the computation graph. We do not want gradients when executing the action.
                                                                                                # .cpu() if the tensor was on the GPU, it is moved to the CPU (GPU is only used for training, for code one uses CPU).
                                                                                                # .numpy() converts the tensor into a NumPy array, which is easier to handle (e.g. for the environment). 
 
        return action, probabilities                                                            #  One action consists of (box_index, position_index, rotation_index)
    

    def explore_environment_multiprocessing(self,
                                            action_queue_list,                                  # Queue for each environment to pass actions from the agent
                                            result_queue_list,                                  # Queue to retrieve the results from the environments
                                            target_step,                                        # How many steps shall be collected?
                                            reward_scale,
                                            discount_factor                                     # The authors set the discount factor 𝛾 is to 0.99
                                        ):                                                      # Run several processes simultaneously to gain a lot of experience more quickly.
        process_num = len(action_queue_list)                                                    # How many processes are running?
        [action_queue_list[process_index].put(False) for process_index in range(process_num)]   # Sends a reset signal, i.e. a kind of synchronous start command, to each parallel environment process before the interaction begins.   
        self.packing_score_average = 0
        episode_num = 0                                                                         # Counter for the number of completed episodes
        srdan_temp = [[] for _ in range(process_num)]                                           # A list that creates a temporary trajectory list for each parallel environment process.
                                                                                                # srdan = (state, reward, done, action, noise)
        last_done = [0] * process_num                                                           # Stores for each process at which iteration step the last done = True episode ended. Used later to extract only the relevant (completed) data per process:

        result_list = [result_queue.get() for result_queue in result_queue_list]                # result_list = [result_0, --> result_x = (state, reward, done, goal). The goal is the packing efficiancy/use ration
                                                                                                #                result_1,
                                                                                                #                ...     ]

        state_list = [result[0] for result in result_list]                                      # state_list = [state_env_0, --> state_env_x = (bin_state, box_state, packing_mask) 
                                                                                                #               state_env_1,
                                                                                                #               ...        ]
        
        for i in range(target_step // process_num):                                             # The division ensures that each environment contributes approximately target_step / process_num steps.
            state = list(map(list, zip(*state_list)))                                           # Transposes the state_list to process all box states together and all bin states together in batches. Then makes it a list
            state = [torch.as_tensor(np.array(s), dtype = torch.float32, device = self.device) for s in state]  # Make state a PyTorch tensor. Firwst numpy array, to ensure that the data is of a uniform type
            action, probabilities = self.actor.get_action_and_probabilities(state)
            action_list = np.array([act.detach().cpu().numpy() for act in actions]).transpose()
            probs_list = list(zip(*[prob.detach().cpu().numpy() for prob in probs]))
            action_int_list = action_list.tolist()
            [action_queue_list[pi].put(action_int_list[pi]) for pi in range(process_num)]
            result_list = [result_queue.get() for result_queue in result_queue_list]

            for process_index in range(process_num):
                if result_list[2][process_index]:                                               # if process is done
                    self.goal_avg = (self.goal_avg * episode_num + result_list[3][process_index]) / (episode_num + 1)
                    episode_num += 1
                    last_done[process_index] = i








