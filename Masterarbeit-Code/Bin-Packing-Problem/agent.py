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
                 learning_rate_actor,                                                               # In the paper it is 1e-5
                 learning_rate_critic,                                                              # In the paper it is 1e-4
                 load_model = None,                                                                 # Whether a pre-trained model shall be used. Useful for checkpoints
                 cwd = None,                                                                        # Current working directory. Where shall the models be saved?
                 env_num = 1
                ):                                                                                  # Number of parallel environments
        super().__init__()

        self.criterion = torch.nn.MSELoss()                                                         # Loss for the critic as in chapter 3.2.2

        self.ratio_clip = params.ratio_clip                                                         # Clip rate of ϵ as in chapter 4.1
        self.lambda_entropy = params.lambda_entropy                                                 # A weight for entropy regularisation. Value not mentioned in the paper, so it is guessed.
        self.lambda_gae_adv = params.lambda_gae_adv                                                 # λ for Generalized Advantage Estimation in PPO as in Schulman et al. - Proximal Policy Optimization Algorithms

        self.device = params.set_device()                                                           # Choose device for training. GPU ist the standard case

        self.trajectory_list = [list() for _ in range(env_num)]                                     # Creates a list for each parallel environment to store trajectories --> sequence of states, actions, and rewards experienced by the agent. Necessary to perform PPO updates.

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
        
        if load_model:                                                                              # If a checkpoint is available and passed, it will be used
            try:
                logging.info(f"Load models from directory: {cwd}")
                self.actor.load_state_dict(torch.load(cwd + "actor.pth"))
                self.critic.load_state_dict(torch.load(cwd + "critic.pth"))
                logging.info(f"Models successfully loaded from {cwd}")
            except Exception as e:
                logging.warning(f"Failed to load models from {cwd}: {e}")

        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), learning_rate_actor)       # The adam optimiser is being use, as stated by the authors
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), learning_rate_critic)


    def explore_environment_multiprocessing(self,
                                            action_queue_list,                                      # Queue for each environment to pass actions from the agent
                                            result_queue_list,                                      # Queue to retrieve the results from the environments
                                            target_step,                                            # How many steps shall be collected?
                                            reward_scale,                                           # Scale the reward, if that is what you fancy (Usually its just 1)
                                            discount_factor                                         # The authors set the discount factor 𝛾 is to 0.99
                                        ):                                                          # Run several processes simultaneously to gain a lot of experience more quickly.
        process_num = len(action_queue_list)                                                        # How many processes are running?
        [action_queue_list[process_index].put(False) for process_index in range(process_num)]       # Sends a reset signal, i.e. a kind of synchronous start command, to each parallel environment process before the interaction begins.   
        self.use_ratio_avg = 0
        episode_num = 0                                                                             # Counter for the number of completed episodes
        srdap_temp = [[] for _ in range(process_num)]                                               # A list that creates a temporary trajectory list for each parallel environment process.
                                                                                                    # srdap = (state, reward, done, action, probabilities)
        last_done = [0] * process_num                                                               # Stores for each process at which iteration step the last done = True episode ended. Used later to extract only the relevant (completed) data per process.

        result_list = [result_queue.get() for result_queue in result_queue_list]                    # result_list = [result_0, --> result_x = (state, reward, done, use_ratio, packing_result).
                                                                                                    #                result_1,
                                                                                                    #                ...     ]

        state_list = [result[0] for result in result_list]                                          # state_list = [state_env_0, --> state_env_x = (bin_state, box_state, rotation_constraints, packing_mask) 
                                                                                                    #               state_env_1,
                                                                                                    #               ...        ]
        
        for i in range(target_step // process_num):                                                 # The division ensures that each environment contributes approximately target_step / process_num steps.
            state = list(map(list, zip(*state_list)))                                               # Transposes the state_list to process all box states together and all bin states together in batches. Then makes it a list
            state[2] = [np.array(state[2][0])]                                                      # Gets rotation_constraints into the right form and removes dimension, that has been added in the line above: [[[...]]] --> [[...]]               
            state = [torch.as_tensor(np.array(s), dtype = torch.float32, device = self.device) for s in state]  # Make state a PyTorch tensor. First NumPy array, to ensure that the data is of a uniform type
            
            action, probabilities = self.actor.get_action_and_probabilities(state)
            
            action_list = np.array([action.detach().cpu().numpy() for action in action]).transpose()# Converts tensors back to NumPy (CPU-compatible) for multiprocess communication. Transpose so that each process gets its own action
            probabilities_list = list(zip(*[probability.detach().cpu().numpy() for probability in probabilities]))
            action_int_list = action_list.tolist()
            [action_queue_list[process_index].put(action_int_list[process_index]) for process_index in range(process_num)]  # Actions are pushed into the action_queue_list. Motivation of those 4 lines: Each parallel environment receives the action selected by the actor.
            
            result_list = [result_queue.get() for result_queue in result_queue_list]                # List of the results

            [srdap_temp[process_index].append(                                                      # srdap saves:
                (state_list[process_index],                                                         # State before action
                 result_list[process_index][1],                                                     # Reward
                 result_list[process_index][2],                                                     # Done flag
                 action_list[process_index],                                                        # Chosen action
                 probabilities_list[process_index])) for process_index in range(process_num)]       # Probabilities for the actions
            
            result_list = list(map(list, zip(*result_list)))                                        # Transposes the result_list so that state_list is updated for the next loop.
            state_list = result_list[0]                                                             # result_list = (state, reward, done, use_ratio)

            for process_index in range(process_num):
                if result_list[2][process_index]:                                                   # Do the following only if process is done
                    self.use_ratio_avg = (self.use_ratio_avg * episode_num + result_list[3][process_index]) / (episode_num + 1)   # Total packing_ratio average so far + the latest
                    episode_num += 1
                    last_done[process_index] = i                                                    # last_done stores the last index i of the loop for each environment at which an episode was completed. Why? Later, only the data up to last_done[process_index] is transferred from srdap_temp. This prevents unfinished episodes or excess steps from being included in the replay data.

        srdap_list = list()                                                                         # Will get the trajectories for the PPO. Has all steps of completed episodes from all processes, while srdap_temp has all steps taken during the exploration for a process
        for process_index in range(process_num):
            srdap_list.extend(srdap_temp[process_index][:last_done[process_index] + 1])             # Put only the finished epidode into the list
        
        srdap_list = list(map(list, zip(*srdap_list)))                                              # Transpose: (state, reward, done, action, probs) becomes [states], [rewards], [dones], [actions], [probs]
        state_list = list(map(list, zip(*(srdap_list[0]))))

        state_array = [np.array(state, dtype = np.float32) for state in state_list]                 # All the states of the finished episodes
        action_list = list(map(list, zip(*(srdap_list[3]))))
        action_array = [np.array(action, dtype = np.float32) for action in action_list]             # All selected actions in NumPy form, prepared for Actor-Critic update.
        probabilities_list = list(map(list, zip(*(srdap_list[4]))))
        probabilities_array = [np.array(probabilities, dtype = np.float32) for probabilities in probabilities_list] # Probability distribution of actions
        reward_array = np.array(srdap_list[1], dtype = np.float32) * reward_scale                   # Scaled rewards
        mask_array = (1.0 - np.array(srdap_list[2], dtype = np.float32)) * discount_factor          # Masking for GAE or discounting: 1.0 - done --> masks completed episodes. Multiplication with discount_factor --> discounts future rewards.
        
        return state_array, action_array, probabilities_array, reward_array, mask_array
    

    def update_net(self,                                                                            # Updates the actor and critic network based on the collected trajectories.
                                                                                                    # Performs PPO updates, including GAE  for advantage calculation.
                                                                                                    # Goal: to train the actor to choose better actions and the critic to predict more accurate state values.
                   buffer,                                                                          # state_array, action_array, probabilities_array, reward_array, mask_array
                   batch_size,
                   repeat_times                                                                     # How often is the network updated per collected trajectory set?
                ):
        with torch.no_grad():                                                                       # No gradients are calculated, only data is converted.
            buffer_length = buffer[3].shape[0]
            state_buffer = [torch.as_tensor(array, device = self.device) for array in buffer[0]]    # Make the buffres PyTorch tensors and put to GPU/CPU
            action_buffer = [torch.as_tensor(array, device = self.device) for array in buffer[1]]
            probabilities_buffer = [torch.as_tensor(array, device = self.device) for array in buffer[2]]

            block_size = batch_size * 2
            value_buffer = [self.critic([s[i:i + block_size] for s in state_buffer]) for i in range(0, buffer_length, block_size)]  # Calculates the V(s)
            value_buffer = torch.cat(value_buffer, dim = 0)                                         # Make all mini batches one tensor again

            logprob_buffer = self.actor.get_old_logprob(action_buffer, probabilities_buffer)        # Log probabilities of old actions under the old policy (See chapter 3.2.2 in the paper)

            array_of_sum_of_rewards, advantage_array = self.get_reward_sum(buffer_length,           # Sum of all the (discounted) rewards and array of advantage values for PPO: 𝐴ₜ = 𝑅ₜ − 𝑉(𝑠ₜ), which shows how good an action was compared to the critic's prediction.
                                                                           reward_array = buffer[3],
                                                                           mask_array = buffer[4],
                                                                           value_array = value_buffer.cpu().numpy())
            buffer_of_sum_of_rewards, advantage_buffer = [torch.as_tensor(array, device = self.device)
                                        for array in (array_of_sum_of_rewards, advantage_array)]    # Make them PyTprch tensors on GPU/CPU
            advantage_buffer = (advantage_buffer - advantage_buffer.mean()) / (advantage_buffer.std() + 1e-8)   # Advantage values are normalised. 1e-8 Prevents division by zero if the standard deviation is exactly 0.

            del probabilities_buffer, buffer[:], array_of_sum_of_rewards, advantage_array

        loss_critic = loss_actor = logprob = None                                               # In case the for loop hat zero iterations, which will likely never happen

        for _ in range(int(buffer_length / batch_size * repeat_times)):                         # Repeat updates repeat_times across the entire buffer and draw random mini-batches (indices) of size batch_size.
            indices = torch.randint(buffer_length, size = (batch_size,), requires_grad = False, device = self.device)   # size is a tuple to make it iterable

            state = [state[indices] for state in state_buffer]                                  # Selection of the mini-batch data
            action = [action[indices] for action in action_buffer]
            sum_of_rewards = buffer_of_sum_of_rewards[indices]
            logprob = logprob_buffer[indices]                                                   # 𝜋_𝜃(𝑎ₜ|𝑠ₜ)
            advantage = advantage_buffer[indices]

            new_logprob, policy_entropy = self.actor.get_logprob_entropy(state, action)         # Actor loss (PPO with clipping & entropy) will be done in the next lines
                                                                                                # Entropy: High entropy --> lots of exploration Low entropy --> policy becomes deterministic
            ratio = (new_logprob - logprob.detach()).exp()                                      # ratio = π_new / π_old --> for PPO clipping (First formula in chapter 3.2.2) exp() removes log(): log(...) --> ...
            surrogate1 = advantage * ratio                                                      # PPO loss                  --> 𝑟ₜ(𝜃)𝐴̂ᵢ
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)      # PPO loss with clipping    --> (𝑟ₜ(𝜃),1-ϵ, 1+ϵ)𝐴̂ᵢ
            surrogate_loss = -torch.min(surrogate1, surrogate2).mean()                          # 𝐿^𝐶𝐿𝐼𝑃(𝜃) = Êₜ[min(𝑟ₜ(𝜃)𝐴̂ₜ, clip(𝑟ₜ(𝜃), 1 − 𝜖, 1+ 𝜖 )𝐴̂ₜ)]. Êₜ = .mean()
                                                                                                # “-” before the tensor because it should be minimised, PPO loss is actually a maximisation problem.
            loss_actor = surrogate_loss + policy_entropy * self.lambda_entropy                  # Loss of the actor
            self.optimiser_update(self.actor_optimiser, loss_actor)                             # Triggers the actual learning of the actor. Calculates backpropagation via the PPO loss and updates the policy weights.

            value = self.critic(state).squeeze(-1)                                              # Critic gives the value
            loss_critic = self.criterion(value, sum_of_rewards) / (sum_of_rewards.std() + 1e-6) # Calculates MSE loss --> How well does V(s) match the feedback?
            self.optimiser_update(self.critic_optimiser, loss_critic)                           # Triggers the actual learning of the actor. Calculates backpropagation via the PPO loss and updates the policy weights.

        return loss_critic.item(), loss_actor.item(), logprob.mean().item()

    def get_reward_sum(self,
                       buffer_length,
                       reward_array,
                       mask_array,
                       value_array
                    ):
        array_of_sum_of_rewards = np.empty(buffer_length, dtype = np.float32)
        advantage_array = np.empty(buffer_length, dtype = np.float32)

        previous_reward_sum = 0
        previous_advantage = 0

        for i in range(buffer_length - 1, -1, -1):                                                  # Iterates backwards over the trajectory, as is customary with discounted rewards.

            array_of_sum_of_rewards[i] = reward_array[i] + mask_array[i] * previous_reward_sum      # Calculates the discounted sum of rewards: Rₜ = rₜ​ + mask (= γ) ⋅ Rₜ₊₁
            previous_reward_sum = array_of_sum_of_rewards[i]

            advantage_array[i] = reward_array[i] + mask_array[i] * previous_advantage - value_array[i]  # Calculates the Generalised Advantage: (Reward) + (ongoing discounted benefits) - (the estimated value function).
            previous_advantage = value_array[i] + advantage_array[i] * self.lambda_gae_adv

        return array_of_sum_of_rewards, advantage_array


    @staticmethod
    def optimiser_update(optimiser, loss):
        optimiser.zero_grad()                                                                       # Sets all gradients of the model parameters (∂(L)/∂(𝜃)) to 0. Otherwise, gradients from previous backpropagation steps would accumulate --> incorrect updates. That is, because PyTorch adds gradients by default
        loss.backward()                                                                             # Performs backpropagation: calculates all gradients ∂(L)/∂(𝜃) for all parameters via autograd
        optimiser.step()                                                                            # Performs the actual parameter-update step: The network is adjusted according to the gradients: 𝜃 ← 𝜃 − 𝛼∇_𝜃 𝐿. 𝛼 is in the optimiser. 𝛼 is learning_rate_actor and learning_rate_critic