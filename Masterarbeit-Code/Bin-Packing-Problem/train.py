from parameters import Parameters
from agent import Agent
from explore_environment import explore_environment

import torch
import numpy as np
import os
import logging
import time

from tensorboardX import SummaryWriter
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Queue

def train_and_evaluate(params, action_queue_list, result_queue_list):
    params.initialise()

# Agent stuff
    save_dir = params.save_dir                                                                  # Folder for TensorBoard checkpoints
    writer = SummaryWriter("runs/" + save_dir)                                                  # Creates "runs" folder automatically
    set_logging(save_dir)

    logging.info(f"{'Step':>12} {'maxGoal':>12} {'expReturn':>12} {'expGoal':>12} {'CriticLoss':>12} {'ActorLoss':>12} {'LogProb':>12}")
    
    agent = Agent(bin_size_x            = params.bin_size_ds_x,
                  bin_size_y            = params.bin_size_ds_y,
                  learning_rate_actor   = params.learning_rate_actor,
                  learning_rate_critic  = params.learning_rate_critic,
                  load_model            = params.load_model,
                  cwd                   = params.cwd
                )

    evaluator = Evaluator(params.cwd)

    buffer = list()

    def update_buffer(saprm):                                                                   # (state, action, probabilities, reward, mask)
        buffer[:]      = saprm
        steps          = saprm[3].shape[0]
        average_reward = saprm[3].mean()
        return steps, average_reward
    
# Start training
    discount_factor        = params.discount_factor
    break_step             = params.break_step
    batch_size             = params.batch_size
    target_step            = params.target_step
    reward_scale           = params.reward_scale
    repeat_times           = params.repeat_times
    evaluator.total_steps += params.load_step
    del params                                                                                  # memory clean-up

    while evaluator.total_steps < break_step:
        print(f"Training: {evaluator.total_steps} out of max. {break_step} steps")
        start_time_explore_environment = time.time()
        with torch.no_grad():
            trajectory_list = agent.explore_environment_multiprocessing(action_queue_list,
                                                                        result_queue_list,
                                                                        target_step,
                                                                        reward_scale,
                                                                        discount_factor
                                                                    )
            steps, average_reward = update_buffer(trajectory_list)
            evaluator.total_steps += steps
            evaluator.save_model_if_it_is_better(agent.actor, agent.critic, agent.use_ratio_avg)

        logging.info(f"Time used for exploring the environment: {int(time.time() - start_time_explore_environment)} second{'s' if int(time.time() - start_time_explore_environment) != 1 else ''}")

        start_time_update_net = time.time()
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times)

        logging.info(f"Time used for updating actor and critic (update_net): {int(time.time() - start_time_update_net)} second{'s' if int(time.time() - start_time_update_net) != 1 else ''}")

        evaluator.tensorboard_writer(average_reward, logging_tuple, agent.use_ratio_avg, writer)# TensorBoard & Logging
    print(f"Done with the training. Trained for {break_step} steps.")


class Evaluator:                                                                                # Used to save the best model during training and record training metrics for TensorBoard and log files.
    def __init__(self,
                 cwd
                ):
                
        self.total_steps = 0
        self.cwd = cwd
        self.max_goal = self.get_max_goal_so_far()

     
    def save_model_if_it_is_better(self, actor, critic, expected_goal):                         # Saves if the current performance is better:
        if expected_goal > self.max_goal:
            self.max_goal = expected_goal
            torch.save(actor.state_dict(), f'{self.cwd}/actor.pth')                             # Save the policy network in *.pth
            torch.save(critic.state_dict(), f'{self.cwd}/critic.pth')
            logging.info(f"{self.total_steps:12} {self.max_goal:12.2f}")                        # Save the policy and print avlues. ONLY if the model is better than the previously best one

            with open(f"{self.cwd}/max_goal.txt", "w") as max_goal_file:                        # Write new max_goal value into max_goal.txt
                max_goal_file.write(str(self.max_goal))


    def tensorboard_writer(self, expected_return, log_tuple, expected_goal, writer):            # Writes training metrics to TensorBoard and the log:
        writer.add_scalar("Expected_goal", expected_goal, self.total_steps)                     # Logging values
        writer.add_scalar("Critic_loss", log_tuple[0], self.total_steps)
        writer.add_scalar("Actor_loss", log_tuple[1], self.total_steps)
        writer.add_scalar("Log_of_prob", log_tuple[2], self.total_steps)                        # p = 𝜋_𝜃(𝑎ₜ|𝑠ₜ) --> log(p)
        
        logging.info(f"{self.total_steps:12} {self.max_goal:12.2f} {expected_return:12.2f} {expected_goal:12.2f} {''.join(f'{n:12.2f}' for n in log_tuple)}")    # Full log of training
        
        with open(self.cwd + "last_step.txt", "w") as f:                                        # File, which contains the last step. Purpose: training can be continued later from the last step without starting from 0.
            f.write("{}".format(self.total_steps))


    def get_max_goal_so_far(self):                                                              # This ensures that even after the training is interrupted and later resumed, the max_goal value remains stored, as otherwise it would not be saved when training is resumed.
        max_goal_file = os.path.join(self.cwd, "max_goal.txt")
        if os.path.exists(max_goal_file):
            with open(max_goal_file, "r") as f:
                max_goal_so_far = float(f.read().strip())
                logging.info(f"Best maxGoal from previous trainings: {max_goal_so_far:.2f}")
        else:
            max_goal_so_far = -np.inf
        return max_goal_so_far


# Helper functions
def set_logging(save_name):
    my_path = Path("./log")
    my_path.mkdir(parents = True, exist_ok = True)
    logging.basicConfig(filename = ("./log/" + save_name + ".log"),
                        filemode = "a",
                        level = logging.INFO,
                        format = "%(asctime)s: %(message)s")                                     # Timestamp and message


if __name__ == '__main__':

    params = Parameters()

    action_queue_list = [Queue(maxsize = 1) for _ in range(params.process_num)]                 # process_num parallel processes, each with its own environment
    result_queue_list = [Queue(maxsize = 1) for _ in range(params.process_num)]                 # Each process produces a transition per iteration (state, action, reward, next_state, done). 
                                                                                                # explore_environment_multiprocessing() in train_and_evaluate() combines all processes and turns them into a joint training session.
    process_list = list()

    for process_index in range(params.process_num):
        process_object = mp.Process(target = explore_environment,
                                    args = (action_queue_list[process_index],
                                            result_queue_list[process_index],
                                            params.bin_size_x,
                                            params.bin_size_y,
                                            params.bin_size_z,
                                            params.bin_size_ds_x,
                                            params.bin_size_ds_y,
                                            params.box_num,
                                            params.min_factor,
                                            params.max_factor,
                                            params.rotation_constraints
                                        )
                                )
        process_list.append(process_object)

    [process_object.start() for process_object in process_list]                             # start() makes the process "alive"

    train_and_evaluate(params, action_queue_list, result_queue_list)
    exit(0)
