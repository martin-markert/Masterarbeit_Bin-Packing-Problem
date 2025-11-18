from parameters import Parameters
from agent import Agent

import torch

import os
import logging
import time

from tensorboardX import SummaryWriter
from pathlib import Path

def train_and_evaluate(params, action_queue_list, result_queue_list):
    params.initialise()

# Agent stuff
    save_dir = params.save_dir                                  # Folder for checkpoints
    writer = SummaryWriter("runs/" + save_dir)
    set_logging(save_dir)

    logging.info(f"{'Step':>10}{'max Reward':>10} |"
                 f"{'exp Return':>10}{'exp Reward':>10}{'CriticLoss':>10}{'ActorLoss':>10}{'LogProb':>10}")
    
    agent = Agent(bin_size_x            = params.bin_size_ds_x,
                  bin_size_y            = params.bin_size_ds_y,
                  box_num               = params.box_num,
                  learning_rate_actor   = params.learning_rate_actor,
                  learning_rate_critic  = params.learning_rate_critic,
                  load_model            = params.load_model,
                  cwd                   = params.cwd
                )

    evaluator = Evaluator(params.cwd)

    buffer = list()

    def update_buffer(saprm):                                   # (state, action, probabilities, reward, mask)
        buffer[:]      = saprm
        steps          = saprm[3].shape[0]
        average_reward = saprm[3].mean()
        return steps, average_reward                            # TODO Or expected reward?
    
# Start training
    discount_factor       = params.discount_factor
    break_step            = params.break_step
    batch_size            = params.batch_size
    target_step           = params.target_step
    reward_scale          = params.reward_scale
    repeat_times          = params.repeat_times
    evaluator.total_step += params.load_step
    del params                                                  # memory clean-up

    while evaluator.total_step < break_step:
        start_time_explore_environment = time.time()
        with torch.no_grad():
            trajectory_list = agent.explore_environment_multiprocessing(action_queue_list,
                                                                        result_queue_list,
                                                                        target_step,
                                                                        reward_scale,
                                                                        discount_factor
                                                                    )
            steps, average_reward = update_buffer(trajectory_list)
            evaluator.total_step += steps
            evaluator.save_model(agent.actor, agent.critic, agent.use_ratio_avg)

        print(f"Time used for exploring the environment: {int(time.time() - start_time_explore_environment)} seconds")

        start_time_update_net = time.time()
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times)

        print(f"Time used for updating actor and critic (update_net): {int(time.time() - start_time_update_net)} seconds")

        evaluator.tensorboard_writer(average_reward, logging_tuple, agent.goal_avg, writer)     # TensorBoard & Logging:


class Evaluator:
    pass


# Helper functions
def set_logging(save_name):
    my_path = Path("./log")
    if not my_path.is_dir():
        os.makedirs(my_path)
    logging.basicConfig(filename=("./log/" + save_name + ".log"),
                        filemode="a",
                        level = logging.INFO,
                        format="%(asctime)s - %(message)s")


def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print(f"GPU device set to: n{str(torch.cuda.get_device_name(device))}")
    else:
        print("Device set to: cpu")
        device = torch.device('cpu')
    return device
                        

