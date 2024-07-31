# Python imports
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sys import getsizeof
import argparse

# Pytorch imports
import torch
from torch.autograd import grad
import torch.optim as optim

# Model learning import
from model_learning.model_learning_transformer import CassieDataset

def get_parser():
    parser = argparse.ArgumentParser(description='Planner hyperparameters')

    # Training hyperparameters
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('--max_future_steps', type=int, default=11,
                        help='Max number of states predicted in the future')
    parser.add_argument('--context_length', type=int, default=40,
                        help='Context length to predict the next state')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')

    # Trained weights path
    parser.add_argument('--representation_path', type=str,
                        default='../results/representation_model_3.pt',
                        help='Trained representation learning weights')
    parser.add_argument('--transformer_path', type=str,
                        default='../results/transformer_model_3.pt',
                        help='Trained transformer learning weights')
    parser.add_argument('--expert_policy_path', type=str,
                        default='../results/actor_iter1143.pt',
                        help='Expert LSTM policy path')

    # Model Hyperparameters
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimensionality to project transformer input')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help='Number of heads in multi-head attention')
    parser.add_argument('--num_encoders', type=int, default=4,
                        help='number of transformer encoders')
    parser.add_argument('--num_decoders', type=int, default=4,
                        help='number of transformer decoders')
    parser.add_argument('--representation_size', type=int, default=64,
                        help='non-linear projection size')
    parser.add_argument('--hidden_one', type=int, default=1000,
                        help='hidden layer #1 units')
    parser.add_argument('--hidden_two', type=int, default=1000,
                        help='hidden layer #2 units')

    # Collected expert trajectory path
    parser.add_argument('--lstm_states_path', type=str,
                        default='../data/lstm_states.pt',
                        help='Perfect walking states')

    parser.add_argument('--lstm_actions_path', type=str,
                        default='../data/lstm_actions.pt',
                        help='Perfect walking actions')

    # Planner hyperparameters
    parser.add_argument('--tau', type=int, default=11,
                        help='Number of steps to plan for')
    parser.add_argument('--num_elites', type=int, default=50,
                        help='Number of elite actions')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of sampled actions')
    parser.add_argument('--num_cem_iterations', type=int, default=5,
                        help='Number of CEM iterations')
    parser.add_argument('--num_gradient_iterations', type=int, default=4,
                        help='Number of Gradient updates per CEM iteration')
    parser.add_argument('--num_iterations', type=int, default=299,
                        help='Maximum number of timesteps in planner epoch')
    parser.add_argument('--dagger_samples', type=int, default=50000,
                        help='Number of DAgger samples to collect')

    return parser

def prepare_epoch_dataset(state, action, history, steps):
    """
    Prepare dataset of states and actions based on epochs
    """
    # Stack them
    state_action_pair = torch.hstack((state, action))
    torch_dataset = CassieDataset(state_action_pair, state_action_pair, history=history, steps=steps)

    return torch_dataset

def prepare_dataset(states, actions, history, steps):
    """
    Prepare the dataset of states and actions based on their episode number
    """
    datasets = []
    count_timestamps = 0
    for episode_num in states.keys():
        # Get the state and episode for that episode number
        state, action = states[episode_num], actions[episode_num]

        # Take a subset of states and actions
        state_subset, action_subset = state[:, :], action[:, :]

        # Create 59 dimension vector (49 state, 10 action)
        state_action_pair = torch.hstack((state_subset, action_subset))

        # Concatenate
        torch_dataset = CassieDataset(state_action_pair, state_action_pair, history=history, steps=steps)
        count_timestamps += len(torch_dataset)
        datasets.append(torch_dataset)

    # stacked_dataset = torch.utils.data.ConcatDataset(datasets)
    return datasets, count_timestamps

def update_state_action_buffer(state_action_buffer, next_state, a_t):
    """
    Update with new states and actions from mujoco simulator
    """
    # Update current action
    state_action_buffer[:, -1, 49: ] = a_t

    # Update for next timestamp
    # Temporary a_{t+1}
    next_action_temp = torch.zeros((1, 1, 10))

    if torch.cuda.is_available():
        next_action_temp = next_action_temp.cuda()
        next_state = next_state.cuda()
        state_action_buffer = state_action_buffer.cuda()
    # [S_{t+1}, 0]^T
    next_pair = torch.cat((next_state, next_action_temp), dim = 2)

    # Update past 10 actions - Remove 0th timestamp and replace with next_pair {t+1}
    state_action_buffer = torch.hstack((state_action_buffer[:, 1:, :], next_pair))

    return state_action_buffer

def get_random_iteration(reset_iteration, offset, phase):
    # Get index to start from
    idx = np.random.randint(low = phase, high = offset + 1 - reset_iteration)
    random_iteration = range(idx, idx + reset_iteration)

    # Return range for mpc step
    return random_iteration

def get_next_mean(reference_step, offset, tau, start):
    """
    Get next mean based on phase
    """
    # Get start and end indices
    start_idx = start
    end_idx = start_idx + offset

    # Use this phase to exploit
    a = torch.load('../data/lstm_actions.pt')
    reference_action = torch.load('../data/lstm_actions.pt') [start_idx : end_idx]

    if (reference_step + tau) <= offset:
        # Use until end
        next_mean = torch.stack(reference_action[reference_step : reference_step + tau])
        # next_mean = next_mean.view(action_dim * tau)
    else:
        # Loop over
        part_one_mean = torch.stack(reference_action[reference_step :])
        part_two_mean = torch.stack(reference_action[ : reference_step + tau - offset])
        next_mean = torch.cat([part_one_mean, part_two_mean], dim = 0)
    return next_mean