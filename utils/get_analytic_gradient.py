# Pytorch imports
import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal

# Simulator imports
from simulator.evaluate_expert import CassieSimulator
from simulator.cassie.cassiemujoco import CassieSim, CassieVis
from simulator.cassie.trajectory import CassieTrajectory

# Python imports
import multiprocessing as mp
import time
import random

# Load train states + actions data (swapping test and train because test has more deviation)
train_states = torch.load('../data/train_gradient_states_large.pt')
train_actions = torch.load('../data/train_gradient_actions_large.pt')

s_init = torch.load('../data/lstm_states.pt') [0]
s_init[-1] = 5.0900e-02

def get_true_reward(s_init, actions, elite_action):
    """
    Given all the actions so far, compute analytical gradient of elite action
    """
    sim = CassieSimulator(s_init, [])
    env = None

    if len(actions) > 0:
        for j in range(len(actions)):
            lstm_action = actions[j]
            if j == 0:
                s_next, reward, env, phase, done, contact_information = sim.env_step(lstm_action, env, False, True)
            else:
                s_next, reward, env, phase, done, contact_information = sim.env_step(lstm_action, env, False, False)

    lstm_action = elite_action.clone().cpu().squeeze()
    # Get reward upto this point
    s_next, reward, env, phase, done, contact_information = sim.env_step(lstm_action, env, False, env is None)
    return reward

def compute_gradients(episode_num):
    print(f"epsiode_num {episode_num}")
    # Get states and actions for that episode
    action_episode = train_actions[episode_num]

    # Store all actions so far
    actions_so_far = []
    gradients = []

    for i in range(action_episode.size()[0]):
        action_i = action_episode[i, :]
        grad_j = []
        for y in range(len(action_i)):
            action = action_i
            # Perturbation
            epsilon = 1e-10
            action_copy = action.clone()

            # Perturb
            action_copy[y] = action[y] + epsilon
            reward_plus = -get_true_reward(s_init, actions_so_far, action_copy)

            # Perturb again
            action_copy = action.clone()
            action_copy[y] = action[y] - epsilon
            reward_minus = -get_true_reward(s_init, actions_so_far, action_copy)

            manual_gradient = (reward_plus - reward_minus) / (2 * epsilon)
            grad_j.append(manual_gradient.item())
        # Add actions
        actions_so_far.append(action_i)
        gradients.append(grad_j)

    return episode_num, torch.Tensor(gradients)

if __name__ == "__main__":
    num_cpu_cores = mp.cpu_count()
    with mp.Pool(processes=num_cpu_cores) as pool:
        results = pool.map(compute_gradients, train_states.keys())

    grad_action = {}
    for episode_num, gradients in results:
        grad_action[episode_num] = gradients
    torch.save(grad_action, 'train_numerical_gradient_large.pt')