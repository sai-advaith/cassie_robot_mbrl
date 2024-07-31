import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal

# Simulator imports
from simulator.evaluate_expert import CassieSimulator
from simulator.cassie.cassiemujoco import CassieSim, CassieVis
from simulator.cassie.trajectory import CassieTrajectory

# Load train states + actions data (swapping test and train because test has more deviation)
train_states = torch.load('data/train_gradient_states_large.pt')
train_actions = torch.load('data/train_gradient_actions_large.pt')

s_init = torch.load('data/lstm_states.pt') [0]
s_init[-1] = 5.0900e-02

phase_action = {}

for episode_num in train_states.keys():
    print(f"episode number  {episode_num}")
    # Get states and actions for that episode
    state_episode = train_states[episode_num]
    action_episode = train_actions[episode_num]

    phase_episode = [14]
    sim = CassieSimulator(s_init, [])
    env = None

    for i in range(action_episode.size() [0]):
        action_i = action_episode[i, :]
        s_next, reward, env, phase, done, contact_information = sim.env_step(action_i, env, False, env is None)
        phase_episode.append(phase)

    phase_action[episode_num] = torch.Tensor(phase_episode)
    torch.save(phase_action, 'data/train_phase_large.pt')
