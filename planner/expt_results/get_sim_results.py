# Simulator imports
from simulator.evaluate_expert import CassieSimulator
import torch

# Initial state
s_0 = torch.load('../../data/states.pt') [0][:1, :]
sim = CassieSimulator(s_0, [])

states = torch.load('../../data/states.pt') [0][:100, :]
actions = torch.load('../../data/actions.pt') [0][:100, :]

action_list = actions.tolist()
action_seq = []
for action in action_list:
    action = torch.Tensor(action)
    action = torch.unsqueeze(action, 0)
    action_seq.append(action)

state_list = states.tolist()
state_seq = []
for state in state_list:
    state = torch.Tensor(state)
    state = torch.unsqueeze(state, 0)
    state_seq.append(state)

# Visualize all the actions now
# sim.visualize_state_sequence(state_seq, action_seq)
# actions
# /home/ubuntu/cassie_project/planner/expt_results/expt_actions3/actions_implemented/true_action_sequence_0_21_256.pt
actions = torch.load('expt_actions3/actions_implemented/true_action_sequence_0_21_256.pt')

# Visualize all the actions now
sim.visualize_sequence(actions)
