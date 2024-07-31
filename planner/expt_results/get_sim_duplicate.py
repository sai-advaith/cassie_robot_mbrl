# Simulator imports
from simulator.evaluate_expert import CassieSimulator
import torch


# Initial state
H = 10

s_0 = torch.stack(torch.load('../../data/lstm_states.pt') [:1])
s_init = torch.stack(torch.load('../../data/lstm_states.pt') [:1])
a_init = torch.stack(torch.load('../../data/lstm_actions.pt') [:1])

sim = CassieSimulator(s_init, a_init, s_0)
env = None
phase = 14

action_seq_two = []
action_lstm = []
action_mpc = []
actions_true = torch.load('../../data/lstm_actions.pt') [:20]
states_true = torch.load('../../data/lstm_actions.pt') [:20]
actions2 = torch.load('../../data/lstm_actions.pt') [20:]
j = 1
for i in range(len(actions_true)):
    lstm_action = actions_true[i]
    if i == 0:
        s_next, reward, env, phase, done = sim.env_step(lstm_action.cpu(), env, True, True)
    else:
        s_next, reward, env, phase, done = sim.env_step(lstm_action.cpu(), env, True, False)
    s_next = torch.Tensor(s_next)

for i in range(len(actions2)):
    lstm_action = actions2[i]
    print(lstm_action)
    s_next, reward, env, phase, done = sim.env_step(lstm_action.cpu(), env, True, False)
    print(i + len(actions_true), reward)
    print()
# for i in range(len(actions_true)):
#     # action = torch.unsqueeze(action, 0)
#     action_seq_two.append(actions_true[i])
# actions = action_seq_two

# #
# # Visualize all the actions now
# sim.visualize_sequence_dup(actions)
