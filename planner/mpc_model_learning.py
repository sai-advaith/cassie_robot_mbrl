# Python imports
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Pytorch imports
import torch

# CEM imports
import mpc_cem as cem_FS

# Model Learning imports
from model_learning.transformer import Transformer
from model_learning.representation import Representation
from model_learning.positional_encoding import PositionalEncoder
from model_learning.model_learning_transformer import CassieDataset

# Simulator imports
from simulator.evaluate_expert import CassieSimulator
from simulator.cassie.cassiemujoco import CassieSim, CassieVis
from simulator.cassie.trajectory import CassieTrajectory

from utils.planner_utility import prepare_dataset, update_state_action_buffer, get_next_mean

class Reward(object):
    def __init__(self, reference_trajectory_path, speed):
        # Reference trajectory
        self.ref_trajectory = CassieTrajectory(reference_trajectory_path)

        # simulate mujoco steps with same pd target
        # 60 brings simulation from 2000Hz to roughly 30Hz
        self.simrate = 60

        # Indices to get joint level errors
        self.pos_idx   = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx   = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        # Speed of the robot
        self.speed = speed

        # Phase length of the robot
        self.phaselen = math.floor(len(self.ref_trajectory) / self.simrate) - 1

        # number of phase cycles completed in episode
        self.counter = 0

    def get_ref_state(self, phase):
        """
        Get the corresponding state from the reference trajectory for the current phase
        """
        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.ref_trajectory.qpos[phase * self.simrate])

        ###### Setting variable speed  #########
        pos[0] *= self.speed
        pos[0] += (self.ref_trajectory.qpos[-1, 0] - self.ref_trajectory.qpos[0, 0]) * self.counter * self.speed

        # setting lateral distance target to 0
        # regardless of reference trajectory
        pos[1] = 0

        vel = np.copy(self.ref_trajectory.qvel[phase * self.simrate])
        vel[0] *= self.speed

        return pos, vel

    def compute_reward(self, phase, cassie_state):
        ref_pos, _ = self.get_ref_state(phase)

        joint_error       = 0
        com_error         = 0
        orientation_error = 0
        spring_error      = 0

        # each joint pos
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
        qpos_joints = cassie_state[:, 0, 5:15]
        for i, j in enumerate(self.pos_idx):
            target = ref_pos[j]
            actual = qpos_joints[:, i].unsqueeze(1)
            error = torch.sub(actual, target)
            joint_error += torch.mul((error ** 2),  30 * weight[i])

        # complete joint error computation
        joint_error = joint_error.squeeze()

        # Q velocity
        qvel = cassie_state[:, 0, 15:18]
        forward_diff = torch.abs(torch.sub(qvel[:, 0].unsqueeze(1), self.speed)).squeeze()
        forward_diff[forward_diff < 0.05] = 0

        # Y direction velocity
        y_vel = torch.abs(qvel[:, 1].unsqueeze(1)).squeeze()
        y_vel[y_vel < 0.03] = 0

        # Make sure cassie orientations are aligned
        actual_q = cassie_state[:, 0, 1:5].double()
        # actual_q = qpos[3:7]
        target_q = torch.Tensor([1., 0., 0., 0.]).double()
        # Convert to cuda
        if torch.cuda.is_available():
            actual_q = actual_q.cuda()
            target_q = target_q.cuda()

        orientation_error = 5 * (1 - torch.inner(actual_q, target_q))**2

        # left and right shin springs positions
        j = 0
        qpos_shin = [cassie_state[:, 0, 34], cassie_state[:, 0, 37]]
        for i in [15, 29]:
            target = ref_pos[i]
            actual = qpos_shin[j]

            spring_error += 1000 * (target - actual) ** 2      
            j += 1

        # Weight
        orientation_weight = torch.tensor(0.300)
        joint_weight = torch.tensor(0.200)
        forward_weight = torch.tensor(0.200)
        yvel_weight = torch.tensor(0.200)
        spring_weight = torch.tensor(0.100)

        # GPU
        if torch.cuda.is_available():
            spring_error = spring_error.cuda()
            orientation_weight = orientation_weight.cuda()
            joint_weight = joint_weight.cuda()
            forward_weight = forward_weight.cuda()
            yvel_weight = yvel_weight.cuda()
            spring_weight = spring_weight.cuda()

        # Accumulate reward based on LSTM policy
        reward = torch.tensor(0.300).cuda() * torch.exp(-orientation_error) + \
                torch.tensor(0.200).cuda() * torch.exp(-joint_error) +       \
                torch.tensor(0.200).cuda() * torch.exp(-forward_diff) +      \
                torch.tensor(0.200).cuda() * torch.exp(-y_vel) +             \
                torch.tensor(0.100).cuda() * torch.exp(-spring_error)      
                # 0.050 * np.exp(-straight_diff) +     \

        return reward

def update_state_action_pairs(state_action_cache, next_state, a_t):
    """
    Update with new states and actions from mujoco simulator
    """
    # Update current action
    state_action_cache[:, -1, 49: ] = a_t

    # Update for next timestamp
    # Temporary a_{t+1}
    next_action_temp = torch.zeros((1, 1, 10))

    if torch.cuda.is_available():
        next_action_temp = next_action_temp.cuda()
        next_state = next_state.cuda()
        state_action_cache = state_action_cache.cuda()

    # [S_{t+1}, 0]^T
    next_pair = torch.cat((next_state, next_action_temp), dim = 2)

    # Update past 10 actions - Remove 0th timestamp and replace with next_pair {t+1}
    state_action_cache = torch.hstack((state_action_cache[:, 1:, :], next_pair))

    return state_action_cache

def retrain(representation, transformer, states, actions, train_datasets, new_train_dataset):
    """
    Retrain models based on data collected from simulator
    """
    # Convert to a dataloader
    state_action_pair = torch.hstack((states, actions))
    new_data = CassieDataset(state_action_pair, state_action_pair, history=10, steps=1)
    new_train_dataset.append(new_data)
    new_stacked_dataset = torch.utils.data.ConcatDataset(new_train_dataset)
    new_data_loader = torch.utils.data.DataLoader(new_stacked_dataset, batch_size = 128, shuffle = True)

    # Big dataset
    stacked_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = torch.utils.data.DataLoader(stacked_dataset, batch_size=128, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(list(representation.parameters()) + list(transformer.parameters()), lr = 1e-5)

    # Dimensionality
    state_dim = 49
    action_dim = 10

    # Loss
    loss_fn = torch.nn.MSELoss()
    epoch_loss = 100

    # max epochs
    max_epochs = 50
    epoch_count = 0

    while epoch_loss > 0.15 and epoch_count < max_epochs:
        epoch_loss = 0
        # Go through dataloader
        for idx, d in enumerate(new_data_loader):
            train, target = d[0], d[1]

            # Get next state only
            target_state = target[:, :, :state_dim]

            # Zero out gradients
            optimizer.zero_grad()

            # Representation
            if torch.cuda.is_available():
                train = train.cuda()
                representation = representation.cuda()

            # Get high dimensional representation
            representations = representation(train)
        
            # Feed to decoder
            target_decoder = representations[:, -1, :].unsqueeze(1)

            # To Cuda
            if torch.cuda.is_available():
                representations = representations.cuda()
                target_decoder = target_decoder.cuda()
                transformer = transformer.cuda()

            # Get output and compute loss
            output = transformer(representations, target_decoder)

            # Compute loss
            if torch.cuda.is_available():
                output = output.cuda()
                target_state = target_state.cuda()
            loss = loss_fn(output, target_state)
            epoch_loss += loss.item()

            # Backprop
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch_count} loss: {epoch_loss}")
            if epoch_loss < 0.15:
                break
            epoch_count += 1
    return representation, transformer, new_train_dataset

def dynamics(representation, transformer, state_action_cache, a_t):
    """
    Feed in input to learned dynamics model from representation and transformer model
    state_action_cache: past H state action pairs to be fed to the models
    """
    # Replace lsat state action pair with a_t action
    if state_action_cache.size() [0] != a_t.size() [0]:
        state_action_cache = state_action_cache.repeat(a_t.size() [0], 1, 1)
    state_action_cache[:, -1:, 49:] = a_t

    # Model in eval mode
    representation.eval()
    transformer.eval()

    # Clear out gradients
    with torch.no_grad():
        if torch.cuda.is_available():
            state_action_cache = state_action_cache.cuda()

        # Representation model
        representations = representation(state_action_cache)
        # Decoder input
        target_decoder = representations[:, -1:, :]

        # Convert to cuda
        if torch.cuda.is_available():
            representations = representations.cuda()
            target_decoder = target_decoder.cuda()

        # s_{t+1}
        next_state = transformer(representations, target_decoder)
        # Temporary a_{t+1}
        next_action_temp = torch.zeros((a_t.size() [0], 1, 10))

        if torch.cuda.is_available():
            next_action_temp = next_action_temp.cuda()

        # [S_{t+1}, 0]^T
        next_pair = torch.cat((next_state, next_action_temp), dim = 2)

        # Update past 10 actions - Remove 0th timestamp and replace with next_pair {t+1}
        state_action_cache = torch.hstack((state_action_cache[:, 1:, :], next_pair))
        return next_state, state_action_cache

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
        state_subset, action_subset = state[:250, :], action[:250, :]

        # Create 59 dimension vector (49 state, 10 action)
        state_action_pair = torch.hstack((state_subset, action_subset))

        # Concatenate
        torch_dataset = CassieDataset(state_action_pair, state_action_pair, history=history, steps=steps)
        count_timestamps += len(torch_dataset)
        datasets.append(torch_dataset)

    # stacked_dataset = torch.utils.data.ConcatDataset(datasets)
    return datasets, count_timestamps


# Load the trained models
if not torch.cuda.is_available():
    map_location = torch.device('cpu')
else:
    map_location = torch.device("cuda:0")

# Define dimensions
state_dim = 49
action_dim = 10
state_action_pair_dim = state_dim + action_dim

# Representation size
representation_size = 100
representation_hidden_one = 1000
representation_hidden_two = 1000

# Transformer parameters
d_model = 512
n_heads = 8
num_decoders = 4
num_encoders = 4

# History (as trained in model)
H = 10

# Representation model
representation_model = Representation(state_action_pair_dim, representation_size, representation_hidden_one, representation_hidden_two)
representation_model.load_state_dict(torch.load('../results/trained_network_representation.pt', map_location))

# Transformer model
transformer_model = Transformer(representation_size, state_dim, d_model, n_heads, num_decoders, num_decoders, H)
transformer_model.load_state_dict(torch.load('../results/trained_network_transformer.pt', map_location))

# Device
d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change device
representation_model.to(d)
transformer_model.to(d)

# Change type
representation_model = representation_model.double()
transformer_model = transformer_model.double()

# Models
models = (representation_model, transformer_model)

# Hyperparams

# Planning horizon (\tau)
TIMESTEPS = 10

# Number of Elites (N)
N_ELITES = 25

# Number of Samples (K)
N_SAMPLES = 1000

# Number of iterations M
SAMPLE_ITER = 10

# Clamp for action vectors
ACTION_LOW = -1.5
ACTION_HIGH = 1.5

# Decay rate of samples per iteration (\gamma)
decay = 0.9

# Iterations for main loop
iterations = 257

# Expert model
model = torch.load('../results/actor_iter1143.pt')
model = model.to(d)

# get model type in place
if model.nn_type == 'policy':
    expert_policy = model
elif model.nn_type == 'extractor':
    expert_policy = torch.load(model.policy_path)

# Hidden states of LSTM
if hasattr(expert_policy, 'init_hidden_state'):
    expert_policy.init_hidden_state()

# Change type
expert_policy = expert_policy.double()

# Get first state of walking as inital state for padding until enough
# s_0 = torch.zeros((1, state_dim))
a_0 = torch.stack(torch.load('../data/perfect_actions.pt') [:H])
s_0 = torch.stack(torch.load('../data/perfect_states.pt') [:H])
s_init = torch.load('../data/perfect_states.pt') [0]
a_init = torch.stack(torch.load('../data/perfect_actions.pt') [: H - 1]).unsqueeze_(dim = 1)
init_state_action_pair = torch.hstack((s_0, a_0))

# Duplicate H such state action pairs
# state_action_pairs = init_state_action_pair.repeat(H, 1)
state_action_pairs = torch.unsqueeze(init_state_action_pair, 0)
state_action_pairs = state_action_pairs.double()

# Reward
trajectory_path = "../simulator/cassie/trajectory/stepdata.bin"
reward_obj = Reward(trajectory_path, 1)
phaselen = reward_obj.phaselen

# Init mean of CEM
init_mean = torch.stack(torch.load('../data/perfect_actions.pt') [: TIMESTEPS])
init_mean = init_mean.view(action_dim * TIMESTEPS)

# Define CEM MPC
cem_gym = cem_FS.CEM(dynamics, reward_obj, state_action_pair_dim, action_dim, models, state_action_pairs, init_mean, retrain,
                    num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER, horizon=TIMESTEPS, device=d,
                    num_elite=N_ELITES, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                    u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d), init_cov_diag=0.8, momentum = 0.02)

# Hyperparameters
H, n = 10, 1

# Load test states + actions data
train_states = torch.load('../data/states.pt')
train_actions = torch.load('../data/actions.pt')

# Training set
train_dataset, train_timestamps = prepare_dataset(train_states, train_actions, history=H, steps=n)

# Reset
reset_iter = 0
retrain_threshold = 50

# Simulator
sim = CassieSimulator(s_init, a_init, s_0)
actions = []
rewards = []

# Environment
env = None

# Phase
phase = H - 1

# Save frequency - How often to save action sequence
save_frequency = 5

# True states
previous_states = s_0
previous_actions = a_0

# Switch to LSTM after these iterations
reset_iteration = 2

new_train_dataset = []

# Phase length
phaselen = 27
offset = 28

# Start recycling the phases:
start_idx = 56

for i in range(10):
    print(f"#### ATTEMPT #{reset_iter}")

    # Start from same initial state
    s_init = torch.load('../data/lstm_states.pt') [0]
    a_init = torch.load('../data/lstm_actions.pt') [: start_idx]
    a_dup = torch.load('../data/lstm_actions.pt') [:]

    actions = []
    states = [s_init]
    lstm_states = [s_init]

    # model free actions
    model_free_actions = []
    mpc_actions = []

    # Model free timesteps
    mpc_timesteps = []

    # Initialize simulator
    sim = CassieSimulator(s_init, actions)
    rewards = []

    # Environment
    env = None

    # Phase (start from where the memory start)
    phase = 14

    # Reward
    trajectory_path = "../simulator/cassie/trajectory/stepdata.bin"
    reward_obj = Reward(trajectory_path, 1)
    phaselen = reward_obj.phaselen

    # Models
    models = (representation_model, transformer_model)

    # Init mean of CEM
    init_mean = torch.stack(torch.load('../data/perfect_actions.pt') [: TIMESTEPS])
    init_mean = init_mean.view(action_dim * TIMESTEPS)

    # Define CEM MPC
    cem_gym = cem_FS.CEM(dynamics, reward_obj, state_action_pair_dim, action_dim, models, state_action_pairs, init_mean, retrain,
                        num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER, horizon=TIMESTEPS, device=d,
                        num_elite=N_ELITES, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                        u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d), init_cov_diag=0.8, momentum = 0.02)

    # Let LSTM do first H for padding reasons
    for i in range(start_idx):
        print(f"Iteration Number: {i}")

        # LSTM action
        lstm_action = a_init[i]

        # First step => initialize the env too
        if i == 0:
            s_next, reward, env, phase, done = sim.env_step(lstm_action.cpu(), env, False, True)
        else:
            s_next, reward, env, phase, done = sim.env_step(lstm_action.cpu(), env, False, False)
        states.append(s_next)
        # Log reward
        print("Low Level Reward after taking MPC action:", reward)
        print("=========================================\n")

        # Give initial state
        actions.append(lstm_action.cpu())
        s_init = torch.Tensor(s_next)
        model_free_actions.append(lstm_action.cpu())
        lstm_states.append(s_init.cpu())
        rewards.append(reward)

    # Stack states and actions
    actions.append(torch.zeros(lstm_action.size()))
    a_0 = torch.stack(actions) [-H:]
    s_0 = torch.stack(lstm_states) [-H:]

    # state action pairs
    init_state_action_pair = torch.hstack((s_0, a_0))
    state_action_buffer = torch.unsqueeze(init_state_action_pair, 0)
    state_action_buffer = state_action_buffer.double()

    # Remove placeholder action
    dummy_action = actions.pop()

    model_free = False

    # True states
    previous_states = s_0
    previous_actions = a_0

    # Reference iteration cap
    ref_idx = 0

    # complete_epoch = True
    phase_count = 0
    iteration_count = 0
    reset_iteration = 10
    for i in range(start_idx, iterations):
        print(f"Iteration Number: {i}") 
        # Reset training
        next_mean = get_next_mean(ref_idx, offset, TIMESTEPS, start_idx)
        next_mean = next_mean.view(action_dim * TIMESTEPS)

        # Refernece mean indices
        ref_idx += 1
        if ref_idx >= offset:
            ref_idx = 0

        # if i % phaselen == 0:
        #     phase_count = 0
        #     model_free = False
        # elif i % phaselen != 0 and phase_count >= reset_iteration:
        #     model_free = True

        # Get action from appropriate policy
        # if model_free:
        #     with torch.no_grad():
        #         if torch.cuda.is_available():
        #             previous_states = previous_states.cuda().double()
        #         # Given current state, provide the next optimal action based on a_next
        #         a_next = expert_policy(previous_states[-1, :])
        #         a_next = a_next.cpu().detach()
        #         a_next = a_next.unsqueeze(dim = 0)
        #         # a_next = a_next.squeeze(dim = 0)
        # else:
        #     phase_count += 1
        #     # Get optimal next step action command from MPC
        #     init_mean = torch.load('../data/perfect_actions.pt') [H + i - 1 : H - 1 + i + TIMESTEPS]

        #     # Get next step from the planner
        #     a_next = cem_gym.command(state_action_pairs, i, N_SAMPLES, decay, phase, next_mean, True)
        #     a_next = a_next.cpu().detach()

        reset = (iteration_count % reset_iteration) == 0
        a_next = cem_gym.command(state_action_pairs, iteration_count, N_SAMPLES, decay, phase, next_mean, reset, True)
        iteration_count += 1
        a_next = a_next.cpu().detach()
        a_next_two = a_dup[i].unsqueeze(dim = 0)
        print(torch.mean(torch.abs(a_next_two - a_next)))

        actions.append(a_next)

        # Phase before sim update
        phase_before = phase

        # Update previous actions
        previous_actions[-1, :] = a_next

        # Next step mean
        s_next, reward, env, phase, done = sim.env_step(a_next, env, False, False)
        # Phase after sim update
        phase_after = phase

        # Get next state and update cache
        s_next = torch.Tensor(s_next)
        s_prev = state_action_pairs[:, -1:, :49]
        # Convert to appropriate size
        # true_next_state = torch.load('../data/states.pt') [0][H + i + 1, :]
        # print("actual phase", phase)

        if np.abs(phase_after - phase_before) != 1:
            print("incrementing")
            reward_obj.counter += 1

        # reward_obj.compute_reward(phase, s_next, True)
        # Log reward

        print("Low Level Reward after taking MPC action:",[reward])
        print("=========================================\n")

        rewards.append(reward)
        # Retrain mechanism
        if reward.item() < 0.65:# or np.abs(phase_after - phase_before) != 1:
            print("retraining")
            break
            # cem_gym.retrain(previous_states, previous_actions, train_dataset)

        # Epoch complete
        if i < (iterations - 1):
            # Set previous states and actions
            if torch.cuda.is_available() and previous_states.get_device() == 0:
                s_next = s_next.cuda()
            # Update previous states and actions
            previous_states = torch.cat((previous_states, s_next.unsqueeze(dim = 0)))
            previous_actions = torch.cat((previous_actions, torch.zeros(a_next.size())))

        # Next states
        s_next = s_next.view(1, 1, -1)

        # Update state_action_pairs
        state_action_pairs = update_state_action_pairs(state_action_pairs, s_next, a_next)

        # if i % save_frequency == 0:
        #     # Save it
        #     torch.save(actions, 'expt_results/expt_actions3/action_sequence_'+str(i)+'.pt')
        #     # Visualize action sequence
        #     # sim.visualize_sequence(actions)
    exit(0)
    representation_model, transformer_model, new_train_dataset = cem_gym.retrain(previous_states.cpu(), previous_actions.cpu(), train_dataset, new_train_dataset)
    reset_iter += 1
    reset_iteration += 1
    print(f"LSTM frequency {reset_iteration}")
    # Increase gradient steps later
    # if reset_iteration < 10:
    #     GRADIENT_UPDATES = 11
    if reset_iteration > phaselen:
        break

    # Visualize all the actions now
    # sim.visualize_sequence(actions)

# # Plot rewards
# plt.plot(range(iterations), rewards, '-o')
# plt.ylabel('Rewards')
# plt.xlabel('Timestamps')
# plt.savefig('reward_results/reward2.png')

