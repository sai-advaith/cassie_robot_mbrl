# Python imports
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sys import getsizeof

# Pytorch imports
import torch
from torch.autograd import grad

# CEM imports
import mpc_cem as cem_FS
import mpc_cem_gd as cem_gd
import gradient_cem as gradient

# Model Learning imports
from model_learning.transformer import Transformer
from model_learning.representation import Representation

# Simulator imports
from simulator.evaluate_expert import CassieSimulator
from simulator.cassie.cassiemujoco import CassieSim, CassieVis
from simulator.cassie.trajectory import CassieTrajectory

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
        orientation_error = 5 * (1 - torch.inner(actual_q, target_q))**2

        # left and right shin springs positions
        j = 0
        qpos_shin = [cassie_state[:, 0, 34], cassie_state[:, 0, 37]]
        for i in [15, 29]:
            target = ref_pos[i]
            actual = qpos_shin[j]

            spring_error += 1000 * (target - actual) ** 2      
            j += 1

        # Accumulate reward based on LSTM policy
        reward = 0.000 + \
                0.300 * torch.exp(-orientation_error) + \
                0.200 * torch.exp(-joint_error) +       \
                0.200 * torch.exp(-forward_diff) +      \
                0.200 * torch.exp(-y_vel) +             \
                0.100 * torch.exp(-spring_error)      
                # 0.050 * np.exp(-straight_diff) +     \

        # Make sure gradients are always required
        reward = reward.requires_grad_()

        return reward

class CassieModelEnv(object):
    """
    Cassie Dynamics model
    """
    def __init__(self, representation, transformer, reward_object, TIMESTEPS, device, nu, samples):
        """
        Rollout action sequences and then return reward for each of the action sequences
        """
        # Learned models
        self.representation = representation
        self.transformer = transformer

        # Reward function
        self.reward = reward_object

        # Hyperparams
        self.dtype = torch.double
        self.T = TIMESTEPS
        self.nu = 10
        self.K = samples
        self.d = device

    def dynamics(self, state_action_buffer, a_t, no_gradient = False):
        """
        Feed in input to learned dynamics model from representation and transformer model
        state_action_buffer: past H state action pairs to be fed to the models
        """
        # Replace lsat state action pair with a_t action
        if state_action_buffer.size() [0] != a_t.size() [0]:
            state_action_buffer = state_action_buffer.repeat(a_t.size() [0], 1, 1)
        a_t = a_t.unsqueeze(1)
        state_action_buffer[:, -1:, 49:] = a_t
        # state_action_buffer.requires_grad_()

        # Model in eval mode
        self.representation.eval()
        self.transformer.eval()

        # TODO: Clean up ASAP
        if no_gradient:
            with torch.no_grad():
                # Convert to GPU if available
                if torch.cuda.is_available():
                    state_action_buffer = state_action_buffer.cuda()

                # Representation model
                representations = self.representation(state_action_buffer)
                # Decoder input
                target_decoder = representations[:, -1:, :]

                # Convert to cuda
                if torch.cuda.is_available():
                    representations = representations.cuda()
                    target_decoder = target_decoder.cuda()

                # s_{t+1}
                next_state = self.transformer(representations, target_decoder)
                # Temporary a_{t+1}
                next_action_temp = torch.zeros((a_t.size() [0], 1, 10))

                if torch.cuda.is_available():
                    next_action_temp = next_action_temp.cuda()

                # [S_{t+1}, 0]^T
                next_pair = torch.cat((next_state, next_action_temp), dim = 2)

                # Update past 10 actions - Remove 0th timestamp and replace with next_pair {t+1}
                state_action_buffer = torch.hstack((state_action_buffer[:, 1:, :], next_pair))
                return next_state, state_action_buffer
        else:
            # Convert to GPU if available
            if torch.cuda.is_available():
                state_action_buffer = state_action_buffer.cuda()

            # Representation model
            representations = self.representation(state_action_buffer)
            # Decoder input
            target_decoder = representations[:, -1:, :]

            # Convert to cuda
            if torch.cuda.is_available():
                representations = representations.cuda()
                target_decoder = target_decoder.cuda()

            # s_{t+1}
            next_state = self.transformer(representations, target_decoder)
            # Temporary a_{t+1}
            next_action_temp = torch.zeros((a_t.size() [0], 1, 10))

            if torch.cuda.is_available():
                next_action_temp = next_action_temp.cuda()

            # [S_{t+1}, 0]^T
            next_pair = torch.cat((next_state, next_action_temp), dim = 2)

            # Update past 10 actions - Remove 0th timestamp and replace with next_pair {t+1}
            state_action_buffer = torch.hstack((state_action_buffer[:, 1:, :], next_pair))
            return next_state, state_action_buffer

    def rollout(self, actions, init_state, phase, no_gradient = False):
        """
        Rollout action sequences at particular state and phase to give the reward of the action sequence
        """

        reward_total = torch.zeros(actions.size() [1], device=self.d, dtype=self.dtype, requires_grad = True)

        # Reset state before run - maintain K copies of the state
        # state = init_state.view(1, -1).repeat(self.K, 1)

        # Rollout states store
        per_rollout_cache = init_state

        # Phase for getting reference position and velocity at the timestamp
        rollout_phase = phase + 1

        # Roll it out
        for t in range(self.T):
            if rollout_phase > self.reward.phaselen:
                rollout_phase = 0
                # TODO: Verify this
                # self.reward.counter += 1

            # Get the action for that timestamp
            u = actions[t, :, :]
            # grad(u.mean(), state, grad_output = v)

            # Get the next state action pair record
            state, per_rollout_cache = self.dynamics(per_rollout_cache, u, no_gradient)

            # Calculate rewards
            reward = self.reward.compute_reward(rollout_phase, state)
            reward_total = reward_total + reward            
            rollout_phase += 1


        # Return all rewards
        return reward_total

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

    # [S_{t+1}, 0]^T
    next_pair = torch.cat((next_state, next_action_temp), dim = 2)

    # Update past 10 actions - Remove 0th timestamp and replace with next_pair {t+1}
    state_action_buffer = torch.hstack((state_action_buffer[:, 1:, :], next_pair))

    return state_action_buffer

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

# Models
models = (representation_model, transformer_model)

# Get first state of walking as inital state for padding until enough
# s_0 = torch.zeros((1, state_dim))
a_0 = torch.load('../data/actions.pt') [0][:1, :]
s_0 = torch.load('../data/states.pt') [0][:1, :]
init_state_action_pair = torch.hstack((s_0, a_0))

# Duplicate H such state action pairs
# state_action_buffer = [init_state_action_pair] * H
state_action_buffer = init_state_action_pair.repeat(H, 1)
state_action_buffer = torch.unsqueeze(state_action_buffer, 0)
state_action_buffer = state_action_buffer.double()

# Device
d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Change device
representation_model.to(d)
transformer_model.to(d)

# Change type
representation_model = representation_model.double()
transformer_model = transformer_model.double()

# Hyperparams

# Planning horizon (\tau)
TIMESTEPS = 10

# Number of Elites (N)
N_ELITES = 2

# Number of Samples (K)
N_SAMPLES = 500

# Number of iterations M
SAMPLE_ITER = 7

# Clamp for action vectors
ACTION_LOW = -1.5
ACTION_HIGH = 1.5

# Decay rate of samples per iteration (\gamma)
decay = 0.9

# Iterations for main loop
iterations = 71

# Gradient Iterations
GRADIENT_UPDATES = 5

# Reward
trajectory_path = "../simulator/cassie/trajectory/stepdata.bin"
reward_obj = Reward(trajectory_path, 1)
phaselen = reward_obj.phaselen

# Init mean of CEM
init_mean = torch.load('../data/actions.pt') [0][: TIMESTEPS, :]
init_mean = init_mean.view(action_dim * TIMESTEPS)

# Simulator
sim = CassieSimulator(s_0)
actions = []
rewards = []

# Environment
env = None

# Phase
phase = 0

# Save frequency - How often to save action sequence
save_frequency = 5

# Cassie's model
CassieModel = CassieModelEnv(representation_model, transformer_model, reward_obj, TIMESTEPS, d, action_dim, N_SAMPLES)
cem_gym = cem_gd.CEM_GD(TIMESTEPS, SAMPLE_ITER, N_SAMPLES,  N_ELITES, CassieModel, d, state_action_buffer, True)

for i in range(iterations):
    print(f"Iteration Number: {i}") 
    # Reset training
    if phase + TIMESTEPS <= phaselen + 1:
        next_mean = torch.load('../data/actions.pt') [0][phase : phase + TIMESTEPS, :]
        # next_mean = next_mean.view(action_dim * TIMESTEPS)
    else:
        part_one_mean = torch.load('../data/actions.pt') [0][phase : phaselen + 1, :]
        part_two_mean = torch.load('../data/actions.pt') [0][ : phase + TIMESTEPS - phaselen - 1, :]
        next_mean = torch.cat([part_one_mean, part_two_mean], dim = 0)
        # next_mean = next_mean.view(action_dim * TIMESTEPS)

    # Get next step from the planner
    a_next = cem_gym.forward(1, phase, state_action_buffer, next_mean)
    actions.append(a_next)

    # Implement in simulator
    phase_before = phase
    if i == 0:
        s_next, reward, env, phase = sim.env_step(a_next, env, False, True)
    else:
        s_next, reward, env, phase = sim.env_step(a_next, env, False, False)
    # print(torch.mean((a_next - init_mean[:10])**2))
    phase_after = phase
    # Get next state and update cache
    s_next = torch.Tensor(s_next)
    s_prev = state_action_buffer[:, -1:, :49]

    # Convert to appropriate size
    s_next = s_next.view(1, 1, -1)
    if np.abs(phase_before - phase_after) != 1:
        reward_obj.counter = reward_obj.counter + 1
    reward_new = reward_obj.compute_reward(phase, s_next, True)

    # Log reward
    print("Low Level Reward after taking MPC action:",[reward, reward_new.item()])
    print("=========================================\n")

    rewards.append(reward)
    # Update state_action_buffer
    state_action_buffer = update_state_action_buffer(state_action_buffer, s_next, a_next)
    if i % save_frequency == 0:
        # Save it
        torch.save(actions, 'expt_results/expt_actions3/action_sequence_'+str(i)+'.pt')
        # Visualize action sequence
        # sim.visualize_sequence(actions)

# Plot rewards
plt.plot(range(iterations), rewards, '-o')
plt.ylabel('Rewards')
plt.xlabel('Timestamps')
plt.savefig('reward_results/reward2.png')

