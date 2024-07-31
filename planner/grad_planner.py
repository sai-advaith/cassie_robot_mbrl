# Python imports
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sys import getsizeof
import time

# Pytorch imports
import torch
from torch.autograd import grad
import torch.optim as optim

# CEM imports
import mpc_cem as cem_FS
import mpc_cem_gd as cem_gd
import gradient_cem as gradient

# Model Learning imports
from model_learning.transformer import Transformer
from model_learning.representation import Representation
from model_learning.positional_encoding import PositionalEncoder
from model_learning.model_learning_transformer import CassieDataset

# Simulator imports
from simulator.evaluate_expert import CassieSimulator
from simulator.cassie.cassiemujoco import CassieSim, CassieVis
from simulator.cassie.trajectory import CassieTrajectory

# Reward import
from planner.reward import Reward

# Dynamics model
from planner.cassie_dynamics import CassieModelEnv

from utils.planner_utility import prepare_dataset, update_state_action_buffer, get_next_mean, get_random_iteration, prepare_epoch_dataset, get_parser

# Load the trained models
if not torch.cuda.is_available():
    map_location = torch.device('cpu')
else:
    map_location = torch.device("cuda:0")

parser = get_parser()
parser.print_help()
args = parser.parse_args()

# History + steps definition
H = args.context_length
n = 1
max_n = args.max_future_steps
batch_size = args.batch_size

# Dimensionality of state action pairs
state_dim = 49
action_dim = 10
state_action_dim = state_dim + action_dim

# Representation size
representation_input_size = 4 * state_action_dim
representation_size = args.representation_size
representation_hidden_one = args.hidden_one
representation_hidden_two = args.hidden_two

# Transformer parameters
d_model = args.d_model
num_attention_heads = args.num_attention_heads
num_decoders = args.num_encoders
num_encoders = args.num_decoders
dropout = args.dropout

torch.cuda.empty_cache()

# Transformer model
transformer_model = Transformer(representation_size=representation_size,
                                output_size=state_dim,
                                d_model=d_model,
                                n_heads=num_attention_heads,
                                num_encoders=num_encoders,
                                num_decoders=num_decoders, history=H,
                                dropout=dropout)

transformer_model = torch.nn.DataParallel(transformer_model)
transformer_model_path = args.transformer_path
transformer_model.load_state_dict(torch.load(transformer_model_path),
                                  map_location)

# Representation model
representation_model = Representation(input_size=representation_input_size,
                                      representation_size=representation_size,
                                      hidden_size1=representation_hidden_one,
                                      hidden_size2=representation_hidden_two,
                                      dropout=dropout)

representation_model_path = args.representation_path
representation_model.load_state_dict(torch.load(representation_model_path),
                                     map_location)

# Device
d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Expert model
lstm_model_path = args.expert_policy_path
model = torch.load(lstm_model_path)
model = model.to(d)

# get model type in place
if model.nn_type == 'policy':
    expert_policy = model
elif model.nn_type == 'extractor':
    expert_policy = torch.load(model.policy_path)

# Lstm state and action path
lstm_state_path = args.lstm_states_path
lstm_action_path = args.lstm_actions_path

# Hidden states of LSTM
if hasattr(expert_policy, 'init_hidden_state'):
    expert_policy.init_hidden_state()

# Change type
expert_policy = expert_policy.double()

# Models
transformer_model.to(d)
representation_model.to(d)

transformer_model = transformer_model.double()
representation_model = representation_model.double()

models = (transformer_model, representation_model)

# Hyperparams

# Planning horizon (\tau)
TIMESTEPS = args.tau

# Number of Elites (N)
N_ELITES = args.num_elites

# Number of Samples (K)
N_SAMPLES = args.num_samples

# Number of iterations M
SAMPLE_ITER = args.num_cem_iterations

# Clamp for action vectors
ACTION_LOW = -1.5
ACTION_HIGH = 1.5

# Decay rate of samples per iteration (\gamma)
decay = 0.9

# Iterations for main loop
# iterations = 299
iterations = args.num_iterations

# Gradient Iterations
GRADIENT_UPDATES = args.num_gradient_iterations

# Load test states + actions data
train_states = torch.load(lstm_state_path)
train_actions = torch.load(lstm_action_path)

# Standard deviation
std = 0.001

# Iteration count
retrain_iteration = 0

# Switch to LSTM after these iterations
reset_iteration = 1

# Number of times to repeat for each n
n_repititions = 40

# Phase length
phaselen = 27
offset = 28

# Start recycling the phases:
start_idx = 56

# Global state and action store
global_state_action_buffer = []

# for i in range(90):
while reset_iteration <= offset:
    print(f"#### ATTEMPT #{retrain_iteration}")
    dataset_store = []

    # Steps completed and required steps
    total_steps = 0
    steps_needed = args.dagger_samples
    iteration = 0

    state_dict = {}
    action_dict = {}
    episode_num = 1

    # TODO: REMOVE
    expt_mpc = start_idx + phaselen

    # for j in range(n_repititions):
    while total_steps < steps_needed:
        # Expert model
        model = torch.load(lstm_model_path)
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

        # print(tot)
        print(f"Run #{iteration}, Steps done: {total_steps}")

        # Start from same initial state
        s_init = torch.load(lstm_state_path) [0]
        s_init[-1] = 5.0900e-02
        s_true = torch.load(lstm_state_path) [start_idx]
        a_init = torch.load(lstm_action_path) [: start_idx]
        a_dup = torch.load(lstm_action_path) [:]

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
        reward_simulator = []
        reward_model = []
        state_action = []
        actions = []

        # Environment
        env = None

        # Phase (start from where the memory start)
        phase = 14

        # Step counter
        number_steps = 0

        # Let LSTM do first H for padding reasons
        for i in range(start_idx):
            number_steps += 1
            print(f"Iteration Number: {i}")
            print(f"Number of steps: {number_steps}")

            # LSTM action
            if torch.cuda.is_available():
                s_init = s_init.cuda()
                s_init = s_init.double()
            lstm_action = expert_policy(s_init).cpu()
            lstm_action = lstm_action.unsqueeze(dim = 0)
            lstm_action = lstm_action.cpu().detach()

            # First step => initialize the env too
            # all_zeros = not a.any()

            s_next, reward, env, phase, done, contact_information = sim.env_step(lstm_action, env, False, env is None)

            print("Low Level Reward after taking MPC action:", reward)
            print("=========================================\n")

            # Give initial state
            s_init = torch.Tensor(s_next)
            model_free_actions.append(lstm_action.squeeze(dim = 0))
            lstm_states.append(s_init.cpu())
            states.append(s_init.cpu())
            actions.append(lstm_action.squeeze(dim = 0))
            # rewards.append(reward)

        # Stack states and actions
        actions.append(torch.zeros(10))
        a_0 = torch.stack(actions) [-H:]
        s_0 = torch.stack(states) [-H:]

        # state action pairs
        init_state_action_pair = torch.hstack((s_0, a_0))
        state_action_buffer = torch.unsqueeze(init_state_action_pair, 0)
        state_action_buffer = state_action_buffer.double()

        # Reward
        trajectory_path = "../simulator/cassie/trajectory/stepdata.bin"
        reward_obj = Reward(trajectory_path, 5.0900e-02)
        phaselen = reward_obj.phaselen

        # Cassie's model
        CassieModel = CassieModelEnv(transformer_model, representation_model,
                                     reward_obj, TIMESTEPS, d, action_dim,
                                     N_SAMPLES)

        cem_gym = gradient.GradCEMPlan(TIMESTEPS, SAMPLE_ITER, std,
                                       GRADIENT_UPDATES, N_SAMPLES, N_ELITES,
                                       CassieModel, d, state_action_buffer,
                                       True, episode_num=episode_num)
        # break

        # Policy
        model_free = False

        # True states
        previous_states = s_0
        previous_actions = a_0

        # Get initial random iteration range
        reset_iter_diff = offset + 1 - reset_iteration
        if phase >= reset_iter_diff:
            random_iteration = range(phase, offset)
        else:
            idx = np.random.randint(low=phase, high=reset_iter_diff)
            random_iteration = range(idx, idx + reset_iteration)

        # MPC timestep
        mpc_timesteps.extend(random_iteration)

        # actions.pop(H-1)
        # Remove placeholder action
        dummy_action = actions.pop()

        # Reference iteration cap
        ref_idx = 0

        for i in range(start_idx, iterations):
            print(f"Iteration Number: {i}")
            print(f"Number of steps: {number_steps}")
            # Reset training
            next_mean = get_next_mean(ref_idx, offset, TIMESTEPS, start_idx)

            # In new phase, we get new random iterations
            if phase == 0:
                # A range of random indices
                random_iteration = get_random_iteration(reset_iteration,
                                                        offset, phase)

            # Determine the policy
            if phase in random_iteration:
                mpc_timesteps.append(i)
                print("Model Based")
                model_free = False
            else:
                print("Model Free")
                model_free = True

            # Refernece mean indices
            ref_idx += 1
            if ref_idx >= offset:
                ref_idx = 0

            # Optimal action
            with torch.no_grad():
                if torch.cuda.is_available():
                    previous_states = previous_states.cuda().double()
                # Given current state, provide the next optimal action based on a_next
                optimal_action = expert_policy(previous_states[-1, :])

            # Get action from appropriate policy
            if model_free:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        previous_states = previous_states.cuda().double()
                    # Given current state, provide the next optimal action based on a_next
                    a_next = expert_policy(previous_states[-1, :])
                    a_next = a_next.cpu().detach()
                    a_next = a_next.unsqueeze(dim = 0)
                    model_free_actions.append(a_next)
            else:
                # Get optimal next step action command from MPC
                init_mean = torch.load(lstm_action_path) [H + i - 1 : H - 1 + i + TIMESTEPS]

                # Get next step from the planner
                a_next = cem_gym.forward(1, phase, state_action_buffer,
                                         next_mean,
                                         verbose=False,
                                         lstm_action=optimal_action,
                                         actions_so_far=actions,
                                         s_init=states[0])

                mpc_actions.append(a_next)

            # Select based on policy
            # print("Model Free action:", action_model_free)
            # print("MPC action:", action_mpc)
            # model_free = ction_mpc

            # Record phase before hand
            phase_before = phase

            # Update previous actions
            previous_actions[-1, :] = a_next

            # Sim step
            s_next, reward, env, phase, done, contact_information = sim.env_step(a_next, env, False, False)

            number_steps += 1
            states.append(torch.Tensor(s_next))
            if done:
                break

            # Add actions
            actions.append(a_next.squeeze(dim = 0))
            a_next_two = a_dup[i].unsqueeze(dim = 0)

            # Compare phase after step
            phase_after = phase

            # Get next state and update cache
            if torch.cuda.is_available():
                s_next = torch.Tensor(s_next)
                s_next = s_next.cuda()

            # Update counter variable
            if np.abs(phase_before - phase_after) != 1:
                reward_obj.counter = reward_obj.counter + 1

            # Log reward
            state_prediction, _ = CassieModel.dynamics(state_action_buffer, a_next)
            diff_prediction = (torch.abs(state_prediction.view(state_dim) - s_next))
            if torch.max(diff_prediction).item() > 0.2:
                print(f"Difference state prediction mpc loop {diff_prediction, torch.max(diff_prediction).item(), torch.argmax(diff_prediction).item()}")
            # reward_predicted_state = reward_obj.compute_reward(phase, state_prediction, verbose = True)
            reward_true_state = reward_obj.compute_reward(phase, torch.Tensor(s_next).view((1, 1, state_dim)), verbose = True).item()
            print("reward difference:", np.abs(reward_true_state - reward))
            # Reward differences
            # reward_model.append(reward_predicted_state)
            reward_simulator.append(reward)

            print("Low Level Reward after taking MPC action:", reward)
            print("=========================================\n")

            # Dont let it fall
            if reward < 0.65 or number_steps >= steps_needed: # or np.abs(phase_before - phase_after) != 1:
                exit(0)
                break

            # Epoch complete
            if i < (iterations - 1):
                # Set previous states and actions
                if torch.cuda.is_available() and previous_states.get_device() == 0:
                    # Update previous states and actions
                    previous_states = torch.cat((previous_states, s_next.unsqueeze(dim = 0).cuda()))
                    previous_actions = torch.cat((previous_actions, torch.zeros(a_next.size())))

            # Next states
            s_next = s_next.view(1, 1, -1)

            # Update state_action_buffer
            state_action_buffer = update_state_action_buffer(state_action_buffer, s_next, a_next)

        state_dict[episode_num] = states
        action_dict[episode_num] = actions
        print("experiment mpc iteration:", expt_mpc)
        print("episode number", episode_num)
        print("tau", TIMESTEPS)

        file_name = str(TIMESTEPS) + "_episode_" + str(episode_num)
        torch.save(action_dict, 'actions_agg/actions_tau_' + file_name + '.pt')
        torch.save(state_dict, 'states_agg/states_tau_' + file_name + '.pt')

        total_steps += number_steps
        print(f"Total steps: {total_steps}")
        # Save everything
        # torch.save(reward_model, 'expt_results/expt_actions3/reward_predicted/model_reward_' + str(iteration) + '.pt')
        # torch.save(reward_simulator, 'expt_results/expt_actions3/reward_true/simulator_reward_' + str(iteration) + '.pt')
        # torch.save(actions, 'expt_results/expt_actions3/actions/actions_' + str(iteration) + '.pt')
        # torch.save(states, 'expt_results/expt_actions3/states/states_' + str(iteration) + '.pt')
        iteration += 1

        # Re initialize to none
        sim = None
        env = None

        complete_epoch = True

        # Store states and actions of each episode if it has enough points to not crash the code
        if previous_states.size() [0] - H - max_n + 1 >= 0:
            epoch_dataset = prepare_epoch_dataset(previous_states.cpu(), previous_actions.cpu(), H, max_n)
            dataset_store.append(epoch_dataset)

    # torch.save(states, 'state_step_' + str(reset_iteration) + '.pt')
    # Global store
    # TODO: Remove
    dagger_states = torch.load('states1.pt')
    dagger_actions = torch.load('actions1.pt')
    # global_state_action_buffer = torch.load('data_store.pt')
    # new_buffer = global_state_action_buffer[0][-80:]
    # global_state_action_buffer = [new_buffer]

    # global_state_action_buffer.append(dataset_store)
    # torch.save(global_state_action_buffer, 'data_store.pt')
    # exit(0)

    # Retrained model
    transformer_model, representation_model = cem_gym.retrain(dagger_states, dagger_actions, reset_iteration)

    # Save the models
    if retrain_iteration % 5 == 0:
        torch.save(transformer_model.state_dict(), '../results3/trained_network_small_transformer' + str(retrain_iteration) + '.pt')

    retrain_iteration += 1
    reset_iteration += 1
    print(f"LSTM frequency {reset_iteration}")