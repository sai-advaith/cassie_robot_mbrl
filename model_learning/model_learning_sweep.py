# Import python packages
import numpy as np
import matplotlib.pyplot as plt
import math
from utils.model_learning_utils import ModelLearningUtility

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

# Wandb
import wandb

# Model Imports
from model_learning.positional_encoding import PositionalEncoder
from model_learning.representation import Representation
from model_learning.transformer import Transformer
from model_learning.scale import TorchStandardScaler, TorchMinMaxScaler

# State-action dataset
class CassieDataset(torch.utils.data.Dataset):
    """
    Create dataset with history and number of steps to predict
    """
    def __init__(self, x, y, history=1, steps=1):
        self.X = x
        self.Y = y
        self.H = history
        self.n = steps

    def __len__(self):
        return self.X.__len__() - self.H - self.n + 1

    def __getitem__(self, index):
        return self.X[index:index + self.H], self.Y[index + self.H: index + self.H + self.n]

def prepare_dataset(states, actions, history, steps, train = True, standard_scaler = None, min_max_scaler = None):
    """
    Prepare the dataset of states and actions based on their episode number
    """
    datasets = []
    count_timestamps = 0
    num_episodes = 0
    if train:
        num_episodes = 6
    else:
        num_episodes = 2
    for episode_num in states.keys():
        # Get the state and episode for that episode number
        state, action = states[episode_num], actions[episode_num]

        # Scale states
        # if train:
            # Only for train
            # standard_scaler.fit(state)
            # min_max_scaler.fit(action)

        # state_transform = standard_scaler.transform(state)
        state_transform = state
        # action_transform = min_max_scaler.transform(action)
        action_transform = action

        # Integer errors
        # assert torch.mean((standard_scaler.inverse_transform(state_transform) - state)**2) < 1e-10
        # assert torch.mean((min_max_scaler.inverse_transform(action_transform) - action)**2) < 1e-10

        # Create 59 dimension vector (49 state, 10 action)
        state_action_pair = torch.hstack((state_transform, action_transform))

        # Concatenate
        torch_dataset = CassieDataset(state_action_pair, state_action_pair, history=history, steps=steps)
        count_timestamps += len(torch_dataset)
        datasets.append(torch_dataset)
        # TODO: Remove
        if episode_num == num_episodes:
            break

    stacked_dataset = torch.utils.data.ConcatDataset(datasets)
    return stacked_dataset, count_timestamps

def weighted_loss_fn(predicted_state, true_state, weight, network_input, include_gradient = True):
    """
    Method to return a weighted L2 loss
    Giving more penalty for errorneous states
    """
    # Convert everything to cuda
    if torch.cuda.is_available():
        weight = weight.cuda()
        predicted_state = predicted_state.cuda()
        true_state = true_state.cuda()

    # Gradient of output wrt input
    # L2_loss = (weight * torch.square(predicted_state - true_state)).mean()

    L2_loss = torch.square((predicted_state - true_state)).mean()
    if include_gradient:
        # Gradient wrt input
        function_gradient = torch.autograd.grad(predicted_state, network_input, grad_outputs=torch.ones_like(predicted_state), create_graph = True) [0]

        # Gradient norm
        gradient_flat = function_gradient.view(function_gradient.size() [0], -1)
        lambda_ = 1
        L2_loss = L2_loss + lambda_ * ((torch.sqrt(torch.sum(gradient_flat ** 2, dim=1) + 1e-12))).mean()

    # Only include if in train
    return L2_loss

def train():
    # History + steps definition
    H = 10
    n = 1
    max_n = 1
    batch_size = 1

    WANDB_USER_NAME = "sai-advaith"
    WANDB_PROJECT_NAME = "model-learning"
    WANDB_RUN_NAME = "model_learning_sweep"

    num_epochs = 250

    # Optimizer
    patience = 75
    factor = 0.9

    # Wandb configuration
    config_dict =   {"epochs": num_epochs,
                    "patience": patience,
                    "scale": False,
                    "factor" : factor}

    # Loss
    loss_fn = torch.nn.MSELoss()

    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_USER_NAME, name = WANDB_RUN_NAME, config = config_dict)

    H = wandb.config.context
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    weight_decay = wandb.config.decay
    dropout = wandb.config.dropout

    print(f"Hyperparameter = context : {H}, batch size : {batch_size}, dropout: {dropout}, learning rate : {learning_rate}, weight decay : {weight_decay}")

    # Dimensionality of state action pairs
    state_dim = 49
    action_dim = 10
    state_action_dim = state_dim + action_dim

    # Representation size
    representation_size = 100
    representation_hidden_one = 1000
    representation_hidden_two = 1000

    # Transformer parameters
    d_model = 512
    n_heads = 8
    num_decoders = 4
    num_encoders = 4

    # Load train states + actions data (swapping test and train because test has more deviation)
    train_states = torch.load('../data/train_states_small.pt')
    train_actions = torch.load('../data/train_actions_small.pt')

    # Load test states + actions data
    test_states = torch.load('../data/test_states_small.pt')
    test_actions = torch.load('../data/test_actions_small.pt')

    # Scaler object
    standard_scaler = TorchStandardScaler()
    min_max_scaler = TorchMinMaxScaler(-1, 1)
    # Training set
    train_dataset, train_timestamps = prepare_dataset(train_states, train_actions, H, max_n, True, standard_scaler, min_max_scaler)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"# Train timesteps {train_timestamps}")
    # Test set
    test_dataset, test_timestamps = prepare_dataset(test_states, test_actions, H, max_n, False, standard_scaler, min_max_scaler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    print(f"# Train timesteps {test_timestamps}")

    # Acceleration and velocity components of Cassie
    acceleration_components = range(31, 34)
    velocity_components = list(range(15, 31)) + list(range(40, 46))

    # Declare object
    transformer = Transformer(state_action_dim, len(acceleration_components), d_model, n_heads, num_encoders, num_decoders, H, dropout)
    representation = Representation(state_action_dim, representation_size, representation_hidden_one, representation_hidden_two)

    # Convert to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transform device
    representation.to(device)
    transformer.to(device)

    # Teacher forcing parameters

    # number of test loss samples to average over
    teacher_forcing_patience = 4
    # a slope greater than this value will trigger the next "step" in multistep loss
    stopping_rule = -0.005

    # Change type
    representation = representation.double()
    transformer = transformer.double()

    # Num epochs
    num_epochs = 500

    representation_layer_names = []
    transformer_layer_names = []
    for idx, (name, param) in enumerate(representation.named_parameters()):
        representation_layer_names.append(name)

    for idx, (name, param) in enumerate(transformer.named_parameters()):
        transformer_layer_names.append(name)


    # Configure parameters
    rate = 0.9
    disp = False

    # Utility object
    model_learning_utils = ModelLearningUtility(representation_layer_names, transformer_layer_names, learning_rate, rate, transformer, representation, disp)
    parameters = model_learning_utils.configure_optimizer()

    # Optimizer
    patience = 75
    factor = 0.9
    optimizer = optim.Adam(transformer.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience = patience)

    # Get parameters
    param1 = model_learning_utils.count_parameters(representation)
    param2 = model_learning_utils.count_parameters(transformer)
    total_parameters = param1 + param2
    # total_parameters = param2

    print("Number of parameters:", total_parameters)

    # Store batch which performs worst
    maximal_training_idx = []
    maximal_testing_idx = []
    test_loss_buffer = []
    teacher_forcing_epochs = [0]

    # Weights
    weight = torch.ones(state_dim)

    for epoch in range(num_epochs):
        # Train mode
        representation.train()
        transformer.train()

        epoch_loss = 0
        train_epoch_loss_store = []
        train_epoch_diff_store = []
        for idx, d in enumerate(train_loader):
            train, target = d[0], d[1]

            # Zero out gradients
            optimizer.zero_grad()
            loop_loss = 0
            diff_tensors = 0
            for j in range(n):
                # Get next state only
                target_state, target_action = target[:, j, acceleration_components], target[:, j, state_dim:]
                # target_state = acc_{t+1} - acc_{t}. if t = 10. target_state = t_10 - t_9
                true_difference = target_state - train[:, -1, acceleration_components]

                # target_state, target_action = target[:, j, :state_dim], target[:, j, state_dim:]
                true_state = target[:, j, :state_dim]
                assert target_state.size() [1] == len(acceleration_components)

                target_state = target_state.unsqueeze(dim = 1)
                target_action = target_action.unsqueeze(dim = 1)
                true_difference = true_difference.unsqueeze(dim = 1)
                true_state = true_state.unsqueeze(dim = 1)

                # Representation
                train = train.cuda()
                train.requires_grad_(True)
                # representations = representation(train)

                # Fed to decoder
                # last output i.e. acc_9 or (acc_9 - acc_8)
                target_decoder = (train[:, -1, acceleration_components] - train[:, -2, acceleration_components]).unsqueeze(1)

                # To Cuda
                target_decoder = target_decoder.cuda()

                # output predicts difference
                output = transformer(train, target_decoder)
                if torch.cuda.is_available():
                    output = output.cuda()
                    target_action = target_action.cuda()
                    true_state = true_state.cuda()
                assert (output.size() [2] == len(acceleration_components)) and (target_state.size() [2] == len(acceleration_components))
                assert output.size() == true_difference.size()

                # loss = weighted_loss_fn(output.cuda(), target_state.cuda(), weight, train, include_gradient = False)
                # loss_weighted = weighted_loss_fn(output.cuda(), true_difference.cuda(), weight, train, include_gradient = False)
                # print(loss_weighted)
                loss = loss_fn(output.cuda(), true_difference.cuda())
                diff_tensors = ((output.cuda() - true_difference.cuda())**2) [0, 0, :]

                # Backprop
                loss.backward()

                # Loss + backprop
                loop_loss += loss.item()

                # Predicted next state + true action at that point
                # Add acceleration predictions to true state
                true_state[:, :, acceleration_components] = output
                next_pair = torch.cat((true_state, target_action.cuda()), dim = 2)
                # next_pair = torch.cat((output, target_action.cuda()), dim = 2)
                next_pair.to(device)

                # Shift the data entries
                train = torch.hstack([train[:, 1:, :], next_pair])
                train = train.detach()

            # Store maximal loss + index
            train_epoch_loss_store.append((loop_loss, idx))
            train_epoch_diff_store.append((diff_tensors, idx))

            # obtain the loss function
            epoch_loss += loop_loss / n

            # Compute loss of all the future state predictions
            optimizer.step()

        # Get loss for epoch
        train_loss_epoch = epoch_loss / len(train_loader)
        max_training_loss_tuple = max(train_epoch_loss_store, key=lambda item:item[0])

        # Maximum loss
        maximal_training_idx.append(max_training_loss_tuple [1])
        maximal_training_loss = max_training_loss_tuple [0]
        max_loss_by_component = train_epoch_diff_store[max_training_loss_tuple [1]] [0]
        worst_component = torch.argmax(max_loss_by_component), torch.max(max_loss_by_component)

        # Give values
        print("Epoch: %d, train loss: %1.5f" % (epoch, train_loss_epoch))
        print("Epoch: %d, max train loss: %1.5f" % (epoch, maximal_training_loss))
        print(f"Epoch {epoch}, worst train prediction: {max_loss_by_component}\n")

        # Test
        test_loss = 0
        test_epoch_loss_store = []
        test_diff_loss_store = []

        # Eval mode
        transformer.eval()
        representation.eval()

        # No grad
        with torch.no_grad():
            # print("Test Epoch started!")
            for i, data in enumerate(test_loader):
                train, target = data

                # Zero out gradients
                test_loop_loss = 0
                test_diff_tensors = 0
                for j in range(n):
                    # Get next state only
                    target_state, target_action = target[:, j, acceleration_components], target[:, j, state_dim:]
                    true_difference = target_state - train[:, -1, acceleration_components]
                    true_state = target[:, j, :state_dim]
                    assert target_state.size() [1] == len(acceleration_components)
                    # target_state, target_action = target[:, j, :state_dim], target[:, j, state_dim:]

                    target_state = target_state.unsqueeze(dim = 1)
                    target_action = target_action.unsqueeze(dim = 1)
                    true_difference = true_difference.unsqueeze(dim = 1)
                    true_state = true_state.unsqueeze(dim = 1)

                    # Representation
                    train = train.cuda()
                    # representations = representation(train)
                    target_decoder = (train[:, -1, acceleration_components] - train[:, -2, acceleration_components]).unsqueeze(1)
                    # target_decoder = train[:, -1, acceleration_components].unsqueeze(1)
                    # target_decoder = representations[:, -1, :].unsqueeze(1)

                    # representations = representations.cuda()
                    target_decoder = target_decoder.cuda()

                    output = transformer(train, target_decoder)
                    # output = transformer(representations, target_decoder)
                    if torch.cuda.is_available():
                        output = output.cuda()
                        target_action = target_action.cuda()
                        true_state = true_state.cuda()
                    assert (output.size() [2] == len(acceleration_components)) and (target_state.size() [2] == len(acceleration_components))
                    loss = weighted_loss_fn(output.cuda(), true_difference.cuda(), weight, train, include_gradient = False)
                    # loss = weighted_loss_fn(output.cuda(), target_state.cuda(), weight, train, include_gradient = False)
                    test_diff_tensors = ((output.cuda() - target_state.cuda())**2) [0, 0, :]

                    # Loss + backprop
                    test_loop_loss += loss.item()

                    # Predicted next state + true action at that point
                    # Add acceleration predictions to true state
                    true_state[:, :, acceleration_components] = output
                    next_pair = torch.cat((true_state, target_action.cuda()), dim = 2)
                    # next_pair = torch.cat((output, target_action.cuda()), dim = 2)
                    next_pair.to(device)

                    # Shift the data entries
                    train = torch.hstack([train[:, 1:, :], next_pair])
                    train = train.detach()

                test_epoch_loss_store.append((test_loop_loss, i))
                test_diff_loss_store.append((test_diff_tensors, i))
                # obtain the loss function
                test_loss += test_loop_loss / n


        # Test loss
        test_loss_epoch = test_loss / len(test_loader)
        max_test_loss_tuple = max(test_epoch_loss_store, key=lambda item:item[0])
        test_loss_buffer.append(test_loss_epoch)
        max_test_loss_by_component = test_diff_loss_store[max_test_loss_tuple [1]] [0]
        test_worst_component = torch.argmax(max_test_loss_by_component), torch.max(max_test_loss_by_component)

        # Maximum loss
        maximal_testing_idx.append(max_test_loss_tuple [1])
        maximal_testing_loss = max_test_loss_tuple [0]

        # Maximum loss
        maximal_training_idx.append(max_training_loss_tuple [1])
        maximal_training_loss = max_training_loss_tuple [0]

        # validation_loss.append(validation_loss_epoch)
        log_dict = {"train_loss": train_loss_epoch,
                    "train_maximal_loss": maximal_training_loss,
                    "train_worst_value": worst_component[1],
                    "train_maximal_idx": max_training_loss_tuple [1],
                    "test_loss" : test_loss_epoch, "epoch": epoch}

        # Send to wandb
        wandb.log(log_dict)

        # Scheduler
        scheduler.step(test_loss_epoch)

        # Give values
        print("Epoch: %d, test loss: %1.5f" % (epoch, test_loss_epoch))
        print("Epoch: %d, max test loss: %1.5f" % (epoch, maximal_testing_loss))
        print(f"Epoch {epoch}, worst test prediction: {max_test_loss_by_component}\n")

        # Teacher forcing
        if epoch % num_epochs == 0 and epoch > 0:
            # torch.save(representation.state_dict(), '../results/acc_decay_representation.pt')
            torch.save(transformer.state_dict(), '../results/acc_decay_transformer.pt')
            exit(0)

    # Close run
    wandb.finish()


if __name__ == "__main__":
    # Wandb
    WANDB_USER_NAME = "sai-advaith"
    WANDB_PROJECT_NAME = "model-learning"
    WANDB_RUN_NAME = "transformer_sweep"

    # History + steps definition
    H = 10
    n = 1
    max_n = 1
    batch_size = 1

    sweep_config = {
        'method': 'grid',
        'parameters': {
            'context': {
                'values': [10, 20, 30, 40, 50]
            },
            'learning_rate': {
                'values': [1e-3, 1e-4, 1e-5]
            },
            'batch_size': {
                'values': [8, 16, 32]
            },
            'decay': {
                'values': [1e-5, 1e-4, 1e-3]
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function = train)