# Python imports
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sys import getsizeof

# Pytorch imports
import torch
from torch.autograd import grad
import torch.optim as optim
import torch.cuda.amp as amp

# Model Learning imports
from model_learning.transformer import Transformer
from model_learning.positional_encoding import PositionalEncoder
from model_learning.model_learning_transformer import CassieDataset

# Utility imports
from utils.model_learning_utils import *

# Wandb
import wandb

from torch.optim.lr_scheduler import _LRScheduler


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

def prepare_dataset(states, actions, history, steps, dataset_keys):
    """
    Prepare the dataset of states and actions based on their episode number
    """
    datasets = []
    count_timestamps = 0
    num_episodes = 0
    for episode_num in dataset_keys:
        # Get the state and episode for that episode number
        state, action = states[episode_num], actions[episode_num]

        state_transform = torch.stack(state[:-1])
        action_transform = torch.stack(action)

        # Create 59 dimension vector (49 state, 10 action)
        state_action_pair = torch.hstack((state_transform, action_transform))

        # Concatenate
        torch_dataset = CassieDataset(state_action_pair, state_action_pair, history=history, steps=steps)
        count_timestamps += len(torch_dataset)
        datasets.append(torch_dataset)

    stacked_dataset = torch.utils.data.ConcatDataset(datasets)
    return stacked_dataset, count_timestamps

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class CassieModelEnv(object):
    """
    Cassie Dynamics model
    """
    def __init__(self, transformer, representation, reward_object, TIMESTEPS, device, nu, samples):
        """
        Rollout action sequences and then return reward for each of the action sequences
        """
        # Learned models
        self.transformer = transformer
        self.representation = representation

        # Reward function
        self.reward = reward_object

        # Hyperparams
        self.dtype = torch.double
        self.batch_size = 64
        self.max_n = 11
        self.T = TIMESTEPS
        self.nu = 10
        self.K = samples
        self.d = device
        self.state_dim = 49
        self.action_dim = 10
        self.state_action_dim = self.state_dim + self.action_dim
        self.H = 40
        self.loss_fn = torch.nn.MSELoss()

        # Get masks
        decoder_mask = torch.triu(torch.ones((self.H - 1, self.H - 1))).transpose(0, 1)
        decoder_mask = decoder_mask.masked_fill(decoder_mask == 0, float('-inf')).masked_fill(decoder_mask == 1, float(0.0))

        # Apply relevant conversions
        self.decoder_mask = decoder_mask.double()
        if torch.cuda.is_available():
            self.decoder_mask = self.decoder_mask.cuda()

        self.memory_mask = None
        self.encoder_mask = torch.zeros((self.H, self.H)).double()

    def evaluate(self, state_action_buffer, a_t):
        """
        Use model and history to predict the next state
        """
        # Convert to GPU if available
        if torch.cuda.is_available():
            state_action_buffer = state_action_buffer.cuda()

        self.transformer.eval()
        self.representation.eval()

        # Decoder input
        target_decoder = get_decoder_input(input_sequence=state_action_buffer,
                                           state_dim=self.state_dim)
        target_decoder = target_decoder.cuda()


        # s_{t+1}
        with amp.autocast():
            encoding = self.representation(state_action_buffer).cuda()
            predicted_difference = self.transformer(x=encoding,
                                                    target=target_decoder,
                                                    src_mask=self.encoder_mask,
                                                    target_mask=self.decoder_mask,
                                                    memory_mask=self.memory_mask)

            # Temporary a_{t+1}
            next_action_temp = torch.zeros((a_t.size() [0], 1, self.action_dim))
            next_state = state_action_buffer[:, -1:, :self.state_dim] + predicted_difference

            if torch.cuda.is_available():
                predicted_difference = predicted_difference.cuda()
                next_action_temp = next_action_temp.cuda()
                next_state = next_state.cuda()

            # [S_{t+1}, 0]^T
            next_pair = torch.cat((next_state, next_action_temp), dim = 2)

            # Update past 10 actions - Remove 0th timestamp and replace with next_pair {t+1}
            state_action_buffer = torch.hstack((state_action_buffer[:, 1:, :], next_pair))

        return next_state, state_action_buffer

    def dynamics(self, state_action_buffer, a_t, no_gradient = False):
        """
        Feed in input to learned dynamics model from transformer model
        state_action_buffer: past H state action pairs to be fed to the models
        """
        # Replace lsat state action pair with a_t action
        if state_action_buffer.size() [0] != a_t.size() [0]:
            state_action_buffer = state_action_buffer.repeat(a_t.size() [0], 1, 1)
        a_t = a_t.unsqueeze(1)
        state_action_buffer[:, -1:, self.state_dim:] = a_t

        # Model in eval mode
        # self.transformer.eval()
        # self.representation.eval()

        # Convert to GPU if available
        if torch.cuda.is_available():
            self.transformer = self.transformer.cuda().double()
            self.representation = self.representation.cuda().double()

        # Predict
        if no_gradient:
            with torch.no_grad():
                next_state, state_action_buffer = self.evaluate(state_action_buffer, a_t)
        else:
            next_state, state_action_buffer = self.evaluate(state_action_buffer, a_t)
        return next_state, state_action_buffer

    def rollout(self, actions, init_state, phase, no_gradient = False,
                return_state = False):
        """
        Rollout action sequences at particular state and phase to give the
        reward of the action sequence
        """
        state_predictions = []
        reward_total = torch.zeros(actions.size() [1], device=self.d, dtype=self.dtype)

        # Rollout states store
        per_rollout_cache = init_state

        # Phase for getting reference position and velocity at the timestamp
        rollout_phase = phase + 1

        # Discount
        gamma = 0.95

        # Roll it out
        for t in range(self.T):
            if rollout_phase > self.reward.phaselen:
                rollout_phase = 0

            # Get the action for that timestamp
            u = actions[t, :, :]

            # Get the next state action pair record
            state, per_rollout_cache = self.dynamics(per_rollout_cache, u, no_gradient)
            state_predictions.append(state.squeeze(dim = 1))

            # Calculate rewards
            reward = self.reward.compute_reward(rollout_phase, state)
            reward = reward * (gamma ** t)
            reward_total = reward_total + reward
            rollout_phase += 1

        # Return all rewards
        if return_state:
            return [reward_total, torch.stack(state_predictions)]
        else:
            return reward_total

    def eval_model(self, data_loader, n, val):
        """
        Evaluate model on either test or validation loader
        """
        # Test
        test_loss = 0
        test_epoch_loss_store = []
        test_worst_per_batch = []
        test_worst_idx = []

        # Eval mode
        self.transformer.eval()

        step_wise_epoch_test = {}
        for i in range(self.max_n):
            step_wise_epoch_test[i] = []

        # No grad
        with torch.no_grad():
            # print("Test Epoch started!")
            for i, data in enumerate(data_loader):
                with amp.autocast():
                    train, target = data

                    # Zero out gradients
                    test_loop_loss = 0
                    test_diff_tensors = 0
                    for j in range(n):
                        if torch.cuda.is_available():
                            train = train.cuda()
                            target = target.cuda()
                        # Get next state only
                        target_state, target_action = target[:, j, :self.state_dim], target[:, j, self.state_dim:]
                        true_difference = target_state - train[:, -1, :self.state_dim]

                        # Tests
                        assert target_state.size() [1] == self.state_dim
                        assert true_difference.size() [1] == self.state_dim
                        assert train.size() [1] == self.H and train.size() [2] == self.state_action_dim

                        # Unsqueeze everything
                        target_state = target_state.unsqueeze(dim = 1)
                        true_difference = true_difference.unsqueeze(dim = 1)
                        target_action = target_action.unsqueeze(dim = 1)

                        # Fed to decoder - cuda
                        # last output i.e. acc_9 or (acc_9 - acc_8)
                        target_decoder = (train[:, -1, :self.state_dim] - train[:, -2, :self.state_dim]).unsqueeze(1)
                        target_decoder = target_decoder.cuda()

                        # Model prediction
                        encoding = self.representation(train).cuda()
                        predicted_difference = self.transformer(x=encoding,
                                                                target=target_decoder,
                                                                src_mask=self.encoder_mask,
                                                                target_mask=self.decoder_mask,
                                                                memory_mask=self.memory_mask)

                        # Cuda
                        if torch.cuda.is_available():
                            predicted_difference = predicted_difference.cuda()
                            target_action = target_action.cuda()
                            target_state = target_state.cuda()

                        # Basic tests
                        assert (predicted_difference.size() [2] == self.state_dim) and (target_state.size() [2] == self.state_dim)
                        assert predicted_difference.size() == true_difference.size()

                        loss = self.loss_fn(predicted_difference.cuda(), true_difference.cuda())
                        next_predicted_state = train[:, -1:, :self.state_dim] + predicted_difference
                        test_diff_tensors = test_diff_tensors + ((predicted_difference.cuda() - true_difference.cuda())**2)
                        assert torch.mean(((predicted_difference.cuda() - true_difference.cuda())**2)) == loss

                        # Test for teacher forcing
                        assert (torch.abs(torch.mean((predicted_difference.cuda() - true_difference.cuda())**2) - torch.mean((next_predicted_state - target_state)**2)) < 1e-5)

                        # Loss
                        test_loop_loss = test_loop_loss + loss.data
                        step_wise_epoch_test[j].append(test_loop_loss.item())

                        # Predicted next state + true action at that point
                        # Add acceleration predictions to true state
                        next_pair = torch.cat((next_predicted_state, target_action.cuda()), dim = 2)
                        next_pair.to(self.d)

                        # Shift the data entries
                        train = torch.hstack([train[:, 1:, :], next_pair])
                        train = train.detach()

                    # obtain the losses
                    test_epoch_loss_store.append((test_loop_loss / n, i))
                    max_val = torch.max(test_diff_tensors)
                    max_idx = (test_diff_tensors==torch.max(test_diff_tensors)).nonzero() [0][2]
                    test_worst_per_batch.append(max_val.item() / n)
                    test_worst_idx.append(max_idx.item())
                    test_loss += (test_loop_loss / n)

        # Test loss
        test_loss_epoch = test_loss / len(data_loader)
        max_test_loss_tuple = max(test_epoch_loss_store, key=lambda item:item[0])
        print(torch.Tensor(test_worst_per_batch))
        print()
        topk_tuple = (torch.topk(torch.Tensor(test_worst_per_batch), k = 10))
        topk_worst, topk_worst_idx = topk_tuple[0], topk_tuple[1]
        test_worst_component, test_worst_component_idx = topk_worst[0], test_worst_idx[topk_worst_idx[0]]

        if val:
            print("VALIDATION (agg)")
        else:
            print("TEST (agg)")
        print(topk_worst)
        print("=======")
        print(test_worst_component.item())
        print()
        print(test_worst_component_idx)
        print()

        # Maximum loss
        maximal_testing_loss = max_test_loss_tuple [0]

        # Return all relevant results
        return (test_worst_component, test_worst_component_idx, test_loss_epoch, maximal_testing_loss, step_wise_epoch_test)

    def weighted_loss_fn(self, predicted_state, true_state, network_input, weight = None, include_gradient = True):
        """
        Method to return a weighted L2 loss
        Giving more penalty for errorneous states
        """
        # Convert everything to cuda
        if torch.cuda.is_available():
            # weight = weight.cuda() if weight is not None
            predicted_state = predicted_state.cuda()
            true_state = true_state.cuda()
            weight = weight.cuda()

        # Gradient of output wrt input
        L2_loss = (weight * torch.square(predicted_state - true_state)).mean()

        # L2_loss = torch.square((predicted_state - true_state)).mean()
        if include_gradient:
            # Gradient wrt input
            function_gradient = torch.autograd.grad(predicted_state, network_input, grad_outputs=torch.ones_like(predicted_state), retain_graph = True) [0]

            # Gradient norm
            gradient_flat = function_gradient.view(function_gradient.size() [0], -1)
            lambda_ = 1e-4
            L2_loss = L2_loss + lambda_ * ((torch.sqrt(torch.sum(gradient_flat ** 2, dim=1) + 1e-12))).mean()

        # Only include if in train
        return L2_loss

    def retrain(self, dagger_states, dagger_actions, iteration):
        """
        Retrain models based on data collected from simulator
        """

        # Wandb
        WANDB_USER_NAME = "sai-advaith"
        WANDB_PROJECT_NAME = "model-learning"
        # WANDB_RUN_NAME = "run3_teacher_forcing_high_lr_"+str(iteration)
        WANDB_RUN_NAME = "scheulder_retraining_"+str(iteration)


        cut = int(0.8 * len(dagger_states)) #80% of the list
        dict_keys = list(dagger_states.keys())
        random.shuffle(dict_keys)
        train_keys = dict_keys[:cut] # first 80% of shuffled list
        test_keys = dict_keys[cut:] # last 20% of shuffled list

        # Training set
        train_dataset, train_timestamps = prepare_dataset(dagger_states, dagger_actions, self.H, self.max_n, train_keys)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size)

        # Test set
        test_dataset, test_timestamps = prepare_dataset(dagger_states, dagger_actions, self.H, self.max_n, test_keys)
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size)

        # Convert to cuda and double
        self.transformer = self.transformer.cuda().double()
        self.representation = self.representation.cuda().double()

        # Optimizer params
        learning_rate = 3e-4
        patience = 75
        factor = 0.9

        # Steps
        n = 1

        # Get new epoch count and max epochs
        epoch = 0
        batch_count = 0
        max_epochs = 2000

        # Get optimizer and scheduler
        params = list(self.representation.parameters()) + list(self.transformer.parameters())
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

        # Loss
        loss_fn = torch.nn.MSELoss()

        # Store batch which performs worst
        validation_loss_buffer = []
        training_loss_buffer = []
        teacher_forcing_min_epochs = 75

        # Teacher forcing buffer
        teacher_forcing_epochs = [0]

        config_dict =   {"base_learning_rate": learning_rate,
                        "batch_size": self.batch_size, "context": self.H, "patience": patience, 
                        "factor" : factor, "training_points" : train_timestamps,
                        "testing_points" : test_timestamps}

        wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_USER_NAME, name = WANDB_RUN_NAME, config = config_dict, settings=wandb.Settings(start_method="fork"))
        print(wandb.config)

        maximal_training_idx = []
        maximal_testing_idx = []
        step_wise_train_loss, step_wise_test_loss = {}, {}

        # Acceleration and velocity components of Cassie
        acceleration_components = range(31, 34)
        velocity_components = list(range(15, 31)) + list(range(40, 46))

        # Weights
        weight = torch.ones(self.state_dim)
        weight[acceleration_components] *= 20
        weight[velocity_components] *= 10

        # Enable benchmark mode
        torch.backends.cudnn.benchmark = True

        for i in range(self.max_n):
            step_wise_train_loss[i] = 0
            step_wise_test_loss[i] = 0

        while n <= self.max_n:
            mse_epoch_loss = 0
            worst_per_batch = []
            worst_idx_batch = []

            # Train mode
            self.transformer.train()

            step_wise_epoch_train = {}
            worst_train = 0
            for i in range(self.max_n):
                step_wise_epoch_train[i] = []

            train_epoch_loss_store = []

            for idx, d in enumerate(train_loader):
                with amp.autocast():
                    train, target = d[0], d[1]

                    # Zero out gradients
                    optimizer.zero_grad()
                    loop_loss = 0
                    diff_tensors = 0
                    mse_train_loop_loss = 0
                    for j in range(n):
                        train.requires_grad_(True)
                        if torch.cuda.is_available():
                            train = train.cuda()
                            target = target.cuda()
                        # Get next state only
                        target_state, target_action = target[:, j, :self.state_dim], target[:, j, self.state_dim:]
                        # target_state = acc_{t+1} - acc_{t}. if t = 10. target_state = t_10 - t_9
                        true_difference = target_state - train[:, -1, :self.state_dim]

                        assert target_state.size() [1] == self.state_dim
                        assert true_difference.size() [1] == self.state_dim
                        assert train.size() [1] == self.H and train.size() [2] == self.state_action_dim

                        # Unsqueeze everything
                        target_state = target_state.unsqueeze(dim = 1)
                        true_difference = true_difference.unsqueeze(dim = 1)
                        target_action = target_action.unsqueeze(dim = 1)

                        # Fed to decoder - cuda
                        # last output i.e. acc_9 or (acc_9 - acc_8)
                        target_decoder = (train[:, -1, :self.state_dim] - train[:, -2, :self.state_dim]).unsqueeze(1)
                        target_decoder = target_decoder.cuda()

                        # output predicts difference
                        encoding = self.representation(train).cuda()
                        predicted_difference = self.transformer(x=encoding,
                                                                target=target_decoder,
                                                                src_mask=self.encoder_mask,
                                                                target_mask=self.decoder_mask,
                                                                memory_mask=self.memory_mask)

                        # Cuda
                        if torch.cuda.is_available():
                            predicted_difference = predicted_difference.cuda()
                            target_action = target_action.cuda()
                            target_state = target_state.cuda()

                        # Basic tests
                        assert (predicted_difference.size() [2] == self.state_dim) and (target_state.size() [2] == self.state_dim)
                        assert predicted_difference.size() == true_difference.size()

                        # Compute all losses
                        loss = self.weighted_loss_fn(predicted_difference.cuda(), true_difference.cuda(), train, weight, include_gradient=True)

                        next_predicted_state = train[:, -1:, :self.state_dim] + predicted_difference
                        diff_tensors = diff_tensors + ((predicted_difference.cuda() - true_difference.cuda())**2)

                        # Test for teacher forcing
                        # assert (torch.abs(torch.mean((predicted_difference.cuda() - true_difference.cuda())**2) - torch.mean((next_predicted_state - target_state)**2)) < 1e-5)

                        # Backprop
                        loss.backward()

                        # Loss + backprop
                        mse_train_loop_loss = mse_train_loop_loss + torch.mean((predicted_difference.cuda() - true_difference.cuda())**2).item()
                        loop_loss = loop_loss + loss.item()
                        step_wise_epoch_train[j].append(loop_loss)

                        # Predicted next state + true action at that point and Add acceleration predictions to true state
                        next_pair = torch.cat((next_predicted_state, target_action.cuda()), dim=2)
                        next_pair.to(self.d)

                        # Shift the data entries
                        train = torch.hstack([train[:, 1:, :], next_pair])
                        train = train.detach()

                    # Compute loss of all the future state predictions and optimize
                    optimizer.step()
                    wandb.log({"learning rate": optimizer.param_groups[0]['lr'], "batch count": batch_count})
                    batch_count += 1

                    # Store maximal loss + index
                    train_epoch_loss_store.append((mse_train_loop_loss / n, idx))
                    max_val = torch.max(diff_tensors)
                    max_idx = (diff_tensors==torch.max(diff_tensors)).nonzero() [0][2]
                    worst_per_batch.append(max_val.item() / n)
                    worst_idx_batch.append(max_idx.item())

                    # obtain the loss function
                    mse_epoch_loss += (mse_train_loop_loss / n)

            # Get loss for epoch
            train_loss_epoch = mse_epoch_loss / len(train_loader)
            max_training_loss_tuple = max(train_epoch_loss_store, key=lambda item:item[0])

            # Maximum loss
            maximal_training_loss = max_training_loss_tuple [0]
            topk_tuple = (torch.topk(torch.Tensor(worst_per_batch), k = 10))
            topk_worst, topk_worst_idx = topk_tuple[0], topk_tuple[1]
            worst_component, worst_component_idx = topk_worst[0], worst_idx_batch[topk_worst_idx[0]]

            # Train loss buffering
            training_loss_buffer.append(train_loss_epoch)

            # Log training loss stats
            print(torch.Tensor(worst_per_batch))
            print()
            print("TRAIN (agg)")
            print(topk_worst)
            print("=======")
            print(worst_component.item())
            print()
            print(worst_component_idx)
            print()

            # Give values
            print("Epoch: %d, train loss: %1.5f" % (epoch, train_loss_epoch))
            print("Epoch: %d, max train loss: %1.5f" % (epoch, maximal_training_loss))

            # Test set
            test_worst_component, test_worst_component_idx, test_loss_epoch, maximal_testing_loss, step_wise_epoch_test = self.eval_model(test_loader, n, val = False)

            # Log test loss stats
            print("Epoch: %d, test loss: %1.5f" % (epoch, test_loss_epoch))
            print("Epoch: %d, max test loss: %1.5f" % (epoch, maximal_testing_loss))
            validation_loss_buffer.append(test_loss_epoch.item())
            scheduler.step(test_loss_epoch)

            # Step wise loss
            for k in range(n):
                step_wise_train_loss[k] = np.sum(step_wise_epoch_train[k]) / len(train_loader)
                step_wise_test_loss[k] = np.sum(step_wise_epoch_test[k]) / len(test_loader)
                print(f"Epoch {epoch}, Step {k + 1}, train (with gradient) loss: {step_wise_train_loss[k]}")
                print(f"Epoch {epoch}, Step {k + 1}, test loss: {step_wise_test_loss[k]}")

            # Log to WANDB
            log_dict = {"train_loss": train_loss_epoch, "train_maximal_loss": maximal_training_loss, "train_worst_value": worst_component,
                        "train_worst_value_idx": worst_component_idx,
                        "test_worst_value" : test_worst_component, "test_worst_value_idx" : test_worst_component_idx, "test_loss" : test_loss_epoch, "test_maximal_loss": maximal_testing_loss,
                        "epoch": epoch}

            # Adding step wise loss to wandb
            for k in range(n):
                log_dict['train_step'+ str(k + 1) + "_loss"] = step_wise_train_loss[k]
                log_dict['test_step'+ str(k + 1) + "_loss"] = step_wise_test_loss[k]

            # Teacher forcing
            if epoch >= teacher_forcing_min_epochs:
                # Flatline conditions (teacher_forcing_min_epochs = 75)
                epochs_past = epoch - teacher_forcing_min_epochs
                # Change in training loss past 75 epochs (if it reduces => change % > 0)
                training_loss_change = (training_loss_buffer[epochs_past] - training_loss_buffer[epoch]) / training_loss_buffer[epochs_past]
                print(f"training loss change {training_loss_change * 100}")
                train_flat_condition = training_loss_change * 100 < 10 and training_loss_change > 0

                # Decrease in validation loss past 75 epochs
                validation_loss_change = (validation_loss_buffer[epochs_past] - validation_loss_buffer[epoch]) / validation_loss_buffer[epochs_past]
                print(f"testing loss change {validation_loss_change * 100}")
                val_flat_condition = validation_loss_change * 100 < 10 and validation_loss_change > 0

                # Threshold conditions
                val_threshold_condition = test_worst_component < 8.0 and test_loss_epoch < 0.04
                train_threshold_condition = worst_component < 8.0 and train_loss_epoch < 0.05

                # Max step condition
                max_step = n <= self.max_n

                # Stopping codition
                threshold_conditions = val_threshold_condition and train_threshold_condition
                change_conditions = val_flat_condition and train_flat_condition
                stopping_rule = threshold_conditions and change_conditions and max_step

                if stopping_rule:
                    n += 1
                    teacher_forcing_epochs.append(epoch)
                    print(f"Stopping rule reached. Moving on to predicting n={n}\n")

                    # Re initialize the optimizer and scheduler
                    optimizer = optim.Adam(self.transformer.parameters(), lr = learning_rate, weight_decay = 1e-4)
                    # scheduler = CosineWarmupScheduler(optimizer = optimizer, warmup=5000, max_iters=20000)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience = patience)
                    if n % 5 == 0:
                        print(f"Model saved after {n} steps")
                        torch.save(self.transformer.state_dict(), '../results/transformer_retrain_' + str(n) + '.pt')
                    if n > self.max_n:
                        print("Model saved")
                        torch.save(self.transformer.state_dict(), '../results/transformer_retrain.pt')
                        break

            # Count for epochs
            epoch += 1
            if epoch >= max_epochs:
                break

            # Send to wandb
            wandb.log(log_dict)

        # Close run
        wandb.finish()
        return self.transformer, self.representation
