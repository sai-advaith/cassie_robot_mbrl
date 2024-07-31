# Utility methods
import torch
import torch.nn as nn
import torch.cuda.amp as amp

import argparse

class ModelLearningUtility(object):
    def __init__(self, models):
        self.models = models

    def count_parameters(self):
        """
        Count parameters in model
        """
        parameter_count = 0
        for model in self.models:
            for param in model.parameters():
                # Only include trainable parameters
                if param.requires_grad:
                    parameter_count += param.numel()
        return parameter_count

# State-action dataset
class CassieDataset(torch.utils.data.Dataset):
    """
    Create dataset with history and number of steps to predict
    """
    def __init__(self, x, y, gradient = None, phase = None, history=1, steps=1):
        self.X = x
        self.Y = y
        self.gradient = gradient
        self.phase = phase
        self.H = history
        self.n = steps

    def __len__(self):
        return self.X.__len__() - self.H - self.n + 1

    def __getitem__(self, index):
        idx_window = index + self.H
        idx_future = idx_window + self.n
        if self.gradient is not None and self.phase is not None:
            # gradient of last action (a_t) and phase we end up in after a_t
            return (self.X[index:idx_window],
                    self.Y[idx_window: idx_future],
                    self.gradient[idx_window: idx_future],
                    self.phase[idx_window: idx_future])
        else:
            return (self.X[index:idx_window], self.Y[idx_window: idx_future])

def prepare_dataset(states, actions, history, steps, gradients = None,
                    phase_store = None, train = True):
    """
    Prepare the dataset of states and actions based on their episode number
    """
    datasets = []
    count_timestamps = 0

    for episode_num in states.keys():
        # Get the state and episode for that episode number
        state, action = states[episode_num], actions[episode_num]

        # Create 59 dimension vector (49 state, 10 action)
        state_action_pair = torch.hstack((state, action))

        # Concatenate
        if train:
            gradient = gradients[episode_num]
            phase = phase_store[episode_num]
            torch_dataset = CassieDataset(state_action_pair, state_action_pair,
                                          gradient=gradient, phase=phase,
                                          history=history, steps=steps)
        else:
            torch_dataset = CassieDataset(state_action_pair, state_action_pair,
                                          history=history, steps=steps)
        count_timestamps += len(torch_dataset)
        datasets.append(torch_dataset)

    stacked_dataset = torch.utils.data.ConcatDataset(datasets)
    return stacked_dataset, count_timestamps

def worst_topk_list(worst_values):
    """
    Given a list of lowest value, return top 10 worst and worst
    """
    topk_worst = torch.topk(torch.Tensor(worst_values), k = 10) [0]
    return topk_worst, topk_worst[0]

def test_model_gradients(model, batch_size, context, state_action_dim,
                         acceleration_components):
    """
    Testing the model's gradients using gradcheck
    """
    # Set in eval mode
    model.eval()

    # Input
    inputs = torch.randn(batch_size, context, state_action_dim, requires_grad=True)
    target_decoder = torch.randn(batch_size, 1, len(acceleration_components), requires_grad=True)

    # Cuda
    if torch.cuda.is_available():
        model = model.cuda().double()
        inputs = inputs.cuda().double()
        target_decoder = target_decoder.cuda().double()
    # Gradcheck
    assert torch.autograd.gradcheck(model, (inputs, target_decoder))

def weighted_loss_fn(predicted_state, true_state, network_input = None,
                     prediction_gradient = None, true_gradient = None,
                     weight = None, include_gradient = True):
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

    if include_gradient:
        # Gradient wrt input
        lambda_1 = 1e-3
        function_gradient = torch.autograd.grad(predicted_state, network_input,
                                                grad_outputs=torch.ones_like(predicted_state),
                                                create_graph=True,
                                                retain_graph=True) [0]

        # Gradient norm
        gradient_flat = function_gradient.view(function_gradient.size() [0], -1)
        gradient_penalty = ((torch.sqrt(torch.sum(gradient_flat ** 2, dim=1) + 1e-12))).mean()
        L2_loss = L2_loss + lambda_1 * gradient_penalty

        # Weight gradient
        weight_gradient = torch.ones(10)
        weight_gradient = weight_gradient * 10
        weight_gradient[6] = 0
        weight_gradient.requires_grad_(True)
        if torch.cuda.is_available():
            weight_gradient = weight_gradient.cuda()

        gradient_diff = (weight_gradient * torch.square(prediction_gradient - true_gradient)).mean()
        L2_loss = L2_loss + gradient_diff

        return L2_loss, gradient_diff, gradient_penalty

    # Only include if in train
    return L2_loss

def get_decoder_input(input_sequence, state_dim):
    """
    Given an input sequence (x_1, ... x_n)
    Output ((x_2 - x_1), ... (x_{n} - x_{n-1}))
    """
    differences = []
    for i in range(1, input_sequence.size() [1]):
        delta = input_sequence[:, i, :state_dim] - input_sequence[:, i - 1, :state_dim]
        differences.append(delta)
    return torch.stack(differences, dim=1)

def eval_model(data_loader, n, H, state_dim, state_action_dim, transformer,
               representation, val, masks):
    """
    Evaluate model on either test or validation loader
    """
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test
    test_loss = 0
    test_epoch_loss_store = []
    test_worst_per_batch = []

    # Get masks
    encoder_mask, decoder_mask, memory_mask = masks

    # Eval mode
    transformer.eval()
    representation.eval()
    loss_fn = nn.MSELoss()

    step_wise_epoch_test = {}
    for i in range(n):
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
                    target_state, target_action = target[:, j, :state_dim], target[:, j, state_dim:]
                    true_difference = target_state - train[:, -1, :state_dim]

                    # Tests
                    assert target_state.size() [1] == state_dim
                    assert true_difference.size() [1] == state_dim
                    assert train.size() [1] == H and train.size() [2] == state_action_dim

                    # Unsqueeze everything
                    target_state = target_state.unsqueeze(dim=1)
                    true_difference = true_difference.unsqueeze(dim=1)
                    target_action = target_action.unsqueeze(dim=1)

                    target_decoder = get_decoder_input(input_sequence=train,
                                                       state_dim=state_dim)
                    target_decoder = target_decoder.cuda()

                    # output predicts difference
                    encoding = representation(train).cuda()
                    predicted_difference = transformer(x=encoding,
                                                       target=target_decoder,
                                                       src_mask=encoder_mask,
                                                       target_mask=decoder_mask,
                                                       memory_mask=memory_mask)

                    # Cuda
                    if torch.cuda.is_available():
                        predicted_difference = predicted_difference.cuda()
                        target_action = target_action.cuda()
                        target_state = target_state.cuda()

                    # Basic tests
                    assert (predicted_difference.size() [2] == state_dim)
                    assert (target_state.size() [2] == state_dim)
                    assert predicted_difference.size() == true_difference.size()

                    loss = loss_fn(predicted_difference.cuda(), true_difference.cuda())
                    next_predicted_state = train[:, -1:, :state_dim] + predicted_difference
                    test_diff_tensors = test_diff_tensors + ((predicted_difference.cuda() - true_difference.cuda())**2)

                    # Test for teacher forcing
                    # assert (torch.abs(torch.mean((predicted_difference.cuda() - true_difference.cuda())**2) - torch.mean((next_predicted_state - target_state)**2)) < 1e-5)

                    # Loss
                    test_loop_loss = test_loop_loss + loss.data
                    step_wise_epoch_test[j].append(test_loop_loss.item())

                    # Predicted next state + true action at that point
                    # Add acceleration predictions to true state
                    next_pair = torch.cat((next_predicted_state, target_action.cuda()), dim = 2)
                    next_pair.to(device)

                    # Shift the data entries
                    train = torch.hstack([train[:, 1:, :], next_pair])

                # obtain the losses
                test_epoch_loss_store.append((test_loop_loss / n, i))
                test_worst_per_batch.append(torch.max(test_diff_tensors).item() / n)
                test_loss += (test_loop_loss / n)

    # Test loss
    test_loss_epoch = test_loss / len(data_loader)
    max_test_loss_tuple = max(test_epoch_loss_store, key=lambda item:item[0])
    test_topk, test_worst_component = worst_topk_list(test_worst_per_batch)
    if val:
        print("VALIDATION (agg)")
    else:
        print("TEST (agg)")
    print(test_topk)
    print("=======")
    print(test_worst_component)
    print()

    # Maximum loss
    maximal_testing_loss = max_test_loss_tuple [0]
    results = [test_worst_component, test_loss_epoch, maximal_testing_loss, step_wise_epoch_test]

    # Return all relevant results
    return results


def get_parser():
    parser = argparse.ArgumentParser(description='Model Training hyperparameters')

    # Learning rate hyperparameters
    parser.add_argument('--beta_1', type=float, default=0.9,
                        help='Beta_1 for optimizer')
    parser.add_argument('--beta_2', type=float, default=0.98,
                        help='Beta_2 for optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-9,
                        help='epsilon for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='optimizer weight decay')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')

    # Scheduler hyperparameters
    parser.add_argument('--patience', type=int, default=75,
                        help='reduce lr on plateau patience')
    parser.add_argument('--factor', type=float, default=0.9,
                        help='learning rate reduction factor')
    parser.add_argument('--warmup_steps', type=int, default=20,
                        help='number of warmup steps for scheduler')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('--max_future_steps', type=int, default=11,
                        help='Max number of states predicted in the future')
    parser.add_argument('--context_length', type=int, default=40,
                        help='Context length to predict the next state')
    parser.add_argument('--gradient_clip_norm', type=float, default=0.9,
                        help='value to clip gradients using norm clipping')

    # Teacher forcing epochs
    parser.add_argument('--val_worst_threshold', type=float, default=15.0,
                        help='teacher forcing worst val prediction threshold')
    parser.add_argument('--val_loss_threshold', type=float, default=0.08,
                        help='teacher forcing val loss threshold')
    parser.add_argument('--val_percent_change', type=float, default=15.0,
                        help='validation loss percentage change threshold')
    parser.add_argument('--teacher_forcing_epochs', type=int, default=75,
                        help='Minimum steps before teacher forcing')
    parser.add_argument('--train_worst_threshold', type=float, default=15.0,
                        help='teacher forcing worst train prediction threshold')
    parser.add_argument('--train_loss_threshold', type=float, default=0.02,
                        help='teacher forcing train loss threshold')
    parser.add_argument('--train_percent_change', type=float, default=15.0,
                        help='training loss percentage change threshold')

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

    # Dataset path
    parser.add_argument('--train_states_path', type=str,
                        default='../data/train_gradient_states_large.pt',
                        help='train states relative path')

    parser.add_argument('--train_actions_path', type=str,
                        default='../data/train_gradient_actions_large.pt',
                        help='train actions path')

    parser.add_argument('--train_analytic_gradient_path', type=str,
                        default='../data/train_numerical_gradient_large.pt',
                        help='analyitcal gradient store path')

    parser.add_argument('--train_action_phase_path', type=str,
                        default='../data/train_phase_large.pt',
                        help='simulator level phase of actions path')

    parser.add_argument('--val_states_path', type=str,
                        default='../data/val_gradient_states_large.pt',
                        help='val states path')

    parser.add_argument('--val_actions_path', type=str,
                        default='../data/val_gradient_actions_large.pt',
                        help='val actions path')

    parser.add_argument('--test_states_path', type=str,
                        default='../data/test_gradient_states_large.pt',
                        help='test states path')

    parser.add_argument('--test_actions_path', type=str,
                        default='../data/test_gradient_actions_large.pt',
                        help='test actions path')

    # Wandb parameters
    parser.add_argument('--user_name', type=str, help='wandb username')
    parser.add_argument('--project_name', type=str, help='wandb project name')
    parser.add_argument('--run_name', type=str, help='wandb run name')

    return parser