# Import python packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import random
import argparse

# Import reward
from planner.reward import Reward

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Utility imports
from utils.model_learning_utils import ModelLearningUtility
from utils.model_learning_utils import *

# Wandb
import wandb

# Model Imports
from model_learning.positional_encoding import PositionalEncoder
from model_learning.representation import Representation
from model_learning.transformer import Transformer

class ReduceOnPlateauWithWarmup(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_steps, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.warmup_steps = warmup_steps
        self.current_lr = self.optimizer.param_groups[0]['lr']
        self.initial_lr = self.current_lr
        self.last_epoch = 0

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.initial_lr * (self.last_epoch + 1) / (self.warmup_steps)]
        else:
            return super().get_lr()

    def step(self, metrics):
        if self.last_epoch < self.warmup_steps:
            self.current_lr = self.get_lr()[0]
            self.optimizer.param_groups[0]['lr'] = self.current_lr
            self.last_epoch += 1
        else:
            super().step(metrics)
            self.current_lr = self.optimizer.param_groups[0]['lr']

def multi_phase_reward(phase_j, next_predicted_state, reward_obj):
    rewards = reward_obj.compute_multiphase_reward(phase_j, next_predicted_state)
    return rewards.sum()

if __name__ == "__main__":
    # Parse the arguments
    parser = get_parser()
    parser.print_help()
    args = parser.parse_args()

    # Wandb
    WANDB_USER_NAME = args.user_name
    WANDB_PROJECT_NAME = args.project_name
    WANDB_RUN_NAME = args.run_name
    print(WANDB_RUN_NAME)

    # History + steps definition
    H = args.context_length

    # Start by predicting one step
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
    num_decoders = args.num_decoders
    num_encoders = args.num_encoders
    dropout = args.dropout

    # Load train states + actions data (swapping test and train because test has more deviation)
    train_states_path = args.test_states_path
    traain_actions_path = args.train_actions_path
    train_states = torch.load(train_states_path)
    train_actions = torch.load(traain_actions_path)

    # Train gradients
    train_analytical_grad_path = args.train_analytic_gradient_path
    train_action_phase_path = args.train_action_phase_path
    train_analytic_gradient = torch.load(train_analytical_grad_path)
    train_action_phase = torch.load(train_action_phase_path)

    # Load test states + actions data
    test_states_path = args.test_states_path
    test_actions_path = args.test_actions_path
    test_states = torch.load(test_states_path)
    test_actions = torch.load(test_actions_path)

    # Load validation states + actions data
    val_states_path = args.val_states_path
    val_actions_path = args.val_actions_path
    validation_states = torch.load(val_states_path)
    validation_actions = torch.load(val_actions_path)

    # Training set
    # Format : (batch_size, num timestamps, size)
    train_dataset, train_timestamps = prepare_dataset(train_states, 
                                                      train_actions, H, max_n,
                                                      train_analytic_gradient,
                                                      train_action_phase,
                                                      train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # Validation set
    # Format : (batch_size, num timestamps, size)
    validation_dataset, validation_timestamps = prepare_dataset(validation_states,
                                                                validation_actions,
                                                                H, max_n,
                                                                train=False)

    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    # Test set
    # Format : (batch_size, num timestamps, size)
    test_dataset, test_timestamps = prepare_dataset(test_states, test_actions,
                                                    H, max_n, train=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size = batch_size,
                                              shuffle=True)

    # Acceleration and velocity components of Cassie
    # Ref: simulator/cassie/cassie.py lines 315-324 for component details
    acceleration_components = range(31, 34)
    velocity_components = list(range(15, 31)) + list(range(40, 46))

    # Declare transformer object
    transformer_model = Transformer(representation_size=representation_size,
                                    output_size=state_dim,
                                    d_model=d_model,
                                    n_heads=num_attention_heads,
                                    num_encoders=num_encoders,
                                    num_decoders=num_decoders, history=H,
                                    dropout=dropout)

    # Representation learning object
    representation_model = Representation(input_size=representation_input_size,
                                          representation_size=representation_size,
                                          hidden_size1=representation_hidden_one,
                                          hidden_size2=representation_hidden_two,
                                          dropout=dropout)

    # Get masks
    decoder_mask = transformer_model.generate_square_subsequent_mask(H - 1)
    memory_mask = None
    encoder_mask = torch.zeros((H, H)).double()
    masks = [encoder_mask, decoder_mask, memory_mask]

    # Distribute
    transformer_model = nn.DataParallel(transformer_model)

    # Convert to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transform device
    representation_model.to(device)
    transformer_model.to(device)

    # Change type
    representation_model = representation_model.double()
    transformer_model = transformer_model.double()

    # Num epochs
    num_epochs = args.num_epochs
    grad_clip = args.gradient_clip_norm

    # Loss
    loss_fn = nn.MSELoss()

    # Initialize weights
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for p in representation_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Configure parameters
    learning_rate = args.learning_rate
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    weight_decay = args.weight_decay
    beta = (beta_1, beta_2)
    epsilon = args.epsilon

    # Utility object
    models = [transformer_model, representation_model]
    model_learning_utils = ModelLearningUtility(models)

    # Optimizer
    params = list(representation_model.parameters()) + list(transformer_model.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate, betas=beta, eps=epsilon,
                            weight_decay=weight_decay)

    # Scheduler + warmup
    patience = args.patience
    factor = args.factor
    warmup_steps = args.warmup_steps
    scheduler = ReduceOnPlateauWithWarmup(optimizer=optimizer, mode='min',
                                          warmup_steps=warmup_steps,
                                          factor=factor, verbose=True,
                                          patience=patience)

    # Get parameters
    total_parameters = model_learning_utils.count_parameters()
    print("Number of parameters:", total_parameters)

    # Wandb configuration
    config_dict =   {"learning_rate": learning_rate, "epochs": num_epochs,
                    "batch_size": batch_size, "context": H, "patience": patience, 
                    "factor" : factor, "training_points" : train_timestamps,
                    "validation_points": validation_timestamps,
                    "testing_points" : test_timestamps,
                    "number_parameters": total_parameters}

    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_USER_NAME, name=WANDB_RUN_NAME)
    wandb.config.update(config_dict)
    print(wandb.config)

    # Gradient open
    viz_gradient = True
    # Only if we want to see gradients
    if viz_gradient:
        wandb.watch(transformer_model, log = 'all')
        wandb.watch(representation_model, log = 'all')

    # Store batch which performs worst
    validation_loss_buffer = []
    training_loss_buffer = []

    teacher_forcing_min_epochs = args.teacher_forcing_epochs
    
    # Validation loss constraints for teacher forcing
    val_worst_threshold = args.val_worst_threshold
    val_loss_threshold = args.val_loss_threshold
    val_loss_percent_change = args.val_percent_change

    # Train loss constraints
    train_worst_threshold = args.train_worst_threshold
    train_loss_threshold = args.train_loss_threshold
    train_loss_percent_change = args.train_percent_change

    teacher_forcing_epochs = [0]

    # Weights
    weight = torch.ones(state_dim)
    # Penalize acceleration and velocity components more
    weight[acceleration_components] *= 20
    weight[velocity_components] *= 10
    weight.requires_grad_(True)

    # problem dim
    problem_dim = 6
    components = list(range(action_dim))
    components = components[:problem_dim] + components[problem_dim + 1:]

    # Reward
    trajectory_path = "../simulator/cassie/trajectory/stepdata.bin"
    reward_obj = Reward(trajectory_path, 5.0900e-02)
    phaselen = reward_obj.phaselen

    # Enable benchmark mode
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        mse_epoch_loss = 0
        loop_loss = 0
        worst_per_batch = []
        worst_gradient_per_batch = []
        # Train mode
        representation_model.train()
        transformer_model.train()

        epoch_loss = 0
        step_wise_epoch_train = {}
        step_wise_gradient_penalty = {}
        step_wise_gradient_diff = {}
        for i in range(max_n):
            step_wise_epoch_train[i] = []
            step_wise_gradient_penalty[i] = []
            step_wise_gradient_diff[i] = []

        train_epoch_loss_store = []

        for idx, d in enumerate(train_loader):
            with amp.autocast():
                # Values from a batch
                train, target, gradient, phase = d
                if torch.cuda.is_available():
                    train = train.cuda()
                    target = target.cuda()
                    gradient = gradient.cuda()

                # Zero out gradients
                optimizer.zero_grad(set_to_none=True)

                # Loss variables
                loop_loss, gradient_diff_loop, gradient_penalty_loop = 0, 0, 0
                diff_tensors, mse_train_loop_loss, gradient_diff_tensors = 0, 0, 0

                for j in range(n):
                    train.requires_grad_(True)
                    # Get next state only
                    target_state, target_action = target[:, j, :state_dim], target[:, j, state_dim:]

                    # target_state = x_{t+1} - x_{t}. if t = 10. target_state = t_10 - t_9
                    true_difference = target_state - train[:, -1, :state_dim]

                    # First order information
                    numerical_gradient = gradient[:, j, :]
                    numerical_gradient.requires_grad_(True)
                    phase_j = phase[:, j]

                    assert target_state.size() [1] == state_dim
                    assert true_difference.size() [1] == state_dim
                    assert train.size() [1] == H and train.size() [2] == state_action_dim

                    # Unsqueeze everything
                    target_state = target_state.unsqueeze(dim=1)
                    true_difference = true_difference.unsqueeze(dim=1)
                    target_action = target_action.unsqueeze(dim=1)

                    # Fed to decoder - cuda
                    target_decoder = get_decoder_input(input_sequence=train,
                                                       state_dim=state_dim)
                    target_decoder = target_decoder.cuda()

                    # output predicts difference
                    encoding = representation_model(train).cuda()
                    predicted_difference = transformer_model(x=encoding,
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
                    assert (predicted_difference.size() [2] == state_dim) and (target_state.size() [2] == state_dim)
                    assert predicted_difference.size() == true_difference.size()

                    next_predicted_state = train[:, -1:, :state_dim] + predicted_difference

                    rewards = multi_phase_reward(phase_j, next_predicted_state, reward_obj)
                    grad_y = torch.autograd.grad(-rewards, train, create_graph=True, retain_graph=True)[0]

                    # With respect to last action
                    gradient_last_action = grad_y[:, -1, state_dim:]

                    # Compute all losses
                    losses = weighted_loss_fn(predicted_difference.cuda(),
                                              true_difference.cuda(),
                                              train,
                                              gradient_last_action,
                                              numerical_gradient,
                                              weight=weight,
                                              include_gradient=True)
                    # Unpack losses
                    loss, gradient_diff, gradient_penalty = losses

                    # Zero out gradient
                    transformer_model.zero_grad(set_to_none=True)
                    representation_model.zero_grad(set_to_none=True)

                    # Accumulate loss for step-wise case
                    loop_loss = loop_loss + loss
                    gradient_diff_loop = gradient_diff_loop + gradient_diff
                    gradient_penalty_loop = gradient_penalty_loop + gradient_penalty

                    # MSE
                    # mask_gradient_diff = torch.ones()
                    diff_tensors = diff_tensors + ((predicted_difference.cuda() - true_difference.cuda())**2)
                    gradient_diff_tensors = gradient_diff_tensors + ((numerical_gradient.cuda() [:, components] - gradient_last_action.cuda() [:, components])**2)

                    # Loss + backprop
                    mse_train_loop_loss = mse_train_loop_loss + torch.mean((predicted_difference.cuda() - true_difference.cuda())**2).item()

                    # Step wise loss
                    step_wise_epoch_train[j].append(loop_loss.data)
                    step_wise_gradient_diff[j].append(gradient_diff_loop.data)
                    step_wise_gradient_penalty[j].append(gradient_penalty_loop.data)

                    # Predicted next state + true action at that point and Add acceleration predictions to true state
                    next_pair = torch.cat((next_predicted_state, target_action.cuda()), dim = 2)
                    next_pair.to(device)

                    # Shift the data entries
                    train = torch.hstack([train[:, 1:, :], next_pair])

                # Compute loss of all the future state predictions and optimize
                loop_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

                # Store maximal loss + index
                train_epoch_loss_store.append((mse_train_loop_loss / n, idx))
                worst_per_batch.append(torch.max(diff_tensors).item() / n)
                worst_gradient_per_batch.append(torch.max(gradient_diff_tensors).item() / n)

                # obtain the loss function
                mse_epoch_loss = mse_epoch_loss + (mse_train_loop_loss / n)

        # Get loss for epoch
        train_loss_epoch = mse_epoch_loss / len(train_loader)
        max_training_loss_tuple = max(train_epoch_loss_store, key=lambda item:item[0])

        # Maximum loss
        maximal_training_loss = max_training_loss_tuple [0]
        topk_worst, worst_component = worst_topk_list(worst_per_batch)
        _, worst_gradient = worst_topk_list(worst_gradient_per_batch)

        # Train loss buffering
        training_loss_buffer.append(train_loss_epoch)

        # Log training loss stats
        print("TRAIN (agg)")
        print(topk_worst)
        print("=======")
        print(worst_component)
        print("=======")
        print(worst_gradient)

        # Give values
        print("Epoch: %d, train loss: %1.5f" % (epoch, train_loss_epoch))
        print("Epoch: %d, max train loss: %1.5f" % (epoch, maximal_training_loss))

        # Validation run
        validation_loss_results = eval_model(validation_loader, n, H,
                                             state_dim, state_action_dim,
                                             transformer_model,
                                             representation_model, val=True,
                                             masks=masks)

        # Unpack losses
        val_worst_component, val_loss_epoch, maximal_val_loss, step_wise_epoch_val = validation_loss_results

        validation_loss_buffer.append(val_loss_epoch.item())

        # Scheduler
        scheduler.step(val_loss_epoch)

        # Log test loss stats
        print("Epoch: %d, val loss: %1.5f" % (epoch, val_loss_epoch))
        print("Epoch: %d, max val loss: %1.5f" % (epoch, maximal_val_loss))

        # Test set
        test_loss_results = eval_model(test_loader, n, H, state_dim,
                                       state_action_dim, transformer_model,
                                       representation_model, val=False,
                                       masks=masks)

        # Unpack losses
        test_worst_component, test_loss_epoch, maximal_testing_loss, step_wise_epoch_test = test_loss_results

        # Log test loss stats
        print("Epoch: %d, test loss: %1.5f" % (epoch, test_loss_epoch))
        print("Epoch: %d, max test loss: %1.5f" % (epoch, maximal_testing_loss))

        # Log to WANDB
        log_dict = {"train_loss": train_loss_epoch,
                    "train_maximal_loss": maximal_training_loss,
                    "train_worst_value": worst_component,
                    "train_worst_gradient": worst_gradient,
                    "validation_loss": val_loss_epoch,
                    "validation_maximal_loss": maximal_val_loss,
                    "validation_worst_value": val_worst_component,
                    "test_worst_value" : test_worst_component,
                    "test_loss" : test_loss_epoch,
                    "test_maximal_loss": maximal_testing_loss, "epoch": epoch}

        for step in range(n):
            step_average = 1 / (step + 1)
            train_loss_k = torch.sum(torch.Tensor(step_wise_epoch_train[step])).item() / len(train_loader)
            log_dict['train_step'+ str(step + 1) + "_loss"] = step_average * train_loss_k

            train_gradient_diff_k = torch.sum(torch.Tensor(step_wise_gradient_diff[step])).item() / len(train_loader)
            log_dict['train_step'+ str(step + 1) + "_gradient_difference"] = step_average * train_gradient_diff_k

            train_gradient_pen_k = torch.sum(torch.Tensor(step_wise_gradient_penalty[step])).item() / len(train_loader)
            log_dict['train_step'+ str(step + 1) + "_gradient_penalty"] = step_average * train_gradient_pen_k

            test_loss_k = np.sum(step_wise_epoch_test[step]) / len(test_loader)
            log_dict['test_step'+ str(step + 1) + "_loss"] = step_average * test_loss_k

            val_loss_k = np.sum(step_wise_epoch_val[step]) / len(validation_loader)
            log_dict['validation_step'+ str(step + 1) + "_loss"] = step_average * val_loss_k

            print(f"Epoch {epoch}, Step {step + 1}, train (with gradient) loss: {train_loss_k}")
            print(f"Epoch {epoch}, Step {step + 1}, train gradient difference: {train_gradient_diff_k}")
            print(f"Epoch {epoch}, Step {step + 1}, train gradient penalty: {train_gradient_pen_k}")
            print(f"Epoch {epoch}, Step {step + 1}, test loss: {test_loss_k}")
            print(f"Epoch {epoch}, Step {step + 1}, validation loss: {val_loss_k}")

        # Teacher forcing
        if epoch >= teacher_forcing_min_epochs:
            # Flatline conditions (teacher_forcing_min_epochs = 75)
            epochs_past = epoch - teacher_forcing_min_epochs
            # Change in training loss past 75 epochs (if it reduces => change % > 0)
            training_loss_change = (training_loss_buffer[epochs_past] - training_loss_buffer[epoch]) / training_loss_buffer[epochs_past]
            print(f"training loss change {training_loss_change * 100}")
            train_flat_condition = training_loss_change * 100 < train_loss_percent_change and training_loss_change > 0

            # Decrease in validation loss past 75 epochs
            validation_loss_change = (validation_loss_buffer[epochs_past] - validation_loss_buffer[epoch]) / validation_loss_buffer[epochs_past]
            print(f"validation loss change {validation_loss_change * 100}")
            val_flat_condition = validation_loss_change * 100 < val_loss_percent_change and validation_loss_change > 0

            # Threshold conditions
            val_threshold_condition = val_worst_component < val_worst_threshold and val_loss_epoch < val_loss_threshold
            train_threshold_condition = worst_component < train_worst_threshold and train_loss_epoch < train_loss_threshold

            # Max step condition
            max_step = n <= max_n

            # Stopping codition
            threshold_conditions = val_threshold_condition and train_threshold_condition
            change_conditions = val_flat_condition and train_flat_condition
            stopping_rule = threshold_conditions and change_conditions and max_step

            if stopping_rule:
                n += 1
                teacher_forcing_epochs.append(epoch)
                print(f"Stopping rule reached. Moving on to predicting n={n}\n")

                # Re initialize the optimizer and scheduler
                optimizer = optim.Adam(transformer_model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      'min',
                                                                      factor=factor,
                                                                      patience=patience)

                if n % 3 == 0:
                    print("Model saved after 5 steps")
                    torch.save(transformer_model.state_dict(), '../results/transformer_model_' + str(n) + '.pt')
                    torch.save(representation_model.state_dict(), '../results/representation_model_' + str(n) + '.pt')
                if n > max_n:
                    print("Model saved")
                    torch.save(transformer_model.state_dict(), '../results/transformer_model.pt')
                    torch.save(representation_model.state_dict(), '../results/representation_model.pt')
                    exit(0)

        # Send to wandb
        wandb.log(log_dict)
    # Close run
    wandb.finish()

    # Save the models
    torch.save(transformer_model.state_dict(), '../results/trained_network_new_transformer4.pt')
    torch.save(representation_model.state_dict(), '../results/trained_network_new_representation4.pt')
