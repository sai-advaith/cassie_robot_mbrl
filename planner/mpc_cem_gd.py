# Pytorch import
import torch
from torch import nn, optim

# Python import
import numpy as np
from copy import deepcopy

class CEM_GD():
    def __init__(self, planning_horizon, opt_iters, samples, top_samples, cassie_env, device, init_state_action_pair, grad_clip=True):
        super().__init__()
        self.env = cassie_env
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K = samples
        self.top_K = top_samples
        self.device = device
        self.grad_clip = grad_clip
        self.a_size = 10
        self.G = 1
        self.J = 1
        self.init_state_action_pair = init_state_action_pair

    def update_learning_rate(self, optimizer, eta):
        for g in optimizer.param_groups:
            g['lr'] = eta
        return optimizer

    def forward(self, batch_size, phase, state_action_buffer, next_mean, return_plan=False, return_plan_each_iter=False):
        # Here batch is strictly if multiple Plans should be performed!
        B = batch_size

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = next_mean.view(next_mean.size() [0], 1, 1, next_mean.size() [1])

        # TODO: Update this
        a_std = torch.ones(self.H, B, 1, self.a_size, device=self.device) * 0.01

        # Sample actions (T x (B*K) x A)
        actions_init = (a_mu + a_std * torch.randn(self.H, B, self.K, self.a_size, device=self.device)).view(self.H, B * self.K, self.a_size)
        actions = actions_init.clone().detach().requires_grad_()

        # Define optimizer
        plan_each_iter = []

        # Gradient update
        rollout_state_action_buffer = state_action_buffer

        # Returns (B*K)actions, init_state, phase
        returns = self.env.rollout(actions, rollout_state_action_buffer, phase, no_gradient = True)
        tot_returns = returns.sum()
        print(f"total return {tot_returns.item()} and average return {returns.mean().item() / self.H}")
        # (-tot_returns).backward()

        # Get elite samples
        _, topk = returns.reshape(B, self.K).topk(self.top_K, dim=1, largest=True, sorted=False)
        topk += self.K * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)

        # Sort these best actions
        best_actions_list = actions[:, topk.view(-1)].reshape(self.H, B, self.top_K, self.a_size)
        best_actions = best_actions_list.clone().detach().requires_grad_()

        optimizer_list = []
        action_sequences_list = []
        # Store relevant action sequences
        for i in range(self.top_K):
            act_seq = best_actions[:, 0, i, :]
            action_seq = act_seq.clone().detach().requires_grad_()
            action_sequences_list.append(action_seq)
            optimizer_list.append({'params' : action_seq}) 

        # Number of action sequences
        n = len(action_sequences_list)

        # Optimization parameters
        base_lr = 1e-8
        factor = 10
        max_tries = 7
        max_iterations = 15

        # Define optimizer
        optimizer = optim.Adam(optimizer_list, lr = base_lr)
        optimizer.zero_grad()

        # Save all parameters
        saved_parameters = [None for i in range(n)]
        saved_opt_states = [None for i in range(n)]
        current_iteration = np.array([0 for i in range(n)])
        done = np.array([False for i in range(n)])

        # Action sequence as a big tensor
        action_sequences_batch = torch.stack(action_sequences_list, dim = 1)
        reward_all = self.env.rollout(action_sequences_batch, rollout_state_action_buffer, phase, no_gradient = False)

        current_reward = [reward_all[i].item() for i in range(n)]

        for i in range(n):
            action_sequences = action_sequences_list[i]
            saved_parameters[i] = action_sequences.detach().clone()
            saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
            (-reward_all[i]).backward(retain_graph=(i != n - 1))

            # Clip
            epsilon = 1e-6
            max_grad_norm = 1.0
            # print(action_sequences.size(), action_sequences)
            actions_grad_norm = action_sequences.grad.norm(2.0, dim=1, keepdim=True) + epsilon
            # print(actions_grad_norm.size())
            # print("before clip", action_sequences.grad.max().cpu().numpy())

            # Normalize by that
            action_sequences.grad.data.div_(actions_grad_norm)
            action_sequences.grad.data.mul_(actions_grad_norm.clamp(min=0, max=max_grad_norm))
            # print("after clip", action_sequences.grad.max().cpu().numpy())
            # exit(0)


        while not np.all(done):
            optimizer.step()

            # Compute objectives of all trajectories after stepping
            action_sequences_batch = torch.stack(action_sequences_list, dim = 1)
            reward_all = self.env.rollout(action_sequences_batch, rollout_state_action_buffer, phase, no_gradient = False)

            backwards_pass = []

            for i in range(n):
                if done[i]:
                    continue
                action_sequences = action_sequences_list[i]
                if reward_all[i].item() > current_reward[i]:
                    # If after the step, the cost is higher, then undo
                    action_sequences.data = saved_parameters[i].data.clone()
                    optimizer.state[action_sequences] = deepcopy(saved_opt_states[i])
                    optimizer.param_groups[i]['lr'] *= factor
                    if optimizer.param_groups[i]['lr'] > factor**max_tries:
                        # line search failed, mark action sequence as done
                        action_sequences.grad = None
                        done[i] = True
                else:
                    # successfully completed step.
                    # Save current state, and compute gradients
                    saved_parameters[i] = action_sequences.detach().clone()
                    saved_opt_states[i] = deepcopy(optimizer.state[action_sequences])
                    current_reward[i] = reward_all[i].item()
                    optimizer.param_groups[i]['lr'] = 1e-8
                    action_sequences.grad = None
                    backwards_pass.append(i)

                    current_iteration[i] += 1
                    if current_iteration[i] > max_iterations:
                        action_sequences.grad = None
                        done[i] = True

            to_compute = [(-reward_all[i]) for i in backwards_pass]
            grads = [(torch.empty_like(-reward_all[i])*0 + 1).to(self.device) for i in backwards_pass]
            torch.autograd.backward(to_compute, grads)

        # Get reward
        action_sequences_batch = torch.stack(action_sequences_list, dim = 1)
        final_reward = self.env.rollout(action_sequences_batch, rollout_state_action_buffer, phase, no_gradient = True)
        # Send best action
        _, topk_idx = torch.topk(final_reward, 1)
        
        # Reset
        best_action = action_sequences_batch[0, topk_idx.item(), :].unsqueeze(dim = 0)
        print(best_action, best_action.size())

        return best_action.cpu().detach()
