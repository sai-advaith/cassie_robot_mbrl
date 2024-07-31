import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal

# Simulator imports
from simulator.evaluate_expert import CassieSimulator
from simulator.cassie.cassiemujoco import CassieSim, CassieVis
from simulator.cassie.trajectory import CassieTrajectory

# Python import
import time

# TODO:
# 1. Multi step elite sample - unfreeze after every timestep
# 2. Does not disrupt the sim and env in grad_planner. idea -> re-execute in new object

class GradCEMPlan():
    def __init__(self, planning_horizon, opt_iters, std, gradient_iters,
                 samples, top_samples, cassie_env, device, init_state_action_pair,
                 grad_clip=False, episode_num=None):
        """
        Gradient-CEM planner for Cassie robot
        """
        super().__init__()
        self.env = cassie_env
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K = samples
        self.top_K = top_samples
        self.device = device
        self.grad_clip = grad_clip
        self.a_size, self.s_size = 10, 49
        self.init_state_action_pair = init_state_action_pair
        self.gradient_iters = gradient_iters
        self.std = std
        self.momentum = 0.1
        self.reset_iteration = 4
        self.episode_num = episode_num

        self.init_cov_diag = torch.Tensor([0.02, 0.02, 0.03, 0.03, 0.01, 0.02, 0.02, 0.03, 0.03, 0.01])
        self.mean = None
        self.std = None

    def filter_elite_samples(self, actions, states, sim, env, last_state, init_clock):
        """
        Given elite actions and results of those elite actions
        determine which are accurately predicted. Reset after every timestep
        """
        state_time_predictions = []
        # Evaluate elite actions in sim
        for elite_k in range(actions.size() [1]):
            elite_sample_prediction = []
            # Rollout every elite action sequence
            for time in range(actions.size() [0]):
                action = actions[time, elite_k, :]
                state, _, _, _, _, _ = sim.env_step(action.cpu(), env,
                                                    visualize=False,
                                                    initial_step=env is None)

                # Convert to cuda
                state = torch.Tensor(state).cuda().double()
                elite_sample_prediction.append(state)

            # Get state action predictions
            state_time_predictions.append(torch.stack(elite_sample_prediction))
            self.reset_time_state(sim, env, last_state, init_clock)

        # Stack true states
        true_state_sim = torch.stack(state_time_predictions, dim=1)

        # Compute best states
        errors = []
        loss_fn = nn.MSELoss()
        for elite_state in range(states.size() [1]):
            # Compare error between prediction and true
            predicted_state = true_state_sim[:,elite_state,:]
            true_state = states[:,elite_state,:]
            prediction_error = loss_fn(predicted_state, true_state)
            errors.append(prediction_error.item())

        # Filter elite samples
        errors = torch.Tensor(errors)
        num_filter = self.top_K // 2
        _, topk_idx = torch.topk(errors, k=num_filter, largest=False)

        # Filter out actions
        return actions[:, topk_idx, :]

    def copy_sim_env(self, s_init, actions_so_far):
        """
        Execute actions so far to get sim and env for that iteration
        """
        sim = CassieSimulator(s_init, [])
        env = None
        for action in actions_so_far:
            action_clone = action.clone().cpu().squeeze()
            _, _, env, _, _, _ = sim.env_step(action_clone, env,
                                              visualize=False,
                                              initial_step=env is None)
        return sim, env

    def get_true_reward(self, sim, env, elite_action, last_state):
        """
        Given all the actions so far, compute reward for the next action
        """
        # Given action a first set the last state
        sim.set_state_wrapper(env, last_state)

        _, reward, _, _, _, _ = sim.env_step(elite_action.cpu().squeeze(),
                                             env, visualize=False,
                                             initial_step=env is None,
                                             freeze_phase=True)

        return reward

    def reset_time_state(self, sim, env, last_state, init_clock):
        """
        Reset the state of the simulator and phase
        """
        init_time, init_phase = init_clock
        sim.set_time_phase(env, init_time, init_phase)
        sim.set_state_wrapper(env, last_state)
        
    def get_numerical_gradient(self, sim, env, actions, last_state, init_clock):
        """
        Compute numerical gradient using finite differences for actions
        based on actions so far. This is done with the help of Mujoco simulator
        """
        best_init_numerical_gradient = []
        init_time, init_phase = init_clock

        for elite_k in range(actions.size() [1]):
            # for kth elite, get numerical gradient
            action_numerical_gradient_k = []
            new_last_state = None

            # Reset timer and phase
            sim.set_time_phase(env, init_time, init_phase)
            for time in range(actions.size() [0]):
                action_time = actions[time, elite_k, :]
                # For each timestep, populate numerical gradient
                action_numerical_gradient_time = []
                reset_state = last_state if time == 0 else new_last_state

                for action_component in range(actions.size() [2]):
                    action = action_time.clone()

                    # Perturbation
                    epsilon = 1e-11

                    # Perturb
                    action_copy = action.clone()
                    action_copy[action_component] = action[action_component] + epsilon

                    # Get reward from simulator
                    reward_p = -self.get_true_reward(sim, env, action_copy,
                                                     reset_state)

                    # Perturb again
                    action_copy = action.clone()
                    action_copy[action_component] = action[action_component] - epsilon

                    # Get reward from simulator
                    reward_n = -self.get_true_reward(sim, env, action_copy,
                                                     reset_state)

                    # Compute numerical gradient
                    finite_differences = (reward_p - reward_n) / (2 * epsilon)
                    action_numerical_gradient_time.append(finite_differences.item())

                action_numerical_gradient_k.append(torch.Tensor(action_numerical_gradient_time).view(action.size()))

                # Step with action_time
                sim.env_step(action_time.cpu().squeeze(), env, visualize=False,
                             initial_step=env is None, freeze_phase=False)

                new_last_state = sim.get_state_wrapper(env)

            # Stack all samples
            best_init_numerical_gradient.append(torch.stack(action_numerical_gradient_k, dim=0))

        gradient = torch.stack(best_init_numerical_gradient, dim=1)
        assert gradient.size() == actions.size()

        # Reset sim and phase
        self.reset_time_state(sim, env, last_state, init_clock)
        return gradient.cuda().double()

    def forward(self, batch_size, phase, state_action_buffer, next_mean,
                return_plan=False, verbose=False, lstm_action=None,
                actions_so_far=None, s_init=None):
        """
        Determine optimal action for next timestep
        """
        # Get simulator level states
        sim, env = self.copy_sim_env(s_init, actions_so_far)
        last_state = sim.get_state_wrapper(env)
        init_clock = sim.get_time_phase(env)

        a = 0.9

        # Here batch is strictly if multiple Plans should be performed!
        B = batch_size

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, 0.01)
        self.mean = next_mean.view(next_mean.size() [0], 1, 1, next_mean.size() [1])

        if torch.cuda.is_available():
            self.mean = self.mean.cuda()

        # TODO: Update this
        self.std = torch.ones(self.H, B, 1, self.a_size, device=self.device) * 1e-3

        # Sample actions (T x (B*K) x A)
        actions = (self.mean + self.std * torch.randn(self.H, B, self.K, self.a_size, device=self.device)).view(self.H, B * self.K, self.a_size)
        actions = torch.clamp(actions, min=-1.5, max=1.5)

        for i in range(self.opt_iters):
            if verbose:
                print("CEM Iteration Number", i)
                print()

            # Reset clock
            self.reset_time_state(sim, env, last_state, init_clock)

            rollout_state_action_buffer = state_action_buffer

            # Returns (B*K)actions, init_state, phase
            org_returns, org_states = self.env.rollout(actions,
                                                       rollout_state_action_buffer,
                                                       phase, no_gradient=True,
                                                       return_state=True)

            # Sort these best actions
            _, topk = org_returns.reshape(B, self.K).topk(self.top_K, dim=1,
                                                          largest=True,
                                                          sorted=False)

            topk += self.K * torch.arange(0, B, dtype=torch.int64,
                                          device=topk.device).unsqueeze(dim=1)

            # Best actions + optimizer
            best_init = actions[:, topk.view(-1)].reshape(self.H, self.top_K, self.a_size)
            best_states = org_states[:, topk.view(-1)].reshape(self.H, self.top_K, self.s_size)

            # Filter out samples
            best_init = self.filter_elite_samples(best_init, best_states, sim,
                                                  env, last_state, init_clock)
            numerical_grad = self.get_numerical_gradient(sim, env, best_init,
                                                         last_state, init_clock)
            best_actions = []
            best_init = best_init.requires_grad_()
            optimizer = optim.Adam([best_init], lr=0.1)

            for k in range(self.gradient_iters):
                optimizer.zero_grad()
                self.env.transformer.zero_grad()
                self.env.representation.zero_grad()

                rollout_state_action_buffer = rollout_state_action_buffer.detach()
                returns = self.env.rollout(best_init,
                                           rollout_state_action_buffer,
                                           phase, no_gradient=False)

                mean_returns = torch.sum(returns)
                if verbose:
                    print(f"Individual Returns after gradient update {k}. Mean returns: {mean_returns}")
                (-mean_returns).backward()
                backprop_grad = best_init.grad.data

                # Clip gradients
                if self.grad_clip:
                    nn.utils.clip_grad_value_(best_init, clip_value=0.125) 

                # Linear combination of simulator grad
                backprop_grad = best_init.grad.data
                best_init.grad = a * numerical_grad + (1-a) * backprop_grad

                # gradient norm
                optimizer.step()

            # Clamp all the best actions
            best_actions = torch.clamp(best_init, min=-1.5, max=1.5)

            # Zero out the gradients
            optimizer.zero_grad()
            self.mean = best_actions.mean(dim=1, keepdim=True)
            self.std = best_actions.std(dim=1, unbiased=False, keepdim=True)

            self.mean = self.mean.cpu().detach()
            self.std = self.std.cpu().detach()

            # Convert to cuda
            if torch.cuda.is_available():
                self.mean = self.mean.cuda()
                self.std = self.std.cuda()

            k_resamp = self.K#-self.top_K
            _, botn_k = org_returns.reshape(B, self.K).topk(k_resamp, dim=1, largest=False, sorted=False)
            botn_k += self.K * torch.arange(0, B, dtype=torch.int64, device=self.device).unsqueeze(dim=1)

            # Resample new actions
            resample_actions = (self.mean + self.std * torch.randn(self.H, k_resamp, self.a_size, device=self.device)).view(self.H, B * k_resamp, self.a_size)
            actions.data[:, botn_k.view(-1)] = resample_actions.data

        best_actions = best_actions.cpu().detach()
        rollout_state_action_buffer = state_action_buffer
        returns = self.env.rollout(best_actions, rollout_state_action_buffer,
                                   phase, no_gradient=True)

        # Re-fit belief to the K best action sequences
        _, topk = returns.reshape(B, returns.size() [0]).topk(1, dim=1, largest=True, sorted=False)
        best_plan = best_actions[:, topk[0]].reshape(self.H, B, self.a_size)

        if return_plan:
            return best_plan
        else:
            return best_plan[0]

    def retrain(self, states, actions, reset_iteration):
        """
        Retrain the model if below a certain threshold
        """
        transformer, representation = self.env.retrain(states, actions, reset_iteration)
        return transformer, representation