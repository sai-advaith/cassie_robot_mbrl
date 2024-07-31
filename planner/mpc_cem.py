#MPC Oracle - CEM modified to support decaying number of samples and shifted mean for warm-stary
#https://github.com/LemonPi/pytorch_cem/blob/master/pytorch_cem/cem.py

# Python Imports
import numpy as np
import time
import logging

# PyTorch Imports
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)


class CEM():
    """
    Cross Entropy Method control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    """

    def __init__(self, dynamics, running_cost, nx, nu, models, init_state_action_pairs, init_mean, retrain_model, num_samples=200, num_iterations=5, num_elite=10, horizon=10,
                 device="cpu",
                 terminal_state_cost=None,
                 u_min=None,
                 u_max=None,
                 choose_best=True,
                 init_cov_diag=0.25, momentum = 0.05):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K x 1) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        """
        # Device
        self.d = device

        # TODO determine dtype
        self.dtype = torch.double

        # N_SAMPLES
        self.K = num_samples

        # TIMESTEPS
        self.T = horizon
        self.M = num_iterations
        self.num_elite = num_elite
        self.choose_best = choose_best

        # Trained models
        self.representation, self.transformer = models
        self.init_state_action_pairs = init_state_action_pairs

        # Retrain function
        self.retrain_model = retrain_model
        # Per rollout
        self.per_rollout_cache = init_state_action_pairs

        # Bias the model
        self.init_mean = init_mean

        # dimensions of state and control
        self.nx = nx
        self.nu = nu

        # Mean and covariance for the CE based planner
        self.mean = None
        self.cov = None

        # Dynamics method and cost function
        self.F = dynamics

        # look at the effect of this
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost

        cov_diag = torch.Tensor([5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5])
        self.init_cov_diag = cov_diag
        cov_increment = 1e-5
        for i in range(1, self.T):
            next_step = cov_diag + cov_increment
            self.init_cov_diag = torch.cat((self.init_cov_diag, next_step), 0)
            cov_diag = next_step
        # self.init_cov_diag = self.init_cov_diag.repeat(self.T)

        self.momentum = momentum

        # Clamp variables
        self.u_min = u_min
        self.u_max = u_max

        # regularize covariance
        self.cov_reg = torch.eye(self.T * self.nu, device=self.d, dtype=self.dtype) * init_cov_diag * 1e-7 

    def shift_means(self, next_step_mean):
        """
        Shift the previous solution by 1(Default) as the mean for the next planning
        """
        # Shift mean by one position back
        # self.mean = new_mean
        previous_means = self.mean[self.nu:]
        updated_mean = torch.cat((previous_means, next_step_mean), 0)
        pre = self.mean
        self.mean = updated_mean

        # self.cov = torch.zeros(self.T * self.nu, device=self.d, dtype=self.dtype) * self.init_cov_diag
        # self.cov = torch.eye(self.T * self.nu, device=self.d, dtype=self.dtype) * 0.01
        # self.cov = torch.diag(self.init_cov_diag)
        # self.cov = self.cov.double()
        self.cov[-self.nu:, -self.nu:] = torch.diag(self.init_cov_diag[-self.nu:])
        self.cov = self.cov.double()

    def reset(self, reset_mean):
        """
        Clear controller state after finishing a trial

        Reset should happen in a new epoch
        """
        # action distribution, initialized as N(0,I)
        # we do Hp x 1 instead of H x p because covariance will be Hp x Hp matrix instead of some higher dim tensor
        # self.mean = torch.zeros(self.T * self.nu, device=self.d, dtype=self.dtype)
        self.mean = reset_mean
        # self.cov = torch.eye(self.T * self.nu, device=self.d, dtype=self.dtype) * self.init_cov_diag
        # self.cov = torch.eye(self.T * self.nu, device=self.d, dtype=self.dtype) * 0.01
        self.cov = torch.diag(self.init_cov_diag)
        self.cov = self.cov.double()

    def _bound_samples(self, samples):
        """
        Utility method
        """
        # Clip the samples to deal with samples more than a bound
        return torch.clamp(samples, self.u_min, self.u_max)

    def _slice_control(self, t):
        """
        Utility method
        """
        # Slice for the timestamp
        return slice(t * self.nu, (t + 1) * self.nu)

    def _evaluate_trajectories(self, samples, init_state, phase):
        """
        Rollout the trajectories and get cost of each of them over a finite horizon
        """
        # Store total cost
        reward_total = torch.zeros(self.K, device=self.d, dtype=self.dtype)
        avg_reward = torch.zeros(self.K, device = self.d, dtype=self.dtype)

        # Initialize states
        state = init_state.view(1, -1).repeat(self.K, 1)
        # Set the rollout cache to be the initial state
        self.per_rollout_cache = init_state

        # Discount
        gamma = 0.95

        # Loop over all samples
        rollout_phase = phase + 1
        for t in range(self.T):
            if rollout_phase > self.running_cost.phaselen:
                rollout_phase = 0

            # T = 0 to 10 and make compatible for the state-action pair
            u = samples[:, self._slice_control(t)]
            u = torch.unsqueeze(u, 1)
            # Get the next state and the cache with previous 10 state action pairs
            state, self.per_rollout_cache = self.F(self.representation, self.transformer, self.per_rollout_cache, u)
            # Get reward wrt reference trajectory
            reward = self.running_cost.compute_reward(rollout_phase, state)
            reward *= gamma ** t
            reward_total += reward

            # Increase phase
            rollout_phase += 1
        avg_reward = reward_total/self.T
        # Reset after one rollout
        self.per_rollout_cache = init_state

        # Print stats after all samples rolled out
        print(f'Minimum total reward {torch.min(reward_total)} and Maximum total reward {torch.max(reward_total)}')
        print(f'Minimum average reward {torch.min(avg_reward)} and Maximum average reward {torch.max(avg_reward)}')
        print()

        return reward_total

    def _sample_top_trajectories(self, state, num_elite, phase):
        """
        Get only the important trajectories
        """
        # sample K action trajectories
        # in case it's singular
        self.cov = self.cov.cuda()
        self.mean = self.mean.cuda()

        self.action_distribution = MultivariateNormal(self.mean, covariance_matrix=self.cov)
        samples = self.action_distribution.sample((self.K,))

        # bound to control maximums
        samples = self._bound_samples(samples)

        # Get reward of each sample trajectory
        reward_total = self._evaluate_trajectories(samples, state, phase)

        # select top k based on score
        top_rewards, topk = torch.topk(reward_total, num_elite, sorted=True)
        top_samples = samples[topk]

        return top_samples

    def command(self, state, epoch, num_samples, decay, phase, next_mean, reset, choose_best=True):
        """
        Give action for next timestamp the the agent
        """

        self.K = num_samples
        # Convert to Tensor
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        state = state.to(dtype=self.dtype, device=self.d)

        if reset:
            self.reset(next_mean)
        else:
            shift_value = next_mean[-self.nu:]
            if torch.cuda.is_available():
                shift_value = shift_value.cuda()
            self.shift_means(shift_value)

        # Run over M times to make sure CEM converges
        for m in range(self.M):
            self.K = torch.maximum(torch.tensor(2*self.num_elite), torch.tensor(num_samples*(decay**m))).to(torch.int)

            # Get top samples
            top_samples = self._sample_top_trajectories(state, self.num_elite, phase)
  
            # fit the gaussian to those top samples
            # self.mean = torch.mean(top_samples, dim = 0)
            self.mean = (1-self.momentum)*torch.mean(top_samples, dim=0) + self.momentum * self.mean
            # new_std = torch.std(top_samples, unbiased = False, dim = 0)
            new_std = (1-self.momentum)*torch.std(top_samples,unbiased=False, dim=0) + self.momentum * torch.sqrt(torch.diag(self.cov))
            self.cov = torch.diag(new_std**2)
        # print("mean: ", self.mean)
        # print("standard deviation", new_std)
        if choose_best and self.choose_best:
            # top_sample = self._sample_top_trajectories(state, 1, phase)
            # Get action for next timestamp
            u = top_samples[0, :10]
            u = torch.unsqueeze(u, 0)

        # only apply the first action from this trajectory
        return u
    def retrain(self, states, actions, train_datasets, new_train_datasets):
        self.representation, self.transformer, new_train_dataset = self.retrain_model(self.representation, self.transformer, states, actions, train_datasets, new_train_datasets)
        return self.representation, self.transformer, new_train_dataset