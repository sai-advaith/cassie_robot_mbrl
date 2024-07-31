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

# Simulator imports
from simulator.evaluate_expert import CassieSimulator
from simulator.cassie.cassiemujoco import CassieSim, CassieVis
from simulator.cassie.trajectory import CassieTrajectory

class Reward(object):
    """
    Reward function to walk along a straight line
    """
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

    def get_multiphase_reference(self, phase):
        """
        Get reference position for multiple walking phases
        """
        reference_phase = []
        phase = phase.detach().cpu().numpy()
        for p in phase:
            ref_pos, _ = self.get_ref_state(int(p))
            reference_phase.append(torch.Tensor(ref_pos))
        return torch.stack(reference_phase).unsqueeze(dim = 1).cuda()

    def compute_multiphase_reward(self, phase, cassie_state):
        """
        Reward function for multiple phases and batch of states
        """
        ref_pos = self.get_multiphase_reference(phase)

        joint_error       = 0
        com_error         = 0
        orientation_error = 0
        spring_error      = 0

        # each joint pos
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
        qpos_joints = cassie_state[:, :, 5:15]
        for i, j in enumerate(self.pos_idx):
            target = ref_pos[:, 0, j]
            actual = qpos_joints[:, 0, i]
            error = torch.sub(actual, target)
            joint_error = joint_error + torch.mul((error ** 2),  30 * weight[i])

        # Q velocity
        qvel = cassie_state[:, 0, 15:18]
        forward_diff = torch.abs(torch.sub(qvel[:, 0], self.speed))
        forward_diff[forward_diff < 0.05] = 0

        # Y direcqtion velocity
        y_vel = torch.abs(qvel[:, 1])
        y_vel[y_vel < 0.03] = 0

        # Make sure cassie orientations are aligned
        actual_q = cassie_state[:, 0, 1:5].double()
        target_q = torch.Tensor([1., 0., 0., 0.]).double()

        # Convert to cuda
        if torch.cuda.is_available():
            actual_q = actual_q.cuda()
            target_q = target_q.cuda()

        # Absolute value
        norm_actual = torch.linalg.norm(actual_q, dim = 1)
        orientation_error = 5 * (1 - (torch.inner(actual_q, target_q) / norm_actual) ** 2)

        straight_diff = torch.abs(cassie_state[:, 0, 0] - 1.0)
        straight_diff[straight_diff < 0.03] = 0

        # left and right shin springs positions
        j = 0
        qpos_shin = [cassie_state[:, 0, 34], cassie_state[:, 0, 37]]
        spring_error = 0
        for i in [15, 29]:
            target = ref_pos[:, 0, i]
            actual = qpos_shin[j]
            spring_error = spring_error + 1000 * (target - actual) ** 2
            j += 1

        # Accumulate reward based on LSTM policy
        reward = torch.tensor(0.000).cuda()                                 + \
                 torch.tensor(0.300).cuda() * torch.exp(-orientation_error) + \
                 torch.tensor(0.200).cuda() * torch.exp(-joint_error) +       \
                 torch.tensor(0.200).cuda() * torch.exp(-forward_diff) +      \
                 torch.tensor(0.200).cuda() * torch.exp(-y_vel) +             \
                 torch.tensor(0.050).cuda() * torch.exp(-straight_diff) +     \
                 torch.tensor(0.050).cuda() * torch.exp(-spring_error)

        return reward

    def compute_reward(self, phase, cassie_state):
        """
        Reward function at specific state and walking phase for cassie
        """
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
            joint_error = joint_error + torch.mul((error ** 2),  30 * weight[i])

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
        # print(f"manual {actual_q}")
        # actual_q = qpos[3:7]
        target_q = torch.Tensor([1., 0., 0., 0.]).double()
        # Convert to cuda
        if torch.cuda.is_available():
            actual_q = actual_q.cuda()
            target_q = target_q.cuda()

        norm_actual = torch.linalg.norm(actual_q, dim = 1)
        orientation_error = 5 * (1 - (torch.inner(actual_q, target_q) / norm_actual) ** 2)

        straight_diff = torch.abs(cassie_state[:, 0, 0] - 1.0)
        straight_diff[straight_diff < 0.03] = 0

        # left and right shin springs positions
        j = 0
        qpos_shin = [cassie_state[:, 0, 34], cassie_state[:, 0, 37]]
        spring_error = 0
        for i in [15, 29]:
            target = ref_pos[i]
            actual = qpos_shin[j]

            spring_error = spring_error + 1000 * (target - actual) ** 2
            j += 1

        # Accumulate reward based on LSTM policy
        reward = torch.tensor(0.000).cuda()                                 + \
                 torch.tensor(0.300).cuda() * torch.exp(-orientation_error) + \
                 torch.tensor(0.200).cuda() * torch.exp(-joint_error) +       \
                 torch.tensor(0.200).cuda() * torch.exp(-forward_diff) +      \
                 torch.tensor(0.200).cuda() * torch.exp(-y_vel) +             \
                 torch.tensor(0.050).cuda() * torch.exp(-straight_diff) +     \
                 torch.tensor(0.050).cuda() * torch.exp(-spring_error)


        return reward
