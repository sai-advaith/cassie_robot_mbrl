import numpy as np
from simulator.cassie.trajectory import CassieTrajectory
import os
from math import floor
import torch
traj_path = "../../simulator/cassie/trajectory/more-poses-trial.bin"
trajectory = CassieTrajectory(traj_path)
phase = 0

simrate = 60 # simulate X mujoco steps with same pd target
                    # 60 brings simulation from 2000Hz to roughly 30Hz

# NOTE: a reference trajectory represents ONE phase cycle
phaselen = floor(len(trajectory) / simrate) - 1
phase_add = 1
speed = 1
counter = 0
def get_ref_state(phase, phaselen, phase_add, trajectory, speed, simrate, counter):

    if phase > phaselen:
        phase = 0

    pos = np.copy(trajectory.qpos[phase * simrate])

    ###### Setting variable speed  #########
    pos[0] *= speed
    pos[0] += (trajectory.qpos[-1, 0] - trajectory.qpos[0, 0]) * counter * speed
    ######                          ########

    # setting lateral distance target to 0
    # regardless of reference trajectory
    pos[1] = 0

    vel = np.copy(trajectory.qvel[phase * simrate])
    vel[0] *= speed

    return pos, vel

states = []
for i in range(100):
    ref_pos, ref_vel = get_ref_state(phase, phaselen, phase_add, trajectory, speed, simrate, counter)
    # Complete state
    pelvis_orientation = ref_pos[3:7]
    motor_position = ref_pos[[7, 8, 9, 14, 20, 21, 22, 23, 28, 34]]

    joint_position = ref_pos[[15, 16, 20, 29, 30, 34]]
    pelvis_velocity = ref_vel[0:6]
    motor_velocity = ref_vel[[6, 7, 8, 12, 18, 19, 20, 21, 25, 31]]

    joint_velocity = ref_vel[[13, 14, 18, 26, 27, 31]]
    pelvis_acceleration = np.zeros(3)

    clock =[np.sin(2 * np.pi *  phase / phaselen), np.cos(2 * np.pi *  phase / phaselen)]
    ext_state = np.concatenate((clock, [speed]))

    height = [ref_pos[2] - 0.001]
    cassie_state = np.concatenate([height, pelvis_orientation, motor_position, pelvis_velocity,
                                motor_velocity, pelvis_acceleration, joint_position,
                                joint_velocity, ext_state])
    states.append(cassie_state)
    phase += phase_add
    if phase > phaselen:
        phase = 0
        counter += 1
torch.save(states, 'states.pt')
# def get_full_state(phase, phase_add):
#     qpos = np.copy(self.sim.qpos())
#     qvel = np.copy(self.sim.qvel()) 
#     ref_pos, _ = self.get_ref_state(phase + phase_add)

#     clock = [np.sin(2 * np.pi *  phase / phaselen),
#             np.cos(2 * np.pi *  phase / phaselen)]
    
#     ext_state = np.concatenate((clock, [self.speed]))
#     # print("Terrain height", self.cassie_state.terrain.height)
#     # Use state estimator [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
    #   robot_state = np.concatenate([
    #       [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height (0)
    #       self.cassie_state.pelvis.orientation[:],                                   # pelvis orientation (1:5) 
    #       self.cassie_state.motor.position[:],                                       # actuated joint positions (5:15)
    #       self.cassie_state.pelvis.translationalVelocity[:],                         # pelvis translational velocity (15:18)
    #       self.cassie_state.pelvis.rotationalVelocity[:],                            # pelvis rotational velocity (18:21)
    #       self.cassie_state.motor.velocity[:],                                       # actuated joint velocities (21:31)
    #       self.cassie_state.pelvis.translationalAcceleration[:],                     # pelvis translational acceleration (31:34)
    #       self.cassie_state.joint.position[:],                                       # unactuated joint positions (34:40)
    #       self.cassie_state.joint.velocity[:]                                        # unactuated joint velocities (40:46)
    #   ])
#     return np.concatenate([robot_state, ext_state])

