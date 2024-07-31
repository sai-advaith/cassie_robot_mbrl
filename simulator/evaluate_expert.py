import torch
import hashlib
import os
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import time
import copy
from functools import partial

# Cassie Imports
from simulator.cassie.cassie import CassieEnv

# Dynamics import
# from planner.mpc_model_learning import dynamics

class CassieSimulator(object):
    def __init__(self, init_state, init_actions, next_states = None, transformer = None, representation = None):
        self.init_state = init_state
        self.transformer = transformer
        self.representation = representation
        self.init_actions = init_actions
        self.speed = 5.0900e-02

    def get_env(self, dynamics_randomization, verbose=False, **kwargs):
        """
        Returns an *uninstantiated* environment constructor.

        Since environments containing cpointers (e.g. Mujoco envs) can't be serialized,
        this allows us to pass their constructors to Ray remote functions instead

        """

        if verbose:
            print("Created cassie env with arguments:")
            print("\tdynamics randomization: {}".format(dynamics_randomization))
        return partial(CassieEnv, dynamics_randomization=dynamics_randomization)

    def env_step(self, action, env, visualize = True, initial_step = False, freeze_phase = False):
        # Get Cassie environment in mujoco
        if env is None and initial_step:
            env = self.get_env(False)()
            env.dynamics_randomization = False
            env.reset()

            # Set speed
            setattr(env, 'speed', self.speed)

            # Set initial state
            self.init_state = self.init_state.squeeze(0)
            self.set_init_state(env)

        # Run in Sim
        action = action.squeeze(0)
        next_state, reward, done, phase, contact_information = env.step(action.numpy(), freeze_phase)

        # Visualize
        if visualize:
            x = env.render()

        # Next state + reward
        return next_state, reward, env, phase, done, contact_information

    def set_init_state(self, env):
        """
        Initialize the initial state of Cassie
        """
        # Set the terrain height
        terrain_height = env.cassie_state.terrain.height

        # Convert from tensor to numpy
        init_state = self.init_state.cpu().detach().numpy()

        # Set robot states
        env.cassie_state.pelvis.position[2] = init_state[0] + terrain_height
        env.cassie_state.pelvis.orientation[:] = init_state[1:5]
        env.cassie_state.motor.position[:] = init_state[5:15]
        env.cassie_state.pelvis.translationalVelocity[:] = init_state[15:18]
        env.cassie_state.pelvis.rotationalVelocity[:] = init_state[18:21]
        env.cassie_state.motor.velocity[:] = init_state[21:31]
        env.cassie_state.pelvis.translationalAcceleration[:] = init_state[31:34]
        env.cassie_state.joint.position[:] = init_state[34:40]
        env.cassie_state.joint.velocity[:] = init_state[40:46]

        # Return environment
        return env

    def get_state_wrapper(self, env):
        """
        Get state of the robot from simulator
        """
        state = env.sim.get_state()
        return state
    def set_state_wrapper(self, env, state):
        """
        Set state of the robot from simulator
        """
        env.sim.set_state(state)
        return env
    def set_time_phase(self, env, new_time, new_phase):
        """
        Given env, reset timer and phase for that timestep
        """
        env.time = new_time
        env.phase = new_phase
    def get_time_phase(self, env):
        """
        Get time and phase of Cassie in simulator
        """
        return [env.time, env.phase]

    def visualize_sequence(self, actions, next_actions = None, render = True, reset = True):
        # Get Cassie environment in mujoco
        if reset:
            env = self.get_env(False)()
            env.dynamics_randomization = False
            env.reset()

            # Set simulator level speed
            setattr(env, 'speed', self.speed)

            # Set initial state
            self.init_state = self.init_state.squeeze(0)
            self.set_init_state(env)

        for i in range(len(actions)):
            # Run the actions in simulator
            action = actions[i].squeeze(0)
            next_state, reward, done, phase, contact_information = env.step(action.numpy())
            print(f"Step {i}, reward {reward}")
            # if render:
            #     env.render()
        return env
