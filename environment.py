import gym
from gym import spaces

import matplotlib as mpl

# Allow displaying stuff
mpl.use("TkAgg")

import matplotlib.pyplot as plt

import numpy as np
import time
import cv2

from shape_generation import Geometry
import core_functions as cf

# Define material parameters
E, nu = 5.4e10, 0.34
rho = 7950.0

# Define maximum length, width and height
Lmax, Bmax, Hmax = 12e-3, 3e-3, 0.2e-3
# Define minimum width
Bmin = 1e-3
# Define initial width
Binit = 1.5e-3
# Defines the number of adjustable parts
grid_size = 2e-3
# Accuracy of the adjustment (defined by the real world, e.g. less than 10 um
# accuracy doesn't make sense)
accuracy = 1000e-6

# Learning rate, step size to determine gradient and maximum number of
# optimization steps
# learning_rate = 2 * 1e-10
# grad_step = 100e-6
no_optimization_steps = 200

# Target frequency and number of eigenstates to compute (for solver)
target_frequency = 100000
no_eigenstates = 10


class MEResonanceEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(MEResonanceEnv, self).__init__()
        # There is the discrete actions of increasing or decreasing each weight

        # Action 0-5 is increase the according weight
        # action 6-11 is decrease the according weight
        no_discrete_actions = 6 * 2
        self.counter = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(no_discrete_actions)

        # Define the boundaries and the size of the observation space
        self.observation_space = spaces.Box(
            low=0, high=1e6, shape=(6 + 1,), dtype=np.float64
        )

        # Init shape
        self.geometry = Geometry(Lmax, Bmax, Hmax, grid_size, accuracy, Bmin)

    def step(self, action):
        # Step to next observation given an action
        if action < 6:
            adjustment_helper = np.zeros(6)
            adjustment_helper[action] = accuracy
            self.geometry.adjust_horizontal_length(
                self.geometry.horizontal_lengths + adjustment_helper
            )
        else:
            adjustment_helper = np.zeros(6)
            adjustment_helper[action % 6] = -accuracy
            self.geometry.adjust_horizontal_length(
                self.geometry.horizontal_lengths + adjustment_helper
            )

        # Generate geometry
        self.geometry.generate_mesh()

        # Calculate resonance frequency
        resonance_frequency, x, y, self.magnitude = cf.do_simulation(
            self.geometry.mesh, E, nu, rho, target_frequency, no_eigenstates
        )

        # Put together observation
        observation = np.append(self.geometry.horizontal_lengths, resonance_frequency)

        # Define reward
        if (self.previous_resonance_frequency - resonance_frequency) > 0:
            self.reward += 1
        elif (self.previous_resonance_frequency - resonance_frequency) < 0:
            self.reward -= 1

        # No info for now
        info = {}

        # Now for the next step set the previous resonance frequency to the current one
        self.previous_resonance_frequency = resonance_frequency

        # Define the stop condition
        self.counter += 1

        if self.counter >= 5:
            self.done = True

        return observation, self.reward, self.done, info

    def reset(self):
        # Reset to given observation
        # Undo environment
        self.done = False
        self.reward = 0
        self.counter = 0

        self.geometry.init_rectangular_geometry(Binit)

        self.geometry.generate_mesh()
        resonance_frequency, x, y, magnitude = cf.do_simulation(
            self.geometry.mesh, E, nu, rho, target_frequency, no_eigenstates
        )
        self.previous_resonance_frequency = resonance_frequency

        observation = np.append(self.geometry.horizontal_lengths, resonance_frequency)

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        # Plot to render for humans
        cf.plot_shape(
            self.geometry.mesh,
            self.previous_resonance_frequency,
            "",
            Lmax,
            self.magnitude,
            save=False,
        )
        # time.sleep(0.5)
        # plt.close()

    def close(self):
        # Close environment
        a = 0
