from geometry_generator import generate_gmsh_mesh
from solver import unified_solving_function
from visualisation import visualise_3D, append_feature

import sys
import os
import psutil

from mpi4py import MPI
import slepc4py

slepc4py.init(sys.argv)

import pyvista
import numpy as np

pyvista.start_xvfb()

import gmsh

gmsh.initialize()

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# For hyper parameter tuning
import optuna

import gymnasium as gym
from gymnasium import spaces

# Define geometry parameters
L, H, B = 12e-3, 0.2e-3, 3e-3

Bmin = 1e-3
Bmax = B
grid_size = 0.5e-3

# Select boundary conditions to apply
bc_z = True
bc_y = True
no_eigenvalues = 20
target_frequency = 120e3

# Set a folder to save the features
folder = "./me-shape-optimization/results/ppo/"
description = "PPO optimization using deep reinforcement learning."

os.makedirs(folder, exist_ok=True)

# Generate a file that contains the meta data in the header
line01 = "Geometry: L = {0} mm, Bmin = {2} mm, Bmax = {1} mm, H = {2} mm, grid size = {3} mm, accuracy = {4} mm".format(
    L * 1e3, Bmax * 1e3, Bmin * 1e3, H * 1e3, grid_size * 1e3
)
line02 = description
line03 = "### Simulation Results ###"
line04 = "Resonance Frequency (Hz)\t Features (m)\n"

header_lines = [line01, line02, line03, line04]

with open(folder + "features.csv", "a+") as csvfile:
    csvfile.write("\n".join(header_lines))


## Define eigensolver
eigensolver = slepc4py.SLEPc.EPS().create(MPI.COMM_WORLD)

# Set problem
eigensolver.setProblemType(slepc4py.SLEPc.EPS.ProblemType.GHEP)
# Shift and invert mode
st = eigensolver.getST()
st.setType(slepc4py.SLEPc.ST.Type.SINVERT)
# target real eigenvalues
eigensolver.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.TARGET_REAL)
# Set the target frequency
eigensolver.setTarget(target_frequency**2 * 2 * np.pi)
# Set no of eigenvalues to compute
eigensolver.setDimensions(nev=no_eigenvalues)

## Plotting
# Now define pyvista plotter
plotter = pyvista.Plotter(off_screen=True)

## Geometry generation
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Box")
model.setCurrent("Box")


class MEResonanceEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(MEResonanceEnv, self).__init__()
        # There is the discrete actions of increasing or decreasing each weight
        self.accuracy = 0.5e-3

        # Action 0-23 is increase the according weight
        # action 24-47 is decrease the according weight
        self.no_discrete_actions = int(L / grid_size * 2)

        # Define counter to stop the episode
        self.episode_length = 100
        self.counter = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = gym.spaces.Discrete(self.no_discrete_actions)

        # Define the boundaries and the size of the observation space
        # The observation space is the current geometry and the resonance frequency
        # The low and high values cannot be used to restrict the action space! I
        # do not konw why they exist at all.
        self.observation_space = gym.spaces.Box(
            low=np.append(np.repeat(Bmin, int(L / grid_size)), 0.5 * target_frequency),
            high=np.append(np.repeat(Bmax, int(L / grid_size)), 2 * target_frequency),
            shape=(int(L / grid_size) + 1,),
            dtype=np.float64,
        )

        # Init shape
        self.horizontal_lengths = np.repeat(Bmax, int(L / grid_size))
        self.gmsh_mesh = generate_gmsh_mesh(model, L, H, B, self.horizontal_lengths)

    def step(self, action):
        # No info for now
        info = {}
        # Step to next observation given an action
        adjustment_helper = np.zeros(int(L / grid_size))
        if action < int(L / grid_size):
            adjustment_helper[action] = self.accuracy
        else:
            adjustment_helper[action % int(L / grid_size)] = -self.accuracy

        self.horizontal_lengths = np.clip(
            self.horizontal_lengths + adjustment_helper, Bmin, Bmax
        )

        """
        self.horizontal_lengths = self.horizontal_lengths + adjustment_helper
        self.horizontal_lengths[self.horizontal_lengths < Bmin] = Bmin
        self.horizontal_lengths[self.horizontal_lengths > Bmax] = Bmax
        """

        # Generate geometry
        self.gmsh_mesh = generate_gmsh_mesh(model, L, H, B, self.horizontal_lengths)

        # Calculate resonance frequency
        (
            self.V,
            self.eigenvalues,
            self.eigenmodes,
            self.first_longitudinal_mode,
        ) = unified_solving_function(
            eigensolver,
            self.gmsh_mesh,
            L,
            H,
            B,
            bc_z,
            bc_y,
            no_eigenvalues,
            target_frequency,
        )

        # after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        # print("Solving function: " + str(after - before))
        # before = after

        # # Plot first longitudinal mode only
        self.eigenfrequency = (
            np.sqrt(self.eigenvalues[self.first_longitudinal_mode].real) / 2 / np.pi
        )

        # Put together observation
        observation = np.append(self.horizontal_lengths, self.eigenfrequency)

        # Define reward function (this is the tricky part)
        # Simple reward function that gives rewards to reducing the resonance
        # frequency. However, this may result in many small steps that do not
        # lead quick to the best solution but rather slowly.
        """ if (self.previous_resonance_frequency - self.eigenfrequency) > 0:
            self.reward += 1
        elif (self.previous_resonance_frequency - self.eigenfrequency) < 0:
            self.reward -= 1
        """

        # Reward function that gives rewards to reducing the resonance frequency
        # depending on how much it was reduced. The reward is simply the
        # reduction in resonance frequency. This obviously results in much
        # higher rewards than the above function.
        self.reward += int((self.previous_resonance_frequency - self.eigenfrequency))

        """
        # Restrict the action space
        if (
            self.horizontal_lengths[action % int(L / grid_size)] < Bmin
            or self.horizontal_lengths[action % int(L / grid_size)] > Bmax
        ):
            self.reward -= 10
        """

        # Now for the next step set the previous resonance frequency to the current one
        self.previous_resonance_frequency = self.eigenfrequency

        # Define the stop condition
        self.counter += 1

        # Render the environment to save a picture of each step
        # self.render()

        if self.counter >= self.episode_length:
            self.done = True

        terminated = self.done
        truncated = False

        return observation, self.reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset to given observation
        # Undo environment
        self.done = False
        self.reward = 0
        self.counter = 0

        self.horizontal_lengths = np.repeat(Bmax, int(L / grid_size))
        self.gmsh_mesh = generate_gmsh_mesh(model, L, H, B, self.horizontal_lengths)

        # Calculate resonance frequency
        (
            self.V,
            self.eigenvalues,
            self.eigenmodes,
            self.first_longitudinal_mode,
        ) = unified_solving_function(
            eigensolver,
            self.gmsh_mesh,
            L,
            H,
            B,
            bc_z,
            bc_y,
            no_eigenvalues,
            target_frequency,
        )

        # after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        # print("Solving function: " + str(after - before))
        # before = after

        # # Plot first longitudinal mode only
        self.eigenfrequency = (
            np.sqrt(self.eigenvalues[self.first_longitudinal_mode].real) / 2 / np.pi
        )

        self.previous_resonance_frequency = self.eigenfrequency

        observation = np.append(self.horizontal_lengths, self.eigenfrequency)

        # No info for now
        info = {}

        return observation, info  # reward, done, info can't be included

    def render(self, mode="human"):
        # Plot to render for humans
        visualise_3D(
            plotter,
            self.V,
            self.eigenvalues,
            self.eigenmodes,
            self.first_longitudinal_mode,
            saving_path=folder + "{i:.2f}.png".format(i=self.eigenfrequency),
            viewup=True,
        )
        append_feature(
            self.eigenfrequency,
            folder + "features.csv",
            self.horizontal_lengths,
        )

    def close(self):
        # Close environment
        a = 0
