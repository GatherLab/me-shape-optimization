import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio

import os

from shape_generation import Geometry
import core_functions as cf

# --------------------
# Parameter Definitions (currently not automatically read from file, although all information is stored there so it has to be manually put in)
# --------------------

# Define material parameters
E, nu = 5.4e10, 0.34
rho = 7950.0

# Define maximum length, width and height
Lmax, Bmax, Hmax = 12e-3, 3e-3, 0.2e-3
# Define minimum width
Bmin = 1e-3
# Define initial width
Binit = 3e-3
# Defines the number of adjustable parts
grid_size = 0.5e-3
# Accuracy of the adjustment (defined by the real world, e.g. less than 10 um
# accuracy doesn't make sense)
accuracy = 10e-6

# Learning rate, step size to determine gradient and maximum number of
# optimization steps
learning_rate = 2 * 1e-10
grad_step = 100e-6
no_optimization_steps = 50

# File to read data from
folder_path = "./simulated_annealing3/"
file = "features.csv"


# --------------------
# Read in data
# --------------------
# Read in features from file
number_features = 24
resonances_and_features = cf.read_features(folder_path + file, number_features)

# Select a feature and generate geometry from it
selected_feature = -1

geometry = Geometry(Lmax, Bmax, Hmax, grid_size, accuracy, Bmin)
geometry.adjust_horizontal_length(
    resonances_and_features.iloc[selected_feature].to_numpy()[1:]
)
geometry.generate_mesh()
cf.plot_shape(
    geometry.mesh,
    resonances_and_features.iloc[selected_feature]["eigenfrequency"],
    folder_path,
    Lmax,
    save=False,
)

# --------------------
# Generate gif
# --------------------

cf.generate_gif(folder_path, resonances_and_features["eigenfrequency"].to_numpy())

# Init geometry
geometry.init_rectangular_geometry(Binit)

filenames = []
for index, row in resonances_and_features.iterrows():
    # Adjust geometry
    horizontal_lengths = row.to_numpy()[1:]
    geometry.adjust_horizontal_length(horizontal_lengths)
    geometry.generate_mesh()


# --------------------
# Initialise Geometry
# --------------------


# --------------------
# Do simulations and Save Stress Data
# --------------------
for row in resonances_and_features.rows:
    # Generate geometry from saved features

    initial_eigenfrequency = cf.do_simulation(geometry.mesh, E, nu, rho)
    #! Find a way of nicely saving the MA data from the simulation (maybe just adding another variable?) !
