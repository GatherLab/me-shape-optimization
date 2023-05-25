import pandas as pd
import numpy as np
import os

from shape_generation import Geometry
import core_functions as cf

# --------------------
# Parameter Definitions
# --------------------

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
grid_size = 0.5e-3
# Accuracy of the adjustment (defined by the real world, e.g. less than 10 um
# accuracy doesn't make sense)
accuracy = 10e-6

# Target frequency and number of eigenstates to compute (for solver)
target_frequency = 100000
no_eigenstates = 15

# Folder to save to
folder = "./img/scipy/shgo2/"
description = "SHGO optimization, scipy.optimize.shgo(opt_function, bounds)"

os.makedirs(folder, exist_ok=True)


# --------------------
# Initialise Data Dump
# --------------------

# Generate a file that contains the meta data in the header
line00 = "Material: Youngs modulus (E) = {0} N/m2, Poissons ratio(nu) = {1}, density (rho) = {2} kg/m3".format(
    E, nu, rho
)
line01 = "Geometry: L = {0} mm, Bmin = {2} mm, Bmax = {1} mm, H = {2} mm, grid size = {3} mm, accuracy = {4} mm".format(
    Lmax * 1e3, Bmax * 1e3, Bmin * 1e3, Hmax * 1e3, grid_size * 1e3, accuracy * 1e3
)
line02 = description
line03 = "### Simulation Results ###"
line04 = "Resonance Frequency (Hz)\t Features (m)\n"

header_lines = [line00, line01, line02, line03, line04]

with open(folder + "features.csv", "a+") as csvfile:
    csvfile.write("\n".join(header_lines))

# --------------------
# Generate initial geometry
# --------------------

# Randomly initialize geometry
geometry = Geometry(Lmax, Bmax, Hmax, grid_size, accuracy, Bmin)
# geometry.generate_random_vertical_length()
# geometry.generate_random_horizontal_length()
geometry.init_rectangular_geometry(Binit)
# geometry.generate_boolean_grid()
geometry.generate_mesh()

"""
# --------------------
# Or read in from file
# --------------------
feature_file = "./simulated_annealing3/features.csv"
# Read in features from file
number_features = 24
resonances_and_features = cf.read_features(feature_file, number_features)

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
    feature_file,
    Lmax,
    save=False,
)
"""


# --------------------
# Setup function to optimize
# --------------------
def opt_function(horizontal_lengths):
    """
    Function that takes in a list containing the horizontal weights and can be
    optimized with the different scipy optimizers.
    """
    # Change geometry
    geometry.adjust_horizontal_length(horizontal_lengths)
    geometry.generate_mesh()

    # Calculate eigenfrequency
    eigenfrequency, x, y, magnitude = cf.do_simulation(
        geometry.mesh, E, nu, rho, target_frequency, no_eigenstates
    )

    cf.plot_shape_with_resonance(x, y, magnitude, eigenfrequency, folder, Lmax)

    # The features should be saved in separate files for each particle!
    cf.append_feature(
        eigenfrequency,
        folder + "features.csv",
        horizontal_lengths,
    )

    return eigenfrequency


# Actual optimization algorithm
# from scipy.optimize import differential_evolution
# from scipy.optimize import dual_annealing
from scipy.optimize import shgo

bounds = (
    (
        Bmin,
        Bmax,
    ),
) * np.size(geometry.horizontal_lengths)

# print(bounds)

final_results = shgo(opt_function, bounds)

print(final_results)
