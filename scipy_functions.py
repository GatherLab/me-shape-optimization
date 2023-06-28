from geometry_generator import generate_geometry, generate_gmsh_mesh
from solver import unified_solving_function, determine_first_longitudinal_mode
from visualisation import visualise_3D, append_feature

from dolfinx.io.gmshio import model_to_mesh

import sys
import os
import numpy as np

from mpi4py import MPI
from slepc4py import SLEPc
import slepc4py

slepc4py.init(sys.argv)

import pyvista

pyvista.start_xvfb()

# Define geometry parameters
L, H, B = 12e-3, 0.2e-3, 3e-3

Bmin = 1e-3
Bmax = B
grid_size = 0.5e-3

# Select boundary conditions to apply
bc_z = False
bc_y = False
no_eigenvalues = 50

# Set a folder to save the features
folder = "./shape_optimization/shape_optimization/results/scipy1/"
description = "SHGO optimization, scipy.optimize.shgo(opt_function, bounds)"

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

eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)


# --------------------
# Setup function to optimize
# --------------------
def opt_function(horizontal_lengths):
    """
    Function that takes in a list containing the horizontal weights and can be
    optimized with the different scipy optimizers.
    """
    # Change geometry
    gmsh_mesh = generate_gmsh_mesh(L, H, B, horizontal_lengths)

    V, eigenvalues, eigenmodes, first_longitudinal_mode = unified_solving_function(
        eigensolver, gmsh_mesh, L, H, B, bc_z, bc_y, no_eigenvalues
    )
    # Plot first longitudinal mode only
    eigenfrequency = np.sqrt(eigenvalues[first_longitudinal_mode].real) / 2 / np.pi
    saving_path = folder + "{i:.2f}.png".format(i=eigenfrequency)
    visualise_3D(
        V, eigenvalues, eigenmodes, first_longitudinal_mode, saving_path, viewup=True
    )

    # The features should be saved in separate files for each particle!
    append_feature(
        eigenfrequency,
        folder + "features.csv",
        horizontal_lengths,
    )

    return eigenfrequency


from scipy.optimize import shgo

bounds = (
    (
        Bmin,
        Bmax,
    ),
) * int(L / grid_size)


final_results = shgo(opt_function, bounds)
