from geometry_generator import generate_gmsh_mesh
from solver import unified_solving_function
from visualisation import visualise_3D, append_feature

import sys
import os
import psutil

from mpi4py import MPI
from slepc4py import SLEPc
import slepc4py

slepc4py.init(sys.argv)

import pyvista
import numpy as np

pyvista.start_xvfb()

from scipy.optimize import (
    shgo,
    differential_evolution,
    dual_annealing,
    direct,
    minimize,
)

import gmsh

gmsh.initialize()

# Track down memory leaks
# from pympler.tracker import SummaryTracker
#
# tracker = SummaryTracker()

# Define geometry parameters
L, H, B = 12e-3, 0.2e-3, 3e-3

Bmin = 1e-3
Bmax = B
grid_size = 0.5e-3

# Select boundary conditions to apply
bc_z = True
bc_y = True
no_eigenvalues = 20
target_frequency = 100e3

# Set a folder to save the features
folder = "./me-shape-optimization/results/scipy2/"
description = "SHGO optimization, scipy.optimize.shgo(opt_function, bounds), bcs on"

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
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)

# Set problem
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
# Shift and invert mode
st = eigensolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)
# target real eigenvalues
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
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

# --------------------
# Setup function to optimize
# --------------------


def opt_function(horizontal_lengths):
    """
    Function that takes in a list containing the horizontal weights and can be
    optimized with the different scipy optimizers.
    """
    print("-------------- New Optimization step --------------")
    # before = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    # Change geometry
    gmsh_mesh = generate_gmsh_mesh(model, L, H, B, horizontal_lengths)

    # after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    # print("Mesh generation: " + str(after - before))
    # before = after

    V, eigenvalues, eigenmodes, first_longitudinal_mode = unified_solving_function(
        eigensolver,
        gmsh_mesh,
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
    eigenfrequency = np.sqrt(eigenvalues[first_longitudinal_mode].real) / 2 / np.pi

    saving_path = folder + "{i:.2f}.png".format(i=eigenfrequency)
    visualise_3D(
        plotter,
        V,
        eigenvalues,
        eigenmodes,
        first_longitudinal_mode,
        saving_path,
        viewup=True,
    )

    # after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    # print("Visualise 3D function: " + str(after - before))
    # before = after

    # The features should be saved in separate files for each particle!
    append_feature(
        eigenfrequency,
        folder + "features.csv",
        horizontal_lengths,
    )

    # after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    # print("Append feature function: " + str(after - before))
    # before = after

    return eigenfrequency


bounds = (
    (
        Bmin,
        Bmax,
    ),
) * int(L / grid_size)


# final_results = differential_evolution(opt_function, bounds)
while True:
    horizontal_lengths = np.random.uniform(Bmin, Bmax, int(L / grid_size))
    opt_function(horizontal_lengths)
