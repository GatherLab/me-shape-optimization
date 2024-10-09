from geometry_generator import (
    generate_gmsh_mesh,
    generate_gmsh_mesh_needle,
    generate_gmsh_mesh_more_crazy_needle,
)
from solver import unified_solving_function
from visualisation import visualise_3D, append_feature

import sys
import os
from mpi4py import MPI
from slepc4py import SLEPc
import slepc4py
import pyvista
import numpy as np
from scipy.optimize import dual_annealing, minimize, basinhopping
import gmsh

# Initialize SLEPc
slepc4py.init(sys.argv)

# Start pyvista with off-screen rendering
pyvista.start_xvfb()

# Initialize gmsh
gmsh.initialize()

# Define geometry parameters
L, H, B = 12e-3, 0.2e-3, 3e-3
Bmin = 1e-3
Bmax = 3e-3
grid_size = 1e-3

# Select boundary conditions to apply
bc_z = True
bc_y = True
no_eigenvalues = 50
target_frequency = 100e3
optimization_target_frequency = 87.5e3

# Set a folder to save the features
folder = "./results/dual-annealing-8/"
description = (
    "Function optimization where it can only take Bmin or Bmax. With grid size of 1 mm."
)
os.makedirs(folder, exist_ok=True)

# Generate a file that contains the meta data in the header
header_lines = [
    f"Geometry: L = {L * 1e3} mm, Bmin = {Bmin * 1e3} mm, Bmax = {Bmax * 1e3} mm, H = {H * 1e3} mm, grid size = {grid_size * 1e3} mm",
    description,
    "### Simulation Results ###",
    "Resonance Frequency (Hz)\t Features (m)\n",
]

with open(folder + "features.csv", "a+") as csvfile:
    csvfile.write("\n".join(header_lines))

# Define eigensolver
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
st = eigensolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
eigensolver.setTarget(target_frequency**2 * 2 * np.pi)
eigensolver.setDimensions(nev=no_eigenvalues)

# Define pyvista plotter
plotter = pyvista.Plotter(off_screen=True)

# Geometry generation
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Box")
model.setCurrent("Box")


def opt_function(horizontal_lengths):
    """
    Function that takes in a list containing the horizontal weights and can be
    optimized with the different scipy optimizers.
    """
    print("-------------- New Optimization step --------------")
    # Make the problem discrete
    # horizontal_lengths = [
    #     Bmin if l < (Bmax - Bmin) * 0.5 + Bmin else Bmax for l in horizontal_lengths
    # ]
    horizontal_lengths = np.round(horizontal_lengths, 4)

    try:
        gmsh_mesh = generate_gmsh_mesh(model, L, H, B, horizontal_lengths)
    except:
        print("Mesh generation failed!")
        return 1e6

    try:
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
    except ValueError:
        print("No longitudinal mode found for lengths!\n")
        print("horizontal_lengths: " + str(horizontal_lengths))
        return 1e6

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

    append_feature(
        eigenfrequency,
        folder + "features.csv",
        horizontal_lengths,
    )

    return eigenfrequency


# Define bounds for the optimization
bounds = (
    (
        Bmin,
        Bmax,
    ),
) * int(L / grid_size)

# Initial geometry width list
geometry_width_list = np.repeat(Bmax, int(L / grid_size))

# Perform optimization using dual annealing
final_results = dual_annealing(opt_function, x0=geometry_width_list, bounds=bounds)
# final_results = basinhopping(
# opt_function,
# x0=geometry_width_list,
# minimizer_kwargs={"bounds": bounds, "method": "Powell"},
# )
# final_results = minimize(
# opt_function, method="Powell", x0=geometry_width_list, bounds=bounds
# )
