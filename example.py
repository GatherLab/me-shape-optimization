from geometry_generator import (
    generate_geometry,
    generate_gmsh_mesh,
    generate_gmsh_mesh_needle,
    generate_gmsh_mesh_more_crazy_needle,
    generate_gmsh_mesh_different_topologies,
)
from solver import unified_solving_function, determine_first_longitudinal_mode
from visualisation import visualise_3D, visualise_mesh, generate_gif

from dolfinx.io.gmshio import model_to_mesh

import sys
import numpy as np
import pandas as pd
import psutil
import os

from mpi4py import MPI
from slepc4py import SLEPc
import slepc4py

slepc4py.init(sys.argv)

import pyvista

pyvista.start_xvfb()

import gmsh

gmsh.initialize()

L, H, B = 12e-3, 0.2e-3, 3e-3

Bmin = 1e-3
Bmax = B
grid_size = 0.5e-3

# Select boundary conditions to apply
bc_z = True
bc_y = True

no_eigenvalues = 10
target_frequency = 100e3

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

# Generate random numbers using numpy library with Bmax as a maximum and Bmin as
# a minimum with L/grid_size entries
# geometry_width_list = np.random.uniform(Bmin, Bmax, int(L / grid_size))

# 0.5 mm grid optimization rectangular structure
# geometry_width_list = [0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.001000,0.003000,0.001000,0.001000,0.001000,0.001000,0.001000,0.001000,0.001000,0.001000,0.003000,0.001000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000]
# 1 mm grid optimization rectangular structure
geometry_width_list = [0.005000,0.005000,0.005000,0.0004000,0.0004000,0.0004000,0.0004000,0.0004000,0.0004000,0.005000,0.005000,0.005000]

# 80122 kHz
# geometry_width_list = [0.001000,0.003000,0.003000,0.001000,0.001000,0.001000,0.001000,0.001000,0.001000,0.001000,0.003000,0.003000]
# 87580
# geometry_width_list = [0.001000,0.001000,0.001000,0.003000,0.003000,0.001000,0.001000,0.001000,0.001000,0.001000,0.003000,0.003000]
# 95014
# geometry_width_list = [0.001000,0.001000,0.001000,0.003000,0.003000,0.001000,0.001000,0.003000,0.003000,0.001000,0.001000,0.003000]
# 102506
# geometry_width_list = [0.001000,0.001000,0.001000,0.003000,0.003000,0.003000,0.003000,0.003000,0.001000,0.001000,0.003000,0.003000]
# 110016
# geometry_width_list = [0.001000,0.001000,0.001000,0.003000,0.003000,0.003000,0.001000,0.001000,0.003000,0.001000,0.001000,0.001000]
# 117464
# geometry_width_list = [0.003000,0.001000,0.001000,0.003000,0.003000,0.003000,0.003000,0.003000,0.001000,0.001000,0.001000,0.001000]
# 124968
# geometry_width_list = [0.001000,0.001000,0.001000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.001000,0.003000,0.001000]
# 133896
# geometry_width_list = [0.001000,0.001000,0.001000,0.001000,0.003000,0.003000,0.003000,0.003000,0.001000,0.001000,0.001000,0.001000]
# 140682
# geometry_width_list = [0.001000,0.001000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.001000,0.001000]
# 144163
# geometry_width_list = [0.001000,0.001000,0.001000,0.001000,0.001000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.001000,0.001000,0.001000,0.001000,0.001000]

## Needle
# 94814
# geometry_width_list = [0.001000,0.001000,0.001000,0.001000,0.001000,0.001000,0.003000,0.001000,0.003000,0.003000,0.003000,0.003000]
# 155709
# geometry_width_list = [0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.003000,0.001000,0.001000,0.001000,0.001000]

# Standard
# geometry_width_list = np.repeat(B, int(L / grid_size)/2)

# With gmsh model
# No memory leak
# geometry_mesh = generate_gmsh_mesh(model, L, H, B, geometry_width_list)
# geometry_mesh = generate_gmsh_mesh_needle(model, L, H, B, geometry_width_list)
geometry_mesh = generate_gmsh_mesh_more_crazy_needle(model, L, H, B, np.array([3e-3]))
# geometry_mesh = generate_gmsh_mesh_different_topologies(model, L, H, B)

"""
# With fenicsx mesh
fenics_mesh = generate_geometry(L, H, B)
"""

# geometry_mesh = gmsh_mesh

# visualise_mesh(geometry_mesh, "mesh.png")

# Solve
V, eigenvalues, eigenmodes, first_longitudinal_mode = unified_solving_function(
    eigensolver, geometry_mesh, L, H, B, bc_z, bc_y, no_eigenvalues
)

# Show first N eigenmodes
# Without constraints, the relevant first longitudinal mode is around the 4th mode
# For only constraining the oscillation in the y-direction, the relevant first longitudinal mode is around the 4th mode
# For constraining the oscillation in the y & z-direction, the relevant first longitudinal mode is around the 19th mode

# Plot all eigenmodes
for mode_no in range(np.size(eigenvalues)):
    freq_3D = np.sqrt(eigenvalues[mode_no].real) / 2 / np.pi
    saving_path = "{i:.2f}Hz.png".format(i=freq_3D)
    visualise_3D(
        plotter, V, eigenvalues, eigenmodes, mode_no, saving_path, viewup=True, high_res=True
    )

# determine_first_longitudinal_mode(V, eigenmodes, eigenvalues, target_frequency)
"""

features = pd.read_csv(
    "./me-shape-optimization/results/dual-annealing-3/features.csv",
    skiprows=4,
    names=np.append("frequency", ["width{0}".format(i) for i in range(24)]),
    sep="\t",
)
# Now reduce number of rows and only take every 100th row
frequencies = np.append(
    features.frequency.to_numpy()[::1000], features.frequency.to_numpy()[-1]
)

generate_gif(
    "./me-shape-optimization/results/dual-annealing-3",
    frequencies,
)

"""


# Plot first longitudinal mode only
# freq_3D = np.sqrt(eigenvalues[first_longitudinal_mode].real) / 2 / np.pi
# saving_path = "{i:.2f}.png".format(i=freq_3D)
# visualise_3D(
# V, eigenvalues, eigenmodes, first_longitudinal_mode, saving_path, viewup=True
# )
