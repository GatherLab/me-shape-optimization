from geometry_generator import generate_geometry, generate_gmsh_mesh
from solver import unified_solving_function, determine_first_longitudinal_mode
from visualisation import visualise_3D, visualise_mesh

from dolfinx.io.gmshio import model_to_mesh

import sys
import numpy as np
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

no_eigenvalues = 20
target_frequency = 130e3

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
geometry_width_list = np.random.uniform(Bmin, Bmax, int(L / grid_size))
# geometry_width_list = [
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.003000,
#     0.001000,
#     0.003000,
#     0.001000,
#     0.003000,
#     0.001000,
#     0.003000,
#     0.003000,
# ]
# geometry_width_list = np.repeat(B, int(L / grid_size))

# With gmsh model
# No memory leak
geometry_mesh = generate_gmsh_mesh(model, L, H, B, geometry_width_list)

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
    saving_path = "deflection{i}.png".format(i=mode_no)
    visualise_3D(plotter, V, eigenvalues, eigenmodes, mode_no, saving_path)

determine_first_longitudinal_mode(V, eigenmodes, eigenvalues, target_frequency)

# Plot first longitudinal mode only
# freq_3D = np.sqrt(eigenvalues[first_longitudinal_mode].real) / 2 / np.pi
# saving_path = "{i:.2f}.png".format(i=freq_3D)
# visualise_3D(
# V, eigenvalues, eigenmodes, first_longitudinal_mode, saving_path, viewup=True
# )
