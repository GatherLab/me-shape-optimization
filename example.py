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

# 1 mm grid optimization rectangular structure
geometry_width_list = [
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
    0.003,
]

# With gmsh model
# No memory leak
geometry_mesh = generate_gmsh_mesh(model, L, H, B, geometry_width_list)

# Solve
V, eigenvalues, eigenmodes, first_longitudinal_mode = unified_solving_function(
    eigensolver, geometry_mesh, L, H, B, bc_z, bc_y, no_eigenvalues
)

# Plot all eigenmodes
for mode_no in range(np.size(eigenvalues)):
    freq_3D = np.sqrt(eigenvalues[mode_no].real) / 2 / np.pi
    saving_path = "{i:.2f}Hz.png".format(i=freq_3D)
    visualise_3D(
        plotter,
        V,
        eigenvalues,
        eigenmodes,
        mode_no,
        saving_path,
        viewup=True,
        high_res=True,
    )
