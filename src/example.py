from geometry_generator import (
    generate_gmsh_mesh,
    generate_gmsh_mesh_needle,
    generate_gmsh_mesh_more_crazy_needle,
)
from solver import unified_solving_function
from visualisation import visualise_3D, visualise_mesh, generate_gif

import sys
import numpy as np
from mpi4py import MPI
from slepc4py import SLEPc
import slepc4py

# Initialize SLEPc
slepc4py.init(sys.argv)

import pyvista

# Start pyvista with off-screen rendering
pyvista.start_xvfb()

import gmsh

# Initialize gmsh
gmsh.initialize()

# Define dimensions of the structure
L, H, B = 12e-3, 0.2e-3, 3e-3

# Define grid size and boundary conditions
Bmin = 1e-3
Bmax = B
grid_size = 0.5e-3
bc_z = False
bc_y = False

# Number of eigenvalues to compute and target frequency
no_eigenvalues = 20
target_frequency = 100e3

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

# Define geometry width list for rectangular structure
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

# Generate mesh using gmsh
geometry_mesh = generate_gmsh_mesh(model, L, H, B, geometry_width_list)

# Solve the eigenvalue problem
V, eigenvalues, eigenmodes, first_longitudinal_mode = unified_solving_function(
    eigensolver, geometry_mesh, L, H, B, bc_z, bc_y, no_eigenvalues
)
# Print dolfinx version
import dolfinx

print(dolfinx.__version__)

# Plot all eigenmodes
for mode_no in range(np.size(eigenvalues)):
    freq_3D = np.sqrt(eigenvalues[mode_no].real) / (2 * np.pi)
    saving_path = "{i:.2f}Hz.png".format(i=freq_3D)
    visualise_3D(
        plotter,
        V,
        eigenvalues,
        eigenmodes,
        mode_no,
        saving_path,
        viewup=False,
        high_res=True,
    )
