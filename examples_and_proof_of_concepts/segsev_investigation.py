from mpi4py import MPI

import slepc4py
import ufl

import numpy as np

from dolfinx import fem, mesh
import sys

# from scipy.optimize import root


# import pyvista
# import pandas as pd

# pyvista.start_xvfb()

# import gmsh

# gmsh.initialize()

# print(
# "memory start: {0}".format(
# psutil.Process(os.getpid()).memory_info().rss / 1024**2
# )
# )
comm = MPI.COMM_WORLD
# comm = MPI.COMM_SELF


def generate_geometry(L, H, B, communicator):
    # Geometry initialization
    geometry_mesh = mesh.create_box(
        communicator,
        [np.array([0, 0, 0]), np.array([L, H, B])],
        [20, 2, 5],
        cell_type=mesh.CellType.tetrahedron,
    )
    # comm.Disconnect()
    return geometry_mesh


def unified_solving_function(eigensolver, geometry_mesh):
    # Define vector space from geometry mesh
    V = fem.VectorFunctionSpace(geometry_mesh, ("CG", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # fdim = geometry_mesh.topology.dim - 1

    """
    # Fixed z
    space, map = V.sub(2).collapse()
    u_D1 = fem.Function(space)

    # Assign all the displacements along y to be zero
    with u_D1.vector.localForm() as loc:
        loc.set(0.0)

    # Locate facets where y = 0 or y = H
    locate_dofs1 = fem.locate_dofs_geometrical(
        (V.sub(1), space),
        lambda x: np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], H)),
    )

    # Create Dirichlet BC
    bc1 = fem.dirichletbc(u_D1, locate_dofs1, V.sub(1))

    # Fixed y
    # Get sub space of y's
    space_2, map_2 = V.sub(2).collapse()
    u_D2 = fem.Function(space_2)

    # Assign all the displacements along y to be zero
    with u_D2.vector.localForm() as loc:
        loc.set(0.0)

    # Locate facets where y = 0 or y = H
    locate_dofs2 = fem.locate_dofs_geometrical(
        (V.sub(2), space_2),
        lambda x: np.logical_or(np.isclose(x[2], 0), np.isclose(x[2], B)),
    )

    # Create Dirichlet BC
    bc2 = fem.dirichletbc(u_D2, locate_dofs2, V.sub(2))
    """

    # Define actual problem
    E, nu = (5.4e10), (0.34)
    rho = 7950.0
    mu = E / 2.0 / (1 + nu)
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)

    def epsilon(u):
        return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

    k_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    m_form = rho * ufl.dot(u, v) * ufl.dx
    k = fem.form(k_form)
    m = fem.form(m_form)

    # One of the big questions is: which boundary conditions do we apply?
    # I am able to retrieve sensible results for either no boundary conditions, only
    # boundary condition 1 and both boundary conditions. The resonance frequency
    # changes quite a bit though.

    ## Takes about 13 MB RAM
    K = fem.petsc.assemble_matrix(k, bcs=[])
    M = fem.petsc.assemble_matrix(m, bcs=[])

    K.assemble()
    M.assemble()

    # Define eigensolver and solve for eigenvalues
    eigensolver.setOperators(K, M)

    eigensolver.solve()
    evs = eigensolver.getConverged()

    # Create dummy vectors for the eigenvectors to store the results in --> Source of segmentation fault (https://fenicsproject.discourse.group/t/setting-snes-solver-with-petsc4py-segfaults/5149/9)
    #
    # vr, vi = K.createVecs()
    # vr, vi = fem.petsc.create_vector()

    eigenvalues = []
    eigenmodes = []

    for mode_number in range(evs):
        eigenmode = fem.Function(V)
        # Get eigenvalue, eigenvector is saved in vr and vi
        # eigenvalue = eigensolver.getEigenpair(mode_number, vr, vi)
        eigenvalue = eigensolver.getEigenpair(mode_number, eigenmode.vector)
        # e_vec = eigensolver.getEigenvector(i, eh.vector)

        if ~np.isclose(eigenvalue.real, 1.0, atol=5):
            eigenvalues.append(eigenvalue)

            # This is also a potential bottleneck according to https://github.com/FEniCS/dolfinx/issues/2308
            # eigenmode.vector[:] = vr.getArray()
            eigenmodes.append(eigenmode)

    # first_longitudinal_eigenmode = determine_first_longitudinal_mode(V, eigenmodes)
    # visualize_eigenmode(eigensolver, K, V)
    # print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


# Choose if Gmsh output is verbose
# gmsh.option.setNumber("General.Terminal", 0)
# model = gmsh.model()
# model.add("Box")
# model.setCurrent("Box")

slepc4py.init()
L, H, B = 12e-3, 0.2e-3, 3e-3
grid_size = 0.5e-3

i = 0
target_frequency = 100e3
no_eigenvalues = 20

# Eigensolver has to be defined once and then updated to prevent RAM leakage
# Usually in the official solution there would have to be an MPI connector in
# the create statement. However, this creates an issue with too many MPI
# connectors being created (it fails after 2048)
eigensolver = slepc4py.SLEPc.EPS().create()

# Shift and invert mode
st = eigensolver.getST()
st.setType(slepc4py.SLEPc.ST.Type.SINVERT)
# target real eigenvalues
eigensolver.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.TARGET_REAL)
# Set the target frequency
eigensolver.setTarget(target_frequency**2 * 2 * np.pi)
# Set no of eigenvalues to compute
eigensolver.setDimensions(nev=no_eigenvalues)
eigensolver.setProblemType(slepc4py.SLEPc.EPS.ProblemType.GHEP)

while True:
    print(i)
    # Define Geometry
    # if i % 2 == 0:

    # else:
    # L, H, B = 11e-3, 0.2e-3, 3e-3
    # Leads to reaching the maxmiuim number of communicators two times faster
    # communicator = comm.Clone()

    geometry_width_list = np.random.uniform(1e-3, 12e-3, 3)
    gmsh_mesh = generate_geometry(L, H, B, comm)

    # geometry_mesh = mesh.create_box(
    # MPI.COMM_WORLD,
    # [np.array([0, 0, 0]), np.array([L, H, B])],
    # [20, 2, 5],
    # cell_type=mesh.CellType.tetrahedron,
    # )
    ##
    unified_solving_function(eigensolver, gmsh_mesh)

    i += 1
