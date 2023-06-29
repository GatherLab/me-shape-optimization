import sys
import psutil
import os
import gc
import tracemalloc

tracemalloc.start()

import numpy as np
import matplotlib.pylab as plt

from geometry_generator import generate_geometry, generate_gmsh_mesh
from scipy.optimize import root
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

from dolfinx import fem, io, plot, mesh
import ufl

import slepc4py

slepc4py.init(sys.argv)

import pyvista
import pandas as pd

pyvista.start_xvfb()

import gmsh

gmsh.initialize()

print(
    "memory start: {0}".format(
        psutil.Process(os.getpid()).memory_info().rss / 1024**2
    )
)


def determine_first_longitudinal_mode(V, eigenmodes):
    """
    One feature - though maybe not distinct - is that the first longitudinal
    mode is symmetric along its long-axis. This should be a first filter to apply.
    Of the modes that are symmetric along the long-axis, the first longitudinal
    mode is the one with the lowest y displacement.
    """
    i = 0
    df_results = pd.DataFrame(columns=["symmetric", "x_max", "y_max", "z_max"])

    for eigenmode in eigenmodes:
        topology, cell_types, geometry = plot.create_vtk_mesh(V)

        # Get the displacement vector
        warp_vector = eigenmode.x.array.reshape((geometry.shape[0], 3))

        # Concatenate both arrays into a pandas dataframe
        norm = np.linalg.norm(warp_vector, axis=1)
        # geometry_plus_norm = np.append(geometry, np.reshape(norm, [511, 1]), axis=1)
        geometry_plus_norm = np.append(geometry, warp_vector, axis=1)
        df = pd.DataFrame(
            geometry_plus_norm, columns=["x", "y", "z", "x_warp", "y_warp", "z_warp"]
        ).round(decimals=5)

        grouped = df.groupby(["x", "y"])

        # Get indices of arrays that have an agreeing min and max z value (min = -max) and are non-zero
        indices = np.logical_and(
            np.isclose(grouped.z.transform(min), -1 * grouped.z.transform(max)),
            ~np.isclose(grouped.z.transform(min), 0),
        )

        # Now get all entries where z is minimum and maximum for a given pair of x and y
        minimum = (
            df[indices]
            .sort_values(["x", "y", "z"], ascending=[True, True, True])
            .groupby(["x", "y"])
            .first()
        )
        maximum = (
            df[indices]
            .sort_values(["x", "y", "z"], ascending=[True, True, False])
            .groupby(["x", "y"])
            .first()
        )

        # Now obtain the max and min z values again for each pair of x and y
        # a = df[indices].groupby(["x", "y"]).z.transform(max) == df[indices].z
        # b = df[indices].groupby(["x", "y"]).z.transform(min) == df[indices].z
        # c = df.loc[a.loc[a == True].index].loc[df.z != 0].sort_values(["x", "y"])
        # d = df.loc[b.loc[b == True].index].loc[df.z != 0].sort_values(["x", "y"])

        # Now get the warp values and subtract them from each other

        """

        df.sort_values(["x", "y", "z"], ignore_index=True).groupby(["x", "y"]).z.max()

        # Only get values where y is max
        df_ymax = df.loc[np.isclose(df.y.to_numpy(), df.y.max())].sort_values(
            ["x", "z"], ignore_index=True
        )

        # The difficulty will now be to determine the longitudinal axis to impose a
        # symmetry
        x_max_boundary = df_ymax.loc[
            np.isclose(df_ymax.z, df_ymax.z.max())
        ].x_warp.to_numpy()
        x_min_boundary = df_ymax.loc[
            np.isclose(df_ymax.z, df_ymax.z.min())
        ].x_warp.to_numpy()

        # y_max_boundary = df_ymax.loc[df_ymax.z == df_ymax.z.max()].y_warp.to_numpy()
        # y_min_boundary = df_ymax.loc[df_ymax.z == df_ymax.z.min()].y_warp.to_numpy()

        z_max_boundary = df_ymax.loc[
            np.isclose(df_ymax.z, df_ymax.z.max())
        ].z_warp.to_numpy()
        z_min_boundary = df_ymax.loc[
            np.isclose(df_ymax.z, df_ymax.z.min())
        ].z_warp.to_numpy()
        # if i == 18:
        # print("Test")

        # print(
        #     i,
        #     np.logical_and(
        #         np.allclose(x_max_boundary, x_min_boundary, atol=10e-0),
        #         np.allclose(z_max_boundary, -1 * z_min_boundary, atol=10e-0),
        #     ),
        #     df.x_warp.max(),
        #     df.y_warp.max(),
        #     df.z_warp.max(),
        # )
        """

        df_results.loc[i] = [
            np.logical_and(
                np.allclose(
                    minimum.x_warp.to_numpy(), maximum.x_warp.to_numpy(), atol=20
                ),
                np.allclose(
                    minimum.z_warp.to_numpy(), -maximum.z_warp.to_numpy(), atol=20
                ),
            ),
            df.x_warp.max(),
            df.y_warp.max(),
            df.z_warp.max(),
        ]
        i += 1

    # Now obtain the first longitudinal mode by selecting for symmetry and
    # minimum y displacement
    no_of_first_longitudinal_mode = df_results.loc[
        df_results.y_max == df_results.loc[df_results.symmetric].y_max.min()
    ].index[0]

    print(
        "Mode {0} is the first longitudinal one.".format(no_of_first_longitudinal_mode)
    )

    return no_of_first_longitudinal_mode


def funct(geometry_mesh):
    # Define vector space from geometry mesh
    V = fem.VectorFunctionSpace(geometry_mesh, ("CG", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = geometry_mesh.topology.dim - 1

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

    # Define actual problem
    E, nu = (5.4e10), (0.34)
    rho = 7950.0
    mu = E / 2.0 / (1 + nu)
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)

    def epsilon(u):
        return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

    T = fem.Constant(
        geometry_mesh, (PETSc.ScalarType(0), PETSc.ScalarType(0), PETSc.ScalarType(0))
    )
    k_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    m_form = rho * ufl.dot(u, v) * ufl.dx
    k = fem.form(k_form)
    m = fem.form(m_form)

    return k, m, bc1, V


def assemble_KM(k, m, bc1):
    # One of the big questions is: which boundary conditions do we apply?
    # I am able to retrieve sensible results for either no boundary conditions, only
    # boundary condition 1 and both boundary conditions. The resonance frequency
    # changes quite a bit though.

    ## Takes about 13 MB RAM
    K = fem.petsc.assemble_matrix(k, bcs=[bc1])
    M = fem.petsc.assemble_matrix(m, bcs=[bc1])

    return K, M


def assemble_eigensolver(eigensolver, K, M):
    ##

    # Define eigensolver and solve for eigenvalues
    no_eigenvalues = 30
    target_frequency = 100000

    eigensolver.setOperators(K, M)

    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    # tol = 1e-9
    # eigensolver.setTolerances(tol=tol)

    # Shift and invert mode
    st = eigensolver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    # target real eigenvalues
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    # Set the target frequency
    eigensolver.setTarget(target_frequency**2 * 2 * np.pi)
    # Set no of eigenvalues to compute
    eigensolver.setDimensions(nev=no_eigenvalues)
    ##
    return eigensolver


def visualize_eigenmode(eigensolver, K, V):
    # Get the number of converged eigenpairs
    evs = eigensolver.getConverged()

    # Create dummy vectors for the eigenvectors to store the results in
    vr, vi = K.createVecs()

    k = 0
    for i in range(evs):
        # e_val = eigensolver.getEigenpair(i, vr, vi)
        # Get the ith eigenvalue and eigenvector (vr and vi are placeholders for
        # the real and complex parts of the eigenvectors) they are then saved in
        # the input variables (vr, vi)
        eigenvalue = eigensolver.getEigenpair(i, vr, vi)
        # e_vec = eigensolver.getEigenvector(i, eh.vector)

        if ~np.isclose(eigenvalue.real, 1.0, atol=5):
            # Calculation of eigenfrequency from real part of eigenvalue
            freq_3D = np.sqrt(eigenvalue.real) / 2 / np.pi

            eigenmode = fem.Function(V)
            eigenmode.vector[:] = vr.getArray()

            # plot_shape_with_resonance(x,y,mode_magnitude, L)
            # Create plotter and pyvista grid
            # p = pyvista.Plotter(off_screen=True)
            # if k == 2:
            #     p = pyvista.Plotter()
            #     topology, cell_types, geometry = plot.create_vtk_mesh(V)
            #     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

            #     # Attach vector values to grid and warp grid by vector
            #     grid["u"] = eigenmode.x.array.reshape((geometry.shape[0], 3))
            #     actor_0 = p.add_mesh(grid, style="wireframe", color="k")
            #     warped = grid.warp_by_vector("u", factor=5e-6)
            #     actor_1 = p.add_mesh(warped, show_edges=True)
            #     p.show_axes()
            #     if not pyvista.OFF_SCREEN:
            #        p.show()
            #     else:
            #         figure_as_array = p.screenshot("deflection.png")

            print("Solid FE: {0:8.5f} [Hz]".format(freq_3D))
            k += 1


def unified_solving_function(eigensolver, geometry_mesh):
    # Define vector space from geometry mesh
    V = fem.VectorFunctionSpace(geometry_mesh, ("CG", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = geometry_mesh.topology.dim - 1

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

    # Define actual problem
    E, nu = (5.4e10), (0.34)
    rho = 7950.0
    mu = E / 2.0 / (1 + nu)
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)

    def epsilon(u):
        return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

    T = fem.Constant(
        geometry_mesh, (PETSc.ScalarType(0), PETSc.ScalarType(0), PETSc.ScalarType(0))
    )
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
    no_eigenvalues = 20
    target_frequency = 100000

    eigensolver.setOperators(K, M)

    eigensolver.solve()
    evs = eigensolver.getConverged()

    # Create dummy vectors for the eigenvectors to store the results in
    vr, vi = K.createVecs()

    eigenvalues = []
    eigenmodes = []

    for mode_number in range(evs):
        # Get eigenvalue, eigenvector is saved in vr and vi
        eigenvalue = eigensolver.getEigenpair(mode_number, vr, vi)
        # e_vec = eigensolver.getEigenvector(i, eh.vector)

        if ~np.isclose(eigenvalue.real, 1.0, atol=5):
            eigenvalues.append(eigenvalue)

            eigenmode = fem.Function(V)
            eigenmode.vector[:] = vr.getArray()
            eigenmodes.append(eigenmode)

    first_longitudinal_eigenmode = determine_first_longitudinal_mode(V, eigenmodes)
    # visualize_eigenmode(eigensolver, K, V)
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


target_frequency = 100e3
no_eigenvalues = 20
# Eigensolver has to be defined once and then updated to prevent RAM leakage
eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)

# Shift and invert mode
st = eigensolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)
# target real eigenvalues
eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
# Set the target frequency
eigensolver.setTarget(target_frequency**2 * 2 * np.pi)
# Set no of eigenvalues to compute
eigensolver.setDimensions(nev=no_eigenvalues)
eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
##

L, H, B = 12e-3, 0.2e-3, 3e-3
# Geometry initialization
geometry_mesh = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([L, H, B])],
    [20, 2, 5],
    cell_type=mesh.CellType.tetrahedron,
)

# Choose if Gmsh output is verbose
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Box")
model.setCurrent("Box")

i = 1
while True:
    print(i)
    # Define Geometry
    # if i % 2 == 0:
    L, H, B = 12e-3, 0.2e-3, 3e-3
    grid_size = 0.5e-3
    # else:
    # L, H, B = 11e-3, 0.2e-3, 3e-3
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    geometry_width_list = np.random.uniform(1e-3, B, int(L / grid_size))
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    gmsh_mesh = generate_gmsh_mesh(model, L, H, B, geometry_width_list)

    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    geometry_mesh = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([L, H, B])],
        [20, 2, 5],
        cell_type=mesh.CellType.tetrahedron,
    )
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    ##
    unified_solving_function(eigensolver, geometry_mesh)
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)

    i += 1


# This is a snippet to test RAM leakage in specific parts of the code
"""
L, H, B = 11e-3, 0.2e-3, 3e-3

# Negligble RAM accumulation
geometry_mesh = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([L, H, B])],
    [20, 2, 5],
    cell_type=mesh.CellType.tetrahedron,
)

after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(after - before)
before = after

# Negligble RAM accumulation
k, m, bc1, V = funct(geometry_mesh)

after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(after - before)
before = after

# ~ 10 MB RAM accumulation still
# Remove K and M does not free the RAM again
K, M = assemble_KM(k, m, bc1)

after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(after - before)
before = after

# Negligble RAM accumulation
K.assemble()
M.assemble()

after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(after - before)
before = after

# Negative RAM usage (-40 MB) - Probably the 40 MB from the eigensolver.solve()
# is freed
eigensolver = assemble_eigensolver(eigensolver, K, M)

after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(after - before)
before = after

# Solve the eigensystem
## Takes about 40 MB RAM
eigensolver.solve()

after = psutil.Process(os.getpid()).memory_info().rss / 1024**2
print(after - before)
before = after

visualize_eigenmode(eigensolver, K, V)
"""
