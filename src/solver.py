import psutil
import os

import numpy as np
import pandas as pd

# from scipy.signal import argrelextrema
from dolfinx import fem, plot, mesh, default_scalar_type
import ufl

from petsc4py import PETSc

# from slepc4py import SLEPc

from visualisation import visualise_3D, visualise_mesh


def unified_solving_function(
    eigensolver,
    geometry_mesh,
    L,
    H,
    B,
    bc_z,
    bc_y,
    no_eigenvalues=25,
    target_frequency=100000,
):

    # PZT
    E, nu = (5.4e10), (0.34)
    rho = 7950.0
    mu = E / 2.0 / (1 + nu)
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)

    # Define vector space from geometry mesh
    V = fem.VectorFunctionSpace(geometry_mesh, ("CG", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    fdim = geometry_mesh.topology.dim - 1

    # Fixed z
    space, map = V.sub(1).collapse()
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

    # Locate facets where z = 0 or z = B
    locate_dofs2 = fem.locate_dofs_geometrical(
        (V.sub(2), space_2),
        lambda x: np.logical_or(np.isclose(x[2], 0), np.isclose(x[2], B)),
    )

    # Create Dirichlet BC
    bc2 = fem.dirichletbc(u_D2, locate_dofs2, V.sub(2))

    # Define actual problem

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

    # Boundary conditions
    if bc_z == True and bc_y == True:
        bcs_applied = [bc1, bc2]
    if bc_z == True and bc_y == False:
        bcs_applied = [bc1]
    if bc_z == False and bc_y == True:
        bcs_applied = [bc2]
    if bc_z == False and bc_y == False:
        bcs_applied = []

    ## Takes about 13 MB RAM
    K = fem.petsc.assemble_matrix(k, bcs=bcs_applied)
    M = fem.petsc.assemble_matrix(m, bcs=bcs_applied)

    K.assemble()
    M.assemble()

    # Define eigensolver and solve for eigenvalues
    eigensolver.setOperators(K, M)

    eigensolver.solve()

    # Presort for sensible modes
    # Get the number of converged eigenpairs
    evs = eigensolver.getConverged()

    eigenvalues = []
    eigenmodes = []

    for mode_number in range(evs):
        eigenmode = fem.Function(V)
        # Get eigenvalue, eigenvector is saved in vr and vi
        eigenvalue = eigensolver.getEigenpair(mode_number, eigenmode.vector)

        if ~np.isclose(eigenvalue.real, 1.0, atol=5):
            eigenvalues.append(eigenvalue)

            eigenmodes.append(eigenmode)

    # If this is selected, only the first longitudinal mode is returned
    try:
        first_longitudinal_eigenmode = determine_first_longitudinal_mode(
            V, eigenmodes, eigenvalues, target_frequency
        )
    except ValueError:
        raise ValueError("No longitudinal mode found.")

    print(
        "Current RAM usage: {0} MB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**2
        )
    )

    return V, eigenvalues, eigenmodes, first_longitudinal_eigenmode


def determine_first_longitudinal_mode(V, eigenmodes, eigenvalues, target_frequency):
    """
    One feature - though maybe not distinct - is that the first longitudinal
    mode is symmetric along its long-axis. This should be a first filter to apply.
    Of the modes that are symmetric along the long-axis, the first longitudinal
    mode is the one with the lowest y displacement.
    """
    i = 0
    df_results = pd.DataFrame(
        columns=["symmetric", "eigenfrequency", "x_max", "y_max", "z_max"]
    )

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

        eigenfrequency = np.sqrt(eigenvalues[i].real) / 2 / np.pi
        min_x_warp = minimum.x_warp.to_numpy(int)
        min_x_warp[np.abs(min_x_warp) < 50] = 0
        max_x_warp = maximum.x_warp.to_numpy(int)
        max_x_warp[np.abs(max_x_warp) < 50] = 0

        min_z_warp = minimum.z_warp.to_numpy(int)
        min_z_warp[np.abs(min_z_warp) < 50] = 0
        max_z_warp = maximum.z_warp.to_numpy(int)
        max_z_warp[np.abs(max_z_warp) < 50] = 0

        df_results.loc[i] = [
            np.all(
                np.logical_or(
                    np.sign(min_x_warp) == np.sign(max_x_warp),
                    np.logical_or(np.sign(min_x_warp) == 0, np.sign(max_x_warp) == 0),
                )
            ),

            eigenfrequency,
            df.x_warp.max(),
            df.y_warp.max(),
            df.z_warp.max(),
        ]
        i += 1

    # Now obtain the first longitudinal mode by selecting for symmetry and
    # minimum y displacement
    df_results = df_results.sort_values("eigenfrequency")
    try:
        no_of_first_longitudinal_mode = df_results.loc[
            np.logical_and(
                df_results.symmetric == True,
                np.logical_and(
                    df_results.eigenfrequency <= target_frequency * 1.5,
                    df_results.eigenfrequency >= target_frequency * 0.5,
                ),
            )
        ].index.to_numpy()[0]
    except:
        raise ValueError("No longitudinal mode found.")


    print(
        "Mode {0} is the first longitudinal mode with a frequency of {1} Hz.".format(
            no_of_first_longitudinal_mode,
            df_results.eigenfrequency[no_of_first_longitudinal_mode],
        )
    )

    return no_of_first_longitudinal_mode
