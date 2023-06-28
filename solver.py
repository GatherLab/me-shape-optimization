import psutil
import os

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from petsc4py import PETSc
from slepc4py import SLEPc

from dolfinx import fem, plot
import ufl
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
    eigensolver.solve()

    """
    Presort for sensible modes
    """
    # Get the number of converged eigenpairs
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

    # If this is selected, only the first longitudinal mode is returned
    first_longitudinal_eigenmode = determine_first_longitudinal_mode(
        V, eigenmodes, eigenvalues, target_frequency
    )
    # first_longitudinal_eigenmode = 1

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

        # Calculate order of mode by getting the number of minima along the x-direction
        """
        x_warp_along_center = (
            df.loc[
                np.logical_and(
                    np.isclose(df.z, (df.z.max() - df.z.min()) / 2),
                    np.isclose(df.y, df.y.max() / 2),
                ),
                "x_warp",
            ]
            .abs()
            .to_numpy()
        )
        no_minima = np.size(argrelextrema(x_warp_along_center, np.less))
        """

        eigenfrequency = np.sqrt(eigenvalues[i].real) / 2 / np.pi
        min_x_warp = minimum.x_warp.to_numpy(int)
        min_x_warp[np.abs(min_x_warp) < 50] = 0
        max_x_warp = maximum.x_warp.to_numpy(int)
        max_x_warp[np.abs(max_x_warp) < 50] = 0

        # min_z_warp = minimum.z_warp.to_numpy(int)
        # min_z_warp[np.abs(min_z_warp) < 10] = 0
        # max_z_warp = maximum.z_warp.to_numpy(int)
        # max_z_warp[np.abs(max_z_warp) < 10] = 0

        df_results.loc[i] = [
            np.all(
                np.logical_or(
                    np.sign(min_x_warp) == np.sign(max_x_warp),
                    np.logical_or(np.sign(min_x_warp) == 0, np.sign(max_x_warp) == 0),
                )
            ),
            # np.logical_or(
            # np.sign(min_x_warp) == np.sign(max_x_warp)),
            # np.sign(min_x_warp) == 0 or np.sign(max_x_warp) == 0,
            # np.all(np.sign(min_z_warp) == -1 * np.sign(max_z_warp)),
            # np.allclose(
            # minimum.x_warp.to_numpy(), maximum.x_warp.to_numpy(), atol=50
            # ),
            # np.allclose(
            # minimum.z_warp.to_numpy(), -maximum.z_warp.to_numpy(), atol=50
            # ),
            # ),
            eigenfrequency,
            df.x_warp.max(),
            df.y_warp.max(),
            df.z_warp.max(),
        ]
        i += 1

    # Now obtain the first longitudinal mode by selecting for symmetry and
    # minimum y displacement
    no_of_first_longitudinal_mode = df_results.loc[
        df_results.y_max
        == df_results.loc[
            np.logical_and(
                df_results.symmetric,
                np.logical_and(
                    df_results.eigenfrequency > target_frequency * 0.5,
                    df_results.eigenfrequency < target_frequency * 1.5,
                ),
            )
        ].y_max.min()
    ].index[0]

    print(
        "Mode {0} is the first longitudinal one.".format(no_of_first_longitudinal_mode)
    )

    """
    if df.eigenfrequency < 50000 or eigenfrequency > 150000:
        for mode_no in range(np.size(eigenvalues)):
            saving_path = "deflection{i}.png".format(i=mode_no)
            visualise_3D(V, eigenvalues, eigenmodes, mode_no, saving_path)
        print("stop")
    """

    return no_of_first_longitudinal_mode
