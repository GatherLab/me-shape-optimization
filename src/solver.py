import psutil
import os
import numpy as np
import pandas as pd
from dolfinx import fem, plot
import ufl
from petsc4py import PETSc


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
    """
    Solves the eigenvalue problem for a given geometry mesh and boundary conditions.

    Parameters:
    eigensolver (SLEPc.EPS): The eigensolver object.
    geometry_mesh (dolfinx.mesh.Mesh): The mesh of the geometry.
    L (float): Length of the geometry.
    H (float): Height of the geometry.
    B (float): Width of the geometry.
    bc_z (bool): Apply boundary condition in the z direction.
    bc_y (bool): Apply boundary condition in the y direction.
    no_eigenvalues (int): Number of eigenvalues to compute.
    target_frequency (float): Target frequency for the eigenvalue problem.

    Returns:
    tuple: Function space, eigenvalues, eigenmodes, and the first longitudinal mode.
    """
    # Material properties for PZT
    E, nu = 5.4e10, 0.34
    rho = 7950.0
    mu = E / 2.0 / (1 + nu)
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)

    # Define vector space from geometry mesh
    V = fem.VectorFunctionSpace(geometry_mesh, ("CG", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Fixed z boundary condition
    space, map = V.sub(1).collapse()
    u_D1 = fem.Function(space)
    with u_D1.vector.localForm() as loc:
        loc.set(0.0)
    locate_dofs1 = fem.locate_dofs_geometrical(
        (V.sub(1), space),
        lambda x: np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], H)),
    )
    bc1 = fem.dirichletbc(u_D1, locate_dofs1, V.sub(1))

    # Fixed y boundary condition
    space_2, map_2 = V.sub(2).collapse()
    u_D2 = fem.Function(space_2)
    with u_D2.vector.localForm() as loc:
        loc.set(0.0)
    locate_dofs2 = fem.locate_dofs_geometrical(
        (V.sub(2), space_2),
        lambda x: np.logical_or(np.isclose(x[2], 0), np.isclose(x[2], B)),
    )
    bc2 = fem.dirichletbc(u_D2, locate_dofs2, V.sub(2))

    # Define strain and stress
    def epsilon(u):
        return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

    # Define forms
    T = fem.Constant(
        geometry_mesh, (PETSc.ScalarType(0), PETSc.ScalarType(0), PETSc.ScalarType(0))
    )
    k_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    m_form = rho * ufl.dot(u, v) * ufl.dx
    k = fem.form(k_form)
    m = fem.form(m_form)

    # Apply boundary conditions
    bcs_applied = []
    if bc_z:
        bcs_applied.append(bc1)
    if bc_y:
        bcs_applied.append(bc2)

    # Assemble matrices
    K = fem.petsc.assemble_matrix(k, bcs=bcs_applied)
    M = fem.petsc.assemble_matrix(m, bcs=bcs_applied)
    K.assemble()
    M.assemble()

    # Solve eigenvalue problem
    eigensolver.setOperators(K, M)
    eigensolver.solve()

    # Get converged eigenpairs
    evs = eigensolver.getConverged()
    eigenvalues = []
    eigenmodes = []

    for mode_number in range(evs):
        eigenmode = fem.Function(V)
        eigenvalue = eigensolver.getEigenpair(mode_number, eigenmode.vector)
        if not np.isclose(eigenvalue.real, 1.0, atol=5):
            eigenvalues.append(eigenvalue)
            eigenmodes.append(eigenmode)

    # Determine the first longitudinal mode
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
    Determine the first longitudinal mode based on symmetry and displacement.

    Parameters:
    V (dolfinx.fem.FunctionSpace): The function space.
    eigenmodes (list): List of eigenmodes.
    eigenvalues (list): List of eigenvalues.
    target_frequency (float): Target frequency for the eigenvalue problem.

    Returns:
    int: Index of the first longitudinal mode.
    """
    # DataFrame to store results
    df_results = pd.DataFrame(
        columns=["symmetric", "eigenfrequency", "x_max", "y_max", "z_max"]
    )

    for i, eigenmode in enumerate(eigenmodes):
        # Create VTK mesh and warp vector
        topology, cell_types, geometry = plot.create_vtk_mesh(V)
        warp_vector = eigenmode.x.array.reshape((geometry.shape[0], 3))
        geometry_plus_norm = np.append(geometry, warp_vector, axis=1)

        # Create DataFrame with geometry and warp vector
        df = pd.DataFrame(
            geometry_plus_norm, columns=["x", "y", "z", "x_warp", "y_warp", "z_warp"]
        ).round(decimals=5)

        # Group by x and y coordinates
        grouped = df.groupby(["x", "y"])

        # Identify symmetric points
        indices = np.logical_and(
            np.isclose(grouped.z.transform(min), -1 * grouped.z.transform(max)),
            ~np.isclose(grouped.z.transform(min), 0),
        )

        # Get minimum and maximum warp values
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

        # Calculate eigenfrequency
        eigenfrequency = np.sqrt(eigenvalues[i].real) / 2 / np.pi

        # Convert warp values to integers and set small values to zero
        min_x_warp = minimum.x_warp.to_numpy(int)
        min_x_warp[np.abs(min_x_warp) < 50] = 0
        max_x_warp = maximum.x_warp.to_numpy(int)
        max_x_warp[np.abs(max_x_warp) < 50] = 0

        # Store results in DataFrame
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

    # Sort results by eigenfrequency
    df_results = df_results.sort_values("eigenfrequency")

    try:
        # Find the first longitudinal mode
        no_of_first_longitudinal_mode = df_results.loc[
            np.logical_and(
                df_results.symmetric == True,
                np.logical_and(
                    df_results.eigenfrequency <= target_frequency * 1.5,
                    df_results.eigenfrequency >= target_frequency * 0.5,
                ),
            )
        ].index.to_numpy()[0]
    except IndexError:
        raise ValueError("No longitudinal mode found.")

    print(
        "Mode {0} is the first longitudinal mode with a frequency of {1} Hz.".format(
            no_of_first_longitudinal_mode,
            df_results.eigenfrequency[no_of_first_longitudinal_mode],
        )
    )

    return no_of_first_longitudinal_mode
