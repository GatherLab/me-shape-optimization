# import fenics as fe
import dolfin as fe
import numpy as np
import pylab as plt
import pandas as pd

import time

from scipy.optimize import curve_fit

import core_functions as cf

# Source: https://fenics-solid-tutorial.readthedocs.io/en/latest/EigenvalueProblem/EigenvalueProblem.html


def func(x, a, b, c):
    return a * (x - b) ** 2 + c


class LinearElasticity:
    def __init__(self, E, nu, rho, mesh):
        # Lame's constants
        self.mu = E / 2.0 / (1 + nu)
        self.lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
        self.rho = rho
        self.mesh = mesh

    def init_geometry(self):
        # --------------------
        # Function spaces
        # --------------------
        self.V = fe.VectorFunctionSpace(self.mesh, "Lagrange", 1)
        u_tr = fe.TrialFunction(self.V)
        u_test = fe.TestFunction(self.V)

        # --------------------
        # Forms & matrices
        # --------------------
        a_form = fe.inner(self.sigma(u_tr), self.epsilon(u_test)) * fe.dx
        m_form = self.rho * fe.inner(u_tr, u_test) * fe.dx

        self.A = fe.PETScMatrix()
        self.M = fe.PETScMatrix()
        self.A = fe.assemble(a_form, tensor=self.A)
        self.M = fe.assemble(m_form, tensor=self.M)
        return self.V

    def init_dirichlet_boundary_conditions(self, dirichlet_bc):
        # --------------------
        # Dirichlet Boundary conditions
        # --------------------
        for bc in dirichlet_bc:
            bc.apply(self.A)
            bc.apply(self.M)

    # --------------------
    # Functions and classes
    # --------------------
    # Strain function
    def epsilon(self, u):
        return 0.5 * (fe.nabla_grad(u) + fe.nabla_grad(u).T)

    # Stress function
    def sigma(self, u):
        return self.lmbda * fe.div(u) * fe.Identity(3) + 2 * self.mu * self.epsilon(u)

    # --------------------
    # Eigensolver
    # --------------------
    def init_eigensolver(self, target_frequency=100000.0):
        """
        Initialise eigensolver
        """
        self.target_frequency = target_frequency
        eigensolver = fe.SLEPcEigenSolver(self.A, self.M)
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["spectrum"] = "target real"
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = target_frequency**2 * 2 * fe.pi
        return eigensolver

    def determine_relevant_mode(self, eigenmode):
        """
        Function to evaluate which of the calculated modes is the most relevant
        """

        # This section is to plot the modes and estimate which one is
        # the one I am interested in
        vector_field = np.reshape(
            eigenmode.vector().get_local(),
            [int(eigenmode.vector().get_local().size / 3), 3],
        )
        # calculate magnitude
        mag = np.linalg.norm(vector_field, axis=1)
        # Get running average of the magnitude
        N = 5000
        averaged_magnitude = np.convolve(mag, np.ones(N) / N, mode="valid")
        # Now I only have to somehow assess the shape of the mode
        # plt.plot(averaged_magnitude, label=str(df.loc[i, "eigenfrequency"]))
        popt, pcov = curve_fit(
            func,
            np.arange(0, np.size(averaged_magnitude), 1),
            averaged_magnitude,
            bounds=(
                [0, 0.4 * np.size(averaged_magnitude), 0],
                [np.inf, 0.6 * np.size(averaged_magnitude), np.inf],
            ),
        )
        # plt.plot(
        #     func(np.arange(0, np.size(averaged_magnitude), 1), *popt), "--"
        # )

        residuals = averaged_magnitude - func(
            np.arange(0, np.size(averaged_magnitude), 1), *popt
        )
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((averaged_magnitude - np.mean(averaged_magnitude)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return r_squared

    def solve_eigenstates(self, eigensolver, N_eig=10):
        """
        Solve for the N first eigenstates
        """
        starting_time = time.time()
        print("Solving for eigenstates")
        eigensolver.solve(N_eig)

        # file_results = fe.XDMFFile("MA.xdmf")
        # file_results.parameters["flush_output"] = True
        # file_results.parameters["functions_share_mesh"] = True

        eigenmodes = []
        df = pd.DataFrame(
            columns=["eigenmode", "eigenfrequency", "norm", "maximum", "r_squared"]
        )

        # Eigenfrequencies
        for i in range(0, N_eig):
            # Get i-th eigenvalue and eigenvector
            # r - real part of eigenvalue
            # c - imaginary part of eigenvalue
            # rx - real part of eigenvector
            # cx - imaginary part of eigenvector
            r, c, rx, cx = eigensolver.get_eigenpair(i)

            # Calculation of eigenfrequency from real part of eigenvalue
            df.loc[i, "eigenfrequency"] = fe.sqrt(r) / 2 / fe.pi
            # Initialize function and assign eigenvector
            # eigenmode = fe.Function(V, name="Eigenvector " + str(i))
            eigenmode = fe.Function(self.V)
            eigenmode.vector()[:] = rx
            df.loc[i, "eigenmode"] = rx

            # Calculate vector norm which can be regarded as a measure for magnitude of elongation
            df.loc[i, "norm"] = fe.norm(eigenmode)
            df.loc[i, "maximum"] = eigenmode.vector().max()

            # Execute

            # This was the attempt of averaging over z-y coordinates to get a clean 1-D graph
            # mesh_coord = self.mesh.coordinates()
            # values, counts = np.unique(np.round(np.sort(mesh_coord, axis = 0)[:,0], decimals = 5), return_counts = True)
            # b = np.split(mag, np.cumsum(counts))
            # a = [np.mean(arr) for arr in b]
            r_squared = 0

            if df.loc[i, "norm"] > 0.005 and (
                df.loc[i, "eigenfrequency"].real > 20000
                and df.loc[i, "eigenfrequency"].real < 220000
            ):
                # and maximum > 1:
                # Write i-th eigenfunction to xdmf file only if the norm surpasses a
                # certain value (otherwise it is probably only jitter)
                # eigenmode.rename(str(df.loc[i, "eigenfrequency"]), "")
                # file_results.write(eigenmode, 0)

                # Determine relevant mode using r_squared of a quadratic fit
                r_squared = self.determine_relevant_mode(eigenmode)

            df.loc[i, "r_squared"] = r_squared

            # print(
            #     "Eigenfrequency {0}: {1:8.5f} [Hz], with norm {2:8.14f}, and max {3} and residuals {4}".format(
            #         i,
            #         df.loc[i, "eigenfrequency"],
            #         df.loc[i, "norm"],
            #         df.loc[i, "maximum"],
            #         df.loc[i, "r_squared"],
            #     )
            # )

        # Sort dataframe in descending order
        df.sort_values("r_squared", inplace=True, ascending=False)
        # print(df)

        # Round the dominant eigenfrequency to about 10 Hz, which seems to be
        # about the spread of the solver accuracy
        dominant_eigenfrequency = cf.myround(df.eigenfrequency.to_numpy()[0], 10)
        time_elapsed = time.time() - starting_time
        # print(df)
        print(
            "Dominant eigenfrequency: {0:.1f} Hz (in {1:.2f} s)".format(
                dominant_eigenfrequency, time_elapsed
            )
        )

        # Sometimes the relevant eigenfrequency is not on the list so
        # retriggering with a higher N is imperative
        if dominant_eigenfrequency < self.target_frequency / 5:
            print(
                "Relevant eigenfrequency not found, repeating to solve with twice the number of eigenvalues."
            )
            dominant_eigenfrequency = self.solve_eigenstates(
                eigensolver, int(2 * N_eig)
            )

        return dominant_eigenfrequency

        # Plot the profiles
        # plt.legend()
        # plt.show()

        # Plot the most relevant mode

    def compute_mises_stress(self, eigenmode):
        # Compute Mises Stress
        deviatoric_stress_tensor = self.sigma(eigenmode) - 1 / 3 * fe.tr(
            self.sigma(eigenmode)
        ) * fe.Identity(eigenmode.geometric_dimension())

        # This is a scalar field
        van_mises_stress = fe.sqrt(
            3 / 2 * fe.inner(deviatoric_stress_tensor, deviatoric_stress_tensor)
        )

        # New function space
        lagrange_scalar_space_first_order = fe.FunctionSpace(self.mesh, "CG", 2)

        van_mises_stress = fe.project(
            van_mises_stress, lagrange_scalar_space_first_order
        )

        return van_mises_stress
