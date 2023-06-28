import numpy as np
import matplotlib.pylab as plt
from dolfinx import fem, plot

from dolfinx import fem

import pyvista

pyvista.start_xvfb()


def visualise_3D(V, eigenvalues, eigenmodes, mode_number, saving_path, viewup=False):
    """
    Function to plot the resonance mode in 3D with a warping according to the
    displacement
    """
    eigenvalue = eigenvalues[mode_number]
    eigenmode = eigenmodes[mode_number]

    # Calculation of eigenfrequency from real part of eigenvalue
    freq_3D = np.sqrt(eigenvalue.real) / 2 / np.pi

    # Create plotter and pyvista grid
    # p = pyvista.Plotter(off_screen=True)
    p = pyvista.Plotter(off_screen=True)
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = eigenmode.x.array.reshape((geometry.shape[0], 3))
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=2e-6)
    actor_1 = p.add_mesh(warped, show_edges=True)
    p.show_axes()
    if viewup:
        p.view_xz()
    if not p.off_screen:
        p.show()
    else:
        figure_as_array = p.screenshot(saving_path)

    pyvista.close_all()

    print("Solid FE: {0:8.5f} [Hz]".format(freq_3D))


def visualise_mesh(mesh, saving_path):
    """
    Function to plot the resonance mode in 3D with a warping according to the
    displacement
    """
    # Attach vector values to grid and warp grid by vector
    V = fem.VectorFunctionSpace(mesh, ("CG", 2))
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    p = pyvista.Plotter(off_screen=True)
    actor_0 = p.add_mesh(grid, style="wireframe")
    p.show_axes()
    if not p.off_screen:
        p.show()
    else:
        figure_as_array = p.screenshot(saving_path)


def append_feature(eigenfrequency, file_path, horizontal_lengths):
    """
    Function to append meta data about the shape (resonance frequency & weights)
    to features file
    """
    # Additionally save the parameters of the optimization in a .txt file
    with open(file_path, "a") as csvfile:
        np.savetxt(
            csvfile,
            [np.append(eigenfrequency, horizontal_lengths)],
            delimiter="\t",
            fmt="%f",
        )
