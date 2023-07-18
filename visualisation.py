import numpy as np
import matplotlib.pylab as plt
from dolfinx import fem, plot

from dolfinx import fem

import pyvista

pyvista.start_xvfb()

import imageio


def visualise_3D(
    plotter, V, eigenvalues, eigenmodes, mode_number, saving_path, viewup=False
):
    """
    Function to plot the resonance mode in 3D with a warping according to the
    displacement
    """
    # Clear plotter before next stuff can be plotted
    plotter.clear()

    eigenvalue = eigenvalues[mode_number]
    eigenmode = eigenmodes[mode_number]

    # Calculation of eigenfrequency from real part of eigenvalue
    freq_3D = np.sqrt(eigenvalue.real) / 2 / np.pi

    # Create plotter and pyvista grid
    if saving_path == None:
        p = pyvista.Plotter(off_screen=True)

    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = eigenmode.x.array.reshape((geometry.shape[0], 3))
    actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=2e-6)
    actor_1 = plotter.add_mesh(warped, show_edges=True)
    plotter.show_axes()
    if viewup:
        plotter.view_xz()

        plotter.add_text(
            "{0:.2f} Hz".format(freq_3D),
            position="upper_left",
            color="white",
            font_size=10,
        )
    if not plotter.off_screen:
        plotter.show()
    else:
        figure_as_array = plotter.screenshot(saving_path)

    # p.close()
    # p.deep_clean()
    # pyvista.close_all()

    print("Resonance Frequency: {0}".format(freq_3D))


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


def append_text(text, file_path):
    """
    Function to append meta data about the shape (resonance frequency & weights)
    to features file
    """
    # Additionally save the parameters of the optimization in a .txt file
    with open(file_path, "a") as csvfile:
        csvfile.write("\n" + text)


def generate_gif(folder_path, frequencies):
    """
    Generate a gif from the generated images
    """
    # Repeat last frame so that it is shown for a while
    frequencies = np.append(frequencies, np.repeat(frequencies[-1], 15))

    with imageio.get_writer(folder_path + "/time_laps.gif", mode="I") as writer:
        for eigenfrequency in frequencies:
            filename = folder_path + "/{0:.2f}.png".format(eigenfrequency)
            image = imageio.imread(filename)
            writer.append_data(image)
