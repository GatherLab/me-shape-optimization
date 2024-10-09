import numpy as np
from dolfinx import fem, plot
import pyvista
import imageio

# Start pyvista with off-screen rendering
pyvista.start_xvfb()


def visualise_3D(
    plotter,
    V,
    eigenvalues,
    eigenmodes,
    mode_number,
    saving_path,
    viewup=False,
    high_res=False,
):
    """
    Plot the resonance mode in 3D with a warping according to the displacement.

    Parameters:
    plotter (pyvista.Plotter): The pyvista plotter object.
    V (dolfinx.fem.FunctionSpace): The function space.
    eigenvalues (np.ndarray): Array of eigenvalues.
    eigenmodes (np.ndarray): Array of eigenmodes.
    mode_number (int): The mode number to visualize.
    saving_path (str): Path to save the visualization.
    viewup (bool): Whether to set the view up direction.
    high_res (bool): Whether to use high resolution for the plot.
    """
    # Clear plotter before next plot
    plotter.clear()

    eigenvalue = eigenvalues[mode_number]
    eigenmode = eigenmodes[mode_number]

    # Calculate eigenfrequency from the real part of the eigenvalue
    freq_3D = np.sqrt(eigenvalue.real) / (2 * np.pi)

    # Create pyvista grid
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = eigenmode.x.array.reshape((geometry.shape[0], 3))

    # Add mesh to plotter and warp by vector
    if high_res:
        actor_0 = plotter.add_mesh(grid, style="wireframe", color="k", line_width=8)
        warped = grid.warp_by_vector("u", factor=2e-6)
        actor_1 = plotter.add_mesh(warped, show_edges=True, line_width=8)
    else:
        actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
        warped = grid.warp_by_vector("u", factor=2e-6)
        actor_1 = plotter.add_mesh(warped, show_edges=True)

    plotter.show_axes()

    # Set view direction
    if viewup:
        plotter.view_xz()
    else:
        actor_0.rotate_x(100)
        actor_1.rotate_x(100)
        actor_0.rotate_z(10)
        actor_1.rotate_z(10)

    # Add frequency text to plot
    plotter.add_text(
        "{0:.2f} Hz".format(freq_3D),
        position="upper_left",
        color="red",
        font_size=10,
    )

    # Show or save the plot
    if not plotter.off_screen:
        plotter.show()
    else:
        if high_res:
            plotter.screenshot(
                saving_path, window_size=[8000, 6400], transparent_background=True
            )
        else:
            plotter.screenshot(saving_path)

    print("Resonance Frequency: {0}".format(freq_3D))


def visualise_mesh(mesh, saving_path):
    """
    Plot the mesh in 3D.

    Parameters:
    mesh (dolfinx.Mesh): The mesh to visualize.
    saving_path (str): Path to save the visualization.
    """
    V = fem.VectorFunctionSpace(mesh, ("CG", 2))
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    p = pyvista.Plotter(off_screen=True)
    p.add_mesh(grid, style="wireframe")
    p.show_axes()

    if not p.off_screen:
        p.show()
    else:
        p.screenshot(saving_path)


def append_feature(eigenfrequency, file_path, horizontal_lengths):
    """
    Append meta data about the shape (resonance frequency & weights) to features file.

    Parameters:
    eigenfrequency (float): The eigenfrequency to append.
    file_path (str): Path to the file where data will be appended.
    horizontal_lengths (np.ndarray): Array of horizontal lengths.
    """
    with open(file_path, "a") as csvfile:
        np.savetxt(
            csvfile,
            [np.append(eigenfrequency, horizontal_lengths)],
            delimiter="\t",
            fmt="%f",
        )


def generate_gif(folder_path, frequencies):
    """
    Generate a GIF from the generated images.

    Parameters:
    folder_path (str): Path to the folder containing images.
    frequencies (np.ndarray): Array of frequencies corresponding to the images.
    """
    print("Started gif generation")
    frequencies = np.append(frequencies, np.repeat(frequencies[-1], 15))
    print(frequencies)

    with imageio.get_writer(folder_path + "/time_laps.gif", mode="I") as writer:
        for i, eigenfrequency in enumerate(frequencies, start=1):
            print("{0}/{1} done".format(i, len(frequencies)))
            filename = f"{folder_path}/{eigenfrequency:.2f}.png"
            image = imageio.imread(filename)
            writer.append_data(image)
