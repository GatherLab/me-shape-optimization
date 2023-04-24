import numpy as np
import pylab as plt
import pandas as pd
import imageio

from solver import LinearElasticity

import dolfin as fe


def myround(x, base=5):
    """
    Function to round to the next arbitrary base
    """
    return base * np.round(x / base)


def do_simulation(mesh, E, nu, rho, target_frequency, no_eigenstates):
    """
    Function that pacakges the simulation (can be just called to receive a
    resonance frequency)
    """

    # Init class
    linear_elasticity = LinearElasticity(E, nu, rho, mesh)

    # Init vector and function spaces using a mesh
    V = linear_elasticity.init_geometry()

    # Init boundary conditions
    # No z movement allowed at all
    def fixed_z(x, on_boundary):
        # return x[0] >=-1
        return on_boundary

    # No x movement allowed at all
    def fixed_y(x, on_boundary):
        # return x[0] >=-1
        return on_boundary

    bc = fe.DirichletBC(V.sub(2), fe.Constant(0.0), fixed_z)
    bc2 = fe.DirichletBC(V.sub(1), fe.Constant(0.0), fixed_y)

    linear_elasticity.init_dirichlet_boundary_conditions([bc, bc2])

    # Solve for eigenstates/values and save to file
    eigensolver = linear_elasticity.init_eigensolver(target_frequency)
    eigenfrequency = linear_elasticity.solve_eigenstates(eigensolver, no_eigenstates)
    return eigenfrequency


# Save the generated geometry together with its resonance frequency for
# later reference
def plot_shape(mesh, eigenfrequency, folder, L, save=True):
    """
    Function to bundles the saving of the data (as a picture with the resonance
    frequency in its name and a text file containing the features.)
    """
    # The built-in plot function of fenics doesn't work well. Here is a custom
    # solution using matplotlib
    xyz = mesh.coordinates()
    x = xyz[:, 0]
    y = xyz[:, 1]
    # z = xyz[:, 2]

    plt.clf()

    # 2D plot is enough for this case since the height cannot be varied
    plt.scatter(y, x, marker="_")
    plt.text(
        0.25 * L,
        0.9 * L,
        str(eigenfrequency) + " Hz",
        fontsize=12,
        color="red",
    )
    plt.ylim([0, L])
    plt.xlim([-L / 2, L / 2])

    # If the user doesn't want to save, just show the image
    if save:
        plt.savefig(
            folder + "shape_{0:.0f}_Hz.png".format(round(eigenfrequency, 0)),
            bbox_inches="tight",
        )
    else:
        plt.show()


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


def read_features(file_path, number_features):
    """ """
    columns = np.append(
        np.array(["eigenfrequency"]),
        np.char.add("w", np.arange(0, number_features, 1).astype("str")),
    )

    resonances_and_features = pd.read_csv(
        file_path, skiprows=5, delimiter="\t", names=columns
    )
    return resonances_and_features


def generate_gif(folder_path, frequencies):
    """
    Generate a gif from the generated images
    """
    # Repeat last frame so that it is shown for a while
    frequencies = np.append(frequencies, np.repeat(frequencies[-1], 15))

    with imageio.get_writer(folder_path + "time_laps.gif", mode="I") as writer:
        for eigenfrequency in frequencies:
            filename = folder_path + "shape_{0:.0f}_Hz.png".format(
                round(eigenfrequency, 0)
            )
            image = imageio.imread(filename)
            writer.append_data(image)
