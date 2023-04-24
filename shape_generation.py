from dolfin import *
from mshr import *

import pylab as plt
import numpy as np
import pandas as pd

import core_functions as cf


class Geometry:
    def __init__(self, Lmax, Bmax, Hmax, grid_size, accuracy=10e-6, Bmin=5e-4):
        """
        Init geometry with length, width, height and its features
        """
        # Define maximum and minimum shape parameters
        self.Lmax = Lmax
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.Hmax = Hmax

        # Define the equally spaced grid that define the geometry
        self.grid_size = grid_size
        self.horizontal_lengths = []

        self.accuracy = accuracy

    def init_rectangular_geometry(self, B):
        """
        Function to generate the "standard" rectangular geometry without any fancy features
        """
        if B > self.Bmax or B < self.Bmin:
            raise ValueError("Width must be in the range set on initialization.")

        self.horizontal_lengths = np.repeat(B, int((self.Lmax) / self.grid_size))

    def generate_random_horizontal_length(self, B):
        """
        Function to initiate the horizontal lengths randomly
        """
        if B > self.Bmax or B < self.Bmin:
            raise ValueError("Width must be in the range set on initialization.")

        self.horizontal_lengths = cf.myround(
            np.random.uniform(self.Bmin, B, size=int((self.Lmax) / self.grid_size)),
            self.accuracy,
        )

    def generate_mesh(self):
        """
        Generate the actual mesh from continuous values of horizontal_lengths
        The idea is to now just add boxes representing the width.
        """
        geometry_gen = Box(
            Point(0, -self.horizontal_lengths[0] / 2, 0),
            Point(self.grid_size, self.horizontal_lengths[0] / 2, self.Hmax),
        )

        i = 1
        for horizontal_length in self.horizontal_lengths[1:]:
            geometry_gen += Box(
                Point(i * self.grid_size, -horizontal_length / 2, 0),
                Point((i + 1) * self.grid_size, horizontal_length / 2, self.Hmax),
            )
            i += 1

        # Create mesh
        self.mesh = generate_mesh(geometry_gen, 200)

    def adjust_horizontal_length(self, new_horizontal_lengths):
        """
        Function to adjust the horizontal lengths bearing in mind to respect the
        boundary conditions of maximum and minimum width.
        """
        self.horizontal_lengths = cf.myround(
            np.maximum(np.minimum(new_horizontal_lengths, self.Bmax), self.Bmin),
            self.accuracy,
        )


"""

# Define length width and height (x, y, z)
L, B, H = 11e-3, 3e-3, 0.2e-3
grid_size = 0.5e-3


# Generate geometry object without features
geometry = Geometry(L, B, H, grid_size)
geometry.generate_random_horizontal_length_continuous()
# geometry.generate_boolean_grid()
geometry.generate_mesh_continuous()

plot(geometry.mesh)
plt.ylim([0, L])
plt.show()

# Generate boolean grid (required for mesh generation in this implementation)
# Plot boolean grid
# plt.imshow(geometry.boolean_grid)
# plt.show()


# Initialize features randomly
for i in range(100):
    geometry.generate_random_vertical_length()
    geometry.generate_random_horizontal_length()
    geometry.generate_boolean_grid()
    plt.imshow(geometry.boolean_grid)
    plt.savefig("./images/shape_test{0}.png".format(i), bbox_inches="tight")


# Generate mesh (from boolean grid)

# Plot mesh
plot(geometry.mesh)
plt.ylim([0, 11])
plt.show()
"""
