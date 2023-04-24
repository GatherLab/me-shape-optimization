import pandas as pd
import numpy as np

from shape_generation import Geometry
import core_functions as cf

# --------------------
# Parameter Definitions
# --------------------

# Define material parameters
E, nu = 5.4e10, 0.34
rho = 7950.0

# Define maximum length, width and height
Lmax, Bmax, Hmax = 12e-3, 3e-3, 0.2e-3
# Define minimum width
Bmin = 1e-3
# Define initial width
Binit = 2e-3
# Defines the number of adjustable parts
grid_size = 0.5e-3
# Accuracy of the adjustment (defined by the real world, e.g. less than 10 um
# accuracy doesn't make sense)
accuracy = 10e-6

# Learning rate, step size to determine gradient and maximum number of
# optimization steps
neighbour_step = 200e-6
no_optimization_steps = 1000
boltzmann_constant = 100

# Target frequency and number of eigenstates to compute (for solver)
target_frequency = 100000
no_eigenstates = 10

# Folder to save to
folder = "./simulated_annealing3/"


# --------------------
# Initialise Data Dump
# --------------------

# Generate a file that contains the meta data in the header
line00 = "Material: Youngs modulus (E) = {0} N/m2, Poissons ratio(nu) = {1}, density (rho) = {2} kg/m3".format(
    E, nu, rho
)
line01 = "Geometry: L = {0} mm, Bmin = {2} mm, Bmax = {1} mm, H = {2} mm, grid size = {3} mm, accuracy = {4} mm".format(
    Lmax * 1e3, Bmax * 1e3, Bmin * 1e3, Hmax * 1e3, grid_size * 1e3, accuracy * 1e3
)
line02 = "Simulation: discrete neighbour step = {0} mm, max optimization steps = {1}, Binit = {2} mm, Boltzmann constant = {3}".format(
    neighbour_step * 1e3, no_optimization_steps, Binit * 1e3, boltzmann_constant
)
line03 = "### Simulation Results ###"
line04 = "Resonance Frequency (Hz)\t Features (m)\n"

header_lines = [line00, line01, line02, line03, line04]

with open(folder + "features.csv", "a+") as csvfile:
    csvfile.write("\n".join(header_lines))

# --------------------
# Initialise Geometry
# --------------------

# Randomly initialize geometry
geometry = Geometry(Lmax, Bmax, Hmax, grid_size, accuracy, Bmin)
# geometry.generate_random_vertical_length()
# geometry.generate_random_horizontal_length()
geometry.init_rectangular_geometry(Binit)

# Dataframe that saves the parameters and the resonance frequencies
resonance_frequency_df = pd.DataFrame(
    columns=np.append(
        ["resonance_frequency"],
        np.char.add(
            "w", np.arange(0, np.size(geometry.horizontal_lengths), 1).astype("str")
        ),
    )
)

# geometry.generate_boolean_grid()
geometry.generate_mesh()

initial_eigenfrequency = cf.do_simulation(
    geometry.mesh, E, nu, rho, target_frequency, no_eigenstates
)

resonance_frequency_df.loc[resonance_frequency_df.shape[0]] = np.append(
    initial_eigenfrequency, geometry.horizontal_lengths
)

cf.plot_shape(geometry.mesh, initial_eigenfrequency, folder, Lmax)
cf.append_feature(
    initial_eigenfrequency, folder + "features.csv", geometry.horizontal_lengths
)

# Set initially guessed solution
approximate_solution = geometry.horizontal_lengths
approximate_best_eigenfrequency = initial_eigenfrequency


# --------------------
# Simulated Annealing
# --------------------

# Define list of temperatures (logspace to allow for quicker cool down at the beginning)
temperatures = np.linspace(100, 0, no_optimization_steps + 1)
# temperatures = np.logspace(2, -1, no_optimization_steps + 1)

for step in range(np.size(temperatures)):
    print("--- Optimization step {0}/{1} ---".format(step + 1, np.size(temperatures)))
    print("Temperature: {0}".format(temperatures[step]))

    # Choose a random neighbour to the current state this is done by choosing
    # random integers from [-5, 5] (numpy has a weird notation for this) and
    # multiply by a step width defined by the user (either decrease, leave or
    # increase the value of the feature).
    geometry.adjust_horizontal_length(
        geometry.horizontal_lengths
        + neighbour_step
        * np.random.randint(-1, 2, np.size(geometry.horizontal_lengths))
    )

    # Generate the geometry again and do a simulation. Save image
    # geometry.generate_boolean_grid()
    geometry.generate_mesh()

    eigenfrequency = cf.do_simulation(
        geometry.mesh, E, nu, rho, target_frequency, no_eigenstates
    )

    if eigenfrequency < approximate_best_eigenfrequency:
        approximate_solution = geometry.horizontal_lengths
        approximate_best_eigenfrequency = eigenfrequency

        resonance_frequency_df.loc[resonance_frequency_df.shape[0]] = np.append(
            initial_eigenfrequency, geometry.horizontal_lengths
        )

        cf.plot_shape(geometry.mesh, eigenfrequency, folder, Lmax)
        cf.append_feature(
            eigenfrequency, folder + "features.csv", geometry.horizontal_lengths
        )
    else:
        # This is always smaller than 1 since eigenfrequency is >= approximate_solution
        # There is an additional factor that serves as a replacement for kB to
        # normalize the results so that the Boltzmann factor yields sensible
        # results for a given set of temperatures. The other option is to use
        # (non physical) higher temperatures instead.
        boltzmann_factor = np.exp(
            -(eigenfrequency - approximate_best_eigenfrequency)
            / boltzmann_constant
            / temperatures[step]
        )
        print("Boltzmann Factor: {0}".format(boltzmann_factor))

        # Accept the solution randomly depending on boltzmann factor. If
        # boltzmann factor = 1, 100% chance the solution is accepted, otherwise
        # it decreases
        if np.random.uniform(0, 1, 1) <= boltzmann_factor:
            approximate_solution = geometry.horizontal_lengths
            approximate_best_eigenfrequency = eigenfrequency

            resonance_frequency_df.loc[resonance_frequency_df.shape[0]] = np.append(
                initial_eigenfrequency, geometry.horizontal_lengths
            )

            cf.plot_shape(geometry.mesh, eigenfrequency, folder, Lmax)
            cf.append_feature(
                eigenfrequency, folder + "features.csv", geometry.horizontal_lengths
            )
