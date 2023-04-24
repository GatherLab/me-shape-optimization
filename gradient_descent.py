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
Binit = 1.5e-3
# Defines the number of adjustable parts
grid_size = 0.5e-3
# Accuracy of the adjustment (defined by the real world, e.g. less than 10 um
# accuracy doesn't make sense)
accuracy = 10e-6

# Learning rate, step size to determine gradient and maximum number of
# optimization steps
learning_rate = 2 * 1e-10
grad_step = 100e-6
no_optimization_steps = 50

# Target frequency and number of eigenstates to compute (for solver)
target_frequency = 100000
no_eigenstates = 15

# Folder to save to
folder = "./sim-24-segments-after-annealing/"


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
line02 = "Simulation: learning rate = {0}, discrete gradient step = {1} mm, max optimization steps = {2}, Binit = {3} mm".format(
    learning_rate, grad_step * 1e3, no_optimization_steps, Binit * 1e3
)
line03 = "### Simulation Results ###"
line04 = "Resonance Frequency (Hz)\t Features (m)\n"

header_lines = [line00, line01, line02, line03, line04]

with open(folder + "features.csv", "a+") as csvfile:
    csvfile.write("\n".join(header_lines))

# --------------------
# Generate initial geometry
# --------------------

"""
# Randomly initialize geometry
geometry = Geometry(Lmax, Bmax, Hmax, grid_size, accuracy, Bmin)
# geometry.generate_random_vertical_length()
# geometry.generate_random_horizontal_length()
geometry.init_rectangular_geometry(Binit)
# geometry.generate_boolean_grid()
geometry.generate_mesh()
"""
# --------------------
# Or read in from file
# --------------------
feature_file = "./simulated_annealing3/features.csv"
# Read in features from file
number_features = 24
resonances_and_features = cf.read_features(feature_file, number_features)

# Select a feature and generate geometry from it
selected_feature = -1

geometry = Geometry(Lmax, Bmax, Hmax, grid_size, accuracy, Bmin)
geometry.adjust_horizontal_length(
    resonances_and_features.iloc[selected_feature].to_numpy()[1:]
)
geometry.generate_mesh()

cf.plot_shape(
    geometry.mesh,
    resonances_and_features.iloc[selected_feature]["eigenfrequency"],
    feature_file,
    Lmax,
    save=False,
)

# --------------------
# Init initial shape
# --------------------

# Dataframe that saves the parameters and the resonance frequencies
resonance_frequency_df = pd.DataFrame(
    columns=np.append(
        ["eigenfrequency"],
        np.char.add(
            "w", np.arange(0, np.size(geometry.horizontal_lengths), 1).astype("str")
        ),
    )
)


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


# --------------------
# Gradient Descent Optimization
# --------------------


def calculate_gradient(geometry, grad_step, initial_eigenfrequency):
    """
    Function to calculate gradient that in this discrete case is defined as
        \nabla f(x) \approx f(x+h) - f(x) / h
    with h = grad_step.
    The function here is the resonance frequency that depends on the shape x
    given as a vector of N dimensions. We slightly perturb it in each direction
    (requires to calculate N values) and in the next step then apply a gradient
    descent.

    The unit is Hz/m and can be converted to kHz/mm by dividing with a factor of 1e6
    """

    gradient_vector = np.empty(np.size(geometry.horizontal_lengths))

    # Iterate over all dimensions and perturb by grad_step
    for i in range(np.size(geometry.horizontal_lengths)):
        # Give a status of how far the calulation of the gradient is
        print(
            ">>> Calculating gradient in dimension {0}/{1} <<<".format(
                i + 1, np.size(geometry.horizontal_lengths)
            )
        )

        # Save old value because we want to revert back after the simulation (since
        # this only calculates the gradient for now)
        old_value = geometry.horizontal_lengths[i]

        # The standard action will be to decrease the value to obtain the
        # gradient. If this is not possible (due to the boundary conditions),
        # increase the value
        if geometry.horizontal_lengths[i] - grad_step >= geometry.Bmin:
            geometry.horizontal_lengths[i] = geometry.horizontal_lengths[i] - grad_step
            sign = 1
        else:
            geometry.horizontal_lengths[i] = geometry.horizontal_lengths[i] + grad_step
            sign = -1

        # Regenerate geometry
        # geometry.generate_boolean_grid()
        geometry.generate_mesh()

        # Do the actual simulation to obtain a resonance frequency
        eigenfrequency = cf.do_simulation(
            geometry.mesh, E, nu, rho, target_frequency, no_eigenstates
        )

        # Calculate partial derivative
        gradient_vector[i] = (
            sign * (eigenfrequency - initial_eigenfrequency) / grad_step
        )

        # Safe data in dataframe
        resonance_frequency_df.loc[resonance_frequency_df.shape[0]] = np.append(
            eigenfrequency, geometry.horizontal_lengths
        )

        # Revert geometry to initial value
        geometry.horizontal_lengths[i] = old_value

    return gradient_vector


for step in range(no_optimization_steps):
    print("--- Optimization step {0}/{1} ---".format(step + 1, no_optimization_steps))

    # Determine gradient
    gradient = calculate_gradient(geometry, grad_step, initial_eigenfrequency)

    # Adjust by gradient descent (this has to be done with the class to respect
    # the boundary conditions)
    geometry.adjust_horizontal_length(
        geometry.horizontal_lengths + learning_rate * gradient
    )
    print(learning_rate * gradient)

    # Generate the geometry again and do a simulation. Save image
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

    # Condition to define the end of the optimization
    # If all values are down to the accuracy value close to the previous
    # solution, the end is reached
    # if np.all(
    #    np.isclose(
    #        resonance_frequency_df.iloc[-2].to_numpy(),
    #        resonance_frequency_df.iloc[-1].to_numpy(),
    #        atol=accuracy,
    #    )
    # ):
    #    break

cf.generate_gif(folder, resonance_frequency_df["eigenfrequency"].to_numpy())
