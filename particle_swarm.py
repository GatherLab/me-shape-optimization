import pandas as pd
import numpy as np

from shape_generation import Geometry
import core_functions as cf

import os

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
# learning_rate = 2 * 1e-10
# grad_step = 100e-6
no_optimization_steps = 500

# Target frequency and number of eigenstates to compute (for solver)
target_frequency = 100000
no_eigenstates = 10


# Update velocity parameters
no_particles = 20

# in m/step
min_velocity = -1e-3
max_velocity = 1e-3

# Loop over different combinations of inertia, cognitive_weight and social weight:
weights = []

weights.append([1.4, 1.0, 1.5])
# weights.append([1.4, 1.0, 1.2])
# weights.append([1.4, 1.0, 1.3])
# weights.append([1.4, 1.0, 1.4])

# --------------------
# Randomly initialize P "particles" with P velocities
# --------------------

weights = np.array(weights)

index = 0
for weight in weights:
    # Create a new folder
    folder = os.path.join("./img/particle_swarm/", "particle-swarm5{0}/".format(index))
    os.makedirs(folder, exist_ok=True)

    # Should be lower than 1 (typically between 0.4 and 0.9)
    inertia = weight[0]

    # Typically between 1.5 and 2
    cognitive_weight = weight[1]
    social_weight = weight[2]

    # Folder to save to
    # folder = "./particle-swarm6/"

    # --------------------
    # Initialise Data Dump
    # --------------------

    # Generate a file that contains the meta data in the header
    line00 = "Material: Youngs modulus (E) = {0} N/m2, Poissons ratio(nu) = {1}, density (rho) = {2} kg/m3".format(
        E, nu, rho
    )
    line01 = "Geometry: L = {0} mm, Bmin = {1} mm, Bmax = {2} mm, H = {3} mm, grid size = {4} mm, accuracy = {5} mm".format(
        Lmax * 1e3, Bmin * 1e3, Bmax * 1e3, Hmax * 1e3, grid_size * 1e3, accuracy * 1e3
    )
    line02 = "Simulation: inertia = {0}, cognitive weight = {1}, social weight = {2}, min velocity = {3} mm/it, max velocity = {4} mm/it, no particles = {5}, no optimization steps = {6}".format(
        inertia,
        cognitive_weight,
        social_weight,
        min_velocity * 1e3,
        max_velocity * 1e3,
        no_particles,
        no_optimization_steps,
    )
    line03 = "### Simulation Results ###"
    line04 = "Resonance Frequency (Hz)\t Features (m)\n"

    header_lines = [line00, line01, line02, line03, line04]

    for i in range(no_particles):
        with open(folder + "features_p" + str(i) + ".csv", "a+") as csvfile:
            csvfile.write("\n".join(header_lines))

    # --------------------
    # Randomly initialize P "particles" with P velocities
    # --------------------
    # Use a standard array first because appending is more straight forward
    particles = []

    for i in range(no_particles):
        # Randomly initialize geometry
        geometry = Geometry(Lmax, Bmax, Hmax, grid_size, accuracy, Bmin)
        geometry.generate_random_horizontal_length(Bmax)
        geometry.generate_mesh()

        # It is best to intialise with zero velocity to prevent random movement out
        # of its search space
        # velocity = cf.myround(
        # np.random.uniform(min_velocity, max_velocity, size=int((Lmax) / grid_size)),
        # accuracy,
        # )
        velocity = np.zeros(int((Lmax) / grid_size))
        eigenfrequency = 0
        particles.append([geometry, velocity, eigenfrequency])

    particles = np.array(particles, dtype=object)

    # --------------------
    # Init initial shape
    # --------------------

    # Dataframe that saves the parameters and the resonance frequencies of the
    # global optimum only!
    resonance_frequency_df = pd.DataFrame(
        columns=np.append(
            ["particle_number", "eigenfrequency"],
            np.char.add(
                "w", np.arange(0, np.size(geometry.horizontal_lengths), 1).astype("str")
            ),
        )
    )

    # Iterate over optimization steps
    for iteration in range(no_optimization_steps):
        print(
            "--- Optimization step {0}/{1} ---".format(
                iteration + 1, no_optimization_steps
            )
        )
        i = 0

        # Go through all geometries
        for geometry, velocity, eigenfrequency in particles:
            print(
                ">>> Calculating eigenfrequency for particle {0}/{1} <<<".format(
                    i + 1, no_particles
                )
            )
            particles[i, 2], x, y, magnitude = cf.do_simulation(
                particles[i, 0].mesh, E, nu, rho, target_frequency, no_eigenstates
            )

            cf.plot_shape_with_resonance(x, y, magnitude, particles[i, 2], folder, Lmax)

            # The features should be saved in separate files for each particle!
            cf.append_feature(
                particles[i, 2],
                folder + "features_p" + str(i) + ".csv",
                particles[i, 0].horizontal_lengths,
            )

            # Probably better to save all results in a dataframe with an additional
            # particle number column
            resonance_frequency_df.loc[resonance_frequency_df.shape[0]] = np.append(
                [int(i), particles[i, 2]],
                particles[i, 0].horizontal_lengths,
            )

            i += 1

        # Update geometries and velocities (this is only possible after the states
        # of each particle are known)
        i = 0
        for geometry, velocity, eigenfrequency in particles:
            # Update particle positions
            particles[i, 0].adjust_horizontal_length(
                particles[i, 0].horizontal_lengths + particles[i, 1]
            )
            particles[i, 0].generate_mesh()

            # Update velocities
            random_factor_1 = cf.myround(
                np.random.uniform(0, 1, size=int((Lmax) / grid_size)), accuracy
            )
            random_factor_2 = cf.myround(
                np.random.uniform(0, 1, size=int((Lmax) / grid_size)), accuracy
            )

            random_factor_1 = cf.myround(np.random.uniform(0, 1, size=1), accuracy)
            # random_factor_2 = cf.myround(np.random.uniform(0, 1, size=1), accuracy)
            # A very complex way to return the individual best that is defined by
            # the minimum resonance frequency achieved for this particle
            individual_best = resonance_frequency_df.loc[
                resonance_frequency_df.loc[
                    resonance_frequency_df.particle_number == i
                ].eigenfrequency.argmin()
            ].to_numpy()[2:]

            global_best = resonance_frequency_df.loc[
                resonance_frequency_df.eigenfrequency.argmin()
            ].to_numpy()[2:]

            # Adjust velocity (with velocity clamping so that it doesnt become too large or small)
            particles[i, 1] = np.maximum(
                np.minimum(
                    max_velocity,
                    inertia * particles[i, 1]
                    + cognitive_weight
                    * random_factor_1
                    * (individual_best - particles[i, 0].horizontal_lengths)
                    + social_weight
                    * random_factor_2
                    * (global_best - particles[i, 0].horizontal_lengths),
                ),
                min_velocity,
            )
            i += 1
    index += 1
