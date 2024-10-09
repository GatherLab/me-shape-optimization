from geometry_generator import generate_gmsh_mesh
from solver import unified_solving_function
from visualisation import visualise_3D, append_feature

import sys
import os
from mpi4py import MPI
import slepc4py
import pyvista
import numpy as np
import gmsh
import pyswarms as ps
import optuna

# Initialize SLEPc
slepc4py.init(sys.argv)

# Start pyvista with off-screen rendering
pyvista.start_xvfb()

# Initialize gmsh
gmsh.initialize()

# Define geometry parameters
L, H, B = 12e-3, 0.2e-3, 3e-3
Bmin = 1e-3
Bmax = B
grid_size = 0.5e-3

# Select boundary conditions to apply
bc_z = True
bc_y = True
no_eigenvalues = 20
target_frequency = 100e3

# Define eigensolver
eigensolver = slepc4py.SLEPc.EPS().create(MPI.COMM_WORLD)
eigensolver.setProblemType(slepc4py.SLEPc.EPS.ProblemType.GHEP)
st = eigensolver.getST()
st.setType(slepc4py.SLEPc.ST.Type.SINVERT)
eigensolver.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.TARGET_REAL)
eigensolver.setTarget(target_frequency**2 * 2 * np.pi)
eigensolver.setDimensions(nev=no_eigenvalues)

# Define pyvista plotter
plotter = pyvista.Plotter(off_screen=True)

# Geometry generation
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Box")
model.setCurrent("Box")

# Particle swarm optimization parameters
no_particles = 20
bounds = (Bmin * np.ones(int(L / grid_size)), Bmax * np.ones(int(L / grid_size)))
min_max_velocities = [-1e-3, 1e-3]

# Set a folder to save the features
folder = "./me-shape-optimization/results/particle-swarm/"
os.makedirs(folder, exist_ok=True)

# Generate a file that contains the meta data in the header
header_lines = [
    f"Geometry: L = {L * 1e3} mm, Bmin = {Bmin * 1e3} mm, Bmax = {Bmax * 1e3} mm, H = {H * 1e3} mm, grid size = {grid_size * 1e3} mm",
    f"Particle swarm optimization with {no_particles} particles, {bounds} mm bounds and {min_max_velocities} min and max velocities",
    "### Simulation Results ###",
    "Resonance Frequency (Hz)\t Features (m)\n",
]

for i in range(no_particles):
    with open(f"{folder}features_p{i}.csv", "a+") as csvfile:
        csvfile.write("\n".join(header_lines))


def opt_function(horizontal_lengths):
    """
    Function that takes in a list containing the horizontal weights and can be
    optimized with the different scipy optimizers.
    """
    print("-------------- New Optimization step --------------")
    n_particles = horizontal_lengths.shape[0]
    eigenfrequencies = []

    for i in range(n_particles):
        print(f"-- Particle {i} --")
        gmsh_mesh = generate_gmsh_mesh(model, L, H, B, horizontal_lengths[i])

        V, eigenvalues, eigenmodes, first_longitudinal_mode = unified_solving_function(
            eigensolver,
            gmsh_mesh,
            L,
            H,
            B,
            bc_z,
            bc_y,
            no_eigenvalues,
            target_frequency,
        )

        eigenfrequency = np.sqrt(eigenvalues[first_longitudinal_mode].real) / (
            2 * np.pi
        )
        saving_path = f"{folder}{eigenfrequency:.2f}.png"

        visualise_3D(
            plotter,
            V,
            eigenvalues,
            eigenmodes,
            first_longitudinal_mode,
            saving_path,
            viewup=True,
        )

        append_feature(
            eigenfrequency, f"{folder}features_p{i}.csv", horizontal_lengths[i]
        )
        eigenfrequencies.append(eigenfrequency)

    return eigenfrequencies


def objective(trial):
    """
    Objective function for hyperparameter optimization using Optuna.
    """
    c1 = trial.suggest_float("c1", 0, 1)
    c2 = trial.suggest_float("c2", 0.5, 1.4)
    w = trial.suggest_float("w", 0.3, 1.0)
    options = {"c1": c1, "c2": c2, "w": w}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=no_particles,
        dimensions=int(L / grid_size),
        options=options,
        bounds=bounds,
        velocity_clamp=min_max_velocities,
    )

    cost, pos = optimizer.optimize(opt_function, iters=200)
    return cost


# Uncomment to run hyperparameter optimization using Optuna
# study = optuna.create_study(
#     direction="minimize",
#     storage="sqlite:///db.sqlite3",
#     study_name="ps_parameter_search_20p_200iters",
# )
# study.optimize(objective, n_trials=10)
# print(study.best_params)

# Set PSO hyperparameters
c1, c2, w = 0.5, 0.8, 0.5
options = {"c1": c1, "c2": c2, "w": w}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(
    n_particles=no_particles,
    dimensions=int(L / grid_size),
    options=options,
    bounds=bounds,
    velocity_clamp=min_max_velocities,
)

# Perform optimization
cost, pos = optimizer.optimize(opt_function, iters=500)
