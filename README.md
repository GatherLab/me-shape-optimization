<h1 align="center">
    Engineering the Shape of ME Transducers via Finite Element Optimization
</h1>

More information about this work can be found in our publication with the DOI <??>.

<p align="center">
   <a href="https://github.com/GatherLab/me-shape-optimization/commits" title="Last Commit"><img src="https://img.shields.io/github/last-commit/GatherLab/me-shape-optimization"></a>
   <a href="https://github.com/GatherLab/me-shape-optimization/issues" title="Open Issues"><img src="https://img.shields.io/github/issues/GatherLab/me-shape-optimization"></a>
   <a href="./LICENSE" title="License"><img src="https://img.shields.io/github/license/GatherLab/me-shape-optimization"></a>
</p>

Software for the simulation of oscillation modes of arbitrary structures using
the finite element method solver of dolfinx. The software solves the equations
of linear elasticity and allows for simple plotting functionalities. 

<p align="center">
  <img src="./examples/bending-mode.png" alt="Image 1" width="30%" style="display: inline-block; margin: 0 10px;">
  <img src="./examples/twisting-mode.png" alt="Image 2" width="30%" style="display: inline-block; margin: 0 10px;">
  <img src="./examples/first-longitudinal-mode.png" alt="Image 3" width="30%" style="display: inline-block; margin: 0 10px;">
</p>

Automatically identify the first longitudinal resonance mode for magnetoelectric
laminate applications.

<p align="center">
  <img src="./examples/first-longitudinal-bar-shape.png" alt="Image 1" width="30%" style="display: inline-block; margin: 0 10px;">
  <img src="./examples/needle-shape-first-longitudinal.png" alt="Image 2" width="30%" style="display: inline-block; margin: 0 10px;">
</p>

Through various state of the art optimization algorithms,
optimize the shape for a specific resonance frequency, minimum or maximum
resonance frequency.

<p align="center">
  <img src="./examples/time_laps.gif" alt="Animation" width="60%" style="display: inline-block; margin: 0 10px;">
</p>


## Setup
1. Install latest dolfinx image (tested under dolfinx:nightly, v. 0.7.0.0)
2. Clone github respository (git clone )
3. Setup a python virtual environment (ideally via venv) (tested on python 3.10.6)

```terminal
py -m venv venv
```

4. Activate the new environement

```
source ./bin/activate
```

5. Install required packages from requirements.txt

```
pip install -r requirements.txt
```

6. Run the example.py to run a basic FEM simulation an a rectangle

```terminal
python3 example.py
```

## Optimization

- Define your own shape generation strategy (geometry_generator.py) based on a number of features that can be transformed into a shape.
- Run particle swarm or scipy optimization algorithms (e.g. dual_annealing, basinhopping or minimize) to optimize the shape for a minimum resonance frequency.
- Modify the opt_func in *_optimization.py to change the target (reverse the return function or e.g. return the difference to a target frequency).
- Visualize the optimization process by generating a .gif from the shapes produced during optimization (visualisation.py) or calculate the resonance modes of a specific shape (example.py).
