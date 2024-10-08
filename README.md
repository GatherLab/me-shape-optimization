<h1 align="center">
    Engineering the Shape of ME Transducers via Finite Element Optimization
</h1>

<p align="center">
   <a href="https://github.com/GatherLab/me-shape-optimization/commits" title="Last Commit"><img src="https://img.shields.io/github/last-commit/GatherLab/me-shape-optimization"></a>
   <a href="https://github.com/GatherLab/me-shape-optimization/issues" title="Open Issues"><img src="https://img.shields.io/github/issues/GatherLab/me-shape-optimization"></a>
   <a href="./LICENSE" title="License"><img src="https://img.shields.io/github/license/GatherLab/me-shape-optimization"></a>
</p>

<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#optimization">Optimization</a> •
  <a href="#development">Development</a>
</p>

Software for the simulation and optimization of the resonance frequency of the
first longitudinal mode of a thin structure.

## Setup
1. Install latest dolfinx image (tested under dolfinx:nightly, v.??)
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
## Development
