# ------------------------------------------------------------------------------
#
#  Gmsh Python tutorial 1
#
#  Geometry basics, elementary entities, physical groups
#
# ------------------------------------------------------------------------------

# The Python API is entirely defined in the `gmsh.py' module (which contains the
# full documentation of all the functions in the API):
import gmsh
import sys
import numpy as np

L = 12e-3
H = 0.1e-3
B = 3e-3
geometry_width_list = [
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
    3e-3,
]

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()

# Next we add a new model named "t1" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add("t1")

wedge1 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
wedge2 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
# gmsh.model.occ.rotate([(3, wedge2)], 0, 0, 0, 0, 0, 1, np.pi / 2)
gmsh.model.occ.mirror([(3, wedge2)], 0, 1, 0, 0)
gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 0, 1, 0, np.pi)
gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 1, 0, 0, np.pi / 2)


N = int(np.size(geometry_width_list))

# The boxes are defined by a point x,y,z and the length in the x,y,z direction
# Define a list of boxes with a unit lenght of L/N, height H and width B
boxes = [
    gmsh.model.occ.addBox(
        i * L / 2 / N,
        0,
        -geometry_width_list[i] / 2,
        L / 2 / N,
        H,
        geometry_width_list[i],
    )
    for i in range(N)
]
boxes_with_dimensions = [(3, box) for box in boxes]
# box1 = gmsh.model.occ.addBox(0, 0, 0, L / 2, H, B)
# box2 = gmsh.model.occ.addBox(L / 2, 0, 0, L / 2, H, B)

# Fuse the two boxes together
gmsh.model.occ.fuse([(3, wedge1)], [(3, wedge2)] + boxes_with_dimensions)

# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-3)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e-3)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# Add physical marker for cells. It is important to call this function
# after OpenCascade synchronization
gmsh.model.add_physical_group(dim=3, tags=[wedge1, wedge2] + boxes)

# We finally generate and save the mesh:
gmsh.model.mesh.generate(dim=3)

gmsh.write("t2.msh")


"""
# Create a DOLFINx mesh (same mesh on each rank)
msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
msh.name = "Box"
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

"""
