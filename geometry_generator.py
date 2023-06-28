import numpy as np
import gmsh

from mpi4py import MPI
from dolfinx import mesh, io

"""
Shape gneration in fenicsx works with gmsh, see e.g.: 
https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_gmsh.html
"""


def generate_geometry(L, H, B):
    # Geometry initialization
    geometry_mesh = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([L, H, B])],
        [20, 2, 5],
        cell_type=mesh.CellType.tetrahedron,
    )
    return geometry_mesh


def generate_gmsh_mesh(L, H, B, geometry_width_list):
    """
    https://jsdokken.com/src/tutorial_gmsh.html
    """
    N = int(np.size(geometry_width_list))

    gmsh.initialize()

    # Choose if Gmsh output is verbose
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model.add("Box")
    model.setCurrent("Box")

    # The boxes are defined by a point x,y,z and the length in the x,y,z direction
    # Define a list of boxes with a unit lenght of L/N, height H and width B
    boxes = [
        model.occ.addBox(
            i * L / N, 0, -geometry_width_list[i] / 2, L / N, H, geometry_width_list[i]
        )
        for i in range(N)
    ]
    boxes_with_dimensions = [(3, box) for box in boxes]
    # box1 = model.occ.addBox(0, 0, 0, L / 2, H, B)
    # box2 = model.occ.addBox(L / 2, 0, 0, L / 2, H, B)

    # Fuse the two boxes together
    model.occ.fuse([(3, boxes[0])], boxes_with_dimensions[1:])

    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-3)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e-3)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical marker for cells. It is important to call this function
    # after OpenCascade synchronization
    model.add_physical_group(dim=3, tags=boxes)

    # Generate the mesh
    model.mesh.generate(dim=3)

    # Create a DOLFINx mesh (same mesh on each rank)
    msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
    msh.name = "Box"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"
    return msh
