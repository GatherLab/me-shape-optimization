import numpy as np
import gmsh
import math


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


def generate_gmsh_mesh(model, L, H, B, geometry_width_list):
    """
    https://jsdokken.com/src/tutorial_gmsh.html
    """
    # Clear gmsh
    gmsh.clear()

    N = int(np.size(geometry_width_list))

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


def generate_gmsh_mesh_needle(model, L, H, B, geometry_width_list):
    """
    https://jsdokken.com/src/tutorial_gmsh.html
    """
    # Clear gmsh
    gmsh.clear()

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

    # Create a DOLFINx mesh (same mesh on each rank)
    msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
    msh.name = "Wedge"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"
    return msh

def generate_gmsh_mesh_more_crazy_needle(model, L, H, B, weights):
    """
    https://jsdokken.com/src/tutorial_gmsh.html
    """
    # Clear gmsh
    gmsh.clear()

    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
    # unnamed model will be created on the fly, if necessary):
    gmsh.model.add("t1")

    wedge1 = gmsh.model.occ.addWedge(0, 0, 0, L, weights[0] / 2, H)
    wedge2 = gmsh.model.occ.addWedge(0, 0, 0, L, weights[0] / 2, H)
    # gmsh.model.occ.rotate([(3, wedge2)], 0, 0, 0, 0, 0, 1, np.pi / 2)
    gmsh.model.occ.mirror([(3, wedge2)], 0, 1, 0, 0)
    gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 0, 1, 0, np.pi)
    gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 1, 0, 0, np.pi / 2)

    """
    wedge3 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
    wedge4 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
    # gmsh.model.occ.rotate([(3, wedge2)], 0, 0, 0, 0, 0, 1, np.pi / 2)
    gmsh.model.occ.mirror([(3, wedge4)], 0, 1, 0, 0)
    gmsh.model.occ.rotate([(3, wedge3), (3, wedge4)], 0, 0, 0, 0, 1, 0, np.pi)
    gmsh.model.occ.rotate([(3, wedge3), (3, wedge4)], 0, 0, 0, 1, 0, 0, np.pi / 2)
    # Mirror along y axis
    gmsh.model.occ.mirror([(3, wedge3), (3, wedge4)], 1, 0, 0, 0)
    """

    ##############################
    # Generate the opposite structure that is thinned at the centre and thick
    # otherwise 
    # gmsh.model.occ.translate([(3, wedge1), (3, wedge2)], L, 0, 0)

    # Add the required box in the centre that ensures a minimum thickness
    # The notation is box coordinates and then the length in each direction
    # Basic pad
    # centre_box = gmsh.model.occ.addBox( -1e-3+L/2, 0, -0.5e-3, 2e-3, H, 1e-3)
    # Seemless pad
    # centre_box = gmsh.model.occ.addBox( -2e-3+L/2, 0, -0.5e-3, 4e-3, H, 1e-3)

    ##############################

    # Fuse the two boxes together
    gmsh.model.occ.fuse([(3, wedge1)], [(3, wedge2)]) #, (3, wedge3), (3, wedge4)]) #, (3, centre_box)])

    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1e-3)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e-3)

    # Synchronize OpenCascade representation with gmsh model
    gmsh.model.occ.synchronize()

    # Add physical marker for cells. It is important to call this function
    # after OpenCascade synchronization
    gmsh.model.add_physical_group(dim=3, tags=[wedge1, wedge2]) #, wedge3, wedge4]) #, centre_box])

    # We finally generate and save the mesh:
    gmsh.model.mesh.generate(dim=3)

    # Create a DOLFINx mesh (same mesh on each rank)
    msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
    msh.name = "Wedge"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"
    return msh


def generate_gmsh_mesh_different_topologies(model, L, H, B):
    """
    https://jsdokken.com/src/tutorial_gmsh.html
    """
    # Clear gmsh
    gmsh.clear()

    # Next we add a new model named "t1" (if gmsh.model.add() is not called a new
    # unnamed model will be created on the fly, if necessary):
    gmsh.model.add("t1")

    # Basic pad
    centre_box = gmsh.model.occ.addBox(-L/2, 0, -B/4, L/3, H, B)
    centre_box2 = gmsh.model.occ.addBox(-L/2+L/3, 0, -B/2, L/3, H, B)
    centre_box3 = gmsh.model.occ.addBox(-L/2+2*L/3, 0, -3*B/4, L/3, H, B)

    # Pads to subtract on both ends 
    # subtract_box1 = gmsh.model.occ.addBox(-L/2, 0, -B/4, 2e-3, H, B/2)
    # subtract_box2 = gmsh.model.occ.addBox(L/2-2e-3, 0, -B/4, 2e-3, H, B/2)

    # Pad cut out in the middle
    # subtract_box1 = gmsh.model.occ.addBox(-2e-3, 0, -B/4, 1e-3, H, B/2)
    # subtract_box2 = gmsh.model.occ.addBox(1e-3, 0, -B/4, 1e-3, H, B/2)


    # Cut out shapes from the standard rectangular box
    # gmsh.model.occ.cut([(3, centre_box)], [(3, cylinder)])

    # Add shapes
    gmsh.model.occ.fuse([(3, centre_box)], [(3, centre_box2), (3, centre_box3)])

    # Synchronize OpenCascade representation with gmsh model
    gmsh.model.occ.synchronize()

    # Add physical marker for cells. It is important to call this function
    # after OpenCascade synchronization
    gmsh.model.add_physical_group(dim=3, tags=[centre_box, centre_box2, centre_box3]) #centre_box, subtract_box1, subtract_box2])

    # We finally generate and save the mesh:
    gmsh.model.mesh.generate(dim=3)

    # Create a DOLFINx mesh (same mesh on each rank)
    msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
    msh.name = "Wedge"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"
    return msh


