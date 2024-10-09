import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx import io

"""
Shape generation in fenicsx works with gmsh, see e.g. for a good demo example: 
https://jsdokken.com/src/tutorial_gmsh.html
"""


def generate_gmsh_mesh(model, L, H, B, geometry_width_list):
    """
    Rectangular shape generation using length L and height H of the overall
    structure. The geometry_width_list is a list of the widths of the individual
    segments of the rectangular structure.
    """
    # Clear gmsh
    gmsh.clear()

    N = int(np.size(geometry_width_list))

    # Define a list of boxes with a unit length of L/N, height H, and width B
    boxes = [
        model.occ.addBox(
            i * L / N, 0, -geometry_width_list[i] / 2, L / N, H, geometry_width_list[i]
        )
        for i in range(N)
    ]
    boxes_with_dimensions = [(3, box) for box in boxes]

    # Fuse the boxes together
    model.occ.fuse([(3, boxes[0])], boxes_with_dimensions[1:])

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical marker for cells
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
    Generate a needle-like structure with a wedge at the top and a rectangular
    shape at the bottom.
    """
    # Clear gmsh
    gmsh.clear()

    # Add a new model named "t1"
    gmsh.model.add("t1")

    # Create two wedges
    wedge1 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
    wedge2 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
    gmsh.model.occ.mirror([(3, wedge2)], 0, 1, 0, 0)
    gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 0, 1, 0, np.pi)
    gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 1, 0, 0, np.pi / 2)

    N = int(np.size(geometry_width_list))

    # Define a list of boxes with a unit length of L/2/N, height H, and width B
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

    # Fuse the wedges and boxes together
    gmsh.model.occ.fuse([(3, wedge1)], [(3, wedge2)] + boxes_with_dimensions)

    # Synchronize OpenCascade representation with gmsh model
    gmsh.model.occ.synchronize()

    # Add physical marker for cells
    gmsh.model.add_physical_group(dim=3, tags=[wedge1, wedge2] + boxes)

    # Generate the mesh
    gmsh.model.mesh.generate(dim=3)

    # Create a DOLFINx mesh (same mesh on each rank)
    msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
    msh.name = "Wedge"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"
    return msh


def generate_gmsh_mesh_more_crazy_needle(model, L, H, B, weights):
    """
    Generate more needle structures such as an ME laminate that has no segments
    but is simply a wedge, or a double wedge (diamond) structure. Comment out
    the according parts in the code to experiment with the more advanced
    structures.
    """
    # Clear gmsh
    gmsh.clear()

    # Add a new model named "t1"
    gmsh.model.add("t1")

    # Create two wedges
    wedge1 = gmsh.model.occ.addWedge(0, 0, 0, L, weights[0] / 2, H)
    wedge2 = gmsh.model.occ.addWedge(0, 0, 0, L, weights[0] / 2, H)
    gmsh.model.occ.mirror([(3, wedge2)], 0, 1, 0, 0)
    gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 0, 1, 0, np.pi)
    gmsh.model.occ.rotate([(3, wedge1), (3, wedge2)], 0, 0, 0, 1, 0, 0, np.pi / 2)

    """
    # Uncomment to add more wedges
    wedge3 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
    wedge4 = gmsh.model.occ.addWedge(0, 0, 0, L / 2, B / 2, H)
    gmsh.model.occ.mirror([(3, wedge4)], 0, 1, 0, 0)
    gmsh.model.occ.rotate([(3, wedge3), (3, wedge4)], 0, 0, 0, 0, 1, 0, np.pi)
    gmsh.model.occ.rotate([(3, wedge3), (3, wedge4)], 0, 0, 0, 1, 0, 0, np.pi / 2)
    gmsh.model.occ.mirror([(3, wedge3), (3, wedge4)], 1, 0, 0, 0)
    """

    """
    # Uncomment to add a box in the center for minimum thickness
    centre_box = gmsh.model.occ.addBox(-2e-3 + L / 2, 0, -0.5e-3, 4e-3, H, 1e-3)
    """

    # Fuse the wedges together
    gmsh.model.occ.fuse(
        [(3, wedge1)], [(3, wedge2)]
    )  # , (3, wedge3), (3, wedge4)]) #, (3, centre_box)])

    # Synchronize OpenCascade representation with gmsh model
    gmsh.model.occ.synchronize()

    # Add physical marker for cells
    gmsh.model.add_physical_group(
        dim=3, tags=[wedge1, wedge2]
    )  # , wedge3, wedge4]) #, centre_box])

    # Generate the mesh
    gmsh.model.mesh.generate(dim=3)

    # Create a DOLFINx mesh (same mesh on each rank)
    msh, cell_markers, facet_markers = io.gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
    msh.name = "Wedge"
    cell_markers.name = f"{msh.name}_cells"
    facet_markers.name = f"{msh.name}_facets"
    return msh
