from dctkit import int_dtype, float_dtype
import gmsh
import numpy as np


def read_mesh(filename=None, format="gmsh"):
    """Reads a mesh from file.

    Args:
        filename: name of the file containing the mesh.
    Returns:
        numNodes: number of mesh points.
    """
    assert format == "gmsh"

    if not gmsh.is_initialized():
        gmsh.initialize()

    if filename is not None:
        gmsh.open(filename)

    # Get nodes and corresponding coordinates
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    numNodes = len(nodeTags)
    # print("# nodes = ", numNodes)

    coords = np.array(coords, dtype=float_dtype)
    # Get 2D elements and associated node tags
    # NOTE: ONLY GET TRIANGLES
    elemTags, nodeTagsPerElem = gmsh.model.mesh.getElementsByType(2)

    # Decrease element IDs by 1 to have node indices starting from 0
    nodeTagsPerElem = np.array(nodeTagsPerElem, dtype=int_dtype) - 1
    nodeTagsPerElem = nodeTagsPerElem.reshape(len(nodeTagsPerElem) // 3, 3)
    # Get number of TRIANGLES
    numElements = len(elemTags)
    # print("# elements = ", numElements)

    # physicalGrps = gmsh.model.getPhysicalGroups()
    # print("physical groups: ", physicalGrps)

    # edgeNodesTags = gmsh.model.mesh.getElementEdgeNodes(2)
    # print("edge nodes tags: ", edgeNodesTags)

    # Position vectors of mesh points
    node_coords = coords.reshape(len(coords)//3, 3)
    return numNodes, numElements, nodeTagsPerElem, node_coords


def generate_square_mesh(lc):
    if not gmsh.is_initialized():
        gmsh.initialize()

    gmsh.model.add("t1")
    gmsh.model.geo.addPoint(1, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(0, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 4, 2)
    gmsh.model.geo.addLine(4, 3, 3)
    gmsh.model.geo.addLine(3, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 1)
    gmsh.model.mesh.generate(2)

    numNodes, numElements, nodeTagsPerElem, node_coords = read_mesh()

    return numNodes, numElements, nodeTagsPerElem, node_coords
