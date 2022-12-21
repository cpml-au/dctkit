import gmsh
import numpy as np


def read_mesh(filename, format="gmsh"):
    """Reads a mesh from file.

    Args:
        filename: name of the file containing the mesh.
    Returns:
        numNodes: number of mesh points.
    """
    if format != "gmsh":
        print("Mesh format NOT IMPLEMENTED!")

    gmsh.initialize()
    gmsh.open(filename)

    # Get nodes and corresponding coordinates
    nodeTags, coords, paramCoords = gmsh.model.mesh.getNodes()
    numNodes = len(nodeTags)
    # print("# nodes = ", numNodes)

    # Get 2D elements and associated node tags
    # NOTE: ONLY GET TRIANGLES
    elemTags, nodeTagsPerElem = gmsh.model.mesh.getElementsByType(2)

    # Decrease element IDs by 1 to have node indices starting from 0
    nodeTagsPerElem = np.array(nodeTagsPerElem) - 1
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


def generate_mesh(lc):
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

    # Get nodes and corresponding coordinates
    nodeTags, coords, paramCoords = gmsh.model.mesh.getNodes()
    numNodes = len(nodeTags)
    # print("# nodes = ", numNodes)

    # Get 2D elements and associated node tags
    # NOTE: ONLY GET TRIANGLES
    elemTags, nodeTagsPerElem = gmsh.model.mesh.getElementsByType(2)

    # Decrease element IDs by 1 to have node indices starting from 0
    nodeTagsPerElem = np.array(nodeTagsPerElem) - 1
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
