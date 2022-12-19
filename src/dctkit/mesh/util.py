import gmsh
import numpy as np


class Mesh:
    def __init__(self, filename, format="gmsh"):
        self.filename = filename
        self.format = format

    def initialize_mesh(self):
        if self.format != "gmsh":
            print("Mesh format NOT IMPLEMENTED!")
        gmsh.initialize()
        gmsh.open(self.filename)

    def refine_mesh(self):
        gmsh.model.mesh.refine()

    def get_mesh(self):
        # Get nodes and corresponding coordinates
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()
        self.numNodes = len(nodeTags)
        # print("# nodes = ", numNodes)

        # Get 2D elements and associated node tags
        # NOTE: ONLY GET TRIANGLES
        elemTags, nodeTagsPerElem = gmsh.model.mesh.getElementsByType(2)

        # Decrease element IDs by 1 to have node indices starting from 0
        nodeTagsPerElem = np.array(nodeTagsPerElem) - 1
        self.nodeTagsPerElem = nodeTagsPerElem.reshape(len(nodeTagsPerElem) // 3, 3)
        # Get number of TRIANGLES
        self.numElements = len(elemTags)
        # print("# elements = ", numElements)

        # physicalGrps = gmsh.model.getPhysicalGroups()
        # print("physical groups: ", physicalGrps)

        # edgeNodesTags = gmsh.model.mesh.getElementEdgeNodes(2)
        # print("edge nodes tags: ", edgeNodesTags)

        # Position vectors of mesh points
        self.node_coords = coords.reshape(len(coords)//3, 3)


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
