from dctkit import int_dtype, float_dtype
import gmsh  # type: ignore
import numpy as np
import numpy.typing as npt
from typing import Tuple


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

    # get node tags per boundary elements
    _, nodeTagsPerBElem = gmsh.model.mesh.getElementsByType(1)
    nodeTagsPerBElem = np.array(nodeTagsPerBElem, dtype=int_dtype) - 1
    nodeTagsPerBElem = nodeTagsPerBElem.reshape(len(nodeTagsPerBElem) // 2, 2)
    # we sort every row to get the orientation used for our simulations
    nodeTagsPerBElem = np.sort(nodeTagsPerBElem)

    return numNodes, numElements, nodeTagsPerElem, node_coords, nodeTagsPerBElem


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
    gmsh.model.addPhysicalGroup(1, [2], 2, name="left")
    gmsh.model.addPhysicalGroup(1, [4], 3, name="right")
    gmsh.model.mesh.generate(2)

    numNodes, numElements, nodeTagsPerElem, node_coords, nodeTagsPerBElem = read_mesh()

    return numNodes, numElements, nodeTagsPerElem, node_coords, nodeTagsPerBElem


def generate_hexagon_mesh(a, lc):
    """Generate a regular hexagonal mesh

       Args:
            a (float): length of the hexagonal edges.
    """
    if not gmsh.is_initialized():
        gmsh.initialize()

    gmsh.model.add("hexagon")
    gmsh.model.geo.addPoint(2*a, np.sqrt(3)/2 * a, 0, lc, 1)
    gmsh.model.geo.addPoint(3/2 * a, np.sqrt(3)*a, 0, lc, 2)
    gmsh.model.geo.addPoint(a/2, np.sqrt(3)*a, 0, lc, 3)
    gmsh.model.geo.addPoint(0, np.sqrt(3)/2 * a, 0, lc, 4)
    gmsh.model.geo.addPoint(a/2, 0, 0, lc, 5)
    gmsh.model.geo.addPoint(3/2 * a, 0, 0, lc, 6)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 1, 6)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6], 1)
    gmsh.model.mesh.generate(2)

    numNodes, numElements, nodeTagsPerElem, node_coords, nodeTagsPerBElem = read_mesh()

    return numNodes, numElements, nodeTagsPerElem, node_coords, nodeTagsPerBElem


def generate_1_D_mesh(num_nodes: int, L: float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Generate a uniform 1D mesh.

    Args:
        num_nodes: number of nodes.
        L: length of the interval.
    Returns:
        a tuple consisting of the matrix of node coordinates (rows = node IDs, cols =
            x,y coords) and a matrix containing the IDs of the nodes belonging to
            each 1-simplex.
    """
    node_coords = np.linspace(0, L, num=num_nodes)
    x = np.zeros((num_nodes, 2))
    x[:, 0] = node_coords
    S_1 = np.empty((num_nodes - 1, 2))
    S_1[:, 0] = np.arange(num_nodes-1)
    S_1[:, 1] = np.arange(1, num_nodes)
    return S_1, x


def get_nodes_from_physical_group(dim: int, tag: int):
    """Wrap-function for gmsh.model.mesh.getNodesForPhysicalGroup that indexes 
    correctly node_tags."""
    node_tags, node_coords_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
    node_tags -= 1
    return node_tags, node_coords_flatten


def get_belonging_elements(dim: int, tag: int, nodeTagsPerElem: npt.NDArray):
    _, elem_coords_in_tag_elem_flatten = gmsh.model.mesh.getElementsByType(dim, tag)
    elem_coords_in_tag_elem_flatten -= 1
    elem_coords_in_tag_elem = elem_coords_in_tag_elem_flatten.reshape(
        len(elem_coords_in_tag_elem_flatten) // 2, 2)
    elem_coords_in_tag_elem = np.sort(elem_coords_in_tag_elem)
    print(elem_coords_in_tag_elem)
    inside_elems_idx = [int(np.argwhere(
        np.all(nodeTagsPerElem == bnd_face, axis=1)))
        for bnd_face in elem_coords_in_tag_elem]
    # inside_elems_idx = np.sort(inside_elems_idx)
    return inside_elems_idx
