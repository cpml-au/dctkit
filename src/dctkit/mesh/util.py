from dctkit import int_dtype, float_dtype
from dctkit.mesh.simplex import SimplicialComplex
import gmsh  # type: ignore
import pygmsh
import meshio as mo
import numpy as np
import numpy.typing as npt
from typing import Tuple, Any


def read_mesh(filename: str | None = None) -> Tuple[int, int, npt.NDArray,
                                                    npt.NDArray, npt.NDArray]:
    """Process the current mesh model. If a filename is provided, read the mesh from a
        .msh file.

    Args:
        filename: name of the .msh file containing the mesh.

    Returns:
        a tuple containing the number of mesh nodes; the number of faces; the matrix
        containing the IDs of the nodes (cols) belonging to each face (rows); the node
        coordinates; the matrix containing the IDs of the nodes (cols) belonging to
        each boundary element (rows).
    """
    if not gmsh.is_initialized():
        gmsh.initialize()

    if filename is not None:
        gmsh.open(filename)

    # Get nodes and corresponding coordinates
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()
    numNodes = len(nodeTags)

    coords = np.array(coords, dtype=float_dtype)
    # Get 2D elements and associated node tags
    elemTags, nodeTagsPerElem = gmsh.model.mesh.getElementsByType(2)

    # Decrease element IDs by 1 to have node indices starting from 0
    nodeTagsPerElem = np.array(nodeTagsPerElem, dtype=int_dtype) - 1
    nodeTagsPerElem = nodeTagsPerElem.reshape(len(nodeTagsPerElem) // 3, 3)
    # Get number of TRIANGLES
    numElements = len(elemTags)

    # Position vectors of mesh points
    node_coords = coords.reshape(len(coords)//3, 3)

    # get node tags per boundary elements
    _, nodeTagsPerBElem = gmsh.model.mesh.getElementsByType(1)
    nodeTagsPerBElem = np.array(nodeTagsPerBElem, dtype=int_dtype) - 1
    nodeTagsPerBElem = nodeTagsPerBElem.reshape(len(nodeTagsPerBElem) // 2, 2)
    # we sort every row to get the orientation used for our simulations
    nodeTagsPerBElem = np.sort(nodeTagsPerBElem)

    return numNodes, numElements, nodeTagsPerElem, node_coords, nodeTagsPerBElem


def build_complex_from_mesh(mesh: mo.Mesh) -> SimplicialComplex:
    node_coords = mesh.points
    # TODO: only works with triangles, detect top-level simplices as tets
    tet_node_tags = mesh.cells_dict["triangle"]
    S = SimplicialComplex(tet_node_tags, node_coords, is_well_centered=True)
    return S


def generate_square_mesh(lc: float, L: float = 1.) -> Tuple[mo.Mesh, Any]:
    """Generate a mesh for the unit square.

    Args:
        lc: target mesh size.
        L: side length.
    """
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        mesh = geom.generate_mesh()

    return mesh, p


def generate_hexagon_mesh(a: float, lc: float) -> Tuple[mo.Mesh, Any]:
    """Generate a mesh for the regular hexagon.

    Args:
        a: edge length.
        lc: target mesh size.
    """
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[2*a, np.sqrt(3)/2*a], [3/2*a, np.sqrt(3)*a],
                              [a/2, np.sqrt(3)*a], [0., np.sqrt(3)/2*a], [a/2, 0.],
                              [3/2*a, 0.]], mesh_size=lc)
        mesh = geom.generate_mesh()

    return mesh, p


def generate_tet_mesh(lc: float):
    """Generate the mesh of a tetrahedron.

    Args:
        lc: target mesh size (lc) close to a given point.

    Returns:
        a tuple containing the number of mesh nodes; the number of faces; the matrix
        containing the IDs of the nodes (cols) belonging to each face (rows); the node
        coordinates; the matrix containing the IDs of the nodes (cols) belonging to
        each boundary element (rows).

    """
    # FIXME: REWRITE USING PYGMSH PRIMITIVES; return meshio.Mesh and polygon objs
    if not gmsh.is_initialized():
        gmsh.initialize()

    gmsh.model.add("tet")
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1/2, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 0, 1, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 1, 3)
    gmsh.model.geo.addLine(1, 4, 4)
    gmsh.model.geo.addLine(2, 4, 5)
    gmsh.model.geo.addLine(3, 4, 6)
    gmsh.model.geo.addCurveLoop([1, 2, 3], 1)
    gmsh.model.geo.addCurveLoop([1, 5, -4], 2)
    gmsh.model.geo.addCurveLoop([-3, 6, -4], 3)
    gmsh.model.geo.addCurveLoop([2, 6, -5], 4)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.addPlaneSurface([3], 3)
    gmsh.model.geo.addPlaneSurface([4], 4)
    gmsh.model.geo.addSurfaceLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addVolume([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)


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


def get_nodes_for_physical_group(mesh: mo.Mesh, dim: int, group_name: str) -> list:
    """Find the IDs of the nodes belonging to a physical group within the mesh object.

    Args:
        mesh: a meshio object.
        dim: dimension of the physical group.
        group_name: tag of the physical group.

    Returns:
        list of the node IDs belonging to the physical group.

    """
    if dim == 1:
        cell_type = "line"
    elif dim == 2:
        cell_type = "triangle"
    else:
        raise NotImplementedError

    group_cells = mesh.cell_sets_dict[group_name].get(cell_type)
    nodes_ids = list(set(mesh.cells_dict[cell_type][group_cells].flatten()))
    return nodes_ids


def get_edges_for_physical_group(S: SimplicialComplex, mesh: mo.Mesh, group_name: str):
    edges_ids_in_mesh = mesh.cell_sets_dict[group_name]["line"]
    edges_nodes_ids = mesh.cells_dict["line"][edges_ids_in_mesh]
    # nodes ids for edges in S[1] are sorted lexicographycally
    edges_nodes_ids.sort()
    edges_ids = [int(np.argwhere(np.all(S.S[1] == edge_nodes, axis=1)))
                 for edge_nodes in edges_nodes_ids]
    return edges_ids
