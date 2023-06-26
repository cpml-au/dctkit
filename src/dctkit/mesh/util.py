from dctkit.mesh.simplex import SimplicialComplex
import numpy as np
from typing import Tuple, List
from pygmsh.geo import Geometry
from meshio import Mesh


def build_complex_from_mesh(mesh: Mesh, is_well_centered=True) -> SimplicialComplex:
    """Build a SimplicialComplex object from a meshio.Mesh object.

    Args:
        mesh: a meshio.Mesh object.
        is_well_centered: True if the mesh is well centered.

    Returns:
        a SimplicialComplex object.
    """
    node_coords = mesh.points
    cell_types = mesh.cells_dict.keys()

    if "tetra" in cell_types:
        tet_node_tags = mesh.cells_dict["tetra"]
    elif "triangle" in cell_types:
        tet_node_tags = mesh.cells_dict["triangle"]
    elif "line" in cell_types:
        tet_node_tags = mesh.cells_dict["line"]

    S = SimplicialComplex(tet_node_tags, node_coords, is_well_centered=is_well_centered)
    return S


def generate_square_mesh(lc: float, L: float = 1.) -> Tuple[Mesh, Geometry]:
    """Generate the mesh of a square.

    Args:
        lc: target mesh size.
        L: side length.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Geometry objects.
    """
    with Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        mesh = geom.generate_mesh()

    return mesh, geom


def generate_hexagon_mesh(a: float, lc: float) -> Tuple[Mesh, Geometry]:
    """Generate the mesh of a regular hexagon.

    Args:
        a: edge length.
        lc: target mesh size.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Geometry objects.
    """
    with Geometry() as geom:
        geom.add_polygon([[2*a, np.sqrt(3)/2*a], [3/2*a, np.sqrt(3)*a],
                          [a/2, np.sqrt(3)*a], [0., np.sqrt(3)/2*a], [a/2, 0.],
                          [3/2*a, 0.]], mesh_size=lc)
        mesh = geom.generate_mesh()

    return mesh, geom


def generate_tet_mesh(lc: float) -> Tuple[Mesh, Geometry]:
    """Generate the mesh of a tetrahedron.

    Args:
        lc: target mesh size.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Geometry objects.
    """
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1/2, 1, 0], [0, 0, 1]])
    lines = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    curve_loops = []
    with Geometry() as geom:
        points = np.array([geom.add_point(node, lc) for node in nodes])
        edges = [geom.add_line(*points[line_idx]) for line_idx in lines]
        curve_loops.append(geom.add_curve_loop([edges[0], edges[3], -edges[1]]))
        curve_loops.append(geom.add_curve_loop([edges[0], edges[4], -edges[2]]))
        curve_loops.append(geom.add_curve_loop([edges[1], edges[5], -edges[2]]))
        curve_loops.append(geom.add_curve_loop([edges[3], edges[5], -edges[4]]))
        surfaces = [geom.add_surface(curve_loops[i]) for i in range(len(curve_loops))]
        surface_loop = geom.add_surface_loop(surfaces)
        geom.add_volume(surface_loop)
        mesh = geom.generate_mesh()

    return mesh, geom


def generate_cube_mesh(lc: float, L: float = 1.) -> Tuple[Mesh, Geometry]:
    """Generate the mesh of a cube.

    Args:
        lc: target mesh size.
        L: side length.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Geometry objects.
    """
    with Geometry() as geom:
        poly = geom.add_polygon([[0.0, 0.0], [L, 0.0], [L, L], [0.0, L]], lc)
        geom.extrude(poly, [0, 0, L])
        mesh = geom.generate_mesh()

    return mesh, geom


def generate_line_mesh(num_nodes: int, L: float = 1.) -> Tuple[Mesh, Geometry]:
    """Generate a uniform mesh in an interval of given length.

    Args:
        num_nodes: number of nodes.
        L: length of the interval.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Geometry objects.
    """
    lc = L/(num_nodes-1)
    points = [None]*num_nodes
    with Geometry() as g:
        points[0] = g.add_point([0., 0.], lc)
        # we have to add one line for each element of the mesh, otherwise if we mesh one
        # line for the whole interval [0, L] the end nodes are going to be the first two
        # items in the mesh.points matrix.
        for i in range(1, num_nodes):
            points[i] = g.add_point([i*lc, 0.], lc)
            # see also test_hex in pygmsh library's tests
            new_line_points = [points[i-1], points[i]]
            g.add_line(*new_line_points)
            mesh = g.generate_mesh()

    return mesh, g


def get_nodes_for_physical_group(mesh: Mesh, dim: int, group_name: str) -> List[int]:
    """Find the IDs of the nodes belonging to a physical group within the mesh object.

    Args:
        mesh: a meshio object.
        dim: dimension of the cells belonging to the physical group.
        group_name: name of the physical group.

    Returns:
        list of the node IDs belonging to the physical group.
    """
    if dim == 1:
        cell_type = "line"
    elif dim == 2:
        cell_type = "triangle"
    elif dim == 3:
        cell_type = "tetra"
    else:
        raise NotImplementedError

    group_cells = mesh.cell_sets_dict[group_name].get(cell_type)
    nodes_ids = list(set(mesh.cells_dict[cell_type][group_cells].flatten()))
    return nodes_ids


def get_edges_for_physical_group(S: SimplicialComplex, mesh: Mesh,
                                 group_name: str) -> List[int]:
    """Find the IDs of the edges belonging to a physical group within the mesh object.

    Args:
        S: SimplicialComplex object associated to the mesh.
        mesh: a meshio object.
        group_name: name of the physical group.

    Returns:
        list of the node IDs belonging to the physical group.
    """
    edges_ids_in_mesh = mesh.cell_sets_dict[group_name]["line"]
    edges_nodes_ids = mesh.cells_dict["line"][edges_ids_in_mesh]
    # nodes ids for edges in S[1] are sorted lexicographycally
    edges_nodes_ids.sort()
    edges_ids = [int(np.argwhere(np.all(S.S[1] == edge_nodes, axis=1)))
                 for edge_nodes in edges_nodes_ids]
    return edges_ids
