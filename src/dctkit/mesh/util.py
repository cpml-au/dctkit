from dctkit.mesh.simplex import SimplicialComplex
import gmsh  # type: ignore
import numpy as np
from typing import Tuple
from pygmsh.geo import Geometry
from meshio import Mesh


def build_complex_from_mesh(mesh: Mesh) -> SimplicialComplex:
    """Build a SimplicialComplex object from a meshio.Mesh object.

    Args:
        mesh: a meshio.Mesh object.

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

    S = SimplicialComplex(tet_node_tags, node_coords, is_well_centered=True)
    return S


def generate_square_mesh(lc: float, L: float = 1.) -> Tuple[Mesh, Geometry]:
    """Generate a mesh for the unit square.

    Args:
        lc: target mesh size.
        L: side length.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Polygon objects.
    """
    with Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        mesh = geom.generate_mesh()

    return mesh, geom


def generate_hexagon_mesh(a: float, lc: float) -> Tuple[Mesh, Geometry]:
    """Generate a mesh for the regular hexagon.

    Args:
        a: edge length.
        lc: target mesh size.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Polygon objects.
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
        a tuple containing a meshio Mesh and a pygmsh Polygon objects.
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


def generate_cube_mesh(lc: float, L: float = 1.):
    with Geometry() as geom:
        poly = geom.add_polygon([[0.0, 0.0], [L, 0.0], [L, L], [0.0, L]], lc)
        geom.extrude(poly, [0, 0, L])
        mesh = geom.generate_mesh()

    return mesh, geom


def generate_line_mesh(num_nodes: int, L: float) -> Tuple[Mesh, Geometry]:
    """Generate a uniform mesh in an interval of given length.

    Args:
        num_nodes: number of nodes.
        L: length of the interval.

    Returns:
        a tuple containing a meshio Mesh and a pygmsh Line objects.
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
            mesh = g.generate_mesh(1)

    return mesh, g


def get_nodes_for_physical_group(mesh: Mesh, dim: int, group_name: str) -> list:
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


def get_edges_for_physical_group(S: SimplicialComplex, mesh: Mesh, group_name: str):
    edges_ids_in_mesh = mesh.cell_sets_dict[group_name]["line"]
    edges_nodes_ids = mesh.cells_dict["line"][edges_ids_in_mesh]
    # nodes ids for edges in S[1] are sorted lexicographycally
    edges_nodes_ids.sort()
    edges_ids = [int(np.argwhere(np.all(S.S[1] == edge_nodes, axis=1)))
                 for edge_nodes in edges_nodes_ids]
    return edges_ids
