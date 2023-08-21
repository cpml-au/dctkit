import numpy as np
import dctkit
from dctkit.mesh import circumcenter as circ, volume
from dctkit.math import shifted_list as sl
from dctkit.math import spmv
import numpy.typing as npt
from jax import Array
import jax.numpy as jnp
from typing import Tuple, Any


class SimplicialComplex:
    """Simplicial complex class.

    Args:
        tet_node_tags: matrix containing the IDs of the nodes (cols) belonging to each
            tetrahedron or top-level simplex (rows).
        node_coords: Cartesian coordinates (columns) of all the nodes (rows) of the
            simplicial complex.
        is_well_centered: True if the mesh is well-centered.

    Attributes:
        dim (int): dimension of the complex.
        S (list): list where each entry p is a matrix containing the IDs of the
            nodes belonging to each p-simplex.
        circ (list): list where each entry p is a matrix containing the
            coordinates of the circumcenters (cols) of all the p-simplexes (rows).
        boundary (list): list of the boundary matrices at all dimensions (0..dim-1).
        node_coords (npt.NDArray): Cartesian coordinates (cols) of the nodes (rows) of
            the simplicial complex.
        primal_volumes (list): list where each entry p is an array containing all the
            volumes of the primal p-simplices.
        dual_volumes (list): list where each entry p is an array containing all
            the volumes of the dual p-simplices.
        simplices_faces (list): list where each entry p is a matrix containing
            the IDs of the (p-1)-simplices (cols) belonging to each p-simplex (rows).
        hodge_star (list): list where each entry is an array containing the
            diagonal of the Hodge star matrix.
    """

    def __init__(self, tet_node_tags: npt.NDArray, node_coords: npt.NDArray,
                 is_well_centered: bool = False):

        self.node_coords = node_coords.astype(dctkit.float_dtype)
        tet_node_tags = tet_node_tags.astype(dctkit.int_dtype)
        self.num_nodes = node_coords.shape[0]
        self.space_dim = node_coords.shape[1]
        self.float_dtype = dctkit.float_dtype
        self.int_dtype = dctkit.int_dtype
        self.is_well_centered = is_well_centered
        self.ref_covariant_basis = None
        self.ref_metric_contravariant = None

        # compute complex dimension from top-level simplices
        self.dim = tet_node_tags.shape[1] - 1

        self.S = [npt.NDArray[Any]] * (self.dim + 1)
        self.S[-1] = tet_node_tags

        self.get_boundary_operators()

        # FIXME: maybe we don't want to compute the metric by default, in some
        # applications is not needed...
        if self.dim == 2:
            self.reference_metric = self.get_current_metric_2D(self.node_coords)

    def get_boundary_operators(self):
        """Compute all the COO representations of the boundary matrices."""
        self.boundary = sl.ShiftedList([None] * self.dim, -1)
        self.simplices_faces = sl.ShiftedList([None] * self.dim, -1)
        for p in range(self.dim):
            boundary, vals, faces_ordered = compute_boundary_COO(self.S[self.dim - p])
            self.boundary[self.dim - p] = boundary
            self.S[self.dim - p - 1] = vals
            self.simplices_faces[self.dim - p] = compute_simplices_faces(
                self.S[self.dim - p], faces_ordered)

    def get_complex_boundary_faces_indices(self):
        """Find the IDs of the boundary faces of the complex, i.e. the row indices of
        the boundary faces in the matrix S[dim-1].
        """
        # boundary faces IDs appear only once in the matrix simplices_faces[dim]
        unique_elements, counts = np.unique(
            self.simplices_faces[self.dim], return_counts=True)
        self.bnd_faces_indices = np.sort(unique_elements[counts == 1])

    def get_tets_containing_a_boundary_face(self):
        """Compute a list in which the i-th element is the index of the top-level
        simplex in which the i-th boundary face belongs."""
        if not hasattr(self, "bnd_faces_indices"):
            self.get_complex_boundary_faces_indices()
        dim = self.dim
        # the index of the top level simplex in which the i-th boundary face belongs
        # is the (only) row index in which i appears in simplices_faces[dim].
        self.tets_cont_bnd_face = [np.nonzero(
            self.simplices_faces[dim] == i)[0][0] for i in self.bnd_faces_indices]

    def get_circumcenters(self):
        """Compute all the circumcenters."""
        self.circ = sl.ShiftedList([None] * (self.dim), -1)
        self.bary_circ = sl.ShiftedList([None] * (self.dim), -1)
        for p in range(1, self.dim + 1):
            S = self.S[p]
            C, B = circ.circumcenter(S, self.node_coords)
            self.circ[p] = C
            self.bary_circ[p] = B

    def get_primal_volumes(self):
        """Compute all the primal volumes."""
        self.primal_volumes = [None]*(self.dim + 1)
        self.primal_volumes[0] = np.ones(self.num_nodes, dtype=self.float_dtype)
        for p in range(1, self.dim + 1):
            S = self.S[p]
            if p == self.space_dim:
                primal_volumes = volume.signed_volume(S, self.node_coords)
            else:
                primal_volumes = volume.unsigned_volume(S, self.node_coords)
            self.primal_volumes[p] = primal_volumes

    def get_dual_volumes(self):
        """Compute all the dual volumes."""
        if not hasattr(self, "circ"):
            self.get_circumcenters()

        self.dual_volumes = [None] * (self.dim+1)
        self.dual_volumes[self.dim] = np.ones(self.S[self.dim].shape[0],
                                              dtype=self.float_dtype)

        # loop over simplices at all dimensions
        for p in range(self.dim, 0, -1):
            num_p, _ = self.simplices_faces[p].shape

            if p == 1:
                # circ_pm1 = circumcenters of the (p-1)-simplices and the circumcenters
                # of the nodes (0-simplices) are the nodes itself.
                circ_pm1 = self.node_coords
                num_pm1 = self.num_nodes
            else:
                circ_pm1 = self.circ[p - 1]
                num_pm1, _ = self.S[p - 1].shape
            dv = np.zeros(num_pm1, dtype=self.float_dtype)
            # Loop over p-simplices
            for i in range(num_p):
                face_id = self.simplices_faces[p][i, :]
                # Distances between circumcenter of the p-simplex and the boundary
                # (p-1)-simplices
                length = np.linalg.norm(self.circ[p][i, :] - circ_pm1[face_id, :],
                                        axis=1)

                # Find opposite vertexes to the (p-1)-simplices
                if p == 1:
                    opp_vert = np.array(
                        [list(set(self.S[p][i]) - set(j.flatten())) for j in face_id])
                else:
                    opp_vert = np.array([list(set(self.S[p][i]) -
                                              set(self.S[p - 1][j])) for j in face_id])
                opp_vert_index = [list(self.S[p][i]).index(j) for j in opp_vert]

                # Sign of the dual volume of the boundary (p-1)-simplex = sign of
                # the barycentric coordinate of the circumcenter of the parent
                # p-simplex relative to the opposite vertex
                sign = np.copysign(1, self.bary_circ[p][i, opp_vert_index])
                # Update dual volume of the boundary (p-1)-simplex
                dv[face_id] += sign * (length*self.dual_volumes[p][i] /
                                       (self.dim - p + 1))

            self.dual_volumes[p - 1] = dv

    def get_hodge_star(self):
        """Compute all the Hodge stars, and their inverses if the mesh is well-centered.
        """
        n = self.dim

        if not hasattr(self, "primal_volumes"):
            self.get_primal_volumes()

        if not hasattr(self, "dual_volumes"):
            self.get_dual_volumes()

        self.hodge_star = [self.dual_volumes[i]/self.primal_volumes[i]
                           for i in range(n + 1)]

        if self.is_well_centered:
            # adjust the sign in order to have star_inv*star = (-1)^(p*(n-p))
            self.hodge_star_inverse = [(-1)**(i*(n-i))/self.hodge_star[i]
                                       for i in range(n + 1)]

    def get_primal_edge_vectors(self):
        """Compute the primal edge vectors."""
        primal_edges = self.S[1]
        node_coords = self.node_coords
        self.primal_edges_vectors = node_coords[primal_edges[:, 1], :] - \
            node_coords[primal_edges[:, 0], :]

    def get_dual_edge_vectors(self):
        """Compute the dual edge vectors."""
        dim = self.dim
        # dual nodes == circumcenters of the n-simplices
        if not hasattr(self, "circ"):
            self.get_circumcenters()
        dual_nodes_coords = self.circ[dim]
        num_dual_edges = self.S[dim-1].shape[0]

        # apply the dual coboundary to the dual vector-valued 0-cochain
        # of the coordinates of the dual nodes
        self.dual_edges_vectors = spmv.spmm(self.boundary[0],
                                            dual_nodes_coords,
                                            shape=num_dual_edges)
        self.dual_edges_vectors *= (-1)**dim

        # dual edges that belong to "incomplete boundary cells" must be treated
        # separately, as described below

        # construct the array consisting of the positions of the circumcenters of the
        # boundary faces arranged by rows, padded with zeros for the non-boundary edges
        if not hasattr(self, "bnd_faces_indices"):
            self.get_complex_boundary_faces_indices()
        circ_faces = self.circ[dim-1]
        circ_bnd_faces = np.zeros(circ_faces.shape, dtype=dctkit.float_dtype)
        circ_bnd_faces[self.bnd_faces_indices] = circ_faces[self.bnd_faces_indices]

        # adjust the signs based on the appropriate entries of the dual coboundary
        # NOTE: here we take the values of the boundary matrix, we fix their signs later
        # to avoid allocating a new matrix for the coboundary.
        rows, _, vals = self.boundary[0]
        # extract rows indices with only one non-zero element, as they correspond to
        # dual edges incident on boundary faces
        _, idx, count = np.unique(rows, return_index=True, return_counts=True)
        boundary_rows_idx = idx[count == 1]

        # the action of the dual coboundary on the collection of the coordinates of the
        # dual nodes produces incomplete results on the dual edges having only one dual
        # node as a boundary (i.e. those who are incident on the boundary faces). To
        # compensate for this, add the coordinates of the circumcenters of the boundary
        # faces with the appropriate sign, given by the orientation of the dual edge
        # contained in the dual coboundary matrix.
        # NOTE: vals must be a COLUMN vector
        # NOTE: the (-1)**dim factor accounts for the correct sign of the dual
        # coboundary matrix
        sign = -vals[boundary_rows_idx][:, None]*(-1)**dim
        complement = circ_bnd_faces
        complement[self.bnd_faces_indices] *= sign

        self.dual_edges_vectors += complement

        self.dual_edges_lengths = np.linalg.norm(self.dual_edges_vectors, axis=1)

    def get_flat_DPD_weights(self):
        """Compute the matrix where each non-negative entry (i,j) is the ratio between
           the length of the j-th dual edge contained in the i-th n-simplex and the
           total length of the j-th dual edge.

           This ratio appears as a weighting factor in the computation of the discrete
           flat operator.
        """
        if not hasattr(self, "dual_edges_lengths"):
            self.get_dual_edge_vectors()

        dim = self.dim
        B = self.simplices_faces[dim]
        num_n_simplices = self.S[dim].shape[0]
        num_nm1_simplices = self.S[dim-1].shape[0]
        self.dual_edges_fractions_lengths = np.zeros(
            (num_n_simplices, num_nm1_simplices), dtype=dctkit.float_dtype)

        for i in range(num_n_simplices):
            # get the indices of the (n-1)-simplices belonging to the i-th n-simplex
            dual_edges_indices = B[i, :]
            # construct the matrix containing the difference vectors between the
            # circumcenter of the i-th n-simplex and the circumcenters of the dual edges
            # intersecting such a simplex, arranged in rows.
            diff_circs = self.circ[dim][i, :] - self.circ[dim-1][dual_edges_indices, :]
            # take the norms of the difference vectors
            self.dual_edges_fractions_lengths[i, :][
                dual_edges_indices] = np.linalg.norm(diff_circs, axis=1)

        self.flat_DPD_weights = self.dual_edges_fractions_lengths / \
            self.dual_edges_lengths
        # in the case of non-well centered mesh an entry of the flat weights matrix
        # can be NaN. In this case, the corresponding dual edge is the null vector,
        # hence we shouldn't take in account dot product with it. We then replace
        # any NaN with 0.
        self.flat_DPD_weights = np.nan_to_num(self.flat_DPD_weights)

    def get_flat_DPP_weights(self):
        # FIXME: extend to 3D case.
        # NOTATION:
        # s^i: generic i-simplex of the simplicial complex self.
        # s^j > s^i: s^i is a proper face of s^j (hence i<j)
        if not hasattr(self, "primal_edge_vectors"):
            self.get_primal_edge_vectors()
            self.get_tets_containing_a_boundary_face()

        if self.dim == 2:
            # in this case the entries of the flat_DPD matrix coincides
            # with the entries of the flat_DPP matrix, since in this case
            # n -1 = 1. Hence summing over s^n > s^1 is the same as summing
            # over s^n > s^{n-1} and moreover |★s^{n-1} ∩ s^n| = |★s^1 ∩ s^n|
            if not hasattr(self, "flat_DPD_weights"):
                self.get_flat_DPD_weights()
            self.flat_DPP_weights = self.flat_DPD_weights

    def get_current_covariant_basis(self, node_coords: npt.NDArray | Array) -> Array:
        """Compute the current covariant basis of each face of a 2D simplicial complex.

        Args:
            node_coords: matrix of shape (n, space_dim) where the i-th row is the
                    vector of coordinates of i-th node of the simplex in the current
                    configuration.

        Returns:
            the multiarray of shape (n, 2, 2), where n is the number of 2-simplices
                    and each 2x2 matrix is the current covariant basis of the
                    corresponding 2-simplex.


        """
        dim = self.dim
        B = self.simplices_faces[dim]
        primal_edges = self.S[1]
        # construct the matrix in which the i-th row corresponds to the vector
        # of coordinates of the i-th primal edge
        primal_edge_vectors = node_coords[primal_edges[:, 1], :2] - \
            node_coords[primal_edges[:, 0], :2]
        # construct the multiarray of shape (n, 2, 2) where any 2x2 matrix represents
        # the coordinates of the first two edge vectors (arranged in rows) belonging to
        # corresponding primal 2-simplex i.e. the rows are the vectors g_i
        current_covariant_basis = primal_edge_vectors[B][:, :2, :]

        # compute the matrix (a_k)r and its transpose
        if self.ref_covariant_basis is None:
            self.ref_covariant_basis = current_covariant_basis
            self.ref_covariant_basis_T = jnp.transpose(
                self.ref_covariant_basis, axes=(0, 2, 1))

        return current_covariant_basis

    def get_current_metric_2D(self, node_coords: npt.NDArray | Array) -> Array:
        """Compute the current metric of a 2D simplicial complex.

            Args:
                node_coords: matrix of shape (n, space_dim) where the i-th row is the
                    vector of coordinates of i-th node of the simplex in the current
                    configuration.

            Returns:
                the multiarray of shape (n, 2, 2), where n is the number of 2-simplices
                and each 2x2 matrix is the current metric of the corresponding
                2-simplex.
        """
        # NOTATION:
        # a_i, reference covariant basis (pairs of edge vectors of a primal 2-simplex)
        # a^i = a^(ik)a_k, reference contravariant basis
        # G = current metric
        # g_(ij), covariant components of the current metric
        # g_i, current covariant basis
        # (a_k)r, r-th Cartesian component of the basis vector a_k
        # e_r, global Cartesian basis
        # g^(ij)_p the contravariant components of the the pull-back of the current
        # metric

        current_covariant_basis = self.get_current_covariant_basis(node_coords)

        # compute the matrix (a_k)r and its transpose
        if self.ref_covariant_basis is None:
            self.ref_covariant_basis = current_covariant_basis
            self.ref_covariant_basis_T = jnp.transpose(
                self.ref_covariant_basis, axes=(0, 2, 1))

        # compute g_(ij) = g_i dot g_j
        current_metric_covariant = current_covariant_basis @ jnp.transpose(
            current_covariant_basis, axes=(0, 2, 1))

        # compute a^(ij)
        if self.ref_metric_contravariant is None:
            ref_metric_covariant = current_metric_covariant
            self.ref_metric_contravariant = jnp.linalg.inv(ref_metric_covariant)

        # compute g^(km)_p = g_(ij) a^(ik) a^(jm)
        pullback_current_metric_contravariant = ((self.ref_metric_contravariant @
                                                 current_metric_covariant) @
                                                 self.ref_metric_contravariant)

        # compute the components of G = g^(km)_p (a_k)r (a_m)s e_r x e_s
        current_cartesian_metric = ((self.ref_covariant_basis_T @
                                     pullback_current_metric_contravariant) @
                                    self.ref_covariant_basis)
        return current_cartesian_metric

    def get_deformation_gradient(self, node_coords: npt.NDArray) -> Array:
        """Compute the deformation gradient of a 2D simplicial complex.

        Args:
                node_coords: matrix of shape (n, space_dim) where the i-th row is the
                    vector of coordinates of i-th node of the simplex in the current
                    configuration.

            Returns:
                the multiarray of shape (n, 2, 2), where n is the number of 2-simplices
                    and each 2x2 matrix is the deformation gradient of the
                    corresponding 2-simplex.

        """
        current_covariant_basis = self.get_current_covariant_basis(node_coords)

        if self.ref_metric_contravariant is None:
            current_metric_covariant = current_covariant_basis @ jnp.transpose(
                current_covariant_basis, axes=(0, 2, 1))
            ref_metric_covariant = current_metric_covariant
            self.ref_metric_contravariant = jnp.linalg.inv(ref_metric_covariant)

        # compute F_(jl) = (a'_i)_j (g_R)^(ik) (a_k)_l
        F = jnp.transpose(current_covariant_basis, axes=(0, 2, 1)
                          ) @ self.ref_metric_contravariant @ self.ref_covariant_basis
        return F


def __simplex_array_parity(s: npt.NDArray) -> npt.NDArray:
    """Compute the number of transpositions needed to sort the array in ascending order
       modulo 2. (Copied from PyDEC, dec/simplex_array.py)

        Args:
            s: array of the simplices.

        Returns:
            array of the transpositions modulo 2.

    """
    s = s.copy()
    M, N = s.shape

    # number of transpositions used to sort the indices of each simplex (row of s)
    trans = np.zeros_like(s[:, 0])
    seq = np.arange(M)

    # count the transpositions
    for _ in range(N - 1):
        pos = s.argmin(axis=1)
        s[seq, pos] = s[:, 0]
        pos.clip(0, 1, pos)
        trans = trans + pos
        s = s[:, 1:]

    # compute parity
    trans %= 2

    return trans


def compute_boundary_COO(S: npt.NDArray) -> Tuple[list, npt.NDArray, npt.NDArray]:
    """Compute the COO representation of the boundary matrix of all p-simplices.

    Args:
        S: matrix of the IDs of the nodes (cols) belonging to each p-simplex (rows).

    Returns:
        a tuple containing a list with the COO representation of the boundary, the
        matrix of node IDs belonging to each (p-1)-face ordered lexicographically,
        and a matrix containing the IDs of the nodes (cols) belonging to
        each p-simplex (rows) counted with repetition and ordered lexicographically.

    """
    # number of p-simplices
    num_simplices = S.shape[0]
    # nodes per p-simplex = p + 1
    nodes_per_simplex = S.shape[1]

    N = num_simplices * nodes_per_simplex

    # compute array of relative orientations of the (p-1)-faces wrt the
    # p-simplices
    orientations = 1 - 2 * __simplex_array_parity(S)

    # sort the rows of S lexicographically
    # FIXME: avoid making a copy and sorting every time
    F = S.copy()
    F.sort(axis=1)

    # S_(p-1) matrix with repeated (p-1)-simplices and with two extra columns
    S_pm1_ext = np.empty((N, nodes_per_simplex + 1), dtype=dctkit.int_dtype)

    # find the node IDs of the (p-1)-simplices and store their relative
    # orientations wrt the parent simplex
    for i in range(nodes_per_simplex):
        # remove the i-th column from the S matrix and put the result in the
        # appropriate block S_pm1_ext
        rows = S_pm1_ext[num_simplices * i:num_simplices * (i + 1)]
        rows[:, :i] = F[:, :i]
        rows[:, i:-2] = F[:, i + 1:]

        # put IDs of the p-simplices in the last column
        rows[:, -1] = np.arange(num_simplices)

        # put the orientations in the next-to-last-column
        rows[:, -2] = ((-1)**i) * orientations

    # order faces lexicographically (copied from PyDEC)
    # FIXME: maybe use sort
    faces_ordered = S_pm1_ext[np.lexsort(S_pm1_ext[:, :-2].T[::-1])]
    values = faces_ordered[:, -2]
    column_index = faces_ordered[:, -1]
    faces = faces_ordered[:, :-2]

    # compute the matrix obtained from faces removing the duplicate rows and mantaining
    # the lexicographically order (unique faces) and the vector of occurences for each
    # non-duplicate row (rows_index); e.g. if faces = [[1,2];[1,3];[1,3];[1,4]],
    # then unique_faces = [[1,2]; [1,3]; [1,4]] and rows_index = [0; 1; 1; 2]
    unique_faces, rows_index = np.unique(faces, axis=0, return_inverse=True)
    rows_index = rows_index.astype(dtype=dctkit.int_dtype)
    boundary_COO = [rows_index, column_index, values]

    return boundary_COO, unique_faces, faces_ordered


def compute_simplices_faces(S: npt.NDArray, faces_ordered:
                            npt.NDArray) -> npt.NDArray:
    """Compute the matrix containing the IDs of the (p-1)-simplices (cols) belonging
    to each p-simplex (rows).

    Args:
        S: matrix of the IDs of the nodes (cols) belonging to each p-simplex (rows).

    Returns:
        a matrix containing the IDs of the (p-1)-simplices (cols) belonging
            to each p-simplex (rows).

    """

    nodes_per_simplex = S.shape[1]
    p = nodes_per_simplex - 1

    # for triangles and tets, compute the matrix explicitly
    if p > 1:
        # order faces_ordered w.r.t last column
        faces_ordered_last = faces_ordered[faces_ordered[:, -1].argsort()]

        # unique returns an array that must be reshaped into a matrix
        _, simplices_faces = np.unique(
            faces_ordered_last[:, :-2], axis=0, return_inverse=True)
        simplices_faces = simplices_faces.reshape(
            faces_ordered.shape[0] // nodes_per_simplex, nodes_per_simplex)

    # for edges, take S[1]
    else:
        simplices_faces = S
    simplices_faces.astype(dtype=dctkit.int_dtype)

    return simplices_faces
