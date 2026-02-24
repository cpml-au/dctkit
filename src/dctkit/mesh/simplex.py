import numpy as np
import dctkit
from dctkit.mesh import circumcenter as circ, volume
from dctkit.math import shifted_list as sl
from dctkit.math import spmm
import numpy.typing as npt
from jax import Array
import jax.numpy as jnp
from typing import Tuple, Any, List
from functools import partial
from jax import vmap, jit
from dctkit.math.permutation import permutation_sign, compute_permutation_vectors


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
                 space_dim: int = 3, is_well_centered: bool = False):

        self.node_coords = node_coords.astype(dctkit.float_dtype)[:, :space_dim]
        tet_node_tags = tet_node_tags.astype(dctkit.int_dtype)
        self.num_nodes = node_coords.shape[0]
        self.space_dim = space_dim
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
            # NOTE: the cofaces of a simplex can be determined by the coboundary matrix
            # maybe we do not really need to store simplices_faces...
            self.simplices_faces[self.dim - p] = compute_simplices_faces(
                self.S[self.dim - p], faces_ordered)

    def get_complex_boundary_simplices_indices(self):
        """Finds boundary k-simplices IDs for any k=0,...,dim."""
        self.boundary_simplices = [None] * (self.dim + 1)

        # top-level boundary (Faces in 3D, Edges in 2D)
        # these are (n-1)-simplices that belong to exactly ONE n-simplex
        boundary_mat = self.boundary[self.dim]
        # indices of (n-1)-simplices
        rows = boundary_mat[0]
        unique, counts = np.unique(rows, return_counts=True)
        bnd_top_faces = unique[counts == 1]
        self.boundary_simplices[self.dim - 1] = bnd_top_faces

        # boundary Nodes (0-simplices)
        # recursively find all nodes contained in the boundary faces
        # get all nodes (S[0]) belonging to the boundary (n-1)-simplices
        # we use S[dim-1] to get the connectivity of the boundary elements
        bnd_nodes = np.unique(self.S[self.dim - 1][bnd_top_faces])
        self.boundary_simplices[0] = bnd_nodes

        # intermediate dimensions (e.g., edges in a 3D mesh boundary)
        if self.dim == 3:
            # Boundary edges are edges that belong to the boundary faces
            # We can find them by looking at S[2] (faces) -> S[1] (edges)
            # NOTE: This requires having a simplices_faces map for faces
            bnd_faces_edges = self.simplices_faces[2][bnd_top_faces]
            self.boundary_simplices[1] = np.unique(bnd_faces_edges)

        # identify boundary n-simplices (Tets in 3D, Tris in 2D)
        # any n-simplex that has at least one face on the boundary
        simplices_faces = self.simplices_faces[self.dim]
        is_boundary_simplex = np.any(np.isin(simplices_faces, bnd_top_faces), axis=1)
        self.boundary_simplices[self.dim] = np.nonzero(is_boundary_simplex)[0]

    def get_tets_containing_a_boundary_face(self):
        """Compute a list in which the i-th element is the index of the top-level
        simplex in which the i-th boundary face belongs."""
        if not hasattr(self, "boundary_simplices"):
            self.get_complex_boundary_simplices_indices()
        dim = self.dim - 1
        self.tets_cont_bnd_face = get_cofaces(
            self.boundary_simplices[dim], dim, self)

    def get_circumcenters(self):
        """Compute all the circumcenters."""
        # self.circ = sl.ShiftedList([None] * (self.dim), -1)
        # self.bary_circ = sl.ShiftedList([None] * (self.dim), -1)
        self.circ = [None] * (self.dim+1)
        self.bary_circ = [None] * (self.dim+1)
        self.circ[0] = self.node_coords
        self.bary_circ[0] = self.node_coords
        for p in range(1, self.dim + 1):
            S = self.S[p]
            C, B = circ.circumcenter(S, self.node_coords)
            self.circ[p] = C
            self.bary_circ[p] = B

    def get_S_dual(self):
        """Compute S_dual[k] for all k = 0.1

        Each S_dual[k] is a matrix where each row contains the indices of dual nodes
        (i.e., circumcenters of top-dimensional simplices) that form a dual k-simplex.

        Stores the result in self.S_dual[k].
        """
        # FIXME: test properly this routine!
        if not hasattr(self, "boundary_simplices"):
            self.get_complex_boundary_simplices_indices()
        dim = self.dim
        self.S_dual = [None]*2

        # store dual 0-simplices
        self.S_dual[0] = np.arange(
            self.S[dim].shape[0], dtype=dctkit.int_dtype).reshape(-1, 1)

        # dual 0-simplices are the circumcenters of top-dimensional primal simplices
        # These are not stored in S_dual but are the "nodes" for the dual complex

        num_codim_1 = self.S[dim - 1].shape[0]

        S_dual_interior_k = []

        for idx in range(num_codim_1):
            # Find all top-simplices (of dim) that contain this codim-k simplex
            cofaces = np.nonzero(self.simplices_faces[dim] == idx)[0]
            if len(cofaces) == 2:
                S_dual_interior_k.append(cofaces)

        S_dual_interior_k = np.array(S_dual_interior_k, dtype=dctkit.int_dtype)
        S_dual_bnd_k_idx = self.boundary_simplices[dim-1]
        S_dual_interior_k_idx = np.setdiff1d(
            np.arange(num_codim_1), S_dual_bnd_k_idx)
        S_dual_k = np.empty((num_codim_1, 2), dtype=dctkit.int_dtype)
        # set placeholder for the boundary
        S_dual_k[S_dual_bnd_k_idx] = 0.
        # set correct value for the interior
        S_dual_k[S_dual_interior_k_idx] = S_dual_interior_k
        self.S_dual[1] = S_dual_k

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
        self.dual_edges_vectors = spmm.spmm(self.boundary[0],
                                            dual_nodes_coords,
                                            shape=num_dual_edges)
        self.dual_edges_vectors *= (-1)**dim

        # dual edges that belong to "incomplete boundary cells" must be treated
        # separately, as described below

        # construct the array consisting of the positions of the circumcenters of the
        # boundary faces arranged by rows, padded with zeros for the non-boundary edges
        if not hasattr(self, "bnd_faces_indices"):
            self.get_complex_boundary_simplices_indices()
        if dim == 1:
            # in this case faces = nodes
            circ_faces = self.node_coords
        else:
            circ_faces = self.circ[dim-1]
        circ_bnd_faces = np.zeros(circ_faces.shape, dtype=dctkit.float_dtype)
        circ_bnd_faces[self.boundary_simplices[dim-1]
                       ] = circ_faces[self.boundary_simplices[dim-1]]

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
        complement[self.boundary_simplices[dim-1]] *= sign

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

        # Preallocate lists for COO format
        rows = []
        cols = []
        data = []

        for i in range(num_n_simplices):
            # get the indices of the (n-1)-simplices belonging to the i-th n-simplex
            dual_edges_indices = B[i, :]
            # construct the matrix containing the difference vectors between the
            # circumcenter of the i-th n-simplex and the circumcenters of the dual edges
            # intersecting such a simplex, arranged in rows.
            diff_circs = self.circ[dim][i, :] - self.circ[dim-1][dual_edges_indices, :]
            # take the norms of the difference vectors
            lengths = np.linalg.norm(diff_circs, axis=1)

            # append nonzero entries
            for j, val in zip(dual_edges_indices, lengths):
                if val != 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(val/self.dual_edges_lengths[j])

        # Convert to NumPy arrays
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        data = np.array(data, dtype=dctkit.float_dtype)
        # in the case of non-well centered mesh an entry of the flat weights matrix
        # can be NaN. In this case, the corresponding dual edge is the null vector,
        # hence we shouldn't take in account dot product with it. We then replace
        # any NaN with 0.
        data = np.nan_to_num(data)

        # Build sparse COO matrix of raw lengths
        self.flat_DPD_weights = (rows, cols, data)

    def get_flat_dual_upw_weights(self):
        """Compute the matrix where each nonzero entry (i, j) corresponds to the
           contribution of the j-th dual (n-1)-cell (dual edge) to the i-th n-simplex.
           The weight represents the fraction of the dual edge length that lies inside
           the n-simplex. In the current implementation, this contribution is treated
           as unitary (value 1.0) when the dual edge intersects the simplex in a
           non-degenerate way, and zero otherwise.

           This ratio appears as a weighting factor in the computation of the discrete
           flat operator.
        """
        if not hasattr(self, "S_dual"):
            self.get_S_dual()

        num_dual_edges = self.S_dual[1].shape[0]

        # Preallocate lists for COO format
        rows = []
        cols = []
        data = []
        for i in range(num_dual_edges):
            dual_edge = self.S_dual[1][i]
            # check that the dual edge is not degenerate, that is not of length 0
            if jnp.sum(dual_edge) > 0:
                # extract only the left value
                rows.append(dual_edge[0])
                cols.append(i)
                data.append(1.)

        # Convert to NumPy arrays
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        data = np.array(data, dtype=dctkit.float_dtype)

        # Build sparse COO matrix of raw lengths
        self.flat_dual_upw_weights = (rows, cols, data)

    def get_flat_DPP_weights(self):
        # FIXME: extend to 3D case.
        # NOTATION:
        # s^i: generic i-simplex of the simplicial complex self.
        # s^j > s^i: s^i is a proper face of s^j (hence i<j)
        if not hasattr(self, "primal_edge_vectors"):
            self.get_primal_edge_vectors()
            # NOT USED IN dim = 2
            # self.get_tets_containing_a_boundary_face()

        if self.dim == 2:
            # in this case the entries of the flat_DPD matrix coincides
            # with the entries of the flat_DPP matrix, since in this case
            # n -1 = 1. Hence summing over s^n > s^1 is the same as summing
            # over s^n > s^{n-1} and moreover |★s^{n-1} ∩ s^n| = |★s^1 ∩ s^n|
            if not hasattr(self, "flat_DPD_weights"):
                self.get_flat_DPD_weights()
            self.flat_DPP_weights = self.flat_DPD_weights

    def get_flat_PDP_weights(self):
        """Construct the primal-dual-primal (PDP) weighting matrix for the
           discrete flat operator."""
        num_edges = self.S[1].shape[0]

        # Preallocate lists for COO format
        rows = []
        cols = []
        data = []

        # FIXME: optimize this routine with jax.vmap
        for i in range(num_edges):
            edge_nodes = self.S[1][i]
            for node in edge_nodes:
                rows.append(node)
                cols.append(i)
                data.append(0.5)

        # Convert to NumPy arrays
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        data = np.array(data, dtype=dctkit.float_dtype)
        self.flat_PDP_weights = (rows, cols, data)

    def get_current_covariant_basis(self, node_coords: npt.NDArray | Array) -> Array:
        """Compute the current covariant basis of each face of a 2D simplicial complex.

        Args:
            node_coords: matrix of shape (n, space_dim) where the i-th row is the
                    vector of coordinates of i-th node of the simplex in the current
                    configuration.

        Returns:
            the multiarray of shape (n, 2, 2), where n is the number of 2-simplices
            and each 2x2 matrix is the current covariant basis of the corresponding
            2-simplex.


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
            node_coords: matrix of shape (n, space_dim) where the i-th row is the vector
                of coordinates of i-th node of the simplex in the current configuration.

        Returns:
            the multiarray of shape (n, 2, 2), where n is the number of 2-simplices
            and each 2x2 matrix is the deformation gradient of the corresponding
            2-simplex.

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

    @partial(vmap, in_axes=(None, 0, None))
    def find_simplex_idx(self, s: Array, S: Array) -> Array:
        """Finds the index of a given simplex in a set of simplices.

        Args:
            s: A 1D array representing a simplex (e.g., a set of vertex indices).
            S: A 2D array where each row is a simplex.

        Returns:
            the index of the simplex `s` in `S`. If `s` is not found,
            returns -1.
        """
        # Broadcast and compare all rows to the given row
        matches = jnp.all(S == s, axis=1)
        # Find the index where all elements match
        simplex_idx = jnp.where(matches, size=1, fill_value=-1)[0][0]
        return simplex_idx

    @partial(vmap,  in_axes=(None, 0, None, None, None, None))
    def compute_cup_product_entry(self, simplex, perm_vec, p, S_p, S_q):
        perm_simplex = simplex[perm_vec]
        # split the perm simplices in vector of indices compatible
        # with c_1 and c_2
        perm_simplex_c_1 = perm_simplex[:, :p+1]
        perm_simplex_c_2 = perm_simplex[:, p:]

        # since the perm simplices may not have the same orientations
        # as the original one, we need to account for that
        perm_ord_c_1 = jnp.argsort(perm_simplex_c_1, axis=1)
        perm_ord_c_2 = jnp.argsort(perm_simplex_c_2, axis=1)
        sgn_orient_c_1 = permutation_sign(perm_ord_c_1)
        sgn_orient_c_2 = permutation_sign(perm_ord_c_2)

        # compute the indexes for every (ordered) perm_simplex
        ord_simplex_c_1 = jnp.take_along_axis(
            perm_simplex_c_1, perm_ord_c_1, axis=1)
        ord_simplex_c_2 = jnp.take_along_axis(
            perm_simplex_c_2, perm_ord_c_2, axis=1)
        perm_idx_c_1 = self.find_simplex_idx(ord_simplex_c_1, S_p)
        perm_idx_c_2 = self.find_simplex_idx(ord_simplex_c_2, S_q)
        lookup_simplex = jnp.array([perm_idx_c_1, perm_idx_c_2])
        sgn_orient_simplex = jnp.array([sgn_orient_c_1, sgn_orient_c_2])
        return lookup_simplex, sgn_orient_simplex

    def get_cup_product_coeffs(self):
        """Precompute lookup tables for the discrete cup product.

        This method builds and stores the coefficient tables required to evaluate
        the cup product between discrete p- and q-cochains, for both the primal
        and dual cell complexes."""
        if not hasattr(self, "S_dual"):
            self.get_S_dual()
        types = ["primal", "dual"]
        S_lists = [self.S, self.S_dual]
        max_dims = [self.dim, 1]
        self.cup_lookup = {}
        # jit the function that compute the cup_product entry
        jitted_cup_prod_entry_fun = jit(
            self.compute_cup_product_entry, static_argnums=(2,))

        for k, type_ in enumerate(types):
            for p in range(max_dims[k]):
                for q in range(p, max_dims[k] - p + 1):
                    if q > 0:
                        # cup_product dim
                        cup_product_dim = p + q
                        S_p = S_lists[k][p]
                        S_q = S_lists[k][q]
                        S_cup_product = S_lists[k][cup_product_dim]

                        # generate the permutation vectors and compute its signs
                        perm_vec = compute_permutation_vectors(cup_product_dim+1)
                        sgn_perm_vec = permutation_sign(perm_vec)

                        # compute lookup table
                        lookup, sgn_orient = jitted_cup_prod_entry_fun(
                            S_cup_product, perm_vec, p, S_p, S_q)

                        sgn_orient = sgn_orient.astype(jnp.int8)
                        perm_vec = perm_vec.astype(jnp.int8)

                        self.cup_lookup[(type_, p, q)] = {"lookup": lookup,
                                                          "sgn_orient": sgn_orient,
                                                          "sgn_perm_vec": sgn_perm_vec}


def get_cofaces(faces_ids: list[int] | npt.NDArray, faces_dim: int,
                S: SimplicialComplex) -> List[npt.NDArray]:
    """Get the IDs of the cofaces of a simplex, i.e. the neighour simplices of one
    higher dimension.

    Args:
        faces_ids: list or array containing the IDs of the faces for which the cofaces
            should be determined.
        faces_dim: dimension of each face in the list.
        S: simplicial complex to which the faces belong.

    Returns:
        list of arrays, where each array contains the IDs of the cofaces of a face.
    """
    # the indices of the parent simplices to which the faces belong are the row indices
    # of the matrix simplices_faces in which the IDs of the faces appear
    cofaces_ids = [np.nonzero(S.simplices_faces[faces_dim+1] == i)[0].flatten()
                   for i in faces_ids]
    return cofaces_ids


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
        a matrix containing the IDs of the (p-1)-simplices (cols) belonging to each
        p-simplex (rows).

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
