import numpy as np
import dctkit
from dctkit.math import circumcenter as circ
from dctkit.math import shifted_list as sl
from dctkit.math import volume, spmv


class SimplicialComplex:
    """Simplicial complex class.

    Args:
        tet_node_tags (int32 np.array): matrix containing the IDs of the nodes
            (cols) belonging to each tetrahedron or top-level simplex (rows).
        node_coord (float np.array): Cartesian coordinates (columns) of all the
        nodes (rows) of the simplicial complex.
        belem_tags (float np.array): matrix containing the IDs of the nodes 
            (cols) belonging to each boundary (n-1)-simplex (rows).
        is_well_centered (bool): True if the mesh is well-centered.
    Attributes:
        dim (int32): dimension of the complex.
        S (list): list where each entry p is a matrix containing the IDs of the
            nodes belonging to each p-simplex.
        circ (list): list where each entry p is a matrix containing the
            coordinates of the circumcenters (cols) of all the p-simplexes (rows).
        boundary (list): list of the boundary matrices at all dimensions (0..dim-1).
        node_coord (float np.array): Cartesian coordinates (columns) of all the
            nodes (rows) of the simplicial complex.
        primal_volumes (list): list where each entry p is an array containing all the
            volumes of the primal p-simplices.
        dual_volumes (list): list where each entry p is an array containing all
            the volumes of the dual p-simplices.
        B (list): list where each entry p is a matrix containing the IDs of the
            (p-1)-simplices (cols) belonging to each p-simplex (rows).
        hodge_star (list): list where each entry is an array containing the
            diagonal of the Hodge star matrix.
    """

    def __init__(self, tet_node_tags, node_coord, belem_tags=None, is_well_centered=False):

        # store the coordinates of the nodes
        node_coord = np.array(node_coord, dtype=dctkit.float_dtype)
        tet_node_tags = np.array(tet_node_tags, dtype=dctkit.int_dtype)
        self.node_coord = node_coord
        self.num_nodes = node_coord.shape[0]
        self.embedded_dim = node_coord.shape[1]
        self.float_dtype = dctkit.float_dtype
        self.int_dtype = dctkit.int_dtype
        self.is_well_centered = is_well_centered
        self.belem_tags = belem_tags

        # compute complex dimension from top-level simplices
        self.dim = tet_node_tags.shape[1] - 1

        self.S = [None] * (self.dim + 1)
        self.S[-1] = tet_node_tags

        # populate boundary operators and boundary elements indices
        self.__get_boundary()
        if belem_tags is not None:
            self.__get_belem_positions()

    def __get_boundary(self):
        """Compute all the COO representations of the boundary matrices.
        """
        self.boundary = sl.ShiftedList([None] * self.dim, -1)
        self.B = sl.ShiftedList([None] * self.dim, -1)
        for p in range(self.dim):
            boundary, vals, B = compute_boundary_COO(self.S[self.dim - p])

            self.boundary[self.dim - p] = boundary
            self.B[self.dim - p] = B
            self.S[self.dim - p - 1] = vals

    def __get_belem_positions(self):
        """Compute positions of boundary elements."""
        elem = self.S[1]
        num_belem = self.belem_tags.shape[0]
        self.bpositions = np.zeros(num_belem, dtype=dctkit.int_dtype)
        for i, bedge in enumerate(self.belem_tags):
            # get the index of bedge in the matrix
            # of all the elements
            row_finder = np.all(elem == bedge, axis=1)*1
            self.bpositions[i] = np.nonzero(row_finder)[0]
        self.bpositions = np.sort(self.bpositions)

    def get_circumcenters(self):
        """Compute all the circumcenters.
        """
        self.circ = sl.ShiftedList([None] * (self.dim), -1)
        self.bary_circ = sl.ShiftedList([None] * (self.dim), -1)
        for p in range(1, self.dim + 1):
            S = self.S[p]
            C = np.empty((S.shape[0], self.embedded_dim), dtype=self.float_dtype)
            B = np.empty((S.shape[0], S.shape[1]), dtype=self.float_dtype)
            for i in range(S.shape[0]):
                C[i, :], B[i, :] = circ.circumcenter(S[i, :], self.node_coord)
            self.circ[p] = C
            self.bary_circ[p] = B

    def get_primal_volumes(self):
        """Compute all the primal volumes.
        """
        # loop over all p-simplices (1..dim + 1)
        # (volume of 0-simplices is 1, we do not store it)
        self.primal_volumes = sl.ShiftedList([None] * (self.dim), -1)
        for p in range(1, self.dim + 1):
            S = self.S[p]
            num_p_simplices, _ = S.shape
            primal_volumes = np.empty(num_p_simplices)
            if p == self.embedded_dim:
                primal_volumes = volume.signed_volume(S, self.node_coord)
            else:
                primal_volumes = volume.unsigned_volume(S, self.node_coord)
            self.primal_volumes[p] = primal_volumes

    def get_dual_volumes(self):
        """Compute all the dual volumes.
        """
        self.dual_volumes = sl.ShiftedList([None] * (self.dim), -1)
        self.dual_volumes[self.dim] = np.ones(self.S[self.embedded_dim - self.dim].
                                              shape[0], dtype=self.float_dtype)
        # loop over simplices at all dimensions
        for p in range(self.dim, 0, -1):
            num_p, num_bnd_simplices = self.B[p].shape
            num_pm1, _ = self.S[p - 1].shape
            dv = np.zeros(num_pm1, dtype=self.float_dtype)
            if p == 1:
                # circ_pm1 = circumcenters of the (p-1)-simplices and the
                # circumcenters of the nodes (0-simplices) are the nodes itself.
                circ_pm1 = self.node_coord
            else:
                circ_pm1 = self.circ[p - 1]
            # Loop over p-simplices
            for i in range(num_p):
                # Loop over boundary simplices of the p-simplex

                for j in range(num_bnd_simplices):
                    # ID of the boundary (p-1)-simplex
                    index = self.B[p][i, j]

                    # Distance between circumcenters of the p-simplex and the
                    # boundary (p-1)-simplex
                    length = np.linalg.norm(self.circ[p][i, :] -
                                            circ_pm1[index, :])

                    # Find opposite vertex to the (p-1)-simplex
                    opp_vert = list(
                        set(self.S[p][i]) - set(self.S[p - 1][index]))[0]
                    opp_vert_index = list(self.S[p][i]).index(opp_vert)

                    # Sign of the dual volume of the boundary (p-1)-simplex = sign
                    # of the barycentric coordinate of the circumcenter of the
                    # parent p-simplex relative to the opposite vertex
                    sign = np.copysign(1, self.bary_circ[p][i, opp_vert_index])

                    # Update dual volume of the boundary (p-1)-simplex
                    dv[index] += sign * (length*self.dual_volumes[p][i] /
                                         (self.dim - p + 1))

            self.dual_volumes[p - 1] = dv

    def get_hodge_star(self):
        """Compute all the hodge stars and their inverse if the mesh is well-centered.
        """
        n = self.dim
        self.hodge_star = [None]*(n + 1)
        if self.is_well_centered:
            self.hodge_star_inverse = [None]*(n + 1)
        for p in range(n + 1):
            if p == 0:
                # volumes of vertices are 1 by definition
                pv = 1
                dv = self.dual_volumes[n - p]
            elif p == n:
                pv = self.primal_volumes[p]
                # volumes of vertices are 1 by definition
                dv = 1
            else:
                pv = self.primal_volumes[p]
                dv = self.dual_volumes[n - p]
            self.hodge_star[p] = dv/pv
            if self.is_well_centered:
                self.hodge_star_inverse[p] = 1.0/self.hodge_star[p]
                # adjust the sign in order to have star_inv*star = (-1)^(p*(n-p))
                self.hodge_star_inverse[p] *= (-1)**(p*(n-p))

    def get_dual_edges(self):
        """Compute dual edges vectors taking into account dual orientation.
        """
        # get the dual edges taking into account dual orientation
        dim = self.dim
        dnodes_coords = self.circ[dim]
        # number of dual edges
        n_dedges = self.S[dim-1].shape[0]
        # this is the dual 0 coboundary up to sign
        cob_d0 = self.boundary[0]
        # compute the dual coboundary on the dual 0-cochain vector-valued
        # in which the i-th entry is the coordinates of the i-th n-simplex
        # circumcenter
        self.dedges = (-1)**dim*spmv.spmm(cob_d0, dnodes_coords,
                                          shape=n_dedges)

        # construct the vector consisting of boundary edges circumcenter
        # on boundary edges indices and 0 otherwise
        circ_elems = self.circ[dim-1]
        circ_belems = np.zeros(circ_elems.shape, dtype=dctkit.float_dtype)
        circ_belems[self.bpositions] = circ_elems[self.bpositions]

        # adjust the sign of the boundary entries of circ_belems
        rows, _, vals = cob_d0
        _, idx, count = np.unique(rows, return_index=True, return_counts=True)
        # extract indices of rows related to boundary edges
        boundary_rows_idx = idx[count == 1]
        sign = -vals[boundary_rows_idx]
        # sign of the boundary_entries is the opposite of the sign
        # in non-zero entry of the corresponding row
        circ_belems[self.bpositions] = (sign * circ_belems[self.bpositions].T).T

        # the real dual edges coordinates are the sum of self.dedges
        # and circ_belems
        self.dedges += circ_belems
        # save the area of each dual edge
        self.dedges_complete_areas = np.linalg.norm(self.dedges, axis=1)

    def get_areas_complementary_duals(self):
        """Compute a matrix in which each row is a given n-simplex and each column is
           the area of the portions of the dual (n-1)-simplices intersecting the
           n-simplex.
        """
        dim = self.dim
        B = self.B[dim]
        num_n_simplices = self.S[dim].shape[0]
        num_nm1_simplices = self.S[dim-1].shape[0]
        self.delements_areas = -np.ones(
            (num_n_simplices, num_nm1_simplices), dtype=dctkit.float_dtype)
        for i in range(num_n_simplices):
            current_delements_areas = self.delements_areas[i, :]
            # get the positions of the (n-1)-simplices belonging
            # to the i-th n-simplex
            delems_i = B[i, :]
            # get the areas of the portions of the dual edges contained
            # in the i-th simplex
            proj_matrix = self.circ[dim][i, :] - self.circ[dim-1][delems_i, :]
            current_delements_areas[delems_i] = np.linalg.norm(proj_matrix, axis=1)

    def get_flat_coeffs_matrix(self):
        opp_areas = 1/self.dedges_complete_areas
        self.flat_coeffs_matrix = opp_areas*self.delements_areas


def __simplex_array_parity(s):
    """Compute the number of transpositions needed to sort the array in
       ascending order modulo 2. (Copied from PyDEC, dec/simplex_array.py)

        Args:
            s (np.array): array of the simplices.

        Returns:
            np.array: array of the transpositions needed modulo 2.

    """
    s = s.copy()
    M, N = s.shape

    # number of transpositions used to sort the
    # indices of each simplex (row of s)
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


def compute_boundary_COO(S):
    """Compute the COO representation of the boundary matrix of all p-simplices.

    Args:
        S (np.array): matrix of the IDs of the nodes (cols) belonging to
            each p-simplex (rows).
    Returns:
        tuple: tuple with the COO representation of the boundary.
        np.array: np.array matrix of node tags per (p-1)-face
        ordered lexicographically.
    """
    # number of p-simplices
    num_simplices = S.shape[0]
    # nodes per p-simplex = p + 1
    nodes_per_simplex = S.shape[1]

    dim = nodes_per_simplex - 1

    N = num_simplices * nodes_per_simplex

    # compute array of relative orientations of the (p-1)-faces wrt the
    # p-simplices
    orientations = 1 - 2 * __simplex_array_parity(S)

    # sort the rows of S lexicographically
    # FIXME: avoid making a copy and sorting every time
    F = S.copy()
    F.sort(axis=1)
    # ic(F)
    # F_2 = S[np.lexsort(S.T[::-1])]
    # ic(F_2)

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

    # FIXME: explain the meaming of vals and give a more descriptive name
    # compute vals and rows_index
    vals, rows_index = np.unique(faces, axis=0, return_inverse=True)
    rows_index = rows_index.astype(dtype=dctkit.int_dtype)
    boundary_COO = (rows_index, column_index, values)

    # for triangles and tets, compute B explicitly
    if dim > 1:
        # order faces_ordered w.r.t last column
        faces_ordered_last = faces_ordered[faces_ordered[:, -1].argsort()]

        # initialize the matrix of the boundary simplex as an array
        B = np.empty(faces.shape[0], dtype=dctkit.int_dtype)

        # compute B
        _, B = np.unique(faces_ordered_last[:, :-2], axis=0, return_inverse=True)
        B = B.reshape(faces.shape[0] // nodes_per_simplex, nodes_per_simplex)

    # for edges, B_1 = S_1
    else:
        B = S
    B.astype(dtype=dctkit.int_dtype)

    return boundary_COO, vals, B
