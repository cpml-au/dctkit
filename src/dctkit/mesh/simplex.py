import numpy as np
from dctkit.math import circumcenter as circ
from dctkit.math import shifted_list as sl
from dctkit.math import volume


class SimplicialComplex:
    """Simplicial complex class.

    Args:
        tet_node_tags (int32 np.array): matrix containing the IDs of the nodes
            (cols) belonging to each tetrahedron or top-level simplex (rows).
        node_coord (float np.array): Cartesian coordinates (columns) of all the
        nodes (rows) of the simplicial complex.
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
        hodge_star (list): list where each entry p is an array containing the
            diagonal of the p-hodge star matrix.
    """

    def __init__(self, tet_node_tags, node_coord):
        # store the coordinates of the nodes
        self.node_coord = node_coord
        self.num_nodes = node_coord.shape[0]
        self.embedded_dim = node_coord.shape[1]

        # compute complex dimension from top-level simplices
        self.dim = tet_node_tags.shape[1] - 1

        self.S = [None] * (self.dim + 1)
        self.S[-1] = tet_node_tags
        # populate boundary operators
        self.__get_boundary()

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

    def get_circumcenters(self):
        """Compute all the circumcenters.
        """
        self.circ = sl.ShiftedList([None] * (self.dim), -1)
        self.bary_circ = sl.ShiftedList([None] * (self.dim), -1)
        for p in range(1, self.dim + 1):
            S = self.S[p]
            C = np.empty((S.shape[0], self.embedded_dim))
            B = np.empty((S.shape[0], S.shape[1]))
            for i in range(S.shape[0]):
                C[i, :], B[i, :] = circ.circumcenter(S[i, :],
                                                     self.node_coord)
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
                                              shape[0])
        # loop over simplices at all dimensions
        for p in range(self.dim, 0, -1):
            num_p, num_bnd_simplices = self.B[p].shape
            num_pm1, _ = self.S[p - 1].shape
            dv = np.zeros(num_pm1)
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
        """Compute all the hodge stars.
        """
        self.hodge_star = [None]*(self.dim + 1)
        for p in range(self.dim + 1):
            if p == 0:
                # volumes of vertices are 1 by definition
                pv = 1
                dv = self.dual_volumes[self.dim - p]
            elif p == self.dim:
                pv = self.primal_volumes[p]
                # volumes of vertices are 1 by definition
                dv = 1
            else:
                pv = self.primal_volumes[p]
                dv = self.dual_volumes[self.dim - p]
            self.hodge_star[p] = dv/pv


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
    for i in range(N - 1):
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
        S (int32 np.array): matrix of the IDs of the nodes (cols) belonging to
            each p-simplex (rows).
    Returns:
        tuple: tuple with the COO representation of the boundary.
        int32 np.array: np.array matrix of node tags per (p-1)-face
        ordered lexicographically.
    """
    # number of p-simplices
    num_simplices = S.shape[0]
    # nodes per p-simplex = p + 1
    nodes_per_simplex = S.shape[1]

    dim = nodes_per_simplex - 1

    # TODO: is there an appropriate name for this?
    N = num_simplices * nodes_per_simplex

    # compute array of relative orientations of the (p-1)-faces wrt the
    # p-simplices
    orientations = 1 - 2 * __simplex_array_parity(S)

    # sort the rows of S lexicographically
    # FIXME: avoid making a copy and sorting every time
    F = S.copy()
    F.sort(axis=1)

    # S_(p-1) matrix with repeated (p-1)-simplices and with two extra columns
    S_pm1_ext = np.empty((N, nodes_per_simplex + 1), dtype=np.int64)

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

    # compute vals and rows_index
    vals, rows_index = np.unique(faces, axis=0, return_inverse=True)
    boundary_COO = (rows_index, column_index, values)

    # for triangles and tets, compute B explicitly
    if dim > 1:
        # order faces_ordered w.r.t last column
        faces_ordered_last = faces_ordered[faces_ordered[:, -1].argsort()]

        # initialize the matrix of the boundary simplex as an array
        B = np.empty(faces.shape[0], dtype=np.int64)

        for i, v in enumerate(vals):
            # find position of vector v in faces_ordered_last[:, :-2]
            position = np.where(
                (faces_ordered_last[:, :-2] == v).all(axis=1))[0]

            # update B in position indices
            B[position] = i

        B = B.reshape(faces.shape[0] // nodes_per_simplex, nodes_per_simplex)

    # for edges, B_1 = S_1
    else:
        B = S

    return boundary_COO, vals, B
