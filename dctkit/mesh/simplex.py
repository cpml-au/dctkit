import numpy as np


def simplex_array_parity(s):
    """Compute the relative parity of an array of simplices
    """
    s = s.copy()

    M, N = s.shape

    # number of transpositions used to sort the
    # indices of each simplex (row of s)
    trans = np.zeros_like(s[:, 0])
    seq = np.arange(M)

    for i in range(N - 1):
        pos = s.argmin(axis=1)
        s[seq, pos] = s[:, 0]
        pos.clip(0, 1, pos)
        trans = trans + pos
        s = s[:, 1:]

    trans %= 2  # compute parity

    return trans


def compute_face_to_edge_connectivity(nodeTagsPerElem):
    # write documentation at the end

    # reshape nodeTagsPerElem to have a matrix
    S_2 = nodeTagsPerElem.reshape(len(nodeTagsPerElem) // 3, 3)
    print(S_2)
    num_simplices = S_2.shape[0]
    faces_per_simplex = S_2.shape[1]
    num_faces = num_simplices * faces_per_simplex
    orientations = 1 - 2 * simplex_array_parity(S_2)  # calculate orientations
    print(type(orientations))
    S_2.sort(axis=1)  # sort S_2 lexicographically
    # S_2_plus_plus = np.c_[S_2_ord, orientations, np.arange(num_simplices)]

    faces = np.empty((num_faces, faces_per_simplex + 1),
                     dtype=S_2.dtype)  # add documentation
    for i in range(faces_per_simplex):
        rows = faces[num_simplices * i:num_simplices * (i + 1)]
        rows[:, :i] = S_2[:, :i]
        rows[:, i:-2] = S_2[:, i + 1:]
        rows[:, -1] = np.arange(num_simplices)
        rows[:, -2] = ((-1)**i) * orientations

    print(faces)
    temp = faces[faces[:, -1].argsort()]
    edge = temp[:, :2]
    C = temp[:, -2]  # orientation

    # compute edge to face matrix
    vals, idx, count = np.unique(edge,
                                 axis=0,
                                 return_index=True,
                                 return_counts=True)
    big = np.c_[vals, count, idx]  # create the matrix
    big = big[big[:, -1].argsort()]  # sort to preserve the initial order
    count = big[:, 2]  # update count w.r.t the original order
    vals = big[:, :2]  # update vals w.r.t the original order
    rep_edges = vals[count > 1]  # save the edge repeated
    position = np.array([
        np.where((edge == i).all(axis=1))[0] for i in rep_edges
    ])  # index position of rep_edges in the original array edge
    position = np.concatenate(position)
    EtF = np.array(range(len(edge)))  # create the vectors of label
    EtF[position[1::2]] = position[::2]  # eliminate duplicate labels
    EtF = EtF.reshape(len(edge) // 3, 3)  # build edge to face matrix

    # compute node to edge
    NtE = faces[np.lexsort(faces[:, :-2].T[::-1])][:, :2]
    NtE = np.unique(NtE, axis=0)

    return C, NtE, EtF
