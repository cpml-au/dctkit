import numpy as np


def circumcenter(s, x):
    simplex_coord = x[s[:]]
    rows, cols = simplex_coord.shape

    assert (rows <= cols + 1)

    A = np.bmat([[2*np.dot(simplex_coord, simplex_coord.T), np.ones((rows, 1))],
                [np.ones((1, rows)) ,  np.zeros((1, 1))]])

    b = np.hstack((np.sum(simplex_coord * simplex_coord, axis=1), np.ones((1))))
    bary_coords = np.linalg.solve(A, b)
    bary_coords = bary_coords[:-1]
    circumcenters = np.dot(bary_coords, simplex_coord)

    return circumcenters
