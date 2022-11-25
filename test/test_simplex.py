import numpy as np
from dctkit.mesh import simplex


def test_compute_face_to_edge_connectivity():
    #test 1
    edge = np.array([[1,3], [0,3], [0,1], [2,3], [1,3], [1,2], [3,4],[2,4], [2,3]])
    print(f"Label edge matrix: \n {simplex.compute_face_to_edge_connectivity(edge)}")

    #test 2
    edge = np.array([[1,2], [0,2], [0,1], [2,3], [0,3], [0,2]])
    print(f"Label edge matrix: \n {simplex.compute_face_to_edge_connectivity(edge)}")