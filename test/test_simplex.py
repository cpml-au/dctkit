from dctkit.mesh import simplex, util


def test_prova():
    numNodes, numElements, nodeTagsPerElem, x = util.read_mesh("test1.msh")
    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {nodeTagsPerElem}")
    print(f"The coordinates of the nodes are \n {x}")
    C, NtE, EtF = simplex.compute_face_to_edge_connectivity(nodeTagsPerElem)
    print(f"The orientation matrix is \n {C}")
    print(f"The NtE matrix is \n {NtE}")
    print(f"The NtE matrix is \n {EtF}")


if __name__ == "__main__":
    test_prova()