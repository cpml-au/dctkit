import numpy as np
import dctkit
from dctkit import config, FloatDtype, IntDtype
from dctkit.mesh import simplex, util
from dctkit.dec import cochain
import os
import matplotlib.tri as tri
import matplotlib.pyplot as plt


cwd = os.path.dirname(simplex.__file__)

# FIXME: tests should involve different dimensions (of cochains and complex)


def test_cochain(fdtype=FloatDtype.float32, idtype=IntDtype.int32):
    config(fdtype=fdtype, idtype=idtype)

    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {S_2}")

    v_0 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    v_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dctkit.float_dtype)
    cpx = simplex.SimplicialComplex(S_2, node_coord)
    c_0 = cochain.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=v_0)
    c_1 = cochain.Cochain(dim=1, is_primal=True, complex=cpx, coeffs=v_1)

    # coboundary test
    dc_0 = cochain.coboundary(c_0)
    dc_1 = cochain.coboundary(c_1)
    dc_v_0_true = np.array([1, 2, 4, 2, 3, 1, 2, 1], dtype=dctkit.float_dtype)
    dc_v_1_true = np.array([3, -6, 7, -7], dtype=dctkit.float_dtype)
    dc_0_true = cochain.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=dc_v_0_true)
    dc_1_true = cochain.Cochain(dim=1, is_primal=True, complex=cpx, coeffs=dc_v_1_true)

    assert dc_0.coeffs.dtype == dctkit.float_dtype

    assert np.allclose(dc_0.coeffs, dc_0_true.coeffs)
    assert np.allclose(dc_1.coeffs, dc_1_true.coeffs)

    # hodge star test
    cpx.get_circumcenters()
    cpx.get_primal_volumes()
    cpx.get_dual_volumes()
    cpx.get_hodge_star()
    star_c_0 = cochain.star(c_0)
    coeffs_0 = star_c_0.coeffs
    coeffs_0_true = np.array([1/8, 1/4, 3/8, 1/2, 5/2], dtype=dctkit.float_dtype)
    star_c_1 = cochain.star(c_1)
    coeffs_1 = star_c_1.coeffs
    coeffs_1_true = np.array([0, 0, 3, 0, 5, 0, 7, 8], dtype=dctkit.float_dtype)

    assert coeffs_0.dtype == dctkit.float_dtype
    assert coeffs_1.dtype == dctkit.float_dtype

    assert np.allclose(coeffs_0, coeffs_0_true)
    assert np.allclose(coeffs_1, coeffs_1_true)

    # codifferential test (we need a well-centered mesh)

    _, _, S_2_new, node_coords_new = util.generate_square_mesh(0.4)
    triang = tri.Triangulation(node_coords_new[:, 0], node_coords_new[:, 1])

    plt.triplot(triang, 'ko-')
    plt.show()

    cpx_new = simplex.SimplicialComplex(S_2_new, node_coords_new, is_well_centered=True)
    cpx_new.get_circumcenters()
    cpx_new.get_primal_volumes()
    cpx_new.get_dual_volumes()
    cpx_new.get_hodge_star()

    num_0 = cpx_new.S[1].shape[0]
    num_1 = cpx_new.S[2].shape[0]
    v = np.random.rand(num_0).astype(dtype=dctkit.float_dtype)
    w = np.random.rand(num_1).astype(dtype=dctkit.float_dtype)
    c = cochain.Cochain(dim=1, is_primal=True, complex=cpx_new, coeffs=v)
    d = cochain.Cochain(dim=2, is_primal=True, complex=cpx_new, coeffs=w)
    inner_product_standard = cochain.inner_product(cochain.coboundary(c), d)
    inner_product_codiff = cochain.inner_product(c, cochain.codifferential(d))

    assert inner_product_standard.dtype == dctkit.float_dtype
    assert np.allclose(inner_product_standard, inner_product_codiff)

    # inner product test
    v_0_2 = np.array([5, 6, 7, 8, 9], dtype=dctkit.float_dtype)
    c_0_2 = cochain.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=v_0_2)
    inner_product = cochain.inner_product(c_0, c_0_2)
    inner_product_true = 5/8 + 3/2 + 21/8 + 4 + 45/2

    assert inner_product.dtype == dctkit.float_dtype
    assert np.allclose(inner_product, inner_product_true)


if __name__ == '__main__':
    test_cochain(fdtype=FloatDtype.float64, idtype=IntDtype.int64)
    test_cochain(fdtype=FloatDtype.float32, idtype=IntDtype.int32)
