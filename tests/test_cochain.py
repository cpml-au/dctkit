import numpy as np
import dctkit
from dctkit.mesh import util
from dctkit.dec import cochain

# FIXME: tests should involve different dimensions (of cochains and complex)

# FIXME: SPLIT INTO MULTIPLE TESTS - ONE FOR EACH FUNCTIONALITY TO BE TESTED!


def test_cochain(setup_test):
    mesh, _ = util.generate_square_mesh(1.0)
    cpx = util.build_complex_from_mesh(mesh)

    v_0 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    v_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dctkit.float_dtype)
    # cpx = simplex.SimplicialComplex(S_2, node_coord)
    c_0 = cochain.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=v_0)
    c_1 = cochain.Cochain(dim=1, is_primal=True, complex=cpx, coeffs=v_1)

    # coboundary test
    dc_0 = cochain.coboundary(c_0)
    dc_1 = cochain.coboundary(c_1)
    dc_v_0_true = np.array([1, 3, 4, 1, 3, 1, 2, 1], dtype=dctkit.float_dtype)
    dc_v_1_true = np.array([3, -7, 6, 7], dtype=dctkit.float_dtype)
    dc_0_true = cochain.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=dc_v_0_true)
    dc_1_true = cochain.Cochain(dim=1, is_primal=True, complex=cpx, coeffs=dc_v_1_true)

    assert dc_0.coeffs.dtype == dctkit.float_dtype

    assert np.allclose(dc_0.coeffs, dc_0_true.coeffs)
    assert np.allclose(dc_1.coeffs, dc_1_true.coeffs)

    # hodge star test
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

    # primal codifferential test (we need a well-centered mesh)

    mesh, _ = util.generate_square_mesh(0.4)
    cpx_new = util.build_complex_from_mesh(mesh)
    cpx_new.get_hodge_star()

    num_0 = cpx_new.num_nodes
    num_1 = cpx_new.S[1].shape[0]
    num_2 = cpx_new.S[2].shape[0]
    v = np.random.rand(num_1).astype(dtype=dctkit.float_dtype)
    w = np.random.rand(num_2).astype(dtype=dctkit.float_dtype)
    c = cochain.Cochain(dim=1, is_primal=True, complex=cpx_new, coeffs=v)
    d = cochain.Cochain(dim=2, is_primal=True, complex=cpx_new, coeffs=w)
    inner_product_standard = cochain.inner_product(cochain.coboundary(c), d)
    inner_product_codiff = cochain.inner_product(c, cochain.codifferential(d))

    assert inner_product_standard.dtype == dctkit.float_dtype
    assert np.allclose(inner_product_standard, inner_product_codiff)

    # dual codifferential test 1D
    mesh, _ = util.generate_line_mesh(10, 1)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    n_0 = S.num_nodes
    n_1 = S.S[1].shape[0]
    a_vec = np.arange(n_1, dtype=dctkit.float_dtype)
    b_vec = np.arange(n_0, dtype=dctkit.float_dtype)
    a = cochain.Cochain(dim=0, is_primal=False, complex=S, coeffs=a_vec)
    b = cochain.Cochain(dim=1, is_primal=False, complex=S, coeffs=b_vec)
    dual_inner = cochain.inner_product(cochain.coboundary(a), b)
    dual_cod_inner = cochain.inner_product(a, cochain.codifferential(b))
    assert np.allclose(dual_inner, dual_cod_inner)

    # dual codifferential test 2D
    alpha0_vec = np.arange(num_2, dtype=dctkit.float_dtype)
    alpha1_vec = np.arange(num_1, dtype=dctkit.float_dtype)
    alpha2_vec = np.arange(num_0, dtype=dctkit.float_dtype)
    alpha0 = cochain.Cochain(dim=0, is_primal=False, complex=cpx_new, coeffs=alpha0_vec)
    alpha1 = cochain.Cochain(dim=1, is_primal=False, complex=cpx_new, coeffs=alpha1_vec)
    alpha2 = cochain.Cochain(dim=2, is_primal=False, complex=cpx_new, coeffs=alpha2_vec)
    dual_inner_1 = cochain.inner_product(cochain.coboundary(alpha0), alpha1)
    dual_inner_2 = cochain.inner_product(cochain.coboundary(alpha1), alpha2)
    dual_cod_inner_1 = cochain.inner_product(alpha0, cochain.codifferential(alpha1))
    dual_cod_inner_2 = cochain.inner_product(alpha1, cochain.codifferential(alpha2))
    assert np.allclose(dual_inner_1, dual_cod_inner_1)
    assert np.allclose(dual_inner_2, dual_cod_inner_2)

    # inner product test
    v_0_2 = np.array([5, 6, 7, 8, 9], dtype=dctkit.float_dtype)
    c_0_2 = cochain.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=v_0_2)
    inner_product = cochain.inner_product(c_0, c_0_2)
    inner_product_true = 5/8 + 3/2 + 21/8 + 4 + 45/2

    assert inner_product.dtype == dctkit.float_dtype
    assert np.allclose(inner_product, inner_product_true)

    # vector-valued cochain test
    c0_v_coeffs = np.arange(15).reshape((5, 3))
    c1_v_coeffs = np.arange(24).reshape((8, 3))
    c0_v = cochain.CochainP0(cpx, c0_v_coeffs)
    c1_v = cochain.CochainP1(cpx, c1_v_coeffs)
    dc0_v = cochain.coboundary(c0_v)
    dc1_v = cochain.coboundary(c1_v)
    dc0_v_true = np.array([[3, 3, 3],
                           [9,  9,  9],
                           [12, 12, 12],
                           [3, 3, 3],
                           [9, 9, 9],
                           [3, 3, 3],
                           [6, 6, 6],
                           [3, 3, 3]], dtype=dctkit.float_dtype)
    dc1_v_true = np.array([[6, 7, 8],
                           [-18, -19, -20],
                           [15, 16, 17],
                           [18, 19, 20]], dtype=dctkit.float_dtype)

    assert np.allclose(dc0_v.coeffs, dc0_v_true)
    assert np.allclose(dc1_v.coeffs, dc1_v_true)

    star_c0_v = cochain.star(c0_v)
    star_c0_v_true = 0.125*c0_v_coeffs
    star_c0_v_true[-1, :] = np.array([6, 6.5, 7])
    assert np.allclose(star_c0_v.coeffs, star_c0_v_true)
