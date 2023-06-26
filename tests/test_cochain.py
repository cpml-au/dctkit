import numpy as np
import dctkit
from dctkit.mesh import util
from dctkit.dec import cochain as C

# FIXME: tests should involve different dimensions (of cochains and complex)

# FIXME: SPLIT INTO MULTIPLE TESTS - ONE FOR EACH FUNCTIONALITY TO BE TESTED!


def test_coboundary(setup_test):
    mesh_1, _ = util.generate_line_mesh(5, 1.)
    mesh_2, _ = util.generate_square_mesh(1.0)
    mesh_3, _ = util.generate_tet_mesh(2.0)
    S_1 = util.build_complex_from_mesh(mesh_1)
    S_2 = util.build_complex_from_mesh(mesh_2, is_well_centered=False)
    S_3 = util.build_complex_from_mesh(mesh_3)

    # 1D test
    vP0 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    vD0 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    cP0 = C.CochainP0(complex=S_1, coeffs=vP0)
    cD0 = C.CochainD0(complex=S_1, coeffs=vD0)
    dcP0 = C.coboundary(cP0)
    dcD0 = C.coboundary(cD0)

    dcP0_true = np.array([1., 1., 1., 1.], dtype=dctkit.float_dtype)
    dcD0_true = np.array([1.,  1.,  1.,  1., -4.], dtype=dctkit.float_dtype)

    assert np.allclose(dcP0.coeffs, dcP0_true)
    assert np.allclose(dcD0.coeffs, dcD0_true)

    # 2D test
    vP0 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    vP1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dctkit.float_dtype)
    vD0 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vD1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dctkit.float_dtype)
    cP0 = C.CochainP0(complex=S_2, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_2, coeffs=vP1)
    cD0 = C.CochainD0(complex=S_2, coeffs=vD0)
    cD1 = C.CochainD1(complex=S_2, coeffs=vD1)

    dcP0 = C.coboundary(cP0)
    dcP1 = C.coboundary(cP1)
    dcD0 = C.coboundary(cD0)
    dcD1 = C.coboundary(cD1)
    dcP0_true = np.array([1, 3, 4, 1, 3, 1, 2, 1], dtype=dctkit.float_dtype)
    dcP1_true = np.array([3, -7, 6, 7], dtype=dctkit.float_dtype)
    dcD0_true = np.array([1, -2, 1, 3, -2, 4, -1, 2], dtype=dctkit.float_dtype)
    dcD1_true = np.array([6,   8,   9,  -0, -23], dtype=dctkit.float_dtype)

    assert dcP0.coeffs.dtype == dctkit.float_dtype

    assert np.allclose(dcP0.coeffs, dcP0_true)
    assert np.allclose(dcP1.coeffs, dcP1_true)
    assert np.allclose(dcD0.coeffs, dcD0_true)
    assert np.allclose(dcD1.coeffs, dcD1_true)

    # 3D test
    vP0 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP1 = np.array([1, 2, 3, 4, 5, 6], dtype=dctkit.float_dtype)
    vP2 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vD0 = np.array([1], dtype=dctkit.float_dtype)
    vD1 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vD2 = np.array([1, 2, 3, 4, 5, 6], dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_3, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_3, coeffs=vP1)
    cP2 = C.CochainP2(complex=S_3, coeffs=vP2)
    cD0 = C.CochainD0(complex=S_3, coeffs=vD0)
    cD1 = C.CochainD1(complex=S_3, coeffs=vD1)
    cD2 = C.CochainD2(complex=S_3, coeffs=vD2)

    dcP0 = C.coboundary(cP0)
    dcP1 = C.coboundary(cP1)
    dcP2 = C.coboundary(cP2)
    dcD0 = C.coboundary(cD0)
    dcD1 = C.coboundary(cD1)
    dcD2 = C.coboundary(cD2)
    dcP0_true = np.array([1, 2, 3, 1, 2, 1], dtype=dctkit.float_dtype)
    dcP1_true = np.array([3, 3, 5, 5], dtype=dctkit.float_dtype)
    dcP2_true = np.array([2], dtype=dctkit.float_dtype)
    dcD0_true = np.array([1, -1,  1, -1], dtype=dctkit.float_dtype)
    dcD1_true = np.array([3,  2, -5,  5, -2,  7], dtype=dctkit.float_dtype)
    dcD2_true = np.array([6,   8,  -0, -14], dtype=dctkit.float_dtype)

    assert np.allclose(dcP0.coeffs, dcP0_true)
    assert np.allclose(dcP1.coeffs, dcP1_true)
    assert np.allclose(dcP2.coeffs, dcP2_true)
    assert np.allclose(dcD0.coeffs, dcD0_true)
    assert np.allclose(dcD1.coeffs, dcD1_true)
    assert np.allclose(dcD2.coeffs, dcD2_true)

    # vector-valued test
    cP0_v_coeffs = np.arange(15, dtype=dctkit.float_dtype).reshape((5, 3))
    cP1_v_coeffs = np.arange(24, dtype=dctkit.float_dtype).reshape((8, 3))
    cD0_v_coeffs = np.arange(12, dtype=dctkit.float_dtype).reshape((4, 3))
    cD1_v_coeffs = np.arange(24, dtype=dctkit.float_dtype).reshape((8, 3))
    cP0_v = C.CochainP0(S_2, cP0_v_coeffs)
    cP1_v = C.CochainP1(S_2, cP1_v_coeffs)
    cD0_v = C.CochainD0(S_2, cD0_v_coeffs)
    cD1_v = C.CochainD1(S_2, cD1_v_coeffs)
    dcP0_v = C.coboundary(cP0_v)
    dcP1_v = C.coboundary(cP1_v)
    dcD0_v = C.coboundary(cD0_v)
    dcD1_v = C.coboundary(cD1_v)
    dcP0_v_true = np.array([[3, 3, 3],
                           [9,  9,  9],
                           [12, 12, 12],
                           [3, 3, 3],
                           [9, 9, 9],
                           [3, 3, 3],
                           [6, 6, 6],
                           [3, 3, 3]], dtype=dctkit.float_dtype)
    dcP1_v_true = np.array([[6, 7, 8],
                           [-18, -19, -20],
                           [15, 16, 17],
                           [18, 19, 20]], dtype=dctkit.float_dtype)
    dcD0_v_true = np.array([[0,  1,  2],
                            [-3, -4, -5],
                            [3,  3,  3],
                            [6,  7,  8],
                            [-6, -6, -6],
                            [9, 10, 11],
                            [-3, -3, -3],
                            [6,  6,  6]], dtype=dctkit.float_dtype)
    dcD1_v_true = np.array([[9,  12,  15],
                            [21,  22,  23],
                            [24,  25,  26],
                            [3,   2,   1],
                            [-57, -61, -65]], dtype=dctkit.float_dtype)

    assert np.allclose(dcP0_v.coeffs, dcP0_v_true)
    assert np.allclose(dcP1_v.coeffs, dcP1_v_true)
    assert np.allclose(dcD0_v.coeffs, dcD0_v_true)
    assert np.allclose(dcD1_v.coeffs, dcD1_v_true)


def test_hodge_star(setup_test):
    mesh_1, _ = util.generate_line_mesh(5, 1.)
    mesh_2, _ = util.generate_square_mesh(0.8)
    mesh_3, _ = util.generate_tet_mesh(2.0)
    S_1 = util.build_complex_from_mesh(mesh_1)
    S_2 = util.build_complex_from_mesh(mesh_2)
    S_3 = util.build_complex_from_mesh(mesh_3)
    S_1.get_hodge_star()
    S_2.get_hodge_star()
    S_3.get_hodge_star()

    # 1D test
    vP0 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    vP1 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_1, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_1, coeffs=vP1)

    star_cP0 = C.star(cP0)
    star_cP1 = C.star(cP1)
    star_invP0 = C.star(star_cP0)
    star_invP1 = C.star(star_cP1)
    star_cP0_true = np.array([0.125, 0.5, 0.75, 1., 0.625], dtype=dctkit.float_dtype)
    star_cP1_true = np.array([4.,  8., 12., 16.], dtype=dctkit.float_dtype)

    assert np.allclose(star_cP0.coeffs, star_cP0_true)
    assert np.allclose(star_cP1.coeffs, star_cP1_true)
    assert np.allclose(star_invP0.coeffs, cP0.coeffs)
    assert np.allclose(star_invP1.coeffs, cP1.coeffs)

    # 2D test
    vP0 = np.arange(1, 13, dtype=dctkit.float_dtype)
    vP1 = np.arange(1, 26, dtype=dctkit.float_dtype)
    vP2 = np.arange(1, 15, dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_2, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_2, coeffs=vP1)
    cP2 = C.CochainP2(complex=S_2, coeffs=vP2)

    star_cP0 = C.star(cP0)
    star_cP1 = C.star(cP1)
    star_cP2 = C.star(cP2)
    star_invP0 = C.star(star_cP0)
    star_invP1 = C.star(star_cP1)
    star_invP2 = C.star(star_cP2)
    star_cP0_true = np.array([0.0546875, 0.07714844, 0.15560322, 0.16492188,
                              0.36824544, 0.42319261, 0.50444237, 0.59590218,
                              1.11520924, 1.37589165, 1.50811261, 1.49166558],
                             dtype=dctkit.float_dtype)
    star_cP1_true = np.array([0.25,  0.5,  1.,  0.25,  0.3125,
                              4.66666667,  1.44642505,  1.68050682,  3.70569916,
                              0.875, 0.9625,  8.42553191,  9.20833333, 11.66666667,
                              11.38033168, 13.24818921, 13.92687454, 13.10409096,
                              15.48630137, 13.89726027, 15.23812097, 13.52634108,
                              10.75613978, 15.94348786, 14.46332164],
                             dtype=dctkit.float_dtype)
    star_cP2_true = np.array([11.36094675,  21.33333333,  32.,  44.9122807,
                              71.11111111,  85.33333333,  95.31914894, 108.93617021,
                              153.2594235, 175.34246575, 196.00928074, 204.8,
                              220.39735099, 239.46547884], dtype=dctkit.float_dtype)

    assert np.allclose(star_cP0.coeffs, star_cP0_true)
    assert np.allclose(star_cP1.coeffs, star_cP1_true)
    assert np.allclose(star_cP2.coeffs, star_cP2_true)
    assert np.allclose(star_invP0.coeffs, cP0.coeffs)
    assert np.allclose(star_invP1.coeffs, -cP1.coeffs)
    assert np.allclose(star_invP2.coeffs, cP2.coeffs)

    # 3D test
    vP0 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP1 = np.array([1, 2, 3, 4, 5, 6], dtype=dctkit.float_dtype)
    vP0 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP1 = np.array([1], dtype=dctkit.float_dtype)
    # FIXME: continue from here...


def test_cochain(setup_test):
    mesh, _ = util.generate_square_mesh(1.0)
    cpx = util.build_complex_from_mesh(mesh, is_well_centered=False)

    v_0 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    v_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dctkit.float_dtype)
    c_0 = C.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=v_0)
    c_1 = C.Cochain(dim=1, is_primal=True, complex=cpx, coeffs=v_1)

    # hodge star test
    cpx.get_hodge_star()
    star_c_0 = C.star(c_0)
    coeffs_0 = star_c_0.coeffs
    coeffs_0_true = np.array([1/8, 1/4, 3/8, 1/2, 5/2], dtype=dctkit.float_dtype)
    star_c_1 = C.star(c_1)
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
    c = C.Cochain(dim=1, is_primal=True, complex=cpx_new, coeffs=v)
    d = C.Cochain(dim=2, is_primal=True, complex=cpx_new, coeffs=w)
    inner_product_standard = C.inner_product(C.coboundary(c), d)
    inner_product_codiff = C.inner_product(c, C.codifferential(d))

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
    a = C.Cochain(dim=0, is_primal=False, complex=S, coeffs=a_vec)
    b = C.Cochain(dim=1, is_primal=False, complex=S, coeffs=b_vec)
    dual_inner = C.inner_product(C.coboundary(a), b)
    dual_cod_inner = C.inner_product(a, C.codifferential(b))
    assert np.allclose(dual_inner, dual_cod_inner)

    # dual codifferential test 2D
    alpha0_vec = np.arange(num_2, dtype=dctkit.float_dtype)
    alpha1_vec = np.arange(num_1, dtype=dctkit.float_dtype)
    alpha2_vec = np.arange(num_0, dtype=dctkit.float_dtype)
    alpha0 = C.Cochain(dim=0, is_primal=False, complex=cpx_new, coeffs=alpha0_vec)
    alpha1 = C.Cochain(dim=1, is_primal=False, complex=cpx_new, coeffs=alpha1_vec)
    alpha2 = C.Cochain(dim=2, is_primal=False, complex=cpx_new, coeffs=alpha2_vec)
    dual_inner_1 = C.inner_product(C.coboundary(alpha0), alpha1)
    dual_inner_2 = C.inner_product(C.coboundary(alpha1), alpha2)
    dual_cod_inner_1 = C.inner_product(alpha0, C.codifferential(alpha1))
    dual_cod_inner_2 = C.inner_product(alpha1, C.codifferential(alpha2))
    assert np.allclose(dual_inner_1, dual_cod_inner_1)
    assert np.allclose(dual_inner_2, dual_cod_inner_2)

    # inner product test
    v_0_2 = np.array([5, 6, 7, 8, 9], dtype=dctkit.float_dtype)
    c_0_2 = C.Cochain(dim=0, is_primal=True, complex=cpx, coeffs=v_0_2)
    inner_product = C.inner_product(c_0, c_0_2)
    inner_product_true = 5/8 + 3/2 + 21/8 + 4 + 45/2

    assert inner_product.dtype == dctkit.float_dtype
    assert np.allclose(inner_product, inner_product_true)

    # vector-valued cochain test
    c0_v_coeffs = np.arange(15).reshape((5, 3))
    c0_v = C.CochainP0(cpx, c0_v_coeffs)

    star_c0_v = C.star(c0_v)
    star_c0_v_true = 0.125*c0_v_coeffs
    star_c0_v_true[-1, :] = np.array([6, 6.5, 7])
    assert np.allclose(star_c0_v.coeffs, star_c0_v_true)
