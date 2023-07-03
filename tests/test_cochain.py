import numpy as np
import dctkit
from dctkit.mesh import util
from dctkit.dec import cochain as C


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
    dc_all = [dcP0, dcD0]
    dcP0_true = np.array([1., 1., 1., 1.], dtype=dctkit.float_dtype)
    dcD0_true = np.array([1.,  1.,  1.,  1., -4.], dtype=dctkit.float_dtype)
    dc_true_all = [dcP0_true, dcD0_true]

    for i in range(2):
        assert np.allclose(dc_all[i].coeffs, dc_true_all[i])

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
    dc_all = [dcP0, dcP1, dcD0, dcD1]
    dcP0_true = np.array([1, 3, 4, 1, 3, 1, 2, 1], dtype=dctkit.float_dtype)
    dcP1_true = np.array([3, -7, 6, 7], dtype=dctkit.float_dtype)
    dcD0_true = np.array([1, -2, 1, 3, -2, 4, -1, 2], dtype=dctkit.float_dtype)
    dcD1_true = np.array([6,   8,   9,  -0, -23], dtype=dctkit.float_dtype)
    dc_true_all = [dcP0_true, dcP1_true, dcD0_true, dcD1_true]

    for i in range(4):
        assert np.allclose(dc_all[i].coeffs, dc_true_all[i])

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
    dc_all = [dcP0, dcP1, dcP2, dcD0, dcD1, dcD2]
    dcP0_true = np.array([1, 2, 3, 1, 2, 1], dtype=dctkit.float_dtype)
    dcP1_true = np.array([3, 3, 5, 5], dtype=dctkit.float_dtype)
    dcP2_true = np.array([2], dtype=dctkit.float_dtype)
    dcD0_true = np.array([1, -1,  1, -1], dtype=dctkit.float_dtype)
    dcD1_true = np.array([3,  2, -5,  5, -2,  7], dtype=dctkit.float_dtype)
    dcD2_true = np.array([6,   8,  -0, -14], dtype=dctkit.float_dtype)
    dc_true_all = [dcP0_true, dcP1_true, dcP2_true, dcD0_true, dcD1_true, dcD2_true]

    for i in range(6):
        assert np.allclose(dc_all[i].coeffs, dc_true_all[i])

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
    dc_all = [dcP0_v, dcP1_v, dcD0_v, dcD1_v]
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
    dc_true_all = [dcP0_v_true, dcP1_v_true, dcD0_v_true, dcD1_v_true]

    for i in range(4):
        assert np.allclose(dc_all[i].coeffs, dc_true_all[i])


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
    c_all = [cP0, cP1]

    star_cP0 = C.star(cP0)
    star_cP1 = C.star(cP1)
    star_invP0 = C.star(star_cP0)
    star_invP1 = C.star(star_cP1)
    star_all = [star_cP0, star_cP1]
    star_inv_all = [star_invP0, star_invP1]
    star_cP0_true = np.array([0.125, 0.5, 0.75, 1., 0.625], dtype=dctkit.float_dtype)
    star_cP1_true = np.array([4.,  8., 12., 16.], dtype=dctkit.float_dtype)
    star_true_all = [star_cP0_true, star_cP1_true]

    for i in range(2):
        assert np.allclose(star_all[i].coeffs, star_true_all[i])
        assert np.allclose(star_inv_all[i].coeffs, (-1)
                           ** (i*(S_1.dim-i))*c_all[i].coeffs)

    # 2D test
    vP0 = np.arange(1, 13, dtype=dctkit.float_dtype)
    vP1 = np.arange(1, 26, dtype=dctkit.float_dtype)
    vP2 = np.arange(1, 15, dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_2, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_2, coeffs=vP1)
    cP2 = C.CochainP2(complex=S_2, coeffs=vP2)
    c_all = [cP0, cP1, cP2]

    star_cP0 = C.star(cP0)
    star_cP1 = C.star(cP1)
    star_cP2 = C.star(cP2)
    star_invP0 = C.star(star_cP0)
    star_invP1 = C.star(star_cP1)
    star_invP2 = C.star(star_cP2)
    star_all = [star_cP0, star_cP1, star_cP2]
    star_inv_all = [star_invP0, star_invP1, star_invP2]
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
    star_true_all = [star_cP0_true, star_cP1_true, star_cP2_true]

    for i in range(3):
        assert np.allclose(star_all[i].coeffs, star_true_all[i])
        assert np.allclose(star_inv_all[i].coeffs, (-1)
                           ** (i*(S_2.dim-i))*c_all[i].coeffs)

    # 3D test
    vP0 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP1 = np.array([1, 2, 3, 4, 5, 6], dtype=dctkit.float_dtype)
    vP2 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP3 = np.array([1], dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_3, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_3, coeffs=vP1)
    cP2 = C.CochainP2(complex=S_3, coeffs=vP2)
    cP3 = C.CochainP3(complex=S_3, coeffs=vP3)
    c_all = [cP0, cP1, cP2, cP3]

    star_cP0 = C.star(cP0)
    star_cP1 = C.star(cP1)
    star_cP2 = C.star(cP2)
    star_cP3 = C.star(cP3)
    star_invP0 = C.star(star_cP0)
    star_invP1 = C.star(star_cP1)
    star_invP2 = C.star(star_cP2)
    star_invP3 = C.star(star_cP3)
    star_all = [star_cP0, star_cP1, star_cP2, star_cP3]
    star_inv_all = [star_invP0, star_invP1, star_invP2, star_invP3]

    star_cP0_true = np.array([0.0859375, 0.06510417, 0.0859375, 0.078125],
                             dtype=dctkit.float_dtype)
    star_cP1_true = np.array([0.1875,  0.25,  0.515625,  0.125, -0.078125, -0.0625],
                             dtype=dctkit.float_dtype)
    star_cP2_true = np.array([1.,  1.5,  1.5, -0.66666667], dtype=dctkit.float_dtype)
    star_cP3_true = np.array([6.], dtype=dctkit.float_dtype)
    star_true_all = [star_cP0_true, star_cP1_true, star_cP2_true, star_cP3_true]

    for i in range(4):
        assert np.allclose(star_all[i].coeffs, star_true_all[i])
        assert np.allclose(star_inv_all[i].coeffs, (-1)
                           ** (i*(S_3.dim-i))*c_all[i].coeffs)

    # vector-valued test
    cP0_v_coeffs = np.arange(36, dtype=dctkit.float_dtype).reshape((12, 3))
    cP1_v_coeffs = np.arange(75, dtype=dctkit.float_dtype).reshape((25, 3))
    cP2_v_coeffs = np.arange(42, dtype=dctkit.float_dtype).reshape((14, 3))

    cP0_v = C.CochainP0(S_2, cP0_v_coeffs)
    cP1_v = C.CochainP1(S_2, cP1_v_coeffs)
    cP2_v = C.CochainP2(S_2, cP2_v_coeffs)
    c_all = [cP0_v, cP1_v, cP2_v]

    star_cP0_v = C.star(cP0_v)
    star_cP1_v = C.star(cP1_v)
    star_cP2_v = C.star(cP2_v)
    star_invP0_v = C.star(star_cP0_v)
    star_invP1_v = C.star(star_cP1_v)
    star_invP2_v = C.star(star_cP2_v)
    star_all = [star_cP0_v, star_cP1_v, star_cP2_v]
    star_inv_all = [star_invP0_v, star_invP1_v, star_invP2_v]
    star_cP0_v_true = np.array([[0., 0.0546875, 0.109375],
                                [0.11572266, 0.15429688, 0.19287109],
                                [0.31120643, 0.36307417, 0.41494191],
                                [0.37107422, 0.41230469, 0.45353516],
                                [0.88378906, 0.95743815, 1.03108724],
                                [1.05798153, 1.12851363, 1.19904573],
                                [1.29713753, 1.36920072, 1.44126392],
                                [1.56424323, 1.638731, 1.71321878],
                                [2.9738913, 3.09780344, 3.22171557],
                                [3.71490746, 3.85249663, 3.99008579],
                                [4.1130344, 4.25013555, 4.38723669],
                                [4.10208035, 4.22638582, 4.35069128]],
                               dtype=dctkit.float_dtype)
    star_cP1_v_true = np.array([[0.,  0.25,  0.5],
                                [0.75,  1.,  1.25],
                                [2.,  2.33333333,  2.66666667],
                                [0.5625,  0.625,  0.6875],
                                [0.75,  0.8125,  0.875],
                                [11.66666667, 12.44444444, 13.22222222],
                                [3.7193787,  3.92601085,  4.132643],
                                [4.41133041,  4.62139376,  4.83145712],
                                [9.88186442, 10.29360878, 10.70535313],
                                [2.3625,  2.45,  2.5375],
                                [2.625,  2.7125,  2.8],
                                [23.17021277, 23.87234043, 24.57446809],
                                [25.5, 26.20833333, 26.91666667],
                                [32.5, 33.33333333, 34.16666667],
                                [31.86492869, 32.62361747, 33.38230625],
                                [37.26053215, 38.08854398, 38.9165558],
                                [39.32293987, 40.14216778, 40.96139569],
                                [37.12825773, 37.85626278, 38.58426783],
                                [44.01369863, 44.82876712, 45.64383562],
                                [39.60719178, 40.30205479, 40.99691781],
                                [43.53748848, 44.26311329, 44.98873809],
                                [38.73452219, 39.34935588, 39.96418956],
                                [30.8654446, 31.33310285, 31.8007611],
                                [45.83752759, 46.50183959, 47.16615158],
                                [41.65436631, 42.23289917, 42.81143204]],
                               dtype=dctkit.float_dtype)
    star_cP2_v_true = np.array([[0.,  11.36094675,  22.72189349],
                                [32.,  42.66666667,  53.33333333],
                                [64.,  74.66666667,  85.33333333],
                                [101.05263158, 112.28070175, 123.50877193],
                                [170.66666667, 184.88888889, 199.11111111],
                                [213.33333333, 227.55555556, 241.77777778],
                                [245.10638298, 258.72340425, 272.34042553],
                                [285.95744681, 299.57446809, 313.19148936],
                                [408.69179601, 425.72062084, 442.74944568],
                                [473.42465753, 490.95890411, 508.49315069],
                                [534.57076566, 552.38979118, 570.20881671],
                                [563.2, 580.26666667, 597.33333333],
                                [610.33112583, 627.28476821, 644.2384106],
                                [667.08240535, 684.18708241, 701.29175947]],
                               dtype=dctkit.float_dtype)
    star_true_all = [star_cP0_v_true, star_cP1_v_true, star_cP2_v_true]

    for i in range(3):
        assert np.allclose(star_all[i].coeffs, star_true_all[i])
        assert np.allclose(star_inv_all[i].coeffs, (-1)
                           ** (i*(S_2.dim-i))*c_all[i].coeffs)


def test_inner_product(setup_test):
    mesh_1, _ = util.generate_line_mesh(5, 1.)
    mesh_2, _ = util.generate_square_mesh(1.0)
    mesh_3, _ = util.generate_tet_mesh(2.0)
    S_1 = util.build_complex_from_mesh(mesh_1)
    S_2 = util.build_complex_from_mesh(mesh_2, is_well_centered=False)
    S_3 = util.build_complex_from_mesh(mesh_3)
    S_1.get_hodge_star()
    S_2.get_hodge_star()
    S_3.get_hodge_star()

    # 1D test
    vP0_1 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    vP0_2 = np.array([6, 7, 8, 9, 10], dtype=dctkit.float_dtype)
    vP1_1 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP1_2 = np.array([5, 6, 7, 8], dtype=dctkit.float_dtype)

    cP0_1 = C.CochainP0(complex=S_1, coeffs=vP0_1)
    cP0_2 = C.CochainP0(complex=S_1, coeffs=vP0_2)
    cP1_1 = C.CochainP1(complex=S_1, coeffs=vP1_1)
    cP1_2 = C.CochainP1(complex=S_1, coeffs=vP1_2)

    inner_productP0 = C.inner_product(cP0_1, cP0_2)
    inner_productP1 = C.inner_product(cP1_1, cP1_2)
    inner_product_all = [inner_productP0, inner_productP1]
    inner_productP0_true = np.dot(vP0_1, S_1.hodge_star[0]*vP0_2)
    inner_productP1_true = np.dot(vP1_1, S_1.hodge_star[1]*vP1_2)
    inner_product_true_all = [inner_productP0_true, inner_productP1_true]

    for i in range(2):
        assert np.allclose(inner_product_all[i], inner_product_true_all[i])

    # 2D test
    vP0_1 = np.array([1, 2, 3, 4, 5], dtype=dctkit.float_dtype)
    vP0_2 = np.array([6, 7, 8, 9, 10], dtype=dctkit.float_dtype)
    vP1_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=dctkit.float_dtype)
    vP1_2 = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=dctkit.float_dtype)
    vP2_1 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP2_2 = np.array([5, 6, 7, 8], dtype=dctkit.float_dtype)

    cP0_1 = C.CochainP0(complex=S_2, coeffs=vP0_1)
    cP0_2 = C.CochainP0(complex=S_2, coeffs=vP0_2)
    cP1_1 = C.CochainP1(complex=S_2, coeffs=vP1_1)
    cP1_2 = C.CochainP1(complex=S_2, coeffs=vP1_2)
    cP2_1 = C.CochainP2(complex=S_2, coeffs=vP2_1)
    cP2_2 = C.CochainP2(complex=S_2, coeffs=vP2_2)

    inner_productP0 = C.inner_product(cP0_1, cP0_2)
    inner_productP1 = C.inner_product(cP1_1, cP1_2)
    inner_productP2 = C.inner_product(cP2_1, cP2_2)
    inner_product_all = [inner_productP0, inner_productP1, inner_productP2]
    inner_productP0_true = np.dot(vP0_1, S_2.hodge_star[0]*vP0_2)
    inner_productP1_true = np.dot(vP1_1, S_2.hodge_star[1]*vP1_2)
    inner_productP2_true = np.dot(vP2_1, S_2.hodge_star[2]*vP2_2)
    inner_product_true_all = [inner_productP0_true,
                              inner_productP1_true, inner_productP2_true]

    for i in range(3):
        assert np.allclose(inner_product_all[i], inner_product_true_all[i])

    # 3D test

    vP0_1 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP0_2 = np.array([5, 6, 7, 8], dtype=dctkit.float_dtype)
    vP1_1 = np.array([1, 2, 3, 4, 5, 6], dtype=dctkit.float_dtype)
    vP1_2 = np.array([7, 8, 9, 10, 11, 12], dtype=dctkit.float_dtype)
    vP2_1 = np.array([1, 2, 3, 4], dtype=dctkit.float_dtype)
    vP2_2 = np.array([5, 6, 7, 8], dtype=dctkit.float_dtype)
    vP3_1 = np.array([1], dtype=dctkit.float_dtype)
    vP3_2 = np.array([2], dtype=dctkit.float_dtype)

    cP0_1 = C.CochainP0(complex=S_3, coeffs=vP0_1)
    cP0_2 = C.CochainP0(complex=S_3, coeffs=vP0_2)
    cP1_1 = C.CochainP1(complex=S_3, coeffs=vP1_1)
    cP1_2 = C.CochainP1(complex=S_3, coeffs=vP1_2)
    cP2_1 = C.CochainP2(complex=S_3, coeffs=vP2_1)
    cP2_2 = C.CochainP2(complex=S_3, coeffs=vP2_2)
    cP3_1 = C.CochainP3(complex=S_3, coeffs=vP3_1)
    cP3_2 = C.CochainP3(complex=S_3, coeffs=vP3_2)

    inner_productP0 = C.inner_product(cP0_1, cP0_2)
    inner_productP1 = C.inner_product(cP1_1, cP1_2)
    inner_productP2 = C.inner_product(cP2_1, cP2_2)
    inner_productP3 = C.inner_product(cP3_1, cP3_2)
    inner_product_all = [inner_productP0,
                         inner_productP1, inner_productP2, inner_productP3]
    inner_productP0_true = np.dot(vP0_1, S_3.hodge_star[0]*vP0_2)
    inner_productP1_true = np.dot(vP1_1, S_3.hodge_star[1]*vP1_2)
    inner_productP2_true = np.dot(vP2_1, S_3.hodge_star[2]*vP2_2)
    inner_productP3_true = np.dot(vP3_1, S_3.hodge_star[3]*vP3_2)
    inner_product_true_all = [inner_productP0_true,
                              inner_productP1_true,
                              inner_productP2_true,
                              inner_productP3_true]

    for i in range(4):
        assert np.allclose(inner_product_all[i], inner_product_true_all[i])


def test_codifferential(setup_test):
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
    n_0 = S_1.num_nodes
    n_1 = S_1.S[1].shape[0]
    vP0 = np.arange(n_0, dtype=dctkit.float_dtype)
    vP1 = np.arange(n_1, dtype=dctkit.float_dtype)
    vD0 = np.arange(n_1, dtype=dctkit.float_dtype)
    vD1 = np.arange(n_0, dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_1, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_1, coeffs=vP1)
    cD0 = C.CochainD0(complex=S_1, coeffs=vD0)
    cD1 = C.CochainD1(complex=S_1, coeffs=vD1)

    innerP0P1 = C.inner_product(C.coboundary(cP0), cP1)
    innerD0D1 = C.inner_product(C.coboundary(cD0), cD1)
    inner_all = [innerP0P1, innerD0D1]
    cod_innerP0P1 = C.inner_product(cP0, C.codifferential(cP1))
    cod_innerD0D1 = C.inner_product(cD0, C.codifferential(cD1))
    cod_inner_all = [cod_innerP0P1, cod_innerD0D1]

    for i in range(2):
        assert np.allclose(inner_all[i], cod_inner_all[i])

    # 2D test
    n_0 = S_2.num_nodes
    n_1 = S_2.S[1].shape[0]
    n_2 = S_2.S[2].shape[0]
    vP0 = np.arange(n_0, dtype=dctkit.float_dtype)
    vP1 = np.arange(n_1, dtype=dctkit.float_dtype)
    vP2 = np.arange(n_2, dtype=dctkit.float_dtype)
    vD0 = np.arange(n_2, dtype=dctkit.float_dtype)
    vD1 = np.arange(n_1, dtype=dctkit.float_dtype)
    vD2 = np.arange(n_0, dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_2, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_2, coeffs=vP1)
    cP2 = C.CochainP2(complex=S_2, coeffs=vP2)
    cD0 = C.CochainD0(complex=S_2, coeffs=vD0)
    cD1 = C.CochainD1(complex=S_2, coeffs=vD1)
    cD2 = C.CochainD2(complex=S_2, coeffs=vD2)

    innerP0P1 = C.inner_product(C.coboundary(cP0), cP1)
    innerP1P2 = C.inner_product(C.coboundary(cP1), cP2)
    innerD0D1 = C.inner_product(C.coboundary(cD0), cD1)
    innerD1D2 = C.inner_product(C.coboundary(cD1), cD2)
    inner_all = [innerP0P1, innerP1P2, innerD0D1, innerD1D2]
    cod_innerP0P1 = C.inner_product(cP0, C.codifferential(cP1))
    cod_innerP1P2 = C.inner_product(cP1, C.codifferential(cP2))
    cod_innerD0D1 = C.inner_product(cD0, C.codifferential(cD1))
    cod_innerD1D2 = C.inner_product(cD1, C.codifferential(cD2))
    cod_inner_all = [cod_innerP0P1, cod_innerP1P2, cod_innerD0D1, cod_innerD1D2]

    for i in range(4):
        assert np.allclose(inner_all[i], cod_inner_all[i])

    # 3D test
    n_0 = S_3.num_nodes
    n_1 = S_3.S[1].shape[0]
    n_2 = S_3.S[2].shape[0]
    n_3 = S_3.S[3].shape[0]
    vP0 = np.arange(n_0, dtype=dctkit.float_dtype)
    vP1 = np.arange(n_1, dtype=dctkit.float_dtype)
    vP2 = np.arange(n_2, dtype=dctkit.float_dtype)
    vP3 = np.arange(n_3, dtype=dctkit.float_dtype)
    vD0 = np.arange(n_3, dtype=dctkit.float_dtype)
    vD1 = np.arange(n_2, dtype=dctkit.float_dtype)
    vD2 = np.arange(n_1, dtype=dctkit.float_dtype)
    vD3 = np.arange(n_0, dtype=dctkit.float_dtype)

    cP0 = C.CochainP0(complex=S_3, coeffs=vP0)
    cP1 = C.CochainP1(complex=S_3, coeffs=vP1)
    cP2 = C.CochainP2(complex=S_3, coeffs=vP2)
    cP3 = C.CochainP3(complex=S_3, coeffs=vP3)
    cD0 = C.CochainD0(complex=S_3, coeffs=vD0)
    cD1 = C.CochainD1(complex=S_3, coeffs=vD1)
    cD2 = C.CochainD2(complex=S_3, coeffs=vD2)
    cD3 = C.CochainD3(complex=S_3, coeffs=vD3)

    innerP0P1 = C.inner_product(C.coboundary(cP0), cP1)
    innerP1P2 = C.inner_product(C.coboundary(cP1), cP2)
    innerP2P3 = C.inner_product(C.coboundary(cP2), cP3)
    innerD0D1 = C.inner_product(C.coboundary(cD0), cD1)
    innerD1D2 = C.inner_product(C.coboundary(cD1), cD2)
    innerD2D3 = C.inner_product(C.coboundary(cD2), cD3)
    inner_all = [innerP0P1, innerP1P2, innerP2P3, innerD0D1, innerD1D2, innerD2D3]
    cod_innerP0P1 = C.inner_product(cP0, C.codifferential(cP1))
    cod_innerP1P2 = C.inner_product(cP1, C.codifferential(cP2))
    cod_innerP2P3 = C.inner_product(cP2, C.codifferential(cP3))
    cod_innerD0D1 = C.inner_product(cD0, C.codifferential(cD1))
    cod_innerD1D2 = C.inner_product(cD1, C.codifferential(cD2))
    cod_innerD2D3 = C.inner_product(cD2, C.codifferential(cD3))
    cod_inner_all = [cod_innerP0P1, cod_innerP1P2,
                     cod_innerP2P3, cod_innerD0D1, cod_innerD1D2, cod_innerD2D3]

    for i in range(6):
        assert np.allclose(inner_all[i], cod_inner_all[i])


def test_coboundary_closure(setup_test):
    mesh_2, _ = util.generate_square_mesh(1.0)
    S_2 = util.build_complex_from_mesh(mesh_2, is_well_centered=False)
    S_2.get_hodge_star()

    c = C.CochainP1(complex=S_2, coeffs=np.arange(1, 9, dtype=dctkit.float_dtype))
    cob_clos_c = C.coboundary_closure(c)
    cob_clos_c_true = np.array([-0.5,  2.5,  5.,  2.,  0.], dtype=dctkit.float_dtype)
    assert np.allclose(cob_clos_c.coeffs, cob_clos_c_true)
