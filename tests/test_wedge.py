import jax.numpy as jnp
import dctkit as dt
from dctkit.mesh import util
from dctkit.dec import cochain as C
from dctkit.dec import wedge as w


def test_wedge(setup_test):
    # 1D test
    mesh_1, _ = util.generate_line_mesh(5, 1.)
    S_1 = util.build_complex_from_mesh(mesh_1)
    S_1.get_hodge_star()
    S_1.get_S_dual()

    vP0_1 = jnp.array([1, 2, 3, 4, 5], dtype=dt.float_dtype)
    vP0_2 = jnp.array([6, 7, 8, 9, 10], dtype=dt.float_dtype)
    vP1 = jnp.array([1, 2, 3, 4], dtype=dt.float_dtype)
    vD0_1 = jnp.array([1, 2, 3, 4], dtype=dt.float_dtype)
    vD0_2 = jnp.array([5, 6, 7, 8], dtype=dt.float_dtype)
    vD1 = jnp.array([1, 2, 3, 4, 5], dtype=dt.float_dtype)

    cP0_1 = C.CochainP0(complex=S_1, coeffs=vP0_1)
    cP0_2 = C.CochainP0(complex=S_1, coeffs=vP0_2)
    cP1 = C.CochainP1(complex=S_1, coeffs=vP1)
    cD0_1 = C.CochainD0(complex=S_1, coeffs=vD0_1)
    cD0_2 = C.CochainD0(complex=S_1, coeffs=vD0_2)
    cD1 = C.CochainD1(complex=S_1, coeffs=vD1)

    wedge_P0_P0 = w.wedge(cP0_1, cP0_2).coeffs.flatten()
    wedge_P0_P1 = w.wedge(cP0_1, cP1).coeffs.flatten()
    wedge_D0_D0 = w.wedge(cD0_1, cD0_2).coeffs.flatten()
    wedge_D0_D1 = w.wedge(cD0_1, cD1).coeffs.flatten()
    wedge_P0_P0_true = jnp.array([6., 14., 24., 36., 50.], dtype=dt.float_dtype)
    wedge_P0_P1_true = jnp.array([1.5,  5., 10.5, 18.], dtype=dt.float_dtype)
    wedge_D0_D0_true = jnp.array([5., 12., 21., 32.], dtype=dt.float_dtype)
    wedge_D0_D1_true = jnp.array([0.,  3.,  7.5, 14.,  0.], dtype=dt.float_dtype)

    assert jnp.allclose(wedge_P0_P0, wedge_P0_P0_true)
    assert jnp.allclose(wedge_P0_P1, wedge_P0_P1_true)
    assert jnp.allclose(wedge_D0_D0, wedge_D0_D0_true)
    assert jnp.allclose(wedge_D0_D1, wedge_D0_D1_true)

    # 2D test
    mesh_2, _ = util.generate_square_mesh(1)
    S_2 = util.build_complex_from_mesh(mesh_2)
    S_2.get_hodge_star()
    S_2.get_S_dual()

    vP0_1 = jnp.array([1, 2, 3, 4, 5], dtype=dt.float_dtype)
    vP0_2 = jnp.array([6, 7, 8, 9, 10], dtype=dt.float_dtype)
    vP1_1 = jnp.arange(1, 9, dtype=dt.float_dtype)
    vP1_2 = jnp.arange(9, 18, dtype=dt.float_dtype)
    vD0_1 = jnp.arange(1, 5, dtype=dt.float_dtype)
    vD1_1 = jnp.arange(5, 9, dtype=dt.float_dtype)
    vD1_1 = jnp.arange(1, 9, dtype=dt.float_dtype)

    cP0_1 = C.CochainP0(complex=S_2, coeffs=vP0_1)
    cP0_2 = C.CochainP0(complex=S_2, coeffs=vP0_2)
    cP1_1 = C.CochainP1(complex=S_2, coeffs=vP1_1)
    cP1_2 = C.CochainP1(complex=S_2, coeffs=vP1_2)
    cD0_1 = C.CochainD0(complex=S_2, coeffs=vD0_1)
    cD0_2 = C.CochainD0(complex=S_2, coeffs=vD0_2)
    cD1_1 = C.CochainD1(complex=S_2, coeffs=vD1_1)

    wedge_P0_P0 = w.wedge(cP0_1, cP0_2).coeffs.flatten()
    wedge_P0_P1 = w.wedge(cP0_1, cP1_1).coeffs.flatten()
    wedge_P1_P1 = w.wedge(cP1_1, cP1_2).coeffs.flatten()
    wedge_D0_D0 = w.wedge(cD0_1, cD0_2).coeffs.flatten()
    wedge_D0_D1 = w.wedge(cD0_1, cD1_1).coeffs.flatten()

    wedge_P0_P0_true = jnp.array([6., 14., 24., 36., 50.], dtype=dt.float_dtype)
    wedge_P0_P1_true = jnp.array(
        [1.5,  5.,  9., 10., 17.5, 21., 28., 36.], dtype=dt.float_dtype)
    wedge_P1_P1_true = jnp.array(
        [-64/6,  16.,  -8.,  -32/6], dtype=dt.float_dtype)
    wedge_D0_D0_true = jnp.array([5., 12., 21., 32.], dtype=dt.float_dtype)
    wedge_D0_D1_true = jnp.array(
        [0.,  0.,  4.5,  0., 10.,  0., 24.5, 24.], dtype=dt.float_dtype)

    assert jnp.allclose(wedge_P0_P0, wedge_P0_P0_true)
    assert jnp.allclose(wedge_P0_P1, wedge_P0_P1_true)
    assert jnp.allclose(wedge_P1_P1, wedge_P1_P1_true)
    assert jnp.allclose(wedge_D0_D0, wedge_D0_D0_true)
    assert jnp.allclose(wedge_D0_D1, wedge_D0_D1_true)
