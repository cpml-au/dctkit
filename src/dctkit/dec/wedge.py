import jax.numpy as jnp
from jax import Array
from dctkit.dec import cochain as C
from scipy.special import factorial


def wedge(c_1: C.Cochain, c_2: C.Cochain, weight: Array = None) -> C.Cochain:
    """Computes the wedge product of two cochains.

    Args:
        c_1: the first cochain.
        c_2: the second cochain.

    Returns:
        a new cochain representing the wedge product of `c_1` and `c_2`.

    Raises:
        Exception: If attempting a primal-dual wedge product, which is
        undefined.
        AssertionError: If computing a dual wedge product with dimension
        greater than 1, which is not defined.
    """
    # if c_2.dim > c_1.dim, use property of the wedge product
    if c_2.dim < c_1.dim:
        return C.scalar_mul(wedge(c_2, c_1, weight), (-1)**(c_1.dim*c_2.dim))
    # so from now on, c_1.dim <= c_2.dim

    # if c_1 and c_2 are 0-cochains, wedge = cochain product
    if c_1.dim == 0 and c_2.dim == 0:
        return C.cochain_mul(c_1, c_2)
    wedge_coch_dim = c_1.dim + c_2.dim
    S = c_1.complex
    # extract the matrix of indices of the wedge_coch_dim+1-simplices (primal/dual)
    if c_1.is_primal and c_2.is_primal:
        # primal wedge
        S_list = S.S
        type_ = "primal"
    elif (not c_1.is_primal) and not (c_2.is_primal):
        # dual wedge is only defined for wedge_coch_dim <=1
        assert wedge_coch_dim <= 1
        S_list = S.S_dual
        type_ = "dual"
    else:
        raise Exception("The primal-dual wedge product is not defined.")
    num_c_2_dim_simplex = S_list[c_2.dim].shape[0]
    if weight is None:
        # standard definition
        weight = 1/(wedge_coch_dim+1)*jnp.ones(num_c_2_dim_simplex)
    weight *= 1/factorial(wedge_coch_dim, True)
    if c_1.dim == 0:
        # in this case the wedge product has an easier formulation
        wedge_vec = jnp.zeros((num_c_2_dim_simplex, c_2.dim+1))
        for i in range(c_2.dim+1):
            wedge_vec = wedge_vec.at[:, i].set(
                c_2.dim*c_1.coeffs[S_list[c_2.dim][:, i]].flatten())
        wedge_vec *= c_2.coeffs
        # lookup = S.cup_lookup[(type_, c_1.dim, c_2.dim)]["lookup"]
    else:
        # extract the permutation vectors and compute its signs
        sgn_perm_vec = S.cup_lookup[(type_, c_1.dim, c_2.dim)]["sgn_perm_vec"]
        # extract cup product and sgn orientations tables
        lookup = S.cup_lookup[(type_, c_1.dim, c_2.dim)]["lookup"]
        sgn_orient = S.cup_lookup[(type_, c_1.dim, c_2.dim)]["sgn_orient"]

        # compute the value of the cup product
        # FIXME: add docs here
        cup_prod_no_sign = c_1.coeffs[lookup[:, 0]]*c_2.coeffs[lookup[:, 1]]
        cup_prod = cup_prod_no_sign.squeeze(-1)*sgn_orient[:, 0]*sgn_orient[:, 1]
        wedge_vec = sgn_perm_vec*cup_prod
    # FIXME: fix this part of the code
    weight = weight[0]
    wedge_coch_coeffs = weight*jnp.sum(wedge_vec, axis=1)
    # if c_1.dim + c_2.dim > 1:
    #     weight = weight[0]
    #     wedge_coch_coeffs = weight*jnp.sum(wedge_vec, axis=1)
    # else:
    #     weight_coeffs = weight[lookup[:, 1, 0]].flatten()
    #     wedge_coch_coeffs = weight_coeffs * \
    #         wedge_vec[:, 0] + (1-weight_coeffs)*wedge_vec[:, 1]

    return C.Cochain(wedge_coch_dim, c_1.is_primal, S, wedge_coch_coeffs)
