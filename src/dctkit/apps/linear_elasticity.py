import numpy as np
from dctkit.mesh.simplex import SimplicialComplex


class LinearElasticity():
    def __init__(self, S: SimplicialComplex, mu_: float, lambda_: float):
        self.S = S
        self.mu_ = mu_
        self.lambda_ = lambda_

    def linear_elasticity_residual(self):
        dim = self.S.dim
        epsilon = 1/2 * (self.S.metric - np.identity(2))
        tr_epsilon = np.trace(epsilon, axis1=1, axis2=2)
        stress = 2*self.mu*epsilon + self.lambda_*tr_epsilon[:, None, None] * \
            np.stack([np.identity(2)]*dim)
