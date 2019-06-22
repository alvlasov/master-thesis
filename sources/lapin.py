import numpy as np
import scipy as sp


class LapinTransformer:

    def __init__(self, sim):
        W = sim - np.diag(sim.diagonal())
        D = np.diag(W.sum(axis=1))

        L = D - W

        inv_sqrt_D = np.linalg.inv(sp.linalg.sqrtm(D))
        Ln = inv_sqrt_D.dot(L).dot(inv_sqrt_D)

        w, v = np.linalg.eig(Ln)
        indices = ~np.isclose(w, 0)

        v_ = v[:, indices]
        w_ = np.diag(w[indices])

        Lp = v_.dot(np.linalg.inv(w_)).dot(v_.T)

        self.W = W
        self.D = D
        self.L = L
        self.Ln = Ln
        self.Lp = Lp
