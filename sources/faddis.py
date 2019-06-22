import logging
import numpy as np


class FADDIS:
    """ Fuzzy Additive Spectral Clustering algorithm

    Parameters
    ----------
    max_clusters : the maximum number of clusters to extract

    """

    def __init__(self, max_clusters):
        self.max_clusters = max_clusters

    @property
    def _logger(self):
        return logging.getLogger('FADDIS')

    @staticmethod
    def _normalize_l2(x: np.ndarray) -> np.ndarray:
        n = np.sqrt((x ** 2).sum())
        return x / (n + (n == 0))

    def predict(self, sim):
        """ Predict fuzzy clusters given the similarity matrix with N objects

        Returns
        -------
        clusters : array of shape (K, N)
        intensities : array of shape (K,)
        contributions : array of shape (K,)
        """
        W = sim.copy()
        total_scatter = (W ** 2).sum()

        clusters = []
        intensities = []
        contributions = []

        for i in range(self.max_clusters):

            w, v = np.linalg.eig(W)

            # fix a bug when eig returns a complex matrix with empty imaginary part
            if w.dtype == 'complex' and np.isclose(w.imag.sum(), 0):
                self._logger.info('empty imag part')
                w, v = w.real, v.real

            max_w = np.argmax(w)
            max_v = v[:, max_w]
            max_v_normed = self._normalize_l2(max_v)

            def get_u_xi_g(vec):
                uu = vec.clip(0, 1).reshape(-1, 1)
                uu = self._normalize_l2(uu)
                xxi = uu.T.dot(W).dot(uu) / (uu.T.dot(uu) ** 2)
                gg = (xxi * uu.T.dot(uu)) ** 2
                return uu, xxi, gg

            if np.all(max_v_normed > 0):
                u, xi, g = get_u_xi_g(max_v_normed)
            elif np.all(max_v_normed < 0):
                u, xi, g = get_u_xi_g(-max_v_normed)
            else:
                u_pos, xi_pos, g_pos = get_u_xi_g(max_v_normed)
                u_neg, xi_neg, g_neg = get_u_xi_g(-max_v_normed)
                if g_pos > g_neg:
                    u, xi, g = u_pos, xi_pos, g_pos
                else:
                    u, xi, g = u_neg, xi_neg, g_neg

            xi = xi.squeeze()

            if xi < 0:
                self._logger.info('xi < 0 -> break')
                break

            self._logger.info(f'i = {i}, xi = {xi:.4f}, sqrt(xi) = {np.sqrt(xi):.4f}')
            self._logger.info(f'E = {((W - xi * u.dot(u.T)) ** 2).sum():.4f}')

            W -= xi * u.dot(u.T)

            c = xi ** 2 / total_scatter / u.T.dot(u)

            clusters.append(u)
            intensities.append(xi)
            contributions.append(c)

        return np.hstack(clusters).T, np.hstack(intensities), np.hstack(contributions)
