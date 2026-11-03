import numpy as np
from scipy.stats import chi2
from utils.utils_io import save_pickle_zst, load_pickle_zst

class Projection:
    def __init__(self, projection_type, Omega, r, D, d):
        self.projection_type = projection_type
        self.Omega = Omega.astype(np.float64)
        self.r = r.astype(np.float64)
        self.D = int(D)
        self.d = int(d)

    
    @staticmethod
    def create(projection_type, D, d, seed):
        rng = np.random.RandomState(seed)

        if projection_type == "rff":
            Omega = rng.normal(0, 1, size=(D, d))
            r = rng.uniform(0, 2 * np.pi, size=D)

        elif projection_type == "orf":
            D = d  # ép D = d
            G = rng.normal(0, 1, size=(d, d))
            Q, _ = np.linalg.qr(G)
            s = np.sqrt(chi2.rvs(df=d, size=d, random_state=rng))
            Omega = (np.diag(s) @ Q) / np.sqrt(d)
            r = rng.uniform(0, 2 * np.pi, size=d)

        else:
            raise ValueError("projection_type must be 'rff' or 'orf'")

        return Projection(
            projection_type=projection_type,
            Omega=Omega,
            r=r,
            D=D,
            d=d
        )

    
    def transform(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.sqrt(2.0 / self.D) * np.cos(np.dot(X, self.Omega.T) + self.r)

    
    def save(self, path):
        save_pickle_zst({
            "projection_type": self.projection_type,
            "Omega": self.Omega,
            "r": self.r,
            "D": self.D,
            "d": self.d
        }, path)

    @staticmethod
    def load(path):
        p = load_pickle_zst(path)
        return Projection(
            projection_type=p["projection_type"],
            Omega=p["Omega"],
            r=p["r"],
            D=p["D"],
            d=p["d"]
        )
