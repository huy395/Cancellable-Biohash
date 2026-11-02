import os
import numpy as np
from sklearn.svm import LinearSVC
from scipy.stats import chi
from utils.utils_io import save_pickle_zst, load_pickle_zst


class FEHash:

    def __init__(self, projection_type="orf", D=512, d=512,
                 seed=None, random_state=None,
                 svm_c=0.01):                           

        self.projection_type = projection_type
        self.D            = D
        self.d            = d
        self.seed         = seed
        self.random_state = random_state
        self.svm_c        = svm_c                       

        self.Omega = None
        self.r     = None
        self.w     = None
        self.b     = None

        if seed is not None:
            self.create_projection()

    def create_projection(self):

        rng = np.random.default_rng(self.seed)

        if self.projection_type == "rff":
            self.Omega = rng.normal(0, 1, size=(self.D, self.d))
            self.r     = rng.uniform(0, 2 * np.pi, size=self.D)

        elif self.projection_type == "orf":
            if self.D % self.d != 0:
                raise ValueError(
                    f"ORF yêu cầu D là bội số của d. "
                    f"Hiện tại D={self.D}, d={self.d}."
                )
            n_blocks = self.D // self.d
            blocks   = []
            for _ in range(n_blocks):
                G    = rng.normal(0, 1, size=(self.d, self.d))
                Q, _ = np.linalg.qr(G)
                norm_samples = rng.normal(0, 1, size=(self.d, self.d))
                s    = np.sqrt(np.sum(norm_samples ** 2, axis=1))
                blocks.append(np.diag(s) @ Q)
            self.Omega = np.vstack(blocks)
            self.r     = rng.uniform(0, 2 * np.pi, size=self.D)

        else:
            raise ValueError("projection_type must be 'rff' or 'orf'")

        self.Omega = self.Omega.astype(np.float64)
        self.r     = self.r.astype(np.float64)

    def transform(self, X):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        phi = np.cos(np.dot(X, self.Omega.T) + self.r)
        phi = np.sqrt(2.0 / self.D) * phi

        norm = np.linalg.norm(phi, axis=1, keepdims=True)
        phi  = phi / np.maximum(norm, 1e-9)

        return phi

    def fit(self, Xtrain, Ytrain):

        Xtrain = np.asarray(Xtrain, dtype=np.float64)
        Ytrain = np.asarray(Ytrain)

        X_proj = self.transform(Xtrain)

        clf = LinearSVC(
            C            = self.svm_c,
            fit_intercept= False,
            max_iter     = 2000,
            dual         = False,
            random_state = self.random_state,
        )
        clf.fit(X_proj, Ytrain)

        self.w = clf.coef_.reshape(self.D).astype(np.float64)

        pos_idx  = np.where(Ytrain ==  1)[0]
        neg_idx  = np.where(Ytrain == -1)[0]
        pos_proj = np.dot(X_proj[pos_idx], self.w) if pos_idx.size else np.array([0.0])
        neg_proj = np.dot(X_proj[neg_idx], self.w) if neg_idx.size else np.array([0.0])
        self.b   = np.float64((-np.max(neg_proj) + np.min(pos_proj)) / 2.0)

    def hash(self, x):

        x   = np.asarray(x, dtype=np.float64).reshape(1, -1)
        phi = self.transform(x)[0]
        return int(np.dot(self.w, phi) + self.b >= 0)

    def save(self, filename):

        if self.seed is None:
            raise ValueError(
                "seed=None: không thể lưu theo seed. "
                "Khởi tạo FEHash với seed cụ thể hoặc dùng save_full()."
            )
        save_pickle_zst(
            {
                "projection_type": self.projection_type,
                "seed":     self.seed,
                "D":        int(self.D),
                "d":        int(self.d),
                "svm_c":    float(self.svm_c),  # ← lưu svm_c
                "w":        self.w.astype(np.float64),
                "b":        np.float64(self.b),
                "_version": 2,
            },
            filename,
            level=10,
        )

    def save_full(self, filename):

        save_pickle_zst(
            {
                "projection_type": self.projection_type,
                "Omega":    self.Omega.astype(np.float64),
                "r":        self.r.astype(np.float64),
                "D":        int(self.D),
                "d":        int(self.d),
                "svm_c":    float(self.svm_c),
                "w":        self.w.astype(np.float64),
                "b":        np.float64(self.b),
                "_version": 1,
            },
            filename,
            level=10,
        )

    @classmethod
    def load(cls, filename):

        params  = load_pickle_zst(filename)
        version = params.get("_version", 1)

        if version == 2:
            fe = cls(
                projection_type = params["projection_type"],
                D               = params["D"],
                d               = params["d"],
                seed            = params["seed"],
                svm_c           = params.get("svm_c", 0.01), 
            )
        else:
            fe = cls(
                projection_type = params["projection_type"],
                D               = params["D"],
                d               = params["d"],
                svm_c           = params.get("svm_c", 0.01),
            )
            fe.Omega = params["Omega"].astype(np.float64)
            fe.r     = params["r"].astype(np.float64)

        fe.w = params["w"].astype(np.float64)
        fe.b = np.float64(params["b"])

        return fe
