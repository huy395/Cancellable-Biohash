import os
import numpy as np
from .secure_matrix import SecureMatrix
from utils.utils_io import save_pickle_zst, load_pickle_zst


class AVET:
    """
    AVET transform
    y = A·(R·x) + B·|R·x|
    - Input:  x ∈ R^m
    - Output: y ∈ R^n
    """

    def __init__(self, m=512, n=256, user_seed=None, save_dir=None):
        """
        Args:
            m, n      : kích thước input/output
            user_seed : seed riêng cho user (bắt buộc để tái tạo ma trận)
            save_dir  : thư mục lưu tham số
        """
        self.m = m
        self.n = n
        self.user_seed = user_seed
        self.save_dir = save_dir or "saved_templates"
        os.makedirs(self.save_dir, exist_ok=True)

        # Sinh seed cho từng ma trận
        rng = np.random.default_rng(user_seed)
        state_R = int(rng.integers(0, 10_000_000))
        state_A = int(rng.integers(0, 10_000_000))
        state_B = int(rng.integers(0, 10_000_000))

        # Sinh ma trận
        self.R = SecureMatrix(
            m_features=m, n_components=n, random_state=state_R
        ).fit().random_weights_

        self.A = SecureMatrix(
            m_features=n, n_components=n, random_state=state_A
        ).fit().random_weights_

        self.B = SecureMatrix(
            m_features=n, n_components=n, random_state=state_B
        ).fit().random_weights_

        # Làm A, B non-invertible
        if self.A.shape[0] > 2:
            for i in range(self.A.shape[1]):
                self.A[2][i] = self.A[0][i] + self.A[1][i]
                self.B[2][i] = self.B[0][i] + self.B[1][i]

    # ==================================================================
    def _save_params(self, user_id):
        """Lưu seed thay vì ma trận R, A, B"""
        if self.user_seed is None:
            raise ValueError(
                "user_seed=None: không thể lưu theo seed. "
                "Khởi tạo AVET với user_seed cụ thể."
            )

        user_dir = os.path.join(self.save_dir, str(user_id))
        os.makedirs(user_dir, exist_ok=True)

        params = {
            "user_seed": self.user_seed,    # ← lưu seed thay vì R, A, B
            "m": int(self.m),
            "n": int(self.n),
            "_avet_version": 8,
        }
        save_pickle_zst(
            params,
            os.path.join(user_dir, "avet_params.zst"),
            level=10,
        )

    def _load_params(self, user_id):
        """Load params và tái tạo AVET từ seed"""
        path = os.path.join(self.save_dir, str(user_id), "avet_params.zst")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy AVET params tại {path}"
            )
        params = load_pickle_zst(path)

        # Backward compatible: file cũ lưu ma trận trực tiếp
        if "_avet_version" not in params or params["_avet_version"] < 8:
            return params   # trả về nguyên, verify sẽ xử lý

        # File mới: tái tạo AVET từ seed → lấy R, A, B
        avet_loaded = AVET(
            m=params["m"],
            n=params["n"],
            user_seed=params["user_seed"],
            save_dir=self.save_dir,
        )
        params["R"] = avet_loaded.R
        params["A"] = avet_loaded.A
        params["B"] = avet_loaded.B
        return params

    # ==================================================================
    def enroll(self, x, binary=False, user_id=None, save=True):
        """Biến đổi embedding khi đăng ký"""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.m:
            raise ValueError(
                f"Kích thước embedding ({x.shape[0]}) khác m={self.m}"
            )

        projected = np.dot(self.R, x)
        y = np.dot(self.A, projected) + np.dot(self.B, np.abs(projected))

        if user_id is not None and save:
            self._save_params(user_id)

        return self.sign_func(y) if binary else y

    def verify(self, user_id, x, binary=False):
        """Biến đổi embedding khi xác thực"""
        params = self._load_params(user_id)
        R = params["R"]
        A = params["A"]
        B = params["B"]

        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != params["m"]:
            raise ValueError(
                f"Kích thước embedding ({x.shape[0]}) ≠ m={params['m']} "
                f"trong saved AVET"
            )

        projected = np.dot(R, x)
        y = np.dot(A, projected) + np.dot(B, np.abs(projected))

        return self.sign_func(y) if binary else y

    # ==================================================================
    def sign_func(self, embedding):
        embedding = np.asarray(embedding)
        return (embedding > 0).astype(np.int8)