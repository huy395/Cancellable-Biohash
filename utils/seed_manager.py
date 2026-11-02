import numpy as np
import random
import hashlib

class SeedManager:
    """
    Quản lý seed cho toàn pipeline:
    - global_seed: dùng cho tái sử dụng toàn hệ thống
    - user_seed: sinh riêng cho từng user từ global_seed
    - bit_seed: sinh riêng cho từng bit của mỗi user
    """

    def __init__(self, global_seed=42):
        self.global_seed = global_seed
        np.random.seed(global_seed)
        random.seed(global_seed)
    
    def stable_hash(self, text):
        
        h = hashlib.sha256(text.encode()).hexdigest()
        return int(h[:8], 16)

    def get_user_seed(self, user_label):
        
        base = self.stable_hash(user_label)
        combined = base + self.global_seed
        rng = np.random.default_rng(combined)
        
        return rng.integers(0, 10_000_000)
        

    def get_bit_seed(self, user_label, bit_idx):
        
        user_seed = self.get_user_seed(user_label)
        combined = user_seed + bit_idx*9973
        rng = np.random.default_rng(combined)
        
        return rng.integers(0, 10_000_000)

    def reseed(self, new_seed=None):
        """Đặt lại global_seed (khi muốn chạy thực nghiệm lặp)"""
        if new_seed is None:
            new_seed = int(np.random.default_rng().integers(0, 10_000_000))
            
        self.global_seed = new_seed
        np.random.seed(new_seed)
        random.seed(new_seed)
        
        return new_seed
