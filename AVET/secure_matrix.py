import numpy as np
import numbers


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


class SecureMatrix:
    """Sinh ma trận Gaussian ngẫu nhiên"""

    def __init__(self, m_features=512, n_components=512, random_state=None):
        self.m_features = m_features
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, y=None):
        random_state = check_random_state(self.random_state)
        scale = 1 / np.sqrt(self.m_features)
        self.random_weights_ = random_state.normal(
            loc=0,
            scale=scale,
            size=(self.n_components, self.m_features),
        )
        return self