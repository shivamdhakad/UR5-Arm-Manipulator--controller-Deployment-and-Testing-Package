import numpy as np
class JointImpedance:
    def __init__(self, K, D, ff=None):
        self.K = np.array(K); self.D = np.array(D)
        self.ff = (lambda q,dq: np.zeros_like(q)) if ff is None else ff
    def update(self, q, dq, q_ref, dq_ref, dt):
        return self.K*(q_ref - q) + self.D*(dq_ref - dq) + self.ff(q, dq)
