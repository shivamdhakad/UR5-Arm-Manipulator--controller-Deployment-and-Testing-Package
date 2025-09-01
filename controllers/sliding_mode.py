import numpy as np
class JointSMC:
    def __init__(self, Lambda, K_sign, Kp=None, Kd=None, eps=1e-3):
        self.Lambda=np.array(Lambda); self.K_sign=np.array(K_sign)
        self.Kp=np.zeros_like(self.Lambda) if Kp is None else np.array(Kp)
        self.Kd=np.zeros_like(self.Lambda) if Kd is None else np.array(Kd)
        self.eps=eps
    def update(self, q, dq, q_ref, dq_ref, dt):
        s = (dq - dq_ref) + self.Lambda*(q - q_ref)
        sign_s = s / (np.abs(s) + self.eps)
        u_eq = self.Kp*(q_ref - q) + self.Kd*(dq_ref - dq)
        return u_eq - self.K_sign*sign_s
