import numpy as np
class JointAdmittance:
    def __init__(self, M, D, K, n_joints):
        self.M = np.array(M); self.D = np.array(D); self.K = np.array(K)
        self.v = np.zeros(n_joints)
    def update(self, q, dq, q_ref, dq_ref, dt, tau_ref=None, tau_ext=None):
        if tau_ref is None: tau_ref = np.zeros_like(q)
        if tau_ext is None: tau_ext = np.zeros_like(q)
        ddq = (tau_ref + tau_ext - self.D*(dq - dq_ref) - self.K*(q - q_ref)) / np.maximum(self.M, 1e-9)
        self.v = dq + ddq*dt
        return self.D*(self.v - dq_ref) + self.K*(q - q_ref)

