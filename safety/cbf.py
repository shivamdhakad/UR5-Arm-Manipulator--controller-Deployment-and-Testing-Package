import numpy as np
from qpsolvers import solve_qp

class CBFProjector:
    """
    min 0.5||u - u_des||^2 + 0.5*rho||s||^2
    s.t.  A u + s >= b,  s >= 0,  u_min <= u <= u_max
    """
    def __init__(self, u_min=None, u_max=None, rho=5e3):
        self.u_min, self.u_max, self.rho = u_min, u_max, rho

    def project_with_slack(self, u_des, A_cbf=None, b_cbf=None):
        n = len(u_des); m = 0 if (A_cbf is None or b_cbf is None) else A_cbf.shape[0]
        H = np.block([[np.eye(n), np.zeros((n,m))],
                      [np.zeros((m,n)), self.rho*np.eye(m)]])
        f = np.hstack([-u_des, np.zeros(m)])
        G_list, h_list = [], []

        # Box: u <= umax ; u >= umin -> -u <= -umin
        if self.u_max is not None:
            G_list.append(np.hstack([np.eye(n), np.zeros((n,m))]));   h_list.append(self.u_max)
        if self.u_min is not None:
            G_list.append(np.hstack([-np.eye(n), np.zeros((n,m))]));  h_list.append(-self.u_min)

        if m > 0:
            # s >= 0  -> -s <= 0
            G_list.append(np.hstack([np.zeros((m,n)), -np.eye(m)]));  h_list.append(np.zeros(m))
            # A u + s >= b -> -A u - s <= -b
            G_list.append(np.hstack([-A_cbf, -np.eye(m)]));           h_list.append(-b_cbf)

        G = np.vstack(G_list) if G_list else None
        h = np.hstack(h_list) if h_list else None

        w = solve_qp(H, f, G, h, solver="osqp")
        if w is None:
            u = u_des.copy()
            if self.u_max is not None: u = np.minimum(u, self.u_max)
            if self.u_min is not None: u = np.maximum(u, self.u_min)
            return u, None
        return w[:n], (w[n:] if m > 0 else None)

    def project(self, u_des, A_cbf=None, b_cbf=None):
        u, _ = self.project_with_slack(u_des, A_cbf, b_cbf)
        return u
