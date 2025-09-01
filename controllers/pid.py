import numpy as np
class JointPID:
    def __init__(self, Kp, Ki, Kd, n_joints):
        self.Kp, self.Ki, self.Kd = map(np.array, (Kp, Ki, Kd))
        self.ei = np.zeros(n_joints)
    def update(self, q, dq, q_ref, dq_ref, dt):
        e, ed = q_ref - q, dq_ref - dq
        self.ei += e * dt
        return self.Kp*e + self.Ki*self.ei + self.Kd*ed



# Each controller (PID / Impedance / SMC / Admittance) uses q_ref, dq_ref to produce a torque command.
# The CBF then projects that torque into the safe set (joint limits + obstacle distance).