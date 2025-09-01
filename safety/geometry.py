# safety/geometry.py
import numpy as np, mujoco

def _ee_pos_site(model, data, site_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[sid].copy()

def _ee_pos_body(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xipos[bid].copy()

def _jac_site(model, data, site_name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, sid)
    return Jp

def _jac_body(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, Jp, Jr, bid)
    return Jp

def obstacle_cbf_rows_for_dofs(model, data, dof_indices, center, min_clearance,
                               zeta=0.8, omega=4.0, site_name=None, body_name=None):
    """
    Relative-degree-2 barrier on distance from EE (site or body) to 'center'.
    Linearized as: (rhat^T Jp) u >= -(2ζω ḋ + ω² (d - dmin)) on selected DOFs.
    Returns (A (1xnd), b (1,), d).
    """
    if site_name is not None:
        p = _ee_pos_site(model, data, site_name)
        Jp_full = _jac_site(model, data, site_name)
    elif body_name is not None:
        p = _ee_pos_body(model, data, body_name)
        Jp_full = _jac_body(model, data, body_name)
    else:
        raise ValueError("Provide either site_name or body_name")

    r = p - center
    d = float(np.linalg.norm(r))
    rhat = r / d if d > 1e-9 else np.array([1.0, 0.0, 0.0])

    Jp = Jp_full[:, dof_indices]
    dq = data.qvel[dof_indices]
    d_dot = float(rhat @ (Jp @ dq))

    A = (rhat @ Jp).reshape(1, -1)
    b = - (2*zeta*omega*d_dot + (omega**2)*(d - min_clearance))
    return A, np.array([b]), d
