# experiments/run_ur5e_multi.py
import time, yaml, numpy as np, mujoco
import mujoco.viewer as viewer
from collections import deque

from envs.ur5e_multi import build_ur5e_multirobot_model, resolve_arm_handles
from controllers.pid import JointPID
from controllers.impedance import JointImpedance
from controllers.sliding_mode import JointSMC
from controllers.admittance import JointAdmittance  # optional
from safety.cbf import CBFProjector
from safety.geometry import obstacle_cbf_rows_for_dofs

def rad(d): return np.deg2rad(np.array(d, dtype=float))

def build_controller(spec):
    typ = spec["type"].upper()
    if typ == "PID":
        return JointPID(spec["Kp"], spec["Ki"], spec["Kd"], n_joints=6)
    if typ == "IMPEDANCE":
        return JointImpedance(spec["K"], spec["D"])
    if typ == "SMC":
        return JointSMC(spec["Lambda"], spec["K_sign"], Kp=spec["Kp"], Kd=spec["Kd"])
    if typ == "ADMITTANCE":
        M = spec.get("M", [1,1,1,1,1,1])
        return JointAdmittance(M, spec["D"], spec["K"], n_joints=6)
    raise ValueError(f"Unknown controller: {typ}")

def joint_limit_cbf_rows(q, dq, qmin, qmax, zeta=0.8, omega=4.0):
    n = len(q); A=[]; b=[]
    for i in range(n):
        h1, dh1 = (qmax[i]-q[i]),  -dq[i]
        rhs1 = -(2*zeta*omega*dh1 + (omega**2)*h1)
        row1 = np.zeros(n); row1[i]=+1; A.append(row1); b.append(rhs1)
        h2, dh2 = (q[i]-qmin[i]),   dq[i]
        rhs2 = -(2*zeta*omega*dh2 + (omega**2)*h2)
        row2 = np.zeros(n); row2[i]=-1; A.append(row2); b.append(rhs2)
    return np.vstack(A), np.array(b)

def main():
    with open("configs/ur5e_multi.yaml","r") as f:
        cfg = yaml.safe_load(f)

    dt = float(cfg["sim"]["dt"])
    T  = float(cfg["sim"]["seconds"])
    ctrl_range = cfg["actuators"]["default_ctrl_range"]
    damping    = float(cfg["actuators"]["damping"])
    rho   = float(cfg["cbf"]["rho"])
    zeta  = float(cfg["cbf"]["joint_barrier"]["zeta"])
    omega = float(cfg["cbf"]["joint_barrier"]["omega"])

    viz = cfg.get("viz", {})
    trail_N   = int(viz.get("trail_points", 0))
    keep_open = bool(viz.get("keep_window_open", True))

    # propagate trail setting so env can preallocate dots
    for inst in cfg["instances"]:
        inst["trail_points"] = trail_N

    # build multi-UR5e world
    ur5e_xml = "assets/universal_robots_ur5e/ur5e.xml"
    model, data = build_ur5e_multirobot_model(ur5e_xml, cfg["instances"], dt=dt)

    # name helpers
    name2geom = lambda nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
    name2site = lambda nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
    name2body = lambda nm: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)

    # per-instance metadata & controllers
    inst_meta, ctrls, amps, freqs, boxes, proj = [], [], [], [], [], []
    for inst in cfg["instances"]:
        name = inst["name"]
        meta = resolve_arm_handles(model, prefix=name)
        meta["name"] = name
        inst_meta.append(meta)

        ctrls.append(build_controller(inst["controller"]))
        amps.append(rad(inst["reference"]["amp_deg"]))
        freqs.append(float(inst["reference"]["freq_hz"]))

        umax = float(inst.get("torque_limit", 80.0)) * np.ones(6)
        boxes.append(umax)
        proj.append(CBFProjector(u_min=-umax, u_max=umax, rho=rho))

    # joint limits (same across arms)
    qmin = np.array([model.jnt_range[j,0] for j in inst_meta[0]["jidx"]])
    qmax = np.array([model.jnt_range[j,1] for j in inst_meta[0]["jidx"]])

    # trail buffers
    trail_bufs = [deque(maxlen=trail_N if trail_N>0 else 1) for _ in cfg["instances"]]

    with viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while (time.time() - t0) < T:
            u = np.zeros(model.nu)
            t = time.time() - t0

            for k, meta in enumerate(inst_meta):
                # state
                q  = np.array([data.qpos[model.jnt_qposadr[j]] for j in meta["jidx"]])
                dq = np.array([data.qvel[d] for d in meta["dofs"]])

                # reference (joint sinusoid)
                w = 2*np.pi*freqs[k]
                q_ref  = amps[k] * np.sin(w*t)
                dq_ref = np.zeros_like(q_ref)

                # raw control
                u_des = ctrls[k].update(q, dq, q_ref, dq_ref, dt)
                u_des = np.clip(u_des, -boxes[k], boxes[k])

                # CBF constraints
                A1, b1 = joint_limit_cbf_rows(q, dq, qmin, qmax, zeta=zeta, omega=omega)
                obst = cfg["instances"][k].get("obstacle", None)
                if obst:
                    center = np.array(obst["center"], dtype=float)
                    dmin   = float(obst.get("d_min", 0.12))
                    if meta.get("site"):
                        A2, b2, _ = obstacle_cbf_rows_for_dofs(
                            model, data, meta["dofs"], center, dmin,
                            zeta=0.9, omega=5.0, site_name=meta["site"], body_name=None
                        )
                    else:
                        A2, b2, _ = obstacle_cbf_rows_for_dofs(
                            model, data, meta["dofs"], center, dmin,
                            zeta=0.9, omega=5.0, site_name=None, body_name=meta["body"]
                        )
                    A = np.vstack([A1, A2]); b = np.hstack([b1, b2])
                else:
                    A, b = A1, b1

                u_safe, s = proj[k].project_with_slack(u_des, A_cbf=A, b_cbf=b)
                slack_sum = float(np.sum(s)) if s is not None else 0.0

                # beacon flash on CBF use
                beacon_id = name2geom(f"{meta['name']}_beacon")
                if beacon_id != -1:
                    base_rgba = np.array(cfg["instances"][k].get("color", [0.8,0.8,0.8,1.0]), dtype=float)
                    rgba = base_rgba.copy()
                    if slack_sum > 1e-8:
                        rgba[3] = 0.95  # brighter alpha
                    model.geom_rgba[beacon_id] = rgba

                # EE pos for trail
                if meta.get("site"):
                    sid = name2site(meta["site"])
                    p = data.site_xpos[sid].copy()
                else:
                    bid = name2body(meta["body"])
                    p = data.xipos[bid].copy()
                trail_bufs[k].append(p)

                # write torques (assumes actuators aligned with DOFs for this model)
                start = k*6
                u[start:start+6] = u_safe

            data.ctrl[:] = u
            mujoco.mj_step(model, data)

            # update trail dot positions by moving trail geoms
            if trail_N > 0:
                for k, inst in enumerate(cfg["instances"]):
                    name = inst["name"]
                    pts = list(trail_bufs[k])
                    if len(pts) < trail_N:
                        pts = [pts[0] if pts else np.zeros(3)]*(trail_N-len(pts)) + pts
                    for i, pos in enumerate(pts[-trail_N:]):
                        gid = name2geom(f"{name}_trail_{i:03d}")
                        if gid != -1:
                            model.geom_pos[gid] = pos
                mujoco.mj_forward(model, data)  # reflect moved geoms

            v.sync()

        # keep window open after sim ends
        if keep_open:
            while v.is_running():
                v.sync()
                time.sleep(0.02)

if __name__ == "__main__":
    main()
