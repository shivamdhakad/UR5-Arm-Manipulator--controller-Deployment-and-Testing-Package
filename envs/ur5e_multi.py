# envs/ur5e_multi.py
import mujoco
import numpy as np

UR5E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# try these site/body names for the end-effector
SITE_CANDIDATES = ["ee", "tcp", "flange", "tool0"]
BODY_CANDIDATES = ["tool0", "ee_link", "wrist_3_link", "flange"]

def build_ur5e_multirobot_model(ur5e_xml_path: str, instances, dt=0.002, obstacle_radius=0.06):
    """
    Load UR5e MJCF once as an MjSpec child, then attach multiple copies
    into a parent MjSpec with unique prefixes and base positions. Also
    spawn per-robot beacons and preallocated trail dots.
    """
    child = mujoco.MjSpec.from_file(ur5e_xml_path)

    parent = mujoco.MjSpec()
    parent.option.timestep = dt
    parent.copy_during_attach = True

    # Ground plane (size length-3)
    g = parent.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.pos  = (0.0, 0.0, 0.0)
    g.size = [5.0, 5.0, 0.1]
    g.rgba = (0.8, 0.8, 0.8, 1.0)

    for inst in instances:
        name   = inst["name"]
        bx, by, bz = map(float, inst["base_pos"])
        rgba   = inst.get("color", [0.8, 0.8, 0.8, 1.0])
        Ntrail = int(inst.get("trail_points", 0)) or 0

        # Base frame
        frame = parent.worldbody.add_frame()
        frame.pos = (bx, by, bz)

        # Attach UR5e under this frame with unique prefix
        parent.attach(child, frame=frame, prefix=f"{name}_")

        # Beacon above base (flashes when CBF active)
        beacon = parent.worldbody.add_geom()
        beacon.name = f"{name}_beacon"
        beacon.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        beacon.pos  = (bx, by, bz + 0.35)
        beacon.size = np.array([0.03, 0.08, 0.0])  # radius, half-height, (unused)
        beacon.rgba = tuple(rgba)

        # Pre-allocate EE trail dots
        for i in range(Ntrail):
            tgeom = parent.worldbody.add_geom()
            tgeom.name = f"{name}_trail_{i:03d}"
            tgeom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            tgeom.pos  = (bx, by, bz)             # will be moved at runtime
            tgeom.size = np.array([0.008, 0.0, 0.0])
            rgba_tr = list(rgba); rgba_tr[3] = 0.35
            tgeom.rgba = tuple(rgba_tr)

        # Optional obstacle (matching color family)
        if "obstacle" in inst and "center" in inst["obstacle"]:
            cx, cy, cz = map(float, inst["obstacle"]["center"])
            s = parent.worldbody.add_geom()
            s.name = f"{name}_obst"
            s.type = mujoco.mjtGeom.mjGEOM_SPHERE
            s.pos  = (cx, cy, cz)
            s.size = np.array([float(obstacle_radius), 0.0, 0.0])  # length-3
            rgba_ob = list(rgba); rgba_ob[3] = 0.35
            s.rgba = tuple(rgba_ob)

    model = parent.compile()
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data

def resolve_arm_handles(model, prefix):
    """
    Return ordered joint IDs, dof indices, and either an EE site or a fallback body.
    """
    # joints
    jids = []
    for suf in UR5E_JOINTS:
        nm = f"{prefix}_{suf}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if jid == -1:
            raise RuntimeError(f"Joint not found: {nm}")
        jids.append(jid)
    dofs = [model.jnt_dofadr[j] for j in jids]

    # prefer a site
    for s in SITE_CANDIDATES:
        nm = f"{prefix}_{s}"
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, nm)
        if sid != -1:
            return dict(jidx=jids, dofs=dofs, site=nm, body=None)

    # fallback to a body
    for b in BODY_CANDIDATES:
        nm = f"{prefix}_{b}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid != -1:
            return dict(jidx=jids, dofs=dofs, site=None, body=nm)

    raise RuntimeError(
        f"No EE site/body found for prefix '{prefix}'. "
        f"Tried sites {SITE_CANDIDATES} and bodies {BODY_CANDIDATES}."
    )
