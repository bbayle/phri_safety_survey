"""
shared/robot_utils.py
─────────────────────
Common MuJoCo helper functions shared across all parts.
"""
import numpy as np
import mujoco


def get_contact_force(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> np.ndarray:
    """
    Return the 3D resultant contact force (N) acting on a named geom.

    Sums contributions from all active contacts involving that geom.
    Returns a zero vector if the geom is not in contact.
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    total_f = np.zeros(3)
    for c in range(data.ncon):
        con = data.contact[c]
        if con.geom1 == geom_id or con.geom2 == geom_id:
            f_con = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, f_con)
            total_f += f_con[:3]
    return total_f


def get_site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    """Return world-frame position of a named site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[sid].copy()


def get_body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    """Return world-frame position of a named body (CoM)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[bid].copy()


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    """
    Return the 3×nv translational Jacobian for a named site.
    Caller is responsible for calling mj_kinematics / mj_step beforehand.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, sid)
    return jacp


def max_contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Return the scalar magnitude of the largest contact force in the scene."""
    worst = 0.0
    for c in range(data.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, f)
        worst = max(worst, np.linalg.norm(f[:3]))
    return worst
