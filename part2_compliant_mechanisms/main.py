"""
Part II — Variable-Impedance Admittance Control
════════════════════════════════════════════════

Survey section: Part II §2.2 Active Compliance via Control

What this demonstrates
──────────────────────
- Task-space admittance control on a 3-DOF torque-controlled arm
- Impedance parameter scheduling: low stiffness in free motion,
  high stiffness in constrained/contact phases
- External force estimation via MuJoCo's cfrc_ext
- Matplotlib live plot of stiffness schedule and end-effector error

Design note (survey §2.2)
──────────────────────────
Admittance control (force input → position output) is preferred over
impedance control (position input → force output) when the robot is
position-controlled at a low level (as with most commercial cobots).
Impedance control is more natural for torque-controlled robots.
This example assumes torque control (direct actuator torques).

Usage
─────
    python part2_compliant_mechanisms/main.py
"""
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.robot_utils import site_jacobian, get_site_position

# ── Config ────────────────────────────────────────────────────────────────────
XML_PATH     = os.path.join(os.path.dirname(__file__), "assets", "planar_arm.xml")
SIM_DURATION = 8.0          # seconds
EE_SITE      = "ee_site"

# Desired end-effector position (world frame)
X_DESIRED = np.array([0.5, 0.0, 0.4])

# Impedance parameters — task space (3×3 diagonal)
K_FREE    = np.diag([60.0,  60.0,  60.0])   # low stiffness: free motion
K_CONTACT = np.diag([250.0, 250.0, 250.0])  # high stiffness: constrained
D_FIXED   = np.diag([18.0,  18.0,  18.0])   # damping (contact-invariant)

CONTACT_THRESHOLD = 2.0   # N — below this we consider "free motion"


def admittance_torques(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    K: np.ndarray,
) -> np.ndarray:
    """
    Compute joint torques from task-space admittance law:
        τ = Jᵀ (K Δx − D ẋ_ee + f_ext)

    where Δx = x_d − x_ee, ẋ_ee = J q̇, f_ext from cfrc_ext.
    """
    J   = site_jacobian(model, data, EE_SITE)          # 3 × nv
    x   = get_site_position(model, data, EE_SITE)       # 3,
    dx  = J @ data.qvel                                  # 3,  ee velocity

    e   = X_DESIRED - x                                  # position error

    # External wrench on end-effector body (world frame, linear part)
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    f_ext = data.cfrc_ext[ee_body_id, 3:6]              # force component only

    # Admittance law
    f_task = K @ e - D_FIXED @ dx + f_ext
    tau    = J.T @ f_task

    # Saturate to actuator limits
    limits = np.array([50.0, 40.0, 25.0])
    return np.clip(tau[:model.nu], -limits, limits)


def run() -> None:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    times, errors, stiffness_trace, f_ext_trace = [], [], [], []
    t = 0.0

    print("[Part II] Variable-impedance admittance control")
    print(f"          Target: {X_DESIRED}  |  K_free diag: {np.diag(K_FREE)}")
    print("          In contact → K switches to K_contact\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -15

        while viewer.is_running() and t < SIM_DURATION:
            # ── Contact-state detection ────────────────────────────────────
            ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
            f_ext_mag  = float(np.linalg.norm(data.cfrc_ext[ee_body_id, 3:6]))
            in_contact = f_ext_mag > CONTACT_THRESHOLD or data.ncon > 0

            K = K_CONTACT if in_contact else K_FREE

            # ── Control ────────────────────────────────────────────────────
            data.ctrl[:] = admittance_torques(model, data, K)
            mujoco.mj_step(model, data)

            # ── Logging ────────────────────────────────────────────────────
            x   = get_site_position(model, data, EE_SITE)
            err = float(np.linalg.norm(X_DESIRED - x))
            times.append(t)
            errors.append(err)
            stiffness_trace.append(float(K[0, 0]))
            f_ext_trace.append(f_ext_mag)

            viewer.sync()
            t += model.opt.timestep

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(times, errors, lw=1.5, color="steelblue", label="EE position error (m)")
    ax1.set_ylabel("Error (m)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(times, stiffness_trace, lw=1.5, color="darkorange", label="K_x (N/m)")
    ax2_r = ax2.twinx()
    ax2_r.plot(times, f_ext_trace, lw=1.0, color="gray", alpha=0.6, label="|f_ext| (N)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Stiffness (N/m)")
    ax2_r.set_ylabel("|f_ext| (N)")
    ax2.legend(loc="upper left"); ax2_r.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle("Part II — Variable-Impedance Admittance Control")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "admittance_log.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[Part II] Plot saved to {out}")


if __name__ == "__main__":
    run()
