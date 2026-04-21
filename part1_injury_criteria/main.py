"""
Part I — Injury Criteria & ISO/TS 15066 Contact Force Monitoring (with Claude)
════════════════════════════════════════════════════════════════════

Survey section: Part I §1.1–1.2

What this demonstrates
──────────────────────
- Real-time contact force extraction using mj_contactForce
- ISO/TS 15066 Table 1 body-region force limits as a runtime check
- Emergency control cutoff when limits are exceeded
- Force time-series logging and matplotlib post-hoc plot

Key concepts
────────────
ISO/TS 15066 defines quasi-static force limits per body region.
This example applies the "arm" region limit (110 N).
Note the known limitation: these are quasi-static limits, and dynamic
impact forces can briefly exceed them without causing injury — and
vice versa. See survey §1.3 for a full critique.

Usage
─────
    python part1_injury_criteria/main.py

Controls (MuJoCo viewer)
────────────────────────
    Space    — pause/resume
    Esc      — quit
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

# ── Path setup so `shared` is importable when running from repo root ──────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.robot_utils import get_contact_force
from shared.safety_monitor import REGION_FORCE_LIMIT

# ── Configuration ─────────────────────────────────────────────────────────────
XML_PATH       = os.path.join(os.path.dirname(__file__), "assets", "two_dof_arm.xml")
BODY_REGION    = "arm"                          # ISO/TS 15066 region being monitored
FORCE_LIMIT    = REGION_FORCE_LIMIT[BODY_REGION]  # 110 N
SIM_DURATION   = 6.0                            # seconds
EE_GEOM        = "end_effector"

# Sinusoidal joint trajectory that drives the arm into the contact pad
def desired_torques(t: float, data: mujoco.MjData) -> np.ndarray:
    """Simple PD controller tracking a sinusoidal shoulder trajectory."""
    q_d    = np.array([0.6 * np.sin(0.8 * t), -1.2])   # desired positions
    dq_d   = np.zeros(2)                                  # desired velocities
    Kp     = np.array([40.0, 30.0])
    Kd     = np.array([4.0,  3.0])
    tau    = Kp * (q_d - data.qpos[:2]) + Kd * (dq_d - data.qvel[:2])
    return np.clip(tau, -40, 40)


def run() -> None:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    times, forces, violations = [], [], []
    stopped = False
    t = 0.0

    print(f"[Part I] Running contact force monitor — ISO/TS 15066 '{BODY_REGION}' limit: {FORCE_LIMIT} N")
    print("         Close the viewer window to stop.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20

        while viewer.is_running() and t < SIM_DURATION:
            step_start = time.time()

            if not stopped:
                data.ctrl[:] = desired_torques(t, data)
            else:
                data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)

            # ── Contact force measurement ──────────────────────────────────
            f_vec  = get_contact_force(model, data, EE_GEOM)
            f_mag  = float(np.linalg.norm(f_vec))

            times.append(t)
            forces.append(f_mag)

            # ── ISO/TS 15066 check ─────────────────────────────────────────
            if f_mag > FORCE_LIMIT and not stopped:
                print(f"[SAFETY t={t:.3f}s] Contact force {f_mag:.1f} N "
                      f"> ISO limit {FORCE_LIMIT} N → cutting motor torques")
                violations.append(t)
                stopped = True
                data.ctrl[:] = 0.0

            viewer.sync()
            t += model.opt.timestep

            # Real-time pacing (optional — comment out for max speed)
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    # ── Post-hoc plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, forces, lw=1.5, label="Contact force (N)")
    ax.axhline(FORCE_LIMIT, color="red", ls="--", lw=1.5,
               label=f"ISO/TS 15066 '{BODY_REGION}' limit ({FORCE_LIMIT} N)")
    for vt in violations:
        ax.axvline(vt, color="orange", ls=":", lw=1.2, label="E-stop triggered")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Part I — Contact Force vs ISO/TS 15066 Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "contact_force_log.png"), dpi=150)
    plt.show()
    print(f"\n[Part I] Plot saved to part1_injury_criteria/contact_force_log.png")


if __name__ == "__main__":
    run()
