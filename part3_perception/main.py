"""
Part III — Human Motion Prediction & Proactive Safety Bubble
═════════════════════════════════════════════════════════════

Survey section: Part III §3.3 Human State Estimation, §3.1 Exteroceptive Sensing

What this demonstrates
──────────────────────
- Rolling-window velocity extrapolation to predict human wrist position
  over a configurable time horizon
- Safety bubble: robot velocity is scaled proportionally to predicted
  proximity — smooth deceleration rather than a binary stop
- Ground-truth vs. predicted trajectory logging and post-hoc plot

Limitations (survey §3.3)
──────────────────────────
Linear extrapolation is adequate for horizon < 300 ms at slow walking
speeds. For longer horizons or fast, non-linear human motion, data-driven
predictors (Trajectron++, Social Force Model variants) are necessary.
This example is intentionally minimal to make the architectural pattern
legible. The survey discusses the full prediction stack.

Usage
─────
    python part3_perception/main.py
"""
import sys, os, time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.robot_utils import get_body_position, get_site_position

# ── Config ────────────────────────────────────────────────────────────────────
XML_PATH         = os.path.join(os.path.dirname(__file__), "assets", "human_robot_scene.xml")
SIM_DURATION     = 10.0      # seconds
DT               = 0.01      # matches XML timestep
HISTORY_WINDOW   = 20        # steps ≈ 200 ms at 100 Hz
PRED_HORIZON     = 0.3       # seconds ahead to predict
SAFETY_MARGIN    = 0.20      # metres — bubble radius
HUMAN_WRIST_BODY = "human_wrist"
ROBOT_EE_SITE    = "robot_ee"

# Human wrist sinusoidal trajectory (drives joint positions directly)
def human_qpos(t: float) -> np.ndarray:
    """Scripted human arm motion: slow reach-and-retract."""
    shoulder = np.array([
        0.4 * np.sin(0.6 * t),           # y-axis swing
        0.3 * np.cos(0.4 * t),           # x-axis rotation
        0.2 * np.sin(0.5 * t + 0.5),    # z-axis twist
        1.0                               # quaternion w (normalised later)
    ])
    shoulder /= np.linalg.norm(shoulder)  # normalise quaternion
    elbow = 0.8 + 0.4 * np.sin(0.7 * t + 1.0)
    return np.concatenate([shoulder, [elbow]])


def predict_linear(history: deque, dt: float, horizon: float) -> np.ndarray:
    """
    Linear extrapolation: x_pred = x[-1] + v * horizon
    where v = (x[-1] - x[-2]) / dt.
    Falls back to last known position if history is too short.
    """
    if len(history) < 2:
        return history[-1].copy()
    v = (history[-1] - history[-2]) / dt
    return history[-1] + v * horizon


def robot_pd_torques(model, data, q_d, Kp=30.0, Kd=4.0) -> np.ndarray:
    """Simple PD torque control toward desired joint configuration."""
    e   = q_d - data.qpos[:model.nu]
    de  = -data.qvel[:model.nu]
    tau = Kp * e + Kd * de
    return np.clip(tau, -60, 60)


def run() -> None:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    history: deque = deque(maxlen=HISTORY_WINDOW)
    times, dists_true, dists_pred, scales = [], [], [], []
    t = 0.0

    # Robot rests at a neutral config, only velocity scales
    q_robot_d = np.array([0.3, -0.5])

    print("[Part III] Human motion prediction + safety bubble")
    print(f"           Horizon: {PRED_HORIZON*1000:.0f} ms  |  Bubble radius: {SAFETY_MARGIN} m\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20

        while viewer.is_running() and t < SIM_DURATION:
            # ── Drive human arm kinematics ─────────────────────────────────
            hq = human_qpos(t)
            # ball joint: 4 DOFs (quaternion), then elbow hinge
            # find human joint addresses
            human_shoulder_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "human_shoulder_j")
            human_elbow_jid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "human_elbow_j")
            qadr_s = model.jnt_qposadr[human_shoulder_jid]
            qadr_e = model.jnt_qposadr[human_elbow_jid]
            data.qpos[qadr_s:qadr_s+4] = hq[:4]
            data.qpos[qadr_e]           = hq[4]

            mujoco.mj_forward(model, data)

            # ── Observe human wrist position ───────────────────────────────
            wrist = get_body_position(model, data, HUMAN_WRIST_BODY)
            history.append(wrist.copy())

            # ── Predict future wrist position ──────────────────────────────
            predicted = predict_linear(history, DT, PRED_HORIZON)

            # ── Compute distances ──────────────────────────────────────────
            ee = get_site_position(model, data, ROBOT_EE_SITE)
            dist_true = float(np.linalg.norm(wrist    - ee))
            dist_pred = float(np.linalg.norm(predicted - ee))

            # ── Velocity scaling from predicted proximity ──────────────────
            scale = min(1.0, dist_pred / SAFETY_MARGIN)
            if scale < 1.0:
                print(f"[t={t:.2f}s] Predicted proximity {dist_pred:.3f} m → scale {scale:.2f}")

            # ── Robot control (scaled) ─────────────────────────────────────
            base_tau = robot_pd_torques(model, data, q_robot_d)
            data.ctrl[:] = base_tau * scale

            mujoco.mj_step(model, data)

            # ── Log ────────────────────────────────────────────────────────
            times.append(t)
            dists_true.append(dist_true)
            dists_pred.append(dist_pred)
            scales.append(scale)

            viewer.sync()
            t += DT

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(times, dists_true, lw=1.5, label="True wrist–EE distance (m)")
    ax1.plot(times, dists_pred, lw=1.5, ls="--", label=f"Predicted (+{PRED_HORIZON*1000:.0f} ms)")
    ax1.axhline(SAFETY_MARGIN, color="red", ls=":", lw=1.5, label=f"Safety bubble ({SAFETY_MARGIN} m)")
    ax1.set_ylabel("Distance (m)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(times, scales, lw=1.5, color="darkorange", label="Velocity scale factor")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Scale [0–1]")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Part III — Human Motion Prediction & Safety Bubble")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "prediction_log.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[Part III] Plot saved to {out}")


if __name__ == "__main__":
    run()
