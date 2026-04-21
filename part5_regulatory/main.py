"""
Part V — Supervisory Safety Envelope with Dual-Channel E-Stop
══════════════════════════════════════════════════════════════

Survey section: Part V §5.2 Verification and Validation,
                §5.1 ISO 10218-1 SRP/CS Category 3

What this demonstrates
──────────────────────
- A supervisory safety layer that runs independently of the functional
  controller — mirroring real SRP/CS (Safety-Related Parts of Control
  System) Category 3 architecture per ISO 13849
- Watchdog timer: controller must call heartbeat() every 5 ms or the
  safety monitor triggers an emergency stop
- Dual-channel E-stop:
    Channel A — zero all control signals
    Channel B — active braking torques opposing current velocity
- Checks: joint velocity, joint torque, and contact force
- Post-hoc violation log and timeline plot

Regulatory context (survey §5.1)
──────────────────────────────────
ISO 10218-1 §5.4 defines four Safety-related Control System (SRP/CS)
categories (B, 1, 2, 3, 4). Category 3 requires that a single fault
does not lead to loss of safety function, achieved here by separating
the safety channel (SafetyMonitor) from the functional channel (PD
controller). The watchdog ensures the functional channel is alive.

Usage
─────
    python part5_regulatory/main.py
"""
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.safety_monitor import SafetyMonitor, SafetyViolation

# ── Config ────────────────────────────────────────────────────────────────────
XML_PATH         = os.path.join(os.path.dirname(__file__), "assets", "cobot_arm.xml")
SIM_DURATION     = 8.0          # seconds

# ISO 10218-1 / ISO/TS 15066 limits for a 7-DOF collaborative robot
MAX_JOINT_VEL    = np.deg2rad([100, 100, 100, 100, 120, 120, 120])   # rad/s
MAX_JOINT_TORQUE = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)  # Nm
MAX_CONTACT_F    = 150.0        # N (ISO/TS 15066 transient, whole-body)
WATCHDOG_TIMEOUT = 0.005        # 5 ms


# ── Functional controller (runs in "functional channel") ──────────────────────

class TrajectoryController:
    """
    Simple joint-space PD controller tracking a sinusoidal trajectory.
    Intentionally ramps velocity to eventually trigger the safety monitor,
    demonstrating that the supervisory layer catches violations regardless
    of what the functional controller does.
    """
    def __init__(self, model: mujoco.MjModel):
        self.nu   = model.nu
        self.Kp   = np.array([60, 60, 50, 50, 15, 15, 10], dtype=float)
        self.Kd   = np.array([8,  8,  6,  6,  2,  2,  1.5], dtype=float)
        self._t   = 0.0

    def compute(self, data: mujoco.MjData, dt: float) -> np.ndarray:
        self._t += dt
        # Amplitude ramps up over time to eventually exceed velocity limits
        amp  = min(1.5, 0.05 * self._t)
        freq = 1.2
        q_d  = amp * np.sin(freq * self._t * np.arange(1, self.nu + 1))
        dq_d = amp * freq * np.cos(freq * self._t * np.arange(1, self.nu + 1))
        tau  = self.Kp * (q_d - data.qpos[:self.nu]) + self.Kd * (dq_d - data.qvel[:self.nu])
        return tau  # deliberately NOT clamped — safety monitor must catch it


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    monitor    = SafetyMonitor(model, MAX_JOINT_VEL, MAX_JOINT_TORQUE,
                               contact_force_limit=MAX_CONTACT_F,
                               watchdog_timeout=WATCHDOG_TIMEOUT)
    controller = TrajectoryController(model)
    dt         = float(model.opt.timestep)

    times, qvel_max, tau_max, fcon_max, estop_times = [], [], [], [], []
    t = 0.0

    print("[Part V] Supervisory safety envelope — ISO 10218-1 SRP/CS Category 3")
    print(f"         Watchdog timeout: {WATCHDOG_TIMEOUT*1000:.0f} ms  |  "
          f"Contact limit: {MAX_CONTACT_F} N\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20

        while viewer.is_running() and t < SIM_DURATION and not monitor._stopped:

            # ── Functional channel ─────────────────────────────────────────
            tau = controller.compute(data, dt)
            data.ctrl[:] = np.clip(tau, -87, 87)

            # Functional channel pings watchdog
            monitor.heartbeat()

            # ── Step simulation ────────────────────────────────────────────
            mujoco.mj_step(model, data)

            # ── Safety channel (independent check) ────────────────────────
            safe, violation = monitor.check(data)
            if not safe:
                reason = str(violation) if violation else "unknown"
                monitor.emergency_stop(data, reason)
                estop_times.append(t)
                print(f"  → E-stop at t={t:.3f}s")
                # Apply one more step with braking torques
                mujoco.mj_step(model, data)
                viewer.sync()
                break

            # ── Logging ────────────────────────────────────────────────────
            times.append(t)
            qvel_max.append(float(np.max(np.abs(data.qvel[:model.nv]))))
            tau_max.append(float(np.max(np.abs(data.actuator_force))))
            fcon = max((np.linalg.norm(data.contact[c].frame[:3])
                        for c in range(data.ncon)), default=0.0)
            fcon_max.append(fcon)

            viewer.sync()
            t += dt

    # ── Final report ──────────────────────────────────────────────────────────
    monitor.report()
    print(f"\n[Part V] E-stops triggered: {len(estop_times)}")
    if estop_times:
        print(f"         First E-stop at: t={estop_times[0]:.3f} s")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not times:
        print("[Part V] No data to plot (stopped before first logged step).")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    lims  = [np.max(MAX_JOINT_VEL), np.max(MAX_JOINT_TORQUE), MAX_CONTACT_F]
    ylabs = ["Max |q̇| (rad/s)", "Max |τ| (Nm)", "Max contact F (N)"]
    data_ = [qvel_max, tau_max, fcon_max]
    cols  = ["steelblue", "darkorange", "seagreen"]

    for ax, y, lim, lab, col in zip(axes, data_, lims, ylabs, cols):
        ax.plot(times, y, lw=1.3, color=col)
        ax.axhline(lim, color="red", ls="--", lw=1.2, label=f"Limit ({lim:.0f})")
        for et in estop_times:
            ax.axvline(et, color="black", ls=":", lw=1.5, label="E-stop")
        ax.set_ylabel(lab)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Part V — Supervisory Safety Monitor\n"
                 "ISO 10218-1 SRP/CS Category 3  |  Dual-channel E-stop")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "safety_monitor_log.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[Part V] Plot saved to {out}")


if __name__ == "__main__":
    run()
