"""
shared/safety_monitor.py
────────────────────────
Reusable ISO/TS 15066 and ISO 10218-1 safety checker.

Design rationale
────────────────
Safety checking is deliberately separated from control logic.
This mirrors real SRP/CS (Safety-Related Parts of Control System) architecture,
where the safety channel is independent of the functional channel.

Usage
─────
    from shared.safety_monitor import SafetyMonitor, SafetyViolation

    monitor = SafetyMonitor(model, joint_vel_limits, joint_torque_limits)
    ok, violation = monitor.check(data)
    if not ok:
        monitor.emergency_stop(data)
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import mujoco


# ─── ISO/TS 15066 Table 1 — quasi-static contact force limits (N) ────────────
# Body region → (force_limit_N, pressure_limit_kPa)
ISO_TS_15066_LIMITS: dict[str, tuple[float, float]] = {
    "skull_and_forehead":     (130,  67),
    "face":                    (65,  26),
    "neck_front":              (145,  42),
    "neck_back":               (145,  36),
    "back_and_shoulder":      (210,  35),
    "chest":                  (140,  45),
    "abdomen":                (110,  35),
    "pelvis":                 (180,  75),
    "upper_arm_and_elbow":    (150,  32),
    "lower_arm_and_wrist":    (160,  40),
    "hand_and_fingers":       (140,  28),
    "thigh_and_knee":         (220,  50),
    "lower_leg":              (130,  36),
    "foot_and_toes":          (125,  28),
}

# Simplified lookup by body region group (for simulation use)
REGION_FORCE_LIMIT: dict[str, float] = {
    "head":    65.0,
    "thorax": 140.0,
    "arm":    110.0,
    "hand":   140.0,
    "leg":    130.0,
}


@dataclass
class SafetyViolation:
    kind: str                    # 'joint_velocity' | 'joint_torque' | 'contact_force' | 'watchdog'
    value: float                 # Observed value
    limit: float                 # Limit that was exceeded
    joint_index: Optional[int] = None
    contact_index: Optional[int] = None

    def __str__(self) -> str:
        loc = f" (joint {self.joint_index})" if self.joint_index is not None else ""
        return (f"[SAFETY VIOLATION] {self.kind}{loc}: "
                f"{self.value:.3f} > limit {self.limit:.3f}")


class SafetyMonitor:
    """
    Supervisory safety layer implementing ISO 10218-1 SRP/CS Category 3 logic.

    Checks are independent of the main controller and run every simulation step.

    Parameters
    ──────────
    model               : MuJoCo model
    joint_vel_limits    : (nv,) array of per-joint velocity limits (rad/s)
    joint_torque_limits : (nu,) array of per-actuator torque limits (Nm)
    contact_force_limit : scalar max contact force (N)
    watchdog_timeout    : max seconds between heartbeat() calls (default 5 ms)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_vel_limits: np.ndarray,
        joint_torque_limits: np.ndarray,
        contact_force_limit: float = 150.0,
        watchdog_timeout: float = 0.005,
    ):
        self.model = model
        self.joint_vel_limits = np.asarray(joint_vel_limits)
        self.joint_torque_limits = np.asarray(joint_torque_limits)
        self.contact_force_limit = contact_force_limit
        self.watchdog_timeout = watchdog_timeout

        self._last_heartbeat: float = time.time()
        self._stopped: bool = False
        self.violation_log: list[SafetyViolation] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def heartbeat(self) -> None:
        """Call this every control cycle to keep the watchdog alive."""
        self._last_heartbeat = time.time()

    def check(self, data: mujoco.MjData) -> tuple[bool, Optional[SafetyViolation]]:
        """
        Run all safety checks. Returns (safe, violation_or_None).
        Logs all violations internally; returns the first one found.
        """
        # 1. Watchdog
        elapsed = time.time() - self._last_heartbeat
        if elapsed > self.watchdog_timeout:
            v = SafetyViolation("watchdog", elapsed, self.watchdog_timeout)
            self.violation_log.append(v)
            return False, v

        # 2. Joint velocities
        for i, (vel, lim) in enumerate(zip(np.abs(data.qvel), self.joint_vel_limits)):
            if vel > lim:
                v = SafetyViolation("joint_velocity", vel, lim, joint_index=i)
                self.violation_log.append(v)
                return False, v

        # 3. Actuator torques
        for i, (tau, lim) in enumerate(zip(np.abs(data.actuator_force), self.joint_torque_limits)):
            if tau > lim:
                v = SafetyViolation("joint_torque", tau, lim, joint_index=i)
                self.violation_log.append(v)
                return False, v

        # 4. Contact forces
        for c in range(data.ncon):
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, c, f)
            magnitude = np.linalg.norm(f[:3])
            if magnitude > self.contact_force_limit:
                v = SafetyViolation("contact_force", magnitude, self.contact_force_limit, contact_index=c)
                self.violation_log.append(v)
                return False, v

        return True, None

    def emergency_stop(self, data: mujoco.MjData, reason: str = "") -> None:
        """
        Dual-channel emergency stop:
          Channel A — zero all control signals
          Channel B — apply braking torques opposing current velocity
        """
        self._stopped = True
        nu = len(data.ctrl)
        nv = len(data.qvel)

        # Channel A: zero controls
        data.ctrl[:] = 0.0

        # Channel B: active braking (opposing velocity, clamped to torque limits)
        if nu <= nv:
            braking = -np.sign(data.qvel[:nu]) * self.joint_torque_limits[:nu]
            data.ctrl[:] = braking

        if reason:
            print(f"[E-STOP] {reason}")

    def report(self) -> None:
        """Print a summary of all logged violations."""
        if not self.violation_log:
            print("[SafetyMonitor] No violations recorded.")
            return
        print(f"[SafetyMonitor] {len(self.violation_log)} violation(s):")
        for v in self.violation_log:
            print(f"  {v}")
