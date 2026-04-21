"""
Part IV — Domain-Randomised Safe Policy Evaluation
════════════════════════════════════════════════════

Survey section: Part IV §4.3 Learning-Based Safe Control, §4.3 Sim-to-Real

What this demonstrates
──────────────────────
- Domain randomisation over mass, joint damping, and geom friction
- Dual-objective evaluation: mean reward AND worst-case contact force
- Pareto plot: the fundamental tension between task performance and safety
- Baseline: random policy vs. a simple gravity-compensating safe policy

Design rationale (survey §4.3)
───────────────────────────────
Safe RL evaluation must report both performance and safety metrics.
Reporting only mean reward is insufficient — a policy can achieve high
reward while occasionally producing dangerous contact forces.
The Pareto frontier of (reward, safety) is the honest evaluation target.

This is headless (no viewer) — it runs N_ROLLOUTS rollouts and produces
a scatter plot. Runtime on Apple M2: ~20–40 s for default settings.

Usage
─────
    python part4_safe_control/main.py [--rollouts 50] [--steps 300]
"""
import sys, os, random, argparse
import numpy as np
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.robot_utils import max_contact_force

XML_PATH   = os.path.join(os.path.dirname(__file__), "assets", "robot_arm.xml")
N_ROLLOUTS = 50
MAX_STEPS  = 300

# ── Domain randomisation ──────────────────────────────────────────────────────

def randomise_dynamics(model: mujoco.MjModel, rng: np.random.Generator) -> None:
    """
    Perturb mass, damping, and sliding friction within ±15–30% of nominal.
    Conservative ranges — wider than typical industrial tolerances.
    """
    for i in range(model.nbody):
        model.body_mass[i] *= rng.uniform(0.85, 1.15)
    for i in range(model.nv):
        model.dof_damping[i] *= rng.uniform(0.70, 1.30)
    for i in range(model.ngeom):
        model.geom_friction[i, 0] *= rng.uniform(0.60, 1.40)   # sliding
        model.geom_friction[i, 1] *= rng.uniform(0.80, 1.20)   # rolling
        model.geom_friction[i, 2] *= rng.uniform(0.80, 1.20)   # spinning


# ── Policies ──────────────────────────────────────────────────────────────────

def random_policy(obs: np.ndarray, nu: int) -> np.ndarray:
    """Unconstrained random policy — worst-case baseline."""
    return np.random.uniform(-30, 30, nu)


def safe_pd_policy(obs: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Gravity-compensating PD policy toward zero configuration.
    Represents a conservative but safe strategy.
    """
    q   = obs[:model.nq]
    dq  = obs[model.nq:model.nq + model.nv]
    Kp  = np.array([40, 40, 30, 30, 10, 10, 8], dtype=float)
    Kd  = np.array([6,  6,  4,  4,  2,  2,  1.5], dtype=float)
    # gravity compensation
    mujoco.mj_forward(model, data)
    grav = data.qfrc_bias[:model.nu].copy()
    tau  = Kp * (0.0 - q[:model.nu]) + Kd * (0.0 - dq[:model.nu]) + grav
    return np.clip(tau, -87, 87)


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout(policy_fn, rng: np.random.Generator, max_steps: int) -> tuple[float, float]:
    """
    Run one episode with randomised dynamics.
    Returns (cumulative_reward, worst_contact_force).
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    randomise_dynamics(model, rng)
    mujoco.mj_resetData(model, data)

    total_reward  = 0.0
    worst_force   = 0.0

    for _ in range(max_steps):
        obs = np.concatenate([data.qpos, data.qvel])
        data.ctrl[:] = policy_fn(obs, model, data)
        mujoco.mj_step(model, data)

        # Reward: penalise joint velocity (energy / safety proxy)
        r = -0.01 * float(np.linalg.norm(data.qvel))
        total_reward += r

        # Safety metric: worst contact force this step
        f = max_contact_force(model, data)
        worst_force = max(worst_force, f)

    return total_reward, worst_force


def evaluate_policy(name: str, policy_fn, n: int, max_steps: int,
                    seed: int = 42) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    rewards, forces = [], []
    print(f"  Evaluating '{name}' over {n} rollouts...")
    for i in range(n):
        r, f = rollout(policy_fn, rng, max_steps)
        rewards.append(r)
        forces.append(f)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n}  mean_r={np.mean(rewards):.2f}  max_f={np.max(forces):.2f} N")
    return rewards, forces


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_rollouts: int = N_ROLLOUTS, max_steps: int = MAX_STEPS) -> None:
    print("[Part IV] Domain-randomised safe policy evaluation (headless)")
    print(f"          {n_rollouts} rollouts × {max_steps} steps each\n")

    # Wrap policies to unified signature (obs, model, data)
    rand_fn = lambda obs, model, data: random_policy(obs, model.nu)
    safe_fn = safe_pd_policy

    r_rand, f_rand = evaluate_policy("random",  rand_fn, n_rollouts, max_steps, seed=0)
    r_safe, f_safe = evaluate_policy("safe_PD", safe_fn, n_rollouts, max_steps, seed=0)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────────────────")
    for name, rewards, forces in [("random", r_rand, f_rand), ("safe_PD", r_safe, f_safe)]:
        print(f"  {name:10s}  mean_reward={np.mean(rewards):7.2f}  "
              f"mean_max_force={np.mean(forces):6.2f} N  "
              f"worst_force={np.max(forces):6.2f} N")

    # ── Pareto plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(r_rand, f_rand, alpha=0.55, s=40, c="tomato",    label="Random policy")
    ax.scatter(r_safe, f_safe, alpha=0.55, s=40, c="steelblue", label="Safe PD policy")

    ax.axhline(150.0, color="black", ls="--", lw=1.2, label="ISO contact limit (150 N)")

    ax.set_xlabel("Cumulative reward (higher = better)")
    ax.set_ylabel("Worst-case contact force per episode (N, lower = safer)")
    ax.set_title("Part IV — Reward vs Safety: Pareto Frontier\n"
                 "(domain-randomised, 7-DOF arm)")
    ax.legend()
    ax.grid(alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "pareto_plot.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\n[Part IV] Pareto plot saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=N_ROLLOUTS)
    parser.add_argument("--steps",    type=int, default=MAX_STEPS)
    args = parser.parse_args()
    run(args.rollouts, args.steps)
