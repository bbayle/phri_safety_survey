# Part IV — Domain-Randomised Safe Policy Evaluation

## Survey section
Part IV §4.3 Learning-Based Safe Control, Sim-to-Real Transfer

## What this simulates
A 7-DOF arm evaluated across 50 rollouts with randomised dynamics
(mass ±15%, damping ±30%, friction ±40%). Two policies are compared:
- **Random policy**: unconstrained torque noise (worst-case baseline)
- **Safe PD policy**: gravity-compensating PD toward zero configuration

Both are evaluated on a dual objective:
- **Cumulative reward**: penalises high joint velocities
- **Worst-case contact force** per episode (safety metric)

The output is a Pareto scatter plot showing the reward–safety tradeoff.

## Key concepts demonstrated
- Domain randomisation harness (`randomise_dynamics`)
- Dual-objective evaluation: you cannot report just mean reward
- Pareto frontier as the honest evaluation target for safe RL
- `--rollouts` and `--steps` CLI flags for quick scaling

## Extending this
To plug in a real learned policy (e.g., trained with SAC or PPO):

```python
# In main.py, replace safe_fn:
import torch
policy_net = torch.load("my_policy.pt")
def learned_fn(obs, model, data):
    with torch.no_grad():
        return policy_net(torch.tensor(obs, dtype=torch.float32)).numpy()
```

## Run

```bash
conda activate phri
python part4_safe_control/main.py
# or with custom settings:
python part4_safe_control/main.py --rollouts 100 --steps 500
```

## Output
- `pareto_plot.png` — reward vs worst-case force scatter (Pareto view)
- Console: per-policy summary statistics

## Runtime (Apple M2)
- Default (50 rollouts × 300 steps × 2 policies): ~25–40 s headless

## Files
```
part4_safe_control/
├── README.md
├── main.py
└── assets/
    └── robot_arm.xml    # 7-DOF arm (base model for randomisation)
```
