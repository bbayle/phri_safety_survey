# Part II — Variable-Impedance Admittance Control

## Survey section
Part II §2.2 Active Compliance via Control

## What this simulates
A 3-DOF torque-controlled planar arm tracking a fixed target position
using task-space admittance control. Impedance stiffness is scheduled
based on contact state: low K in free motion, high K in contact.

## Key concepts demonstrated
- Task-space admittance control: τ = Jᵀ (K Δx − D ẋ + f_ext)
- Contact-state detection via `cfrc_ext` (external wrench per body)
- Stiffness switching: free ↔ constrained without discontinuity
- Post-hoc plot: position error and stiffness schedule vs time

## Design note
Admittance vs. impedance: this example uses direct torque commands,
which is impedance control in the strict sense. For a position-controlled
cobot (Kuka KMR, UR series), you would wrap an outer admittance loop
that converts measured force into a Δx reference, then pass that to the
inner position controller. See survey §2.2 for the distinction.

## Run

```bash
conda activate phri
python part2_compliant_mechanisms/main.py
```

## Output
- Live viewer with arm motion
- `admittance_log.png` — EE error and stiffness trace vs time

## Files
```
part2_compliant_mechanisms/
├── README.md
├── main.py
└── assets/
    └── planar_arm.xml    # 3-DOF torque-controlled arm
```
