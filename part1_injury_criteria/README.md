# Part I — Injury Criteria & ISO/TS 15066 Contact Force Monitoring

## Survey section
Part I §1.1 Contact Taxonomy, §1.2 Biomechanical Injury Criteria

## What this simulates
A 2-DOF planar manipulator whose end-effector makes contact with a
compliant pad (representing a human arm). The simulation monitors
contact force in real time against ISO/TS 15066 Table 1 body-region
limits, and cuts motor torques the moment the threshold is exceeded.

## Key concepts demonstrated
- `mj_contactForce` for per-contact force extraction
- ISO/TS 15066 quasi-static force limits as a runtime supervisory check
- Emergency control cutoff (Channel A of a dual-channel stop)
- Post-hoc force time-series plot

## Known limitation (see survey §1.3)
ISO/TS 15066 limits are defined for quasi-static contact.
In dynamic (impact) scenarios, peak forces can briefly exceed these limits
without causing injury, and conversely a sustained quasi-static force just
below the limit can cause harm. This example uses the limits as-is to
demonstrate the monitoring architecture; the survey discusses alternatives
(probabilistic injury models, HIC-based dynamic limits).

## Run

```bash
conda activate phri
python part1_injury_criteria/main.py
```

## Output
- Live viewer with the arm motion and contact event
- `contact_force_log.png` — force vs time with ISO limit overlay

## Files
```
part1_injury_criteria/
├── README.md
├── main.py
└── assets/
    └── two_dof_arm.xml    # 2-DOF arm + compliant contact pad
```
