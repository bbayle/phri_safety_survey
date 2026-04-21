# Part III — Human Motion Prediction & Proactive Safety Bubble

## Survey section
Part III §3.1 Exteroceptive Sensing, §3.3 Human State Estimation

## What this simulates
A robot arm and a kinematically scripted human arm share a workspace.
The robot monitors the human wrist position via a rolling buffer and
predicts its location 300 ms into the future using linear extrapolation.
When the predicted wrist falls within a 20 cm safety bubble around the
robot end-effector, the robot's control torques are scaled down
proportionally to distance.

## Key concepts demonstrated
- Rolling-window velocity extrapolation (linear predictor baseline)
- Safety bubble: smooth velocity scaling vs. binary stop
- Separation of perception (predict) and control (scale) layers
- Post-hoc plot: true vs. predicted distance and velocity scale factor

## Limitations
Linear extrapolation is adequate for horizon < 300 ms at slow speeds.
Real systems require:
- Learned trajectory predictors (Trajectron++, GRIP, MID)
- Probabilistic prediction with occupancy distributions
- Multi-hypothesis tracking when intent is ambiguous
See survey §3.3 for the full prediction stack and benchmark results.

## Run

```bash
conda activate phri
python part3_perception/main.py
```

## Output
- Live viewer: human arm (skin-toned), robot arm (blue)
- `prediction_log.png` — distance traces and velocity scale vs time

## Files
```
part3_perception/
├── README.md
├── main.py
└── assets/
    └── human_robot_scene.xml    # robot + kinematic human arm
```
