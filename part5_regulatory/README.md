# Part V — Supervisory Safety Envelope & Dual-Channel E-Stop

## Survey section
Part V §5.1 ISO 10218-1 Regulatory Landscape, §5.2 Verification & Validation

## What this simulates
A 7-DOF collaborative arm controlled by a PD trajectory tracker whose
amplitude intentionally ramps up over time — eventually triggering the
supervisory safety monitor. The safety layer runs independently of the
functional controller, mirroring ISO 13849 SRP/CS Category 3 architecture.

## Key concepts demonstrated
- **Dual-channel E-stop**: Channel A (zero torques) + Channel B (active braking)
- **Watchdog timer**: functional controller must ping `heartbeat()` every 5 ms
- **Three independent checks**: joint velocity, joint torque, contact force
- **Separation of concerns**: `SafetyMonitor` class is agnostic to the controller
- Post-hoc plot: all three safety metrics vs time, with E-stop marker

## Regulatory context
| Standard | Relevance |
|---|---|
| ISO 10218-1 §5.4 | SRP/CS Categories B–4 |
| ISO 13849-1 | PLd architecture for Category 3 |
| ISO/TS 15066 | Contact force limits used as thresholds |

Category 3 requires: *a single fault shall not lead to loss of the safety
function*. Here this is achieved by the safety channel being an independent
software process that applies braking regardless of what the functional
channel does.

## Run

```bash
conda activate phri
python part5_regulatory/main.py
```

## Output
- Live viewer: arm accelerates until safety limit is hit, then braking
- `safety_monitor_log.png` — joint velocity, torque, and contact force
  vs time with ISO limit overlays and E-stop marker

## Files
```
part5_regulatory/
├── README.md
├── main.py
└── assets/
    └── cobot_arm.xml    # 7-DOF cobot with fixed obstacle
```
