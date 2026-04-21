# pHRI Safety Survey — MuJoCo Simulation Examples

Companion repository for the survey:
**"Physical Human–Robot Interaction Safety: A Research Survey"**

Each `partN_*/` folder is a self-contained experiment corresponding to one part of the survey.
They share a common conda environment and a small set of utilities in `shared/`.

---

## Repository Structure

```
phri_safety_survey/
├── environment.yml              # conda environment (Python 3.11, MuJoCo 3.x)
├── shared/
│   ├── robot_utils.py           # common helpers (geom force, Jacobian, etc.)
│   └── safety_monitor.py        # reusable ISO/TS 15066 safety checker
├── part1_injury_criteria/
│   ├── README.md
│   ├── assets/two_dof_arm.xml   # MuJoCo scene
│   └── main.py                  # contact force monitoring + ISO threshold check
├── part2_compliant_mechanisms/
│   ├── README.md
│   ├── assets/planar_arm.xml
│   └── main.py                  # variable-impedance admittance control
├── part3_perception/
│   ├── README.md
│   ├── assets/human_robot_scene.xml
│   └── main.py                  # human motion prediction + safety bubble
├── part4_safe_control/
│   ├── README.md
│   ├── assets/robot_arm.xml
│   └── main.py                  # domain-randomised safe policy evaluation
└── part5_regulatory/
    ├── README.md
    ├── assets/cobot_arm.xml
    └── main.py                  # supervisory safety envelope + dual-channel stop
```

---

## Installation (macOS — Apple Silicon M1/M2/M3/M4)

### Prerequisites

- **Homebrew**: https://brew.sh
- **Miniforge** (Apple Silicon native conda — do NOT use Anaconda on M-series):

```bash
brew install miniforge
conda init zsh        # or bash if you use bash
# restart your terminal
```

> ⚠️ If you already have Anaconda installed, make sure you're using the `conda` from
> Miniforge (`which conda` should point to `/opt/homebrew/Caskroom/miniforge3/...`).
> Anaconda's x86 builds will run under Rosetta and break MuJoCo's Metal renderer.

### 1 — Clone and enter the repo

```bash
git clone <your-remote-url> phri_safety_survey
cd phri_safety_survey
```

### 2 — Create the conda environment

```bash
conda env create -f environment.yml
conda activate phri
```

This installs:
- Python 3.11 (arm64 native)
- MuJoCo 3.x via `pip` (DeepMind's official wheel, supports macOS Metal)
- NumPy, SciPy, Matplotlib
- `mujoco-python-viewer` for the passive viewer

### 3 — Verify the install

```bash
python -c "import mujoco; print(mujoco.__version__)"
# Expected: 3.x.x
```

If you see a segfault or Metal errors, check:
```bash
python -c "import mujoco; mujoco.gl_context()"
# Should print: EGL / Metal context initialised
```

---

## Running the Examples

Each part is independent. From the repo root:

```bash
conda activate phri

# Part 1 — Injury criteria & ISO force thresholds
python part1_injury_criteria/main.py

# Part 2 — Variable-impedance admittance control
python part2_compliant_mechanisms/main.py

# Part 3 — Human motion prediction & safety bubble
python part3_perception/main.py

# Part 4 — Domain-randomised safe policy evaluation
python part4_safe_control/main.py

# Part 5 — Supervisory safety envelope & dual-channel stop
python part5_regulatory/main.py
```

Each script opens a MuJoCo passive viewer window (except Part 4, which is headless).
Press `Esc` or close the window to stop.

---

## Key Dependencies and Notes

| Package | Version | Notes |
|---|---|---|
| `mujoco` | ≥ 3.1 | Metal renderer on Apple Silicon |
| `numpy` | ≥ 1.26 | Required by MuJoCo Python bindings |
| `scipy` | ≥ 1.12 | Used in Part 3 (Kalman filter) |
| `matplotlib` | ≥ 3.8 | Used in Part 4 (Pareto plot) |

MuJoCo 3.x breaking changes vs 2.x:
- `mj_contactForce` signature unchanged, but contact array indexing changed
- `viewer` module is now `mujoco.viewer` (no separate `mujoco-viewer` package needed for passive viewer)
- XML `<option>` gravity default changed — all XMLs in this repo set it explicitly

---

## Citation

If you use these examples in your own work:

```bibtex
@misc{phri_safety_survey_2026,
  title  = {Physical Human–Robot Interaction Safety: A Research Survey},
  year   = {2026},
  note   = {Companion simulation repository}
}
```

---

## License

MIT — see `LICENSE`.
