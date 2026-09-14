# MuJoCo Go1 MJX Playground

This project runs a pre-trained policy on a Go1 robot model using MuJoCo MJX for physics, with a passive MuJoCo viewer and optional joystick/keyboard controls.

## Requirements

- Python 3.12
- MuJoCo (Can be installed separately)
- A GPU-capable JAX install if you want acceleration (CPU works too)
- Optional: a game controller

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

## Quick Start

Run the main loop:

```
python "./main.py"
```

On first run, it will download the MuJoCo Menagerie.

If you need a different entry point, check these files:
- `main.py`: main simulator loop and viewer integration
- `inference.py`: policy loading and inference
- `simulator.py`: environment stepping
- `viewer_wrapper.py`: viewer integration

## Config

The `config.json` file contains parameters for the program.

Custom heightfields should be a 640x640 grayscale PNG, where white (255) is the highest point and black (0) is the lowest.

## Controls

- Keyboard controls: Arrows for movement, period (`.`) and comma (`,`) for yaw (I had to use keys not used by any mujoco shortcut).
- Gamepad controls: Left stick for movement, right x-axis for yaw (some controller may need to change `yawAxis` from `2` to `3` in `config.json`).
- Gamepad is recommended, keyboard controls are only enabled when no controller is detected.

