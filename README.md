# OpenXR Controller Retargeters for Isaac Lab

An [Isaac Lab](https://github.com/isaac-sim/IsaacLab) extension that provides retargeters for **OpenXR motion controllers**, enabling VR teleoperation of robot arms in simulation via **SteamVR** and **ALVR**.

Compatible with any VR headset and controllers that work through the OpenXR/SteamVR pipeline (e.g., HTC Vive, Meta Quest, Pico, etc.).

## What This Provides

| Class | Description |
|---|---|
| `OpenXrControllerSe3Retargeter` | Single-arm retargeter — converts one controller to 7-DOF SE3 delta commands `[pos(3), rot(3), gripper(1)]`. |
| `OpenXrControllerDualArmRetargeter` | Dual-arm retargeter — converts both controllers to 14-DOF commands with **delta** or **absolute** orientation tracking. |

Both retargeters plug into Isaac Lab's OpenXR device pipeline and work with the standard `teleop_se3_agent.py` script.

## Installation

```bash
# From the extension root directory:
pip install -e .
```

The extension depends on the `isaaclab` package (provided by your Isaac Lab installation).

## Hardware Prerequisites

- A VR headset with two motion controllers (e.g., HTC Vive XR Elite, Meta Quest, Pico)
- A PC running **SteamVR**
- **ALVR** (Air Light VR) for wireless streaming from the headset to the PC

## Setup

### 1. ALVR

[ALVR](https://github.com/alvr-org/ALVR) streams SteamVR content to standalone VR headsets wirelessly and forwards controller tracking data back to the PC.

1. Install the ALVR **PC dashboard** (v20+) on your host machine.
2. Install the ALVR **client APK** on the headset.
3. Connect both devices to the same local network.
4. In the ALVR dashboard, pair the headset and verify the connection.
5. Ensure **controller tracking** is enabled in the ALVR settings.

### 2. SteamVR

1. Install **SteamVR** via Steam.
2. Launch SteamVR — it should detect the headset and controllers via ALVR.
3. Verify that both controllers appear as **green** (tracked) in the SteamVR status window.

### 3. OpenXR Runtime

Isaac Sim uses OpenXR to communicate with SteamVR. Ensure SteamVR is set as the active runtime:

```bash
cat ~/.config/openxr/1/active_runtime.json
```

This should point to the SteamVR OpenXR runtime (typically `/home/<user>/.steam/steam/steamapps/common/SteamVR/steamxr_linux64.json`).

## Usage

### Quick Start — Standalone Example

Run the example script to see both retargeters process synthetic data (no Isaac Sim required):

```bash
python scripts/example_teleop.py
```

### Isaac Lab Integration

Add a retargeter to your environment config's `teleop_devices`:

```python
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab_openxr_retargeters import OpenXrControllerDualArmRetargeterCfg

# Inside your environment config class:
teleop_devices = {
    "openxr": OpenXRDeviceCfg(
        retargeters=[
            OpenXrControllerDualArmRetargeterCfg(
                tracking_mode="absolute",
                pos_sensitivity=5.0,
                rot_sensitivity=5.0,
                absolute_mode_rot_scale=0.2,  # 1.0 / RMPFlowActionCfg.scale
            ),
        ],
        sim_device=self.sim.device,
        xr_cfg=self.xr,
    ),
}
```

Then run teleoperation:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task <YourTask> --teleop_device openxr
```

### Overriding Tracking Mode via CLI

```bash
# Force absolute mode regardless of config:
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task <YourTask> --teleop_device openxr --tracking_mode absolute
```

## Configuration Reference

### `OpenXrControllerSe3RetargeterCfg`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hand_side` | `str` | `"left"` | Which controller: `"left"` or `"right"` |
| `pos_sensitivity` | `float` | `1.0` | Position delta multiplier |
| `rot_sensitivity` | `float` | `1.0` | Rotation delta multiplier |
| `trigger_threshold` | `float` | `0.5` | Trigger value (0–1) to close gripper |

### `OpenXrControllerDualArmRetargeterCfg`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tracking_mode` | `str` | `"delta"` | `"delta"` or `"absolute"` |
| `pos_sensitivity` | `float` | `1.0` | Position delta multiplier |
| `rot_sensitivity` | `float` | `1.0` | Rotation delta multiplier (delta mode) |
| `trigger_threshold` | `float` | `0.5` | Trigger value (0–1) to close gripper |
| `left_base_quat` | `tuple` | `(1,0,0,0)` | Left arm base rotation `[w,x,y,z]` |
| `right_base_quat` | `tuple` | `(1,0,0,0)` | Right arm base rotation `[w,x,y,z]` |
| `left_controller_to_gripper_rot` | `tuple` | `(1,0,0,0)` | Left controller-to-gripper offset `[w,x,y,z]` |
| `right_controller_to_gripper_rot` | `tuple` | `(1,0,0,0)` | Right controller-to-gripper offset `[w,x,y,z]` |
| `absolute_mode_rot_scale` | `float` | `1.0` | Scale for absolute rotation deltas. Set to `1.0 / RMPFlowActionCfg.scale` for 1:1 tracking. |

## Tracking Modes

### Delta Mode (`tracking_mode="delta"`)

Computes frame-to-frame rotation deltas. The gripper orientation changes based on how much the controller rotated since the previous frame.

- **Pros**: Responsive to small movements.
- **Cons**: Can accumulate drift over long sessions.

### Absolute Mode (`tracking_mode="absolute"`)

Tracks the controller's orientation relative to a calibration reference established on the first valid frame (or after reset). The gripper orientation directly follows the controller's absolute rotation.

- **Pros**: No drift, consistent 1:1 orientation mapping.
- **Cons**: Requires calibration and correct `controller_to_gripper_rot` offsets.

Calibration is automatically performed when:
- The retargeter is first initialized.
- `reset()` is called (e.g., on environment reset).
- Teleoperation is reactivated after being deactivated.

### Base Quaternions

If your robot arms are mounted at an angle (e.g., a dual-arm setup with arms angled at ±45°), set `left_base_quat` and `right_base_quat` to match the arm base orientations. This transforms controller deltas from world frame to each arm's local frame.

## Troubleshooting

| Problem | Solution |
|---|---|
| No controller data | Check SteamVR is running and controllers are tracked (green). Verify ALVR connection. |
| VR view not rendering | Ensure `--xr` flag is set. Restart SteamVR. |
| Gripper orientation wrong (absolute mode) | Adjust `left/right_controller_to_gripper_rot` quaternions. Start with identity and apply 90° rotations until aligned. |
| Orientation drifts in absolute mode | Check `absolute_mode_rot_scale` — it should be `1.0 / RMPFlowActionCfg.scale`. |
| Orientation drifts in delta mode | Expected behaviour. Switch to absolute mode or periodically reset. |
| OpenXR runtime error | Verify `~/.config/openxr/1/active_runtime.json` points to SteamVR. |

## License

BSD-3-Clause. See [LICENSE](LICENSE).
