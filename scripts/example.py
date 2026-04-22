#!/usr/bin/env python3
# Copyright (c) 2025-2026 Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example: Teleoperation with an example task for a dual-arm setup based on UR5e arms (Isaac-Ur5e-Dual-v0)

This script demonstrates how to wire the ``OpenXrControllerDualArmRetargeter``
into an Isaac Lab environment and run VR teleoperation.  It mirrors the workflow of the
original script used for a similar setup

The retargeter is now hardware-agnostic, therefore changes have been made, which I cannot 
test on the actual hardware unfortunately (as I do not have access anymore)

Usage
-----

Run inside Isaac Lab (requires Isaac Sim + SteamVR/ALVR + a VR headset)::

    # Delta tracking mode (default):
    ./isaaclab.sh -p scripts/example.py --task Isaac-Ur5e-Dual-v0

    # Absolute tracking mode:
    ./isaaclab.sh -p scripts/example.py --task Isaac-Ur5e-Dual-v0 --tracking_mode absolute

    # With a different task:
    ./isaaclab.sh -p scripts/example.py --task Isaac-Your-Task
"""

from __future__ import annotations

import argparse
import logging

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Teleoperation example using the OpenXR retargeter",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--task", type=str, default="Isaac-Ur5e-Dual-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="openxr",
    help="Key under env_cfg.teleop_devices.devices (default: 'openxr').",
)
parser.add_argument(
    "--tracking_mode",
    type=str,
    default="delta",
    choices=["delta", "absolute"],
    help="Retargeter tracking mode: 'delta' (frame-to-frame) or 'absolute' (calibrated 1:1).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable XR — required for the OpenXR/SteamVR/ALVR pipeline.
app_launcher_args = vars(args_cli)
app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports that depend on Isaac Sim
# ---------------------------------------------------------------------------

import gymnasium as gym
import torch

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401  — registers gym tasks

from isaaclab_openxr_retargeters import OpenXrControllerDualArmRetargeterCfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retargeter configuration
# ---------------------------------------------------------------------------

def build_retargeter_cfg(tracking_mode: str) -> OpenXrControllerDualArmRetargeterCfg:
    """Build the dual-arm retargeter config for the exemplary UR5e dual setup

    Tune these parameters for your own robot/task
    """
    return OpenXrControllerDualArmRetargeterCfg(
        tracking_mode=tracking_mode,
        pos_sensitivity=5.0,
        rot_sensitivity=5.0,
        trigger_threshold=0.5,
        # Arm base rotations for the UR5e dual setup.
        # Left:  Euler (180, -45, 90) deg in XYZ convention.
        # Right: Euler (180,  45, 90) deg in XYZ convention.
        left_base_quat=(-0.270523, 0.653339, 0.653251, 0.270609),
        right_base_quat=(0.270583, 0.653314, 0.653276, -0.270549),
        # Controller-to-gripper rotation offsets (identity = no offset).
        # Adjust these if the gripper orientation does not match how you hold
        # the controller.
        left_controller_to_gripper_rot=(1.0, 0.0, 0.0, 0.0),
        right_controller_to_gripper_rot=(1.0, 0.0, 0.0, 0.0),
        # Scale compensation for absolute mode: 1.0 / RMPFlowActionCfg.scale.
        # RMPFlowActionCfg.scale = 5.0, so this should be 0.2.
        absolute_mode_rot_scale=0.2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Build environment config -----------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None
    env_cfg = remove_camera_configs(env_cfg)
    env_cfg.sim.render.antialiasing_mode = "DLSS"

    # ---- Inject the cleaned retargeter ------------------------------------
    # This replaces whatever teleop_devices the task config originally
    # defined (e.g. ViveControllerDualArmRetargeterCfg) with the cleaned,
    # hardware-agnostic OpenXrControllerDualArmRetargeterCfg.
    retargeter_cfg = build_retargeter_cfg(args_cli.tracking_mode)
    env_cfg.teleop_devices = DevicesCfg(
        devices={
            args_cli.teleop_device: OpenXRDeviceCfg(
                retargeters=[retargeter_cfg],
                sim_device=env_cfg.sim.device,
                xr_cfg=env_cfg.xr,
            ),
        }
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # ---- Teleop callbacks -------------------------------------------------
    should_reset = False
    teleop_active = False  # XR starts inactive; press SQUEEZE to toggle

    def on_reset():
        nonlocal should_reset
        should_reset = True
        print("[teleop] Reset requested — environment will reset on next step.")

    def on_start():
        nonlocal teleop_active
        teleop_active = True
        print("[teleop] Teleoperation STARTED.")

    def on_stop():
        nonlocal teleop_active
        teleop_active = False
        print("[teleop] Teleoperation STOPPED.")

    callbacks = {
        "R": on_reset,
        "RESET": on_reset,
        "START": on_start,
        "STOP": on_stop,
    }

    teleop_interface = create_teleop_device(
        args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks
    )
    print(f"[teleop] Device: {teleop_interface}")
    print(f"[teleop] Tracking mode: {args_cli.tracking_mode}")
    print("[teleop] Press SQUEEZE on either controller to toggle teleoperation.")

    env.reset()
    teleop_interface.reset()

    # ---- Simulation loop --------------------------------------------------
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                action = teleop_interface.advance()

                if teleop_active:
                    actions = action.repeat(env.num_envs, 1)
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    print("[teleop] Environment reset complete.")
    except KeyboardInterrupt:
        print("\n[teleop] Interrupted — shutting down.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
