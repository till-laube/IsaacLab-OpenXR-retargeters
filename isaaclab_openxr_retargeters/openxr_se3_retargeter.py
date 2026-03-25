# Copyright (c) 2025-2026 Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-arm OpenXR controller retargeter for SE3 commands.

Converts OpenXR motion controller data (received via SteamVR/ALVR or any
compatible OpenXR runtime) into SE3 delta commands:
[position(3), rotation_vector(3), gripper(1)].

Compatible with any VR controllers that provide data through the OpenXR
pipeline (e.g., HTC Vive, Meta Quest, Pico, etc.).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase, RetargeterBase, RetargeterCfg
from isaaclab.utils import configclass


class OpenXrControllerSe3Retargeter(RetargeterBase):
    """Retargeter that converts a single OpenXR motion controller to SE3 delta commands.

    Takes OpenXR controller data (pose + inputs) and outputs SE3 format:
    [position_delta(3), rotation_delta(3), gripper(1)] = 7 elements.

    The gripper is controlled by the trigger input.
    Position and rotation are computed as frame-to-frame deltas.
    """

    cfg: "OpenXrControllerSe3RetargeterCfg"

    def __init__(self, cfg: "OpenXrControllerSe3RetargeterCfg"):
        """Initialize the retargeter.

        Args:
            cfg: Configuration for the retargeter.
        """
        super().__init__(cfg)
        self._hand_side = cfg.hand_side
        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._trigger_threshold = cfg.trigger_threshold

        # Track previous pose for computing deltas
        self._prev_position = None
        self._prev_quaternion = None

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Return required data features for this retargeter."""
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, device_output: dict) -> torch.Tensor:
        """Retarget OpenXR controller data to SE3 delta command.

        Args:
            device_output: Dictionary with TrackingTarget keys.
                CONTROLLER_LEFT/RIGHT values are 2D arrays: [pose(7), inputs(7+)].

        Returns:
            Tensor of shape (7,): [pos_delta(3), rot_delta(3), gripper(1)].
        """
        # Select the appropriate controller based on hand_side
        if self._hand_side == "left":
            controller_key = DeviceBase.TrackingTarget.CONTROLLER_LEFT
        else:
            controller_key = DeviceBase.TrackingTarget.CONTROLLER_RIGHT

        # Get controller data
        controller_data = device_output.get(controller_key, np.array([]))

        # Default output if no controller data
        default_output = np.zeros(7)
        default_output[6] = -1.0  # Gripper open by default

        if len(controller_data) == 0:
            return torch.tensor(default_output, dtype=torch.float32)

        # Extract pose (row 0)
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return torch.tensor(default_output, dtype=torch.float32)

        pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        if len(pose) < 7:
            return torch.tensor(default_output, dtype=torch.float32)

        # Extract current position and quaternion
        current_position = pose[:3]
        current_quaternion = pose[3:7]  # [qw, qx, qy, qz]

        # Compute deltas
        if self._prev_position is None:
            # First frame: initialize, output zero delta
            self._prev_position = current_position.copy()
            self._prev_quaternion = current_quaternion.copy()
            position_delta = np.zeros(3)
            rotation_delta = np.zeros(3)
        else:
            # Position delta
            position_delta = (current_position - self._prev_position) * self._pos_sensitivity

            # Rotation delta via relative quaternion
            quat_prev_scipy = np.array([
                self._prev_quaternion[1], self._prev_quaternion[2],
                self._prev_quaternion[3], self._prev_quaternion[0],
            ])
            quat_curr_scipy = np.array([
                current_quaternion[1], current_quaternion[2],
                current_quaternion[3], current_quaternion[0],
            ])
            rot_prev = Rotation.from_quat(quat_prev_scipy)
            rot_curr = Rotation.from_quat(quat_curr_scipy)
            rotation_delta = (rot_curr * rot_prev.inv()).as_rotvec() * self._rot_sensitivity

            # Update previous pose
            self._prev_position = current_position.copy()
            self._prev_quaternion = current_quaternion.copy()

        # Extract gripper state from trigger input
        gripper = -1.0  # Default: open
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
            if len(inputs) > DeviceBase.MotionControllerInputIndex.TRIGGER.value:
                trigger_value = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
                gripper = 1.0 if trigger_value > self._trigger_threshold else -1.0

        output = np.concatenate([position_delta, rotation_delta, [gripper]])
        return torch.tensor(output, dtype=torch.float32)


@configclass
class OpenXrControllerSe3RetargeterCfg(RetargeterCfg):
    """Configuration for the single-arm OpenXR controller SE3 retargeter.

    Attributes:
        hand_side: Which controller to use: ``"left"`` or ``"right"``.
        pos_sensitivity: Position delta multiplier.
        rot_sensitivity: Rotation delta multiplier.
        trigger_threshold: Trigger value (0.0--1.0) above which the gripper closes.
    """

    retargeter_type: type = OpenXrControllerSe3Retargeter
    hand_side: str = "left"
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    trigger_threshold: float = 0.5
