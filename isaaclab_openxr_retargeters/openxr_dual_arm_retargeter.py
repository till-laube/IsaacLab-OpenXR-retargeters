# Copyright (c) 2025-2026 Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-arm OpenXR controller retargeter for SE3 commands.

Converts both OpenXR motion controllers (received via SteamVR/ALVR or any
compatible OpenXR runtime) into dual-arm SE3 commands with support for delta
and absolute tracking modes.

Compatible with any VR controllers that provide data through the OpenXR
pipeline (e.g., HTC Vive, Meta Quest, Pico, etc.).
"""

import logging

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices.device_base import DeviceBase, RetargeterBase, RetargeterCfg
from isaaclab.utils import configclass

logger = logging.getLogger(__name__)


class OpenXrControllerDualArmRetargeter(RetargeterBase):
    """Retargeter that converts both OpenXR motion controllers to dual-arm SE3 commands.

    Outputs 14 DOF: [left_pos(3), left_rot(3), left_grip(1),
                      right_pos(3), right_rot(3), right_grip(1)].

    Supports two tracking modes:

    **Delta mode** (``tracking_mode="delta"``):
        Computes frame-to-frame rotation deltas. The gripper orientation changes
        based on how much the controller rotated since the previous frame. This
        mode can accumulate drift over time but is responsive to small movements.

    **Absolute mode** (``tracking_mode="absolute"``):
        Tracks the controller's orientation relative to a calibration reference
        established on the first valid frame (or after reset). The gripper
        orientation directly follows the controller's rotation from this
        calibration point. This eliminates drift and provides consistent
        orientation mapping throughout the teleoperation session.

        Calibration is automatically performed when:

        - The retargeter is first initialized.
        - The ``reset()`` method is called (e.g., on environment reset).
        - Teleoperation is reactivated after being deactivated.

    **Controller-to-gripper alignment** (absolute mode only):
        The controller's coordinate frame may not match the gripper's coordinate
        frame. The config parameters ``left_controller_to_gripper_rot`` and
        ``right_controller_to_gripper_rot`` define rotation offsets (as quaternions
        ``[w, x, y, z]``) to align these frames.

        To find the correct offset:

        1. Start with identity ``(1, 0, 0, 0)``.
        2. Hold the controller in a natural position.
        3. Observe the gripper orientation mismatch.
        4. Apply rotations until aligned. Common offsets:

           - 90 deg around X: ``(0.707, 0.707, 0, 0)``
           - 90 deg around Y: ``(0.707, 0, 0.707, 0)``
           - 90 deg around Z: ``(0.707, 0, 0, 0.707)``
           - 180 deg around Z: ``(0, 0, 0, 1)``

    Position tracking uses delta mode in both cases.
    """

    cfg: "OpenXrControllerDualArmRetargeterCfg"

    TRACKING_MODES = ("delta", "absolute")

    def __init__(self, cfg: "OpenXrControllerDualArmRetargeterCfg"):
        super().__init__(cfg)

        if cfg.tracking_mode not in self.TRACKING_MODES:
            raise ValueError(
                f"Invalid tracking_mode '{cfg.tracking_mode}'. "
                f"Must be one of: {self.TRACKING_MODES}"
            )
        self._tracking_mode = cfg.tracking_mode
        logger.info("Initialized with tracking_mode='%s'", self._tracking_mode)

        self._pos_sensitivity = cfg.pos_sensitivity
        self._rot_sensitivity = cfg.rot_sensitivity
        self._trigger_threshold = cfg.trigger_threshold
        self._absolute_mode_rot_scale = cfg.absolute_mode_rot_scale

        # Base rotations for coordinate transformation (world -> arm base frame)
        self._left_base_rot = self._quat_wxyz_to_rotation(cfg.left_base_quat)
        self._right_base_rot = self._quat_wxyz_to_rotation(cfg.right_base_quat)
        self._left_world_to_base = self._left_base_rot.inv()
        self._right_world_to_base = self._right_base_rot.inv()

        # Controller-to-gripper rotation offsets (absolute mode)
        self._left_controller_to_gripper = self._quat_wxyz_to_rotation(cfg.left_controller_to_gripper_rot)
        self._right_controller_to_gripper = self._quat_wxyz_to_rotation(cfg.right_controller_to_gripper_rot)

        # Delta tracking state
        self._prev_left_position = None
        self._prev_left_quaternion = None
        self._prev_right_position = None
        self._prev_right_quaternion = None

        # Absolute tracking calibration state
        self._left_calibration_quat = None
        self._right_calibration_quat = None
        self._left_prev_target_rot = None
        self._right_prev_target_rot = None

    @staticmethod
    def _quat_wxyz_to_rotation(quat_wxyz: tuple[float, float, float, float]) -> Rotation:
        """Convert a [w, x, y, z] quaternion to a scipy Rotation."""
        return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

    @property
    def tracking_mode(self) -> str:
        """Current tracking mode (``"delta"`` or ``"absolute"``)."""
        return self._tracking_mode

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Return required data features for this retargeter."""
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def reset(self):
        """Reset the retargeter state.

        Clears previous-pose tracking and (in absolute mode) the calibration
        reference, so the next frame will establish a new calibration point.
        """
        self._prev_left_position = None
        self._prev_left_quaternion = None
        self._prev_right_position = None
        self._prev_right_quaternion = None

        self._left_calibration_quat = None
        self._right_calibration_quat = None
        self._left_prev_target_rot = None
        self._right_prev_target_rot = None

        if self._tracking_mode == "absolute":
            logger.debug("Reset — calibration will be re-established on next valid frame")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_controller_pose_and_gripper(
        self, controller_data: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Extract position, quaternion, and gripper state from raw controller data.

        Returns:
            ``(position [x,y,z], quaternion [w,x,y,z], gripper_value)``.
            Position and quaternion are ``None`` if data is invalid.
        """
        if len(controller_data) == 0 or len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return None, None, -1.0

        pose = controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        if len(pose) < 7:
            return None, None, -1.0

        position = pose[:3]
        quaternion = pose[3:7]  # [qw, qx, qy, qz]

        gripper = -1.0
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
            if len(inputs) > DeviceBase.MotionControllerInputIndex.TRIGGER.value:
                trigger_value = inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
                gripper = 1.0 if trigger_value > self._trigger_threshold else -1.0

        return position, quaternion, gripper

    def _process_controller_delta(
        self,
        controller_data: np.ndarray,
        prev_position: np.ndarray | None,
        prev_quaternion: np.ndarray | None,
        world_to_base: Rotation,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Process a single controller in delta tracking mode.

        Returns:
            ``(output_7dof, new_position, new_quaternion)``.
        """
        default_output = np.zeros(7)
        default_output[6] = -1.0

        current_position, current_quaternion, gripper = self._extract_controller_pose_and_gripper(controller_data)
        if current_position is None:
            return default_output, prev_position, prev_quaternion

        if prev_position is None:
            position_delta = np.zeros(3)
            rotation_delta = np.zeros(3)
        else:
            position_delta = (current_position - prev_position) * self._pos_sensitivity

            quat_prev_scipy = np.array([
                prev_quaternion[1], prev_quaternion[2], prev_quaternion[3], prev_quaternion[0],
            ])
            quat_curr_scipy = np.array([
                current_quaternion[1], current_quaternion[2], current_quaternion[3], current_quaternion[0],
            ])
            rot_delta = Rotation.from_quat(quat_curr_scipy) * Rotation.from_quat(quat_prev_scipy).inv()
            rotation_delta = rot_delta.as_rotvec() * self._rot_sensitivity

            # Transform deltas from world frame to arm base frame
            position_delta = world_to_base.apply(position_delta)
            rotation_delta = world_to_base.apply(rotation_delta)

        output = np.concatenate([position_delta, rotation_delta, [gripper]])
        return output, current_position.copy(), current_quaternion.copy()

    def _process_controller_absolute(
        self,
        controller_data: np.ndarray,
        prev_position: np.ndarray | None,
        prev_quaternion: np.ndarray | None,
        is_left: bool,
        world_to_base: Rotation,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Process a single controller in absolute orientation tracking mode.

        The approach:

        1. Compute controller rotation from calibration (world frame).
        2. Apply controller-to-gripper offset to align coordinate frames.
        3. Convert to rotation vector and transform to base frame.
        4. Compute incremental delta from previous output.
        5. Output the delta (accumulated by RMPFlow).

        Returns:
            ``(output_7dof, new_position, new_quaternion)``.
        """
        default_output = np.zeros(7)
        default_output[6] = -1.0

        current_position, current_quaternion, gripper = self._extract_controller_pose_and_gripper(controller_data)
        if current_position is None:
            return default_output, prev_position, prev_quaternion

        # Select per-arm state
        if is_left:
            calibration_quat = self._left_calibration_quat
            prev_target_rotvec = self._left_prev_target_rot
            controller_to_gripper = self._left_controller_to_gripper
        else:
            calibration_quat = self._right_calibration_quat
            prev_target_rotvec = self._right_prev_target_rot
            controller_to_gripper = self._right_controller_to_gripper

        # Position: delta tracking (same as delta mode)
        if prev_position is None:
            position_delta = np.zeros(3)
        else:
            position_delta = (current_position - prev_position) * self._pos_sensitivity
            position_delta = world_to_base.apply(position_delta)

        # Orientation: absolute tracking
        if calibration_quat is None:
            # First frame — establish calibration reference
            calibration_quat = current_quaternion.copy()
            prev_target_rotvec = np.zeros(3)

            if is_left:
                self._left_calibration_quat = calibration_quat
                self._left_prev_target_rot = prev_target_rotvec
            else:
                self._right_calibration_quat = calibration_quat
                self._right_prev_target_rot = prev_target_rotvec

            side = "LEFT" if is_left else "RIGHT"
            logger.info("%s arm calibrated — controller orientation captured", side)
            rotation_delta = np.zeros(3)
        else:
            # Compute rotation from calibration to current
            calib_scipy = np.array([
                calibration_quat[1], calibration_quat[2], calibration_quat[3], calibration_quat[0],
            ])
            curr_scipy = np.array([
                current_quaternion[1], current_quaternion[2], current_quaternion[3], current_quaternion[0],
            ])
            controller_rot_from_calib = Rotation.from_quat(curr_scipy) * Rotation.from_quat(calib_scipy).inv()

            # Apply controller-to-gripper offset
            aligned_rot = controller_rot_from_calib * controller_to_gripper

            # Transform to base frame
            target_rotvec_base = world_to_base.apply(aligned_rot.as_rotvec())

            # Incremental delta from previous output, with scale compensation
            rotation_delta = (target_rotvec_base - prev_target_rotvec) * self._absolute_mode_rot_scale

            # Store unscaled target for next frame
            if is_left:
                self._left_prev_target_rot = target_rotvec_base
            else:
                self._right_prev_target_rot = target_rotvec_base

        output = np.concatenate([position_delta, rotation_delta, [gripper]])
        return output, current_position.copy(), current_quaternion.copy()

    def _process_controller(
        self,
        controller_data: np.ndarray,
        prev_position: np.ndarray | None,
        prev_quaternion: np.ndarray | None,
        is_left: bool,
        world_to_base: Rotation,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Dispatch to the appropriate processing method based on tracking mode."""
        if self._tracking_mode == "delta":
            return self._process_controller_delta(
                controller_data, prev_position, prev_quaternion, world_to_base,
            )
        else:
            return self._process_controller_absolute(
                controller_data, prev_position, prev_quaternion, is_left, world_to_base,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retarget(self, device_output: dict) -> torch.Tensor:
        """Retarget both controllers to dual-arm SE3 delta commands.

        Args:
            device_output: Dictionary with ``CONTROLLER_LEFT`` and
                ``CONTROLLER_RIGHT`` keys from the OpenXR device.

        Returns:
            Tensor of shape (14,): ``[left(7), right(7)]`` where each block
            is ``[pos_delta(3), rot_delta(3), gripper(1)]``.
        """
        left_data = device_output.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_data = device_output.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))

        left_output, new_left_pos, new_left_quat = self._process_controller(
            left_data, self._prev_left_position, self._prev_left_quaternion,
            is_left=True, world_to_base=self._left_world_to_base,
        )
        self._prev_left_position = new_left_pos
        self._prev_left_quaternion = new_left_quat

        right_output, new_right_pos, new_right_quat = self._process_controller(
            right_data, self._prev_right_position, self._prev_right_quaternion,
            is_left=False, world_to_base=self._right_world_to_base,
        )
        self._prev_right_position = new_right_pos
        self._prev_right_quaternion = new_right_quat

        output = np.concatenate([left_output, right_output])
        return torch.tensor(output, dtype=torch.float32)


@configclass
class OpenXrControllerDualArmRetargeterCfg(RetargeterCfg):
    """Configuration for the dual-arm OpenXR controller retargeter.

    Attributes:
        tracking_mode: ``"delta"`` for relative rotations or ``"absolute"`` for
            orientation tracking relative to a calibration reference.
        pos_sensitivity: Position delta multiplier.
        rot_sensitivity: Rotation delta multiplier (delta mode only).
        trigger_threshold: Trigger value (0.0--1.0) above which the gripper closes.
        left_base_quat: Quaternion ``[w, x, y, z]`` of the left arm base in world frame.
        right_base_quat: Quaternion ``[w, x, y, z]`` of the right arm base in world frame.
        left_controller_to_gripper_rot: Quaternion ``[w, x, y, z]`` rotation offset from the
            left controller frame to the desired gripper frame (absolute mode).
        right_controller_to_gripper_rot: Same for the right controller.
        absolute_mode_rot_scale: Scale factor applied to absolute-mode rotation deltas.
            Set to ``1.0 / RMPFlowActionCfg.scale`` for correct 1:1 orientation tracking.
    """

    retargeter_type: type = OpenXrControllerDualArmRetargeter
    tracking_mode: str = "delta"
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    trigger_threshold: float = 0.5
    left_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    right_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    left_controller_to_gripper_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    right_controller_to_gripper_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    absolute_mode_rot_scale: float = 1.0
