# Copyright (c) 2025-2026 Till Laube
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR motion controller retargeters for Isaac Lab teleoperation.

This extension provides retargeters that convert OpenXR motion controller data
(received from SteamVR via ALVR or any compatible OpenXR runtime) into SE3
robot commands compatible with Isaac Lab's teleoperation pipeline.

Compatible with any VR headset and controllers that work through the
OpenXR/SteamVR pipeline (e.g., HTC Vive, Meta Quest, Pico, etc.).

Classes:
    OpenXrControllerSe3Retargeter: Single-arm retargeter (delta mode, 7 DOF output).
    OpenXrControllerSe3RetargeterCfg: Configuration for the single-arm retargeter.
    OpenXrControllerDualArmRetargeter: Dual-arm retargeter (delta + absolute modes, 14 DOF output).
    OpenXrControllerDualArmRetargeterCfg: Configuration for the dual-arm retargeter.
"""

from .openxr_dual_arm_retargeter import OpenXrControllerDualArmRetargeter, OpenXrControllerDualArmRetargeterCfg
from .openxr_se3_retargeter import OpenXrControllerSe3Retargeter, OpenXrControllerSe3RetargeterCfg

__all__ = [
    "OpenXrControllerSe3Retargeter",
    "OpenXrControllerSe3RetargeterCfg",
    "OpenXrControllerDualArmRetargeter",
    "OpenXrControllerDualArmRetargeterCfg",
]
