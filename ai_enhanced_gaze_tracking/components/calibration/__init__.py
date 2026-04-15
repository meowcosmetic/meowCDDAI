"""
Automatic calibration system components.

This module provides automatic calibration capabilities including:
- Face characteristic detection and parameter adaptation
- Reference point calibration using known gaze targets
- Child-specific parameter storage and retrieval
- Real-time environmental adaptation
"""

from .personal_calibration import PersonalCalibrationSystem

__all__ = ['PersonalCalibrationSystem']
