"""
Backward compatibility layer for the AI-Enhanced Gaze Tracking System.

Provides adapters that translate between the enhanced system's data models
and the legacy GazeAnalysisResponse format expected by existing API consumers.
"""

from .legacy_adapter import LegacyResponseAdapter, build_legacy_response
from .config_bridge import ConfigBridge

__all__ = [
    'LegacyResponseAdapter',
    'build_legacy_response',
    'ConfigBridge',
]
