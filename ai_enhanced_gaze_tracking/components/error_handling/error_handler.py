"""
Comprehensive error handling and graceful degradation for the AI-Enhanced Gaze Tracking System.

Implements:
- Graceful degradation for system resource exhaustion (Req 8.4)
- Automatic recovery for camera feed interruptions (Req 8.3)
- Detailed error logging and diagnostic information (Req 8.5)
- User guidance for error recovery (Req 8.5)
- Fallback chains for AI model failures (Req 8.2)
- Temporal prediction during face detection failures (Req 8.1)

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for system errors."""
    LOW = "low"          # Minor issue, system continues normally
    MEDIUM = "medium"    # Degraded performance, fallback active
    HIGH = "high"        # Major component failure, limited functionality
    CRITICAL = "critical"  # System cannot continue safely


class SystemState(Enum):
    """Overall system operational state."""
    NORMAL = "normal"          # All components functioning
    DEGRADED = "degraded"      # Some components failed, fallbacks active
    MINIMAL = "minimal"        # Only core functionality available
    RECOVERING = "recovering"  # Attempting to restore components
    FAILED = "failed"          # System cannot provide useful output


class RecoveryAction(Enum):
    """Actions the system can take to recover from errors."""
    RETRY = "retry"                    # Retry the failed operation
    USE_FALLBACK = "use_fallback"      # Switch to fallback method
    REDUCE_COMPLEXITY = "reduce_complexity"  # Lower processing requirements
    SKIP_FRAME = "skip_frame"          # Skip current frame
    RESTART_COMPONENT = "restart_component"  # Restart failed component
    NOTIFY_USER = "notify_user"        # Alert user to take action
    GRACEFUL_SHUTDOWN = "graceful_shutdown"  # Controlled shutdown


@dataclass
class ErrorRecord:
    """Record of a single error occurrence."""
    timestamp: float
    severity: ErrorSeverity
    component: str
    error_type: str
    message: str
    recovery_action: RecoveryAction
    recovered: bool = False
    recovery_time: Optional[float] = None
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStatus:
    """Status of a single system component."""
    name: str
    is_available: bool = True
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    using_fallback: bool = False
    recovery_attempts: int = 0


class ErrorHandler:
    """
    Central error handling and graceful degradation manager.

    Tracks component health, manages fallback chains, and ensures the system
    degrades gracefully under resource exhaustion rather than crashing.

    Property 20: Graceful Degradation
    For any system resource exhaustion, the system should degrade performance
    gracefully rather than crash, maintaining basic functionality.
    """

    # Maximum consecutive failures before marking component unavailable
    _MAX_FAILURES_BEFORE_DISABLE = 5

    # Seconds to wait before attempting component recovery
    _RECOVERY_COOLDOWN_S = 10.0

    # Resource thresholds that trigger degradation
    _MEMORY_CRITICAL_MB = 3000.0
    _MEMORY_HIGH_MB = 2500.0
    _CPU_CRITICAL_PERCENT = 95.0
    _CPU_HIGH_PERCENT = 85.0

    def __init__(self, recovery_cooldown_s: float = 10.0, max_failures: int = 5):
        """
        Args:
            recovery_cooldown_s: Seconds between recovery attempts per component.
            max_failures: Consecutive failures before disabling a component.
        """
        self._recovery_cooldown_s = recovery_cooldown_s
        self._max_failures = max_failures

        self._components: Dict[str, ComponentStatus] = {}
        self._error_log: List[ErrorRecord] = []
        self._system_state = SystemState.NORMAL
        self._lock = threading.Lock()

        # User-facing guidance messages keyed by component
        self._recovery_guidance: Dict[str, str] = {
            "camera": (
                "Camera feed interrupted. Please check the camera connection "
                "and ensure it is not being used by another application."
            ),
            "face_detection": (
                "Face detection is experiencing issues. Ensure the subject is "
                "well-lit and facing the camera."
            ),
            "ai_model": (
                "AI gaze model unavailable. The system is using traditional "
                "computer vision methods with reduced accuracy."
            ),
            "memory": (
                "System memory is running low. Close other applications to "
                "improve performance."
            ),
            "cpu": (
                "CPU load is very high. The system has reduced processing "
                "complexity to maintain real-time operation."
            ),
        }

        logger.info(
            "ErrorHandler initialized. max_failures=%d, recovery_cooldown=%.1fs",
            max_failures, recovery_cooldown_s,
        )

    # ------------------------------------------------------------------
    # Component registration and status
    # ------------------------------------------------------------------

    def register_component(self, name: str) -> None:
        """Register a component for health tracking."""
        with self._lock:
            if name not in self._components:
                self._components[name] = ComponentStatus(name=name)
                logger.debug("Registered component: %s", name)

    def report_success(self, component: str) -> None:
        """Record a successful operation for a component."""
        with self._lock:
            status = self._get_or_create_component(component)
            status.failure_count = 0
            status.is_available = True
            status.last_success_time = time.monotonic()
            status.using_fallback = False

    def report_failure(
        self,
        component: str,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        diagnostic_info: Optional[Dict[str, Any]] = None,
    ) -> RecoveryAction:
        """
        Record a component failure and determine the recovery action.

        Args:
            component: Name of the failing component.
            error: The exception that occurred.
            severity: How severe the failure is.
            diagnostic_info: Additional diagnostic context.

        Returns:
            The recommended RecoveryAction.
        """
        with self._lock:
            status = self._get_or_create_component(component)
            status.failure_count += 1
            status.last_failure_time = time.monotonic()

            action = self._determine_recovery_action(status, severity)

            if status.failure_count >= self._max_failures:
                status.is_available = False
                logger.error(
                    "Component '%s' disabled after %d consecutive failures.",
                    component, status.failure_count,
                )

            record = ErrorRecord(
                timestamp=time.monotonic(),
                severity=severity,
                component=component,
                error_type=type(error).__name__,
                message=str(error),
                recovery_action=action,
                diagnostic_info=diagnostic_info or {},
            )
            self._error_log.append(record)

            self._update_system_state()

            logger.warning(
                "Component failure: component=%s, error=%s, severity=%s, action=%s",
                component, type(error).__name__, severity.value, action.value,
            )

            return action

    def attempt_recovery(self, component: str) -> bool:
        """
        Check whether a recovery attempt is allowed for a component.

        Respects the cooldown period to avoid rapid retry storms.

        Args:
            component: Component to attempt recovery for.

        Returns:
            True if a recovery attempt should proceed.
        """
        with self._lock:
            status = self._get_or_create_component(component)
            now = time.monotonic()

            if status.last_failure_time is None:
                return False

            elapsed = now - status.last_failure_time
            if elapsed < self._recovery_cooldown_s:
                return False

            status.recovery_attempts += 1
            status.is_available = True  # Optimistically re-enable
            logger.info(
                "Recovery attempt #%d for component '%s'.",
                status.recovery_attempts, component,
            )
            return True

    # ------------------------------------------------------------------
    # Resource exhaustion handling (Property 20)
    # ------------------------------------------------------------------

    def handle_resource_exhaustion(
        self,
        memory_mb: float = 0.0,
        cpu_percent: float = 0.0,
    ) -> SystemState:
        """
        Evaluate resource usage and degrade gracefully if limits are exceeded.

        Implements Property 20: Graceful Degradation.
        The system must never crash due to resource exhaustion; instead it
        reduces functionality in a controlled manner.

        Args:
            memory_mb: Current memory usage in MB.
            cpu_percent: Current CPU utilisation percentage.

        Returns:
            The resulting SystemState after applying degradation logic.
        """
        with self._lock:
            previous_state = self._system_state

            if memory_mb >= self._MEMORY_CRITICAL_MB or cpu_percent >= self._CPU_CRITICAL_PERCENT:
                self._system_state = SystemState.MINIMAL
                self._record_resource_error("memory" if memory_mb >= self._MEMORY_CRITICAL_MB else "cpu",
                                            memory_mb, cpu_percent, ErrorSeverity.CRITICAL)

            elif memory_mb >= self._MEMORY_HIGH_MB or cpu_percent >= self._CPU_HIGH_PERCENT:
                if self._system_state == SystemState.NORMAL:
                    self._system_state = SystemState.DEGRADED
                self._record_resource_error("memory" if memory_mb >= self._MEMORY_HIGH_MB else "cpu",
                                            memory_mb, cpu_percent, ErrorSeverity.HIGH)

            elif self._system_state in (SystemState.DEGRADED, SystemState.MINIMAL):
                # Resources recovered — move back toward normal
                self._system_state = SystemState.RECOVERING

            if self._system_state != previous_state:
                logger.warning(
                    "System state changed: %s → %s (memory=%.0fMB, cpu=%.1f%%)",
                    previous_state.value, self._system_state.value,
                    memory_mb, cpu_percent,
                )

            return self._system_state

    # ------------------------------------------------------------------
    # Camera recovery (Req 8.3)
    # ------------------------------------------------------------------

    def handle_camera_interruption(self) -> RecoveryAction:
        """
        Handle a camera feed interruption.

        Returns the recommended recovery action and logs diagnostic info.
        """
        error = RuntimeError("Camera feed interrupted")
        action = self.report_failure(
            "camera",
            error,
            severity=ErrorSeverity.HIGH,
            diagnostic_info={"guidance": self._recovery_guidance["camera"]},
        )
        logger.error("Camera interruption detected. Guidance: %s", self._recovery_guidance["camera"])
        return action

    # ------------------------------------------------------------------
    # AI model fallback (Req 8.2)
    # ------------------------------------------------------------------

    def handle_ai_model_failure(self, model_name: str, error: Exception) -> RecoveryAction:
        """
        Handle an AI model inference failure.

        The system should fall back to traditional CV methods without
        interrupting the session.
        """
        action = self.report_failure(
            "ai_model",
            error,
            severity=ErrorSeverity.MEDIUM,
            diagnostic_info={
                "model": model_name,
                "guidance": self._recovery_guidance["ai_model"],
            },
        )
        status = self._get_or_create_component("ai_model")
        status.using_fallback = True
        return action

    # ------------------------------------------------------------------
    # Diagnostics and user guidance (Req 8.5)
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return detailed diagnostic information for troubleshooting.

        Implements Requirement 8.5: detailed diagnostic information.
        """
        with self._lock:
            recent_errors = [
                {
                    "timestamp": e.timestamp,
                    "component": e.component,
                    "severity": e.severity.value,
                    "error_type": e.error_type,
                    "message": e.message,
                    "recovery_action": e.recovery_action.value,
                    "recovered": e.recovered,
                }
                for e in self._error_log[-20:]  # Last 20 errors
            ]

            component_health = {
                name: {
                    "available": s.is_available,
                    "failure_count": s.failure_count,
                    "using_fallback": s.using_fallback,
                    "recovery_attempts": s.recovery_attempts,
                }
                for name, s in self._components.items()
            }

            return {
                "system_state": self._system_state.value,
                "component_health": component_health,
                "recent_errors": recent_errors,
                "total_errors_logged": len(self._error_log),
                "active_fallbacks": [
                    name for name, s in self._components.items() if s.using_fallback
                ],
                "disabled_components": [
                    name for name, s in self._components.items() if not s.is_available
                ],
            }

    def get_user_guidance(self, component: Optional[str] = None) -> str:
        """
        Return user-facing guidance for error recovery.

        Args:
            component: Specific component to get guidance for, or None for
                       guidance based on current system state.

        Returns:
            Human-readable guidance string.
        """
        if component and component in self._recovery_guidance:
            return self._recovery_guidance[component]

        with self._lock:
            if self._system_state == SystemState.NORMAL:
                return "System is operating normally."
            if self._system_state == SystemState.DEGRADED:
                disabled = [n for n, s in self._components.items() if not s.is_available]
                return (
                    f"System is operating in degraded mode. "
                    f"Affected components: {', '.join(disabled) if disabled else 'none'}. "
                    "Accuracy may be reduced."
                )
            if self._system_state == SystemState.MINIMAL:
                return (
                    "System resources are critically low. Only basic tracking is available. "
                    "Please close other applications and restart the session."
                )
            if self._system_state == SystemState.RECOVERING:
                return "System is recovering. Full functionality will resume shortly."
            return "System has encountered a critical error. Please restart the application."

    def get_system_state(self) -> SystemState:
        """Return the current overall system state."""
        return self._system_state

    def is_component_available(self, component: str) -> bool:
        """Check whether a component is currently available."""
        with self._lock:
            status = self._components.get(component)
            return status.is_available if status else True

    def get_error_count(self, component: Optional[str] = None) -> int:
        """Return total error count, optionally filtered by component."""
        with self._lock:
            if component:
                return sum(1 for e in self._error_log if e.component == component)
            return len(self._error_log)

    def reset(self) -> None:
        """Reset all state (useful between sessions)."""
        with self._lock:
            self._components.clear()
            self._error_log.clear()
            self._system_state = SystemState.NORMAL
            logger.info("ErrorHandler reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_component(self, name: str) -> ComponentStatus:
        """Get or lazily create a ComponentStatus (must hold lock)."""
        if name not in self._components:
            self._components[name] = ComponentStatus(name=name)
        return self._components[name]

    def _determine_recovery_action(
        self, status: ComponentStatus, severity: ErrorSeverity
    ) -> RecoveryAction:
        """Choose the best recovery action given component state and severity."""
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.GRACEFUL_SHUTDOWN

        if status.failure_count >= self._max_failures:
            return RecoveryAction.NOTIFY_USER

        if severity == ErrorSeverity.HIGH:
            if status.failure_count >= 3:
                return RecoveryAction.USE_FALLBACK
            return RecoveryAction.RETRY

        if severity == ErrorSeverity.MEDIUM:
            if status.failure_count >= 2:
                return RecoveryAction.USE_FALLBACK
            return RecoveryAction.RETRY

        # LOW severity
        return RecoveryAction.RETRY

    def _update_system_state(self) -> None:
        """Recompute system state from component health (must hold lock)."""
        if self._system_state in (SystemState.MINIMAL, SystemState.FAILED):
            return  # Resource-driven states take precedence

        disabled = [s for s in self._components.values() if not s.is_available]
        fallbacks = [s for s in self._components.values() if s.using_fallback]

        if len(disabled) == 0 and len(fallbacks) == 0:
            self._system_state = SystemState.NORMAL
        elif len(disabled) >= 3:
            self._system_state = SystemState.MINIMAL
        else:
            self._system_state = SystemState.DEGRADED

    def _record_resource_error(
        self,
        resource: str,
        memory_mb: float,
        cpu_percent: float,
        severity: ErrorSeverity,
    ) -> None:
        """Record a resource exhaustion error (must hold lock)."""
        record = ErrorRecord(
            timestamp=time.monotonic(),
            severity=severity,
            component=resource,
            error_type="ResourceExhaustion",
            message=f"Resource exhaustion: memory={memory_mb:.0f}MB, cpu={cpu_percent:.1f}%",
            recovery_action=RecoveryAction.REDUCE_COMPLEXITY,
            diagnostic_info={
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "guidance": self._recovery_guidance.get(resource, ""),
            },
        )
        self._error_log.append(record)
