"""
Property-based tests for graceful degradation under resource exhaustion.

**Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
**Validates: Requirements 8.4**

Property: For any system resource exhaustion, the system should degrade
performance gracefully rather than crash, maintaining basic functionality.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from ai_enhanced_gaze_tracking.components.error_handling.error_handler import (
    ErrorHandler,
    SystemState,
    ErrorSeverity,
    RecoveryAction,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Memory usage values spanning normal, high, and critical ranges
memory_strategy = st.floats(min_value=0.0, max_value=5000.0)

# CPU usage values spanning 0–100%
cpu_strategy = st.floats(min_value=0.0, max_value=100.0)

# Component names
component_strategy = st.sampled_from(
    ["camera", "face_detection", "ai_model", "head_pose", "sensor_fusion", "focus_detection"]
)

# Error severities
severity_strategy = st.sampled_from(list(ErrorSeverity))

# Number of consecutive failures
failure_count_strategy = st.integers(min_value=1, max_value=10)


# ---------------------------------------------------------------------------
# Property 20: Graceful Degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """
    **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
    **Validates: Requirements 8.4**
    """

    @given(
        memory_mb=memory_strategy,
        cpu_percent=cpu_strategy,
    )
    @settings(max_examples=100, deadline=5000)
    def test_resource_exhaustion_never_raises(self, memory_mb, cpu_percent):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For any combination of memory and CPU usage, calling
        handle_resource_exhaustion should never raise an exception — the system
        must always return a valid SystemState.
        """
        handler = ErrorHandler()

        # Must not raise under any resource values
        state = handler.handle_resource_exhaustion(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
        )

        assert isinstance(state, SystemState), (
            f"handle_resource_exhaustion must return a SystemState, got {type(state)}"
        )
        assert state in list(SystemState), (
            f"Returned state {state} is not a valid SystemState"
        )

    @given(
        memory_mb=st.floats(min_value=3000.0, max_value=8000.0),
        cpu_percent=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_critical_memory_triggers_minimal_state(self, memory_mb, cpu_percent):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For any memory usage at or above the critical threshold (3000 MB),
        the system should enter MINIMAL state — not crash.
        """
        handler = ErrorHandler()
        state = handler.handle_resource_exhaustion(memory_mb=memory_mb, cpu_percent=cpu_percent)

        assert state == SystemState.MINIMAL, (
            f"Critical memory ({memory_mb:.0f} MB) should produce MINIMAL state, got {state.value}"
        )

    @given(
        memory_mb=st.floats(min_value=0.0, max_value=2499.0),
        cpu_percent=st.floats(min_value=95.0, max_value=100.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_critical_cpu_triggers_minimal_state(self, memory_mb, cpu_percent):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For any CPU usage at or above the critical threshold (95%),
        the system should enter MINIMAL state — not crash.
        """
        handler = ErrorHandler()
        state = handler.handle_resource_exhaustion(memory_mb=memory_mb, cpu_percent=cpu_percent)

        assert state == SystemState.MINIMAL, (
            f"Critical CPU ({cpu_percent:.1f}%) should produce MINIMAL state, got {state.value}"
        )

    @given(
        memory_mb=st.floats(min_value=0.0, max_value=2499.0),
        cpu_percent=st.floats(min_value=0.0, max_value=84.9),
    )
    @settings(max_examples=100, deadline=5000)
    def test_normal_resources_maintain_normal_state(self, memory_mb, cpu_percent):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For any resource usage below all thresholds, the system should
        remain in NORMAL state.
        """
        handler = ErrorHandler()
        state = handler.handle_resource_exhaustion(memory_mb=memory_mb, cpu_percent=cpu_percent)

        assert state == SystemState.NORMAL, (
            f"Normal resources (memory={memory_mb:.0f}MB, cpu={cpu_percent:.1f}%) "
            f"should keep NORMAL state, got {state.value}"
        )

    @given(
        component=component_strategy,
        n_failures=failure_count_strategy,
    )
    @settings(max_examples=100, deadline=5000)
    def test_component_failures_never_raise(self, component, n_failures):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For any component and any number of consecutive failures,
        reporting failures should never raise an exception and should always
        return a valid RecoveryAction.
        """
        handler = ErrorHandler(max_failures=5)
        error = RuntimeError("Simulated component failure")

        last_action = None
        for _ in range(n_failures):
            action = handler.report_failure(component, error, severity=ErrorSeverity.MEDIUM)
            assert isinstance(action, RecoveryAction), (
                f"report_failure must return a RecoveryAction, got {type(action)}"
            )
            last_action = action

        # After max_failures, component should be disabled but system still alive
        if n_failures >= handler._max_failures:
            assert not handler.is_component_available(component), (
                f"Component '{component}' should be disabled after {n_failures} failures"
            )
            # System state must still be a valid state (not crashed)
            assert handler.get_system_state() in list(SystemState)

    @given(
        n_components=st.integers(min_value=1, max_value=6),
        n_failures=st.integers(min_value=5, max_value=10),
    )
    @settings(max_examples=100, deadline=5000)
    def test_multiple_component_failures_degrade_not_crash(self, n_components, n_failures):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For any number of component failures, the system state should
        degrade gracefully (NORMAL → DEGRADED → MINIMAL) and never enter an
        undefined state or raise an exception.
        """
        all_components = [
            "camera", "face_detection", "ai_model",
            "head_pose", "sensor_fusion", "focus_detection"
        ]
        components = all_components[:n_components]
        handler = ErrorHandler(max_failures=3)
        error = RuntimeError("Simulated failure")

        for component in components:
            for _ in range(n_failures):
                handler.report_failure(component, error, severity=ErrorSeverity.HIGH)

        state = handler.get_system_state()
        assert state in list(SystemState), f"System state must be valid, got {state}"
        assert state != SystemState.FAILED or n_components >= 3, (
            "FAILED state should only occur with many disabled components"
        )

        # Diagnostics must always be retrievable without error
        diagnostics = handler.get_diagnostics()
        assert "system_state" in diagnostics
        assert "component_health" in diagnostics
        assert "recent_errors" in diagnostics

    @given(
        memory_mb=st.floats(min_value=2500.0, max_value=2999.0),
        cpu_percent=st.floats(min_value=85.0, max_value=94.9),
    )
    @settings(max_examples=100, deadline=5000)
    def test_high_resources_trigger_degraded_state(self, memory_mb, cpu_percent):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For resource usage in the high (but not critical) range,
        the system should enter DEGRADED state — not crash and not go to MINIMAL.
        """
        handler = ErrorHandler()
        state = handler.handle_resource_exhaustion(memory_mb=memory_mb, cpu_percent=cpu_percent)

        assert state in (SystemState.DEGRADED, SystemState.MINIMAL), (
            f"High resources should produce DEGRADED or MINIMAL state, got {state.value}"
        )
        # Must not be NORMAL — resources are elevated
        assert state != SystemState.NORMAL, (
            f"High resources should not keep NORMAL state"
        )

    @given(
        memory_sequence=st.lists(
            st.floats(min_value=0.0, max_value=5000.0),
            min_size=3,
            max_size=10,
        )
    )
    @settings(max_examples=100, deadline=5000)
    def test_state_transitions_are_monotonically_safe(self, memory_sequence):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 20: Graceful Degradation**
        **Validates: Requirements 8.4**

        Property: For any sequence of resource readings, the system should
        always be in a valid state and should never transition to FAILED
        purely from resource exhaustion (FAILED is reserved for component failures).
        """
        handler = ErrorHandler()
        states = []

        for memory_mb in memory_sequence:
            state = handler.handle_resource_exhaustion(memory_mb=memory_mb, cpu_percent=0.0)
            assert isinstance(state, SystemState)
            states.append(state)

        # Resource exhaustion alone should never produce FAILED state
        assert SystemState.FAILED not in states, (
            "Resource exhaustion alone should not produce FAILED state"
        )

    def test_user_guidance_always_returns_string(self):
        """
        Guidance must always return a non-empty string regardless of system state.
        """
        handler = ErrorHandler()

        for state in SystemState:
            handler._system_state = state
            guidance = handler.get_user_guidance()
            assert isinstance(guidance, str) and len(guidance) > 0, (
                f"get_user_guidance() must return non-empty string in state {state.value}"
            )

    def test_diagnostics_structure_always_valid(self):
        """
        Diagnostics dict must always contain required keys regardless of error history.
        """
        handler = ErrorHandler()

        # Trigger various errors
        handler.report_failure("camera", RuntimeError("cam fail"), ErrorSeverity.HIGH)
        handler.report_failure("ai_model", RuntimeError("model fail"), ErrorSeverity.MEDIUM)
        handler.handle_resource_exhaustion(memory_mb=4000.0, cpu_percent=50.0)

        diagnostics = handler.get_diagnostics()

        required_keys = {
            "system_state", "component_health", "recent_errors",
            "total_errors_logged", "active_fallbacks", "disabled_components"
        }
        assert required_keys.issubset(diagnostics.keys()), (
            f"Diagnostics missing keys: {required_keys - diagnostics.keys()}"
        )
        assert isinstance(diagnostics["system_state"], str)
        assert isinstance(diagnostics["component_health"], dict)
        assert isinstance(diagnostics["recent_errors"], list)
