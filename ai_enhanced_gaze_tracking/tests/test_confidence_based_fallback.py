"""
Property-based tests for confidence-based fallback requirements.

**Feature: ai-enhanced-gaze-tracking, Property 6: Confidence-Based Fallback**
**Validates: Requirements 3.4**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import cv2

from ..components.gaze_estimation.ai_gaze_estimator import AIGazeEstimator, MockAIGazeModel, AIGazeModel
from ..core.data_models import FaceDetection, GazeEstimate


class LowConfidenceAIModel(AIGazeModel):
    """AI model that always returns low confidence predictions."""
    
    def __init__(self, confidence: float = 0.2):
        self.model_loaded = False
        self.confidence = confidence
        self.model_info = {
            "name": "LowConfidenceModel",
            "version": "1.0.0",
            "accuracy": 0.5,
            "input_size": (64, 64),
            "description": "Model that returns low confidence for testing fallback"
        }
    
    def load_model(self, model_path: str) -> None:
        """Load the model."""
        self.model_loaded = True
    
    def predict_gaze(self, eye_image: np.ndarray, landmarks: np.ndarray) -> tuple:
        """Predict gaze with low confidence."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Return a random gaze direction with low confidence
        gaze_vector = np.array([0.1, 0.1, 1.0])
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        
        return gaze_vector, self.confidence
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return self.model_info.copy()


def create_face_detection(bbox_center: tuple, landmarks: np.ndarray) -> FaceDetection:
    """Create a face detection for testing."""
    x, y = bbox_center
    w, h = 100, 120
    bbox = (x - w//2, y - h//2, w, h)
    
    return FaceDetection(
        bbox=bbox,
        landmarks=landmarks,
        confidence=0.9,
        quality_score=0.9
    )


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame."""
    frame = np.random.randint(80, 180, (height, width, 3), dtype=np.uint8)
    return frame


@given(
    bbox_center=st.tuples(
        st.integers(min_value=100, max_value=540),
        st.integers(min_value=100, max_value=380)
    ),
    landmarks=arrays(
        dtype=np.float32,
        shape=(6, 2),
        elements=st.floats(min_value=50, max_value=590, allow_nan=False, allow_infinity=False)
    ),
    low_confidence=st.floats(min_value=0.0, max_value=0.49)  # Below default threshold of 0.5
)
@settings(max_examples=100, deadline=5000)
def test_confidence_based_fallback_triggers(bbox_center, landmarks, low_confidence):
    """
    **Feature: ai-enhanced-gaze-tracking, Property 6: Confidence-Based Fallback**
    
    For any AI prediction with confidence below threshold, the system should 
    automatically fallback to computer vision methods and maintain tracking continuity.
    
    **Validates: Requirements 3.4**
    """
    # Create test data
    face_detection = create_face_detection(bbox_center, landmarks)
    frame = create_test_frame()
    
    # Initialize AI gaze estimator with low confidence model
    low_conf_model = LowConfidenceAIModel(confidence=low_confidence)
    estimator = AIGazeEstimator(
        models=[low_conf_model],
        confidence_threshold=0.5  # Default threshold
    )
    
    # Get gaze estimate
    gaze_estimate = estimator.estimate_gaze(face_detection, frame)
    
    # Verify fallback was triggered
    assert isinstance(gaze_estimate, GazeEstimate), "Should return valid GazeEstimate"
    
    # When confidence is below threshold, should use fallback method
    assert "fallback" in gaze_estimate.method.lower(), \
        f"Expected fallback method for confidence {low_confidence}, got {gaze_estimate.method}"
    
    # Tracking continuity: should still produce valid gaze vector
    assert gaze_estimate.gaze_vector_3d is not None, "Should maintain tracking with fallback"
    assert np.linalg.norm(gaze_estimate.gaze_vector_3d) > 0, "Gaze vector should be non-zero"
    
    # Should have valid 2D gaze point
    assert gaze_estimate.gaze_point_2d is not None, "Should have 2D gaze point"
    assert len(gaze_estimate.gaze_point_2d) == 2, "2D point should have x,y coordinates"


@given(
    bbox_center=st.tuples(
        st.integers(min_value=100, max_value=540),
        st.integers(min_value=100, max_value=380)
    ),
    landmarks=arrays(
        dtype=np.float32,
        shape=(6, 2),
        elements=st.floats(min_value=50, max_value=590, allow_nan=False, allow_infinity=False)
    ),
    high_confidence=st.floats(min_value=0.5, max_value=1.0)  # Above threshold
)
@settings(max_examples=100, deadline=5000)
def test_no_fallback_for_high_confidence(bbox_center, landmarks, high_confidence):
    """
    Test that fallback is NOT triggered when confidence is above threshold.
    
    **Feature: ai-enhanced-gaze-tracking, Property 6: Confidence-Based Fallback**
    **Validates: Requirements 3.4**
    """
    # Create test data
    face_detection = create_face_detection(bbox_center, landmarks)
    frame = create_test_frame()
    
    # Create a model that returns high confidence
    class HighConfidenceModel(AIGazeModel):
        def __init__(self, conf):
            self.model_loaded = False
            self.conf = conf
            self.model_info = {"name": "HighConf", "version": "1.0"}
        
        def load_model(self, path):
            self.model_loaded = True
        
        def predict_gaze(self, eye_image, landmarks):
            return np.array([0, 0, 1]), self.conf
        
        def get_model_info(self):
            return self.model_info
    
    high_conf_model = HighConfidenceModel(high_confidence)
    estimator = AIGazeEstimator(
        models=[high_conf_model],
        confidence_threshold=0.5
    )
    
    # Get gaze estimate
    gaze_estimate = estimator.estimate_gaze(face_detection, frame)
    
    # Should NOT use fallback for high confidence
    assert "fallback" not in gaze_estimate.method.lower(), \
        f"Should not use fallback for confidence {high_confidence}, got {gaze_estimate.method}"
    
    # Should use AI method
    assert "ai" in gaze_estimate.method.lower(), \
        f"Should use AI method for high confidence, got {gaze_estimate.method}"


@given(
    bbox_center=st.tuples(
        st.integers(min_value=100, max_value=540),
        st.integers(min_value=100, max_value=380)
    ),
    landmarks=arrays(
        dtype=np.float32,
        shape=(6, 2),
        elements=st.floats(min_value=50, max_value=590, allow_nan=False, allow_infinity=False)
    ),
    threshold=st.floats(min_value=0.1, max_value=0.9),
    confidence=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100, deadline=5000)
def test_fallback_respects_threshold(bbox_center, landmarks, threshold, confidence):
    """
    Test that fallback behavior correctly respects the configured threshold.
    
    **Feature: ai-enhanced-gaze-tracking, Property 6: Confidence-Based Fallback**
    **Validates: Requirements 3.4**
    """
    # Create test data
    face_detection = create_face_detection(bbox_center, landmarks)
    frame = create_test_frame()
    
    # Create model with specific confidence
    class ConfigurableConfidenceModel(AIGazeModel):
        def __init__(self, conf):
            self.model_loaded = False
            self.conf = conf
            self.model_info = {"name": "ConfigConf", "version": "1.0"}
        
        def load_model(self, path):
            self.model_loaded = True
        
        def predict_gaze(self, eye_image, landmarks):
            return np.array([0, 0, 1]), self.conf
        
        def get_model_info(self):
            return self.model_info
    
    model = ConfigurableConfidenceModel(confidence)
    estimator = AIGazeEstimator(
        models=[model],
        confidence_threshold=threshold
    )
    
    # Get gaze estimate
    gaze_estimate = estimator.estimate_gaze(face_detection, frame)
    
    # Verify fallback behavior matches threshold
    if confidence < threshold:
        # Should use fallback
        assert "fallback" in gaze_estimate.method.lower(), \
            f"Expected fallback for confidence {confidence} < threshold {threshold}, got {gaze_estimate.method}"
    else:
        # Should use AI
        assert "ai" in gaze_estimate.method.lower(), \
            f"Expected AI for confidence {confidence} >= threshold {threshold}, got {gaze_estimate.method}"


def test_fallback_maintains_tracking_continuity():
    """
    Unit test to verify tracking continuity during fallback.
    
    **Feature: ai-enhanced-gaze-tracking, Property 6: Confidence-Based Fallback**
    **Validates: Requirements 3.4**
    """
    # Create test data
    landmarks = np.array([
        [200, 200], [220, 200], [240, 200],
        [280, 200], [300, 200], [320, 200]
    ], dtype=np.float32)
    
    face_detection = create_face_detection((260, 240), landmarks)
    frame = create_test_frame()
    
    # Use low confidence model
    low_conf_model = LowConfidenceAIModel(confidence=0.1)
    estimator = AIGazeEstimator(models=[low_conf_model], confidence_threshold=0.5)
    
    # Get multiple estimates to test continuity
    estimates = []
    for _ in range(5):
        estimate = estimator.estimate_gaze(face_detection, frame)
        estimates.append(estimate)
    
    # All estimates should be valid
    for estimate in estimates:
        assert isinstance(estimate, GazeEstimate)
        assert estimate.gaze_vector_3d is not None
        assert np.linalg.norm(estimate.gaze_vector_3d) > 0
        assert estimate.gaze_point_2d is not None
    
    # Tracking should be continuous (all using same fallback method)
    methods = [est.method for est in estimates]
    assert all("fallback" in m.lower() for m in methods), \
        "All estimates should use fallback for consistent low confidence"


def test_fallback_disabled():
    """
    Test that fallback can be disabled when configured.
    
    **Feature: ai-enhanced-gaze-tracking, Property 6: Confidence-Based Fallback**
    **Validates: Requirements 3.4**
    """
    landmarks = np.array([
        [200, 200], [220, 200], [240, 200],
        [280, 200], [300, 200], [320, 200]
    ], dtype=np.float32)
    
    face_detection = create_face_detection((260, 240), landmarks)
    frame = create_test_frame()
    
    # Use low confidence model with fallback disabled
    low_conf_model = LowConfidenceAIModel(confidence=0.1)
    estimator = AIGazeEstimator(models=[low_conf_model], confidence_threshold=0.5)
    estimator.enable_fallback(False)
    
    # Get estimate
    estimate = estimator.estimate_gaze(face_detection, frame)
    
    # Should still use AI method even with low confidence
    assert "ai" in estimate.method.lower(), \
        "Should use AI method when fallback is disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
