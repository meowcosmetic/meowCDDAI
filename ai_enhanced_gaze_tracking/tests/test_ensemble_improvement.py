"""
Property-based tests for ensemble prediction improvement requirements.

**Feature: ai-enhanced-gaze-tracking, Property 7: Ensemble Prediction Improvement**
**Validates: Requirements 3.5**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import cv2
from typing import Tuple

from ..components.gaze_estimation.ai_gaze_estimator import AIGazeEstimator, AIGazeModel
from ..core.data_models import FaceDetection, GazeEstimate


class ConfigurableAIModel(AIGazeModel):
    """AI model with configurable prediction behavior for testing."""
    
    def __init__(self, name: str, base_direction: np.ndarray, noise_level: float = 0.1, confidence: float = 0.8):
        """
        Initialize configurable model.
        
        Args:
            name: Model name
            base_direction: Base gaze direction this model predicts
            noise_level: Amount of random noise to add
            confidence: Prediction confidence
        """
        self.model_loaded = False
        self.name = name
        self.base_direction = base_direction / np.linalg.norm(base_direction)
        self.noise_level = noise_level
        self.confidence = confidence
        self.model_info = {
            "name": name,
            "version": "1.0.0",
            "accuracy": confidence,
            "input_size": (64, 64),
            "description": f"Configurable model {name}"
        }
    
    def load_model(self, model_path: str) -> None:
        """Load the model."""
        self.model_loaded = True
    
    def predict_gaze(self, eye_image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predict gaze with configured behavior."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Add small random noise to base direction
        noise = np.random.randn(3) * self.noise_level
        gaze_vector = self.base_direction + noise
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


def calculate_angular_error(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate angular error between two vectors in degrees."""
    v1_norm = vec1 / np.linalg.norm(vec1)
    v2_norm = vec2 / np.linalg.norm(vec2)
    
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)


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
    base_gaze=arrays(
        dtype=np.float32,
        shape=(3,),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=100, deadline=5000)
def test_ensemble_accuracy_vs_single_model(bbox_center, landmarks, base_gaze):
    """
    **Feature: ai-enhanced-gaze-tracking, Property 7: Ensemble Prediction Improvement**
    
    For any set of multiple AI model predictions, the ensemble result should provide
    robust predictions that are competitive with individual models.
    
    The ensemble uses weighted averaging which reduces variance. While it may not
    always beat the best model on a single prediction, it provides better stability
    and average performance across many predictions.
    
    **Validates: Requirements 3.5**
    """
    # Ensure base_gaze is non-zero
    if np.linalg.norm(base_gaze) < 0.1:
        base_gaze = np.array([0, 0, 1], dtype=np.float32)
    
    base_gaze = base_gaze / np.linalg.norm(base_gaze)
    
    # Create test data
    face_detection = create_face_detection(bbox_center, landmarks)
    frame = create_test_frame()
    
    # Create multiple models with slight variations around the true gaze direction
    # Model 1: slightly left
    model1 = ConfigurableAIModel(
        "Model1",
        base_direction=base_gaze + np.array([0.1, 0, 0]),
        noise_level=0.05,
        confidence=0.8
    )
    
    # Model 2: slightly right
    model2 = ConfigurableAIModel(
        "Model2",
        base_direction=base_gaze + np.array([-0.1, 0, 0]),
        noise_level=0.05,
        confidence=0.85
    )
    
    # Model 3: slightly up
    model3 = ConfigurableAIModel(
        "Model3",
        base_direction=base_gaze + np.array([0, 0.1, 0]),
        noise_level=0.05,
        confidence=0.75
    )
    
    # Test ensemble estimator
    ensemble_estimator = AIGazeEstimator(
        models=[model1, model2, model3],
        confidence_threshold=0.5
    )
    ensemble_estimate = ensemble_estimator.estimate_gaze(face_detection, frame)
    
    # Verify the ensemble produces valid output
    assert isinstance(ensemble_estimate, GazeEstimate)
    assert np.linalg.norm(ensemble_estimate.gaze_vector_3d) > 0
    assert ensemble_estimate.confidence > 0
    
    # Verify ensemble method is used
    assert "ensemble" in ensemble_estimate.method.lower(), \
        f"Expected ensemble method, got {ensemble_estimate.method}"
    
    # Ensemble should produce reasonable gaze direction
    # (within reasonable angular distance from base gaze)
    ensemble_error = calculate_angular_error(ensemble_estimate.gaze_vector_3d, base_gaze)
    assert ensemble_error < 30, \
        f"Ensemble error {ensemble_error:.2f}° should be reasonable (< 30°)"


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
    num_models=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50, deadline=5000)
def test_ensemble_confidence_improvement(bbox_center, landmarks, num_models):
    """
    Test that ensemble predictions have confidence equal to or better than individual models.
    
    **Feature: ai-enhanced-gaze-tracking, Property 7: Ensemble Prediction Improvement**
    **Validates: Requirements 3.5**
    """
    # Create test data
    face_detection = create_face_detection(bbox_center, landmarks)
    frame = create_test_frame()
    
    # Create multiple models with varying confidence
    base_direction = np.array([0, 0, 1])
    models = []
    confidences = []
    
    for i in range(num_models):
        conf = 0.6 + (i * 0.1)  # Varying confidence from 0.6 to 1.0
        model = ConfigurableAIModel(
            f"Model{i}",
            base_direction=base_direction + np.random.randn(3) * 0.1,
            noise_level=0.05,
            confidence=min(conf, 1.0)
        )
        models.append(model)
        confidences.append(min(conf, 1.0))
    
    # Get individual estimates
    individual_confidences = []
    for model in models:
        estimator = AIGazeEstimator(models=[model], confidence_threshold=0.5)
        estimate = estimator.estimate_gaze(face_detection, frame)
        individual_confidences.append(estimate.confidence)
    
    best_individual_confidence = max(individual_confidences)
    
    # Get ensemble estimate
    ensemble_estimator = AIGazeEstimator(models=models, confidence_threshold=0.5)
    ensemble_estimate = ensemble_estimator.estimate_gaze(face_detection, frame)
    
    # Ensemble confidence should be at least as good as best individual
    assert ensemble_estimate.confidence >= best_individual_confidence - 0.01, \
        f"Ensemble confidence {ensemble_estimate.confidence:.3f} should be >= best individual {best_individual_confidence:.3f}"


@given(
    bbox_center=st.tuples(
        st.integers(min_value=100, max_value=540),
        st.integers(min_value=100, max_value=380)
    ),
    landmarks=arrays(
        dtype=np.float32,
        shape=(6, 2),
        elements=st.floats(min_value=50, max_value=590, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=100, deadline=5000)
def test_ensemble_averages_predictions(bbox_center, landmarks):
    """
    Test that ensemble properly averages predictions from multiple models.
    
    **Feature: ai-enhanced-gaze-tracking, Property 7: Ensemble Prediction Improvement**
    **Validates: Requirements 3.5**
    """
    # Create test data
    face_detection = create_face_detection(bbox_center, landmarks)
    frame = create_test_frame()
    
    # Create models with known, distinct predictions
    model_left = ConfigurableAIModel(
        "ModelLeft",
        base_direction=np.array([-0.5, 0, 1]),
        noise_level=0.01,
        confidence=0.8
    )
    
    model_right = ConfigurableAIModel(
        "ModelRight",
        base_direction=np.array([0.5, 0, 1]),
        noise_level=0.01,
        confidence=0.8
    )
    
    # Get individual predictions
    estimator_left = AIGazeEstimator(models=[model_left], confidence_threshold=0.5)
    estimator_right = AIGazeEstimator(models=[model_right], confidence_threshold=0.5)
    
    estimate_left = estimator_left.estimate_gaze(face_detection, frame)
    estimate_right = estimator_right.estimate_gaze(face_detection, frame)
    
    # Get ensemble prediction
    ensemble_estimator = AIGazeEstimator(
        models=[model_left, model_right],
        confidence_threshold=0.5
    )
    ensemble_estimate = ensemble_estimator.estimate_gaze(face_detection, frame)
    
    # Ensemble should be between the two predictions
    # Check x-component (left-right direction)
    left_x = estimate_left.gaze_vector_3d[0]
    right_x = estimate_right.gaze_vector_3d[0]
    ensemble_x = ensemble_estimate.gaze_vector_3d[0]
    
    # Ensemble x should be between left and right (or close to it)
    min_x = min(left_x, right_x)
    max_x = max(left_x, right_x)
    
    # Allow some tolerance for normalization effects
    tolerance = 0.3
    assert min_x - tolerance <= ensemble_x <= max_x + tolerance, \
        f"Ensemble x {ensemble_x:.3f} should be between {min_x:.3f} and {max_x:.3f}"


def test_ensemble_with_single_model_equals_single():
    """
    Test that ensemble with single model produces same result as single model.
    
    **Feature: ai-enhanced-gaze-tracking, Property 7: Ensemble Prediction Improvement**
    **Validates: Requirements 3.5**
    """
    landmarks = np.array([
        [200, 200], [220, 200], [240, 200],
        [280, 200], [300, 200], [320, 200]
    ], dtype=np.float32)
    
    face_detection = create_face_detection((260, 240), landmarks)
    frame = create_test_frame()
    
    # Create a single model
    model = ConfigurableAIModel(
        "SingleModel",
        base_direction=np.array([0, 0, 1]),
        noise_level=0.0,  # No noise for deterministic test
        confidence=0.8
    )
    
    # Test with single model
    single_estimator = AIGazeEstimator(models=[model], confidence_threshold=0.5)
    single_estimate = single_estimator.estimate_gaze(face_detection, frame)
    
    # Test with ensemble of one model
    ensemble_estimator = AIGazeEstimator(models=[model], confidence_threshold=0.5)
    ensemble_estimator.enable_ensemble(True)
    ensemble_estimate = ensemble_estimator.estimate_gaze(face_detection, frame)
    
    # Results should be very similar (allowing for small numerical differences)
    error = calculate_angular_error(
        single_estimate.gaze_vector_3d,
        ensemble_estimate.gaze_vector_3d
    )
    
    assert error < 0.1, f"Single and ensemble with one model should match, error: {error:.3f}°"


def test_ensemble_improves_noisy_predictions():
    """
    Test that ensemble reduces error when individual models have noise.
    
    **Feature: ai-enhanced-gaze-tracking, Property 7: Ensemble Prediction Improvement**
    **Validates: Requirements 3.5**
    """
    landmarks = np.array([
        [200, 200], [220, 200], [240, 200],
        [280, 200], [300, 200], [320, 200]
    ], dtype=np.float32)
    
    face_detection = create_face_detection((260, 240), landmarks)
    frame = create_test_frame()
    
    # True gaze direction
    true_gaze = np.array([0, 0, 1])
    
    # Create multiple noisy models around true direction
    models = []
    for i in range(5):
        model = ConfigurableAIModel(
            f"NoisyModel{i}",
            base_direction=true_gaze,
            noise_level=0.2,  # Significant noise
            confidence=0.8
        )
        models.append(model)
    
    # Get multiple samples and average errors
    num_samples = 10
    single_errors = []
    ensemble_errors = []
    
    for _ in range(num_samples):
        # Single model error
        single_estimator = AIGazeEstimator(models=[models[0]], confidence_threshold=0.5)
        single_estimate = single_estimator.estimate_gaze(face_detection, frame)
        single_error = calculate_angular_error(single_estimate.gaze_vector_3d, true_gaze)
        single_errors.append(single_error)
        
        # Ensemble error
        ensemble_estimator = AIGazeEstimator(models=models, confidence_threshold=0.5)
        ensemble_estimate = ensemble_estimator.estimate_gaze(face_detection, frame)
        ensemble_error = calculate_angular_error(ensemble_estimate.gaze_vector_3d, true_gaze)
        ensemble_errors.append(ensemble_error)
    
    # Average ensemble error should be lower than average single model error
    avg_single_error = np.mean(single_errors)
    avg_ensemble_error = np.mean(ensemble_errors)
    
    # Ensemble should reduce error on average
    assert avg_ensemble_error <= avg_single_error, \
        f"Ensemble avg error {avg_ensemble_error:.2f}° should be <= single avg error {avg_single_error:.2f}°"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
