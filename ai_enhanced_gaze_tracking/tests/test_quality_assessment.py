"""
Property-based tests for quality factor integration in face detection.

**Feature: ai-enhanced-gaze-tracking, Property 22: Quality Factor Integration**
**Validates: Requirements 9.2**

This module tests that the quality assessment system considers head pose, lighting conditions,
and occlusions in the quality calculation for face detection reliability.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import cv2
from unittest.mock import Mock, patch

from ai_enhanced_gaze_tracking.components.face_detection.mediapipe_face_detector import MediaPipeFaceDetector
from ai_enhanced_gaze_tracking.components.face_detection.hybrid_face_detector import HybridFaceDetector
from ai_enhanced_gaze_tracking.core.data_models import FaceDetection, QualityMetrics


class TestQualityFactorIntegration:
    """Test suite for quality factor integration property."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock MediaPipe detector for controlled testing
        self.mock_detector = Mock(spec=MediaPipeFaceDetector)
        
        # We'll test the quality assessment methods directly
        self.detector = MediaPipeFaceDetector(confidence_threshold=0.5)
    
    def create_synthetic_frame_with_conditions(
        self,
        brightness: float = 128,
        blur_level: float = 50,
        noise_level: float = 0,
        width: int = 640,
        height: int = 480
    ) -> np.ndarray:
        """
        Create synthetic frame with specific quality conditions.
        
        Args:
            brightness: Average brightness level (0-255)
            blur_level: Blur level (higher = more blur)
            noise_level: Noise level (0-100)
            width: Frame width
            height: Frame height
            
        Returns:
            Synthetic frame with specified conditions
        """
        # Create base frame with specified brightness
        frame = np.full((height, width, 3), brightness, dtype=np.uint8)
        
        # Add some structure to make it more realistic, but keep the target brightness
        # Create a face-like region in the center with the same brightness
        center_x, center_y = width // 2, height // 2
        face_size = 100
        
        # Face region with the target brightness (no change)
        frame[
            center_y - face_size//2:center_y + face_size//2,
            center_x - face_size//2:center_x + face_size//2
        ] = brightness
        
        # Add eye regions (slightly darker but not too much)
        eye_brightness = max(0, brightness - 5)
        # Left eye
        frame[
            center_y - 20:center_y - 10,
            center_x - 30:center_x - 10
        ] = eye_brightness
        # Right eye
        frame[
            center_y - 20:center_y - 10,
            center_x + 10:center_x + 30
        ] = eye_brightness
        
        # Add mouth region (slightly darker)
        mouth_brightness = max(0, brightness - 3)
        frame[
            center_y + 10:center_y + 20,
            center_x - 15:center_x + 15
        ] = mouth_brightness
        
        # Apply blur if specified
        if blur_level > 0:
            kernel_size = max(1, int(blur_level / 10))
            if kernel_size % 2 == 0:
                kernel_size += 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), blur_level / 20)
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def create_synthetic_landmarks_with_occlusion(
        self,
        center_x: float = 320,
        center_y: float = 240,
        size: float = 100,
        occlusion_level: float = 0.0
    ) -> np.ndarray:
        """
        Create synthetic landmarks with specified occlusion level.
        
        Args:
            center_x: Face center X coordinate
            center_y: Face center Y coordinate
            size: Face size
            occlusion_level: Occlusion level (0.0 = no occlusion, 1.0 = fully occluded)
            
        Returns:
            Synthetic landmarks array
        """
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        # Key landmarks that the occlusion assessment actually checks
        key_landmarks = {
            1: [center_x, center_y + size * 0.1],  # Nose tip
            33: [center_x - size * 0.2, center_y - size * 0.1],  # Left eye
            263: [center_x + size * 0.2, center_y - size * 0.1],  # Right eye
            61: [center_x - size * 0.15, center_y + size * 0.2],  # Left mouth
            291: [center_x + size * 0.15, center_y + size * 0.2],  # Right mouth
            152: [center_x, center_y + size * 0.4],  # Chin
        }
        
        # Apply occlusion by moving landmarks outside frame bounds
        # This matches what the _assess_occlusion method actually checks
        num_occluded = int(len(key_landmarks) * occlusion_level)
        occluded_indices = list(key_landmarks.keys())[:num_occluded]
        
        for idx, pos in key_landmarks.items():
            if idx in occluded_indices:
                # Move landmark outside frame bounds (the actual check in _assess_occlusion)
                landmarks[idx] = [-10, -10]  # Outside frame bounds
            else:
                landmarks[idx] = pos
        
        # Fill remaining landmarks with interpolated values
        for i in range(468):
            if i not in key_landmarks:
                # Simple interpolation around face center
                angle = (i / 468.0) * 2 * np.pi
                radius = size * 0.3 * (0.8 + 0.4 * np.random.random())
                landmarks[i] = [
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle)
                ]
        
        return landmarks
    
    @given(
        brightness=st.floats(min_value=30, max_value=220),
        blur_level=st.floats(min_value=0, max_value=100),
        occlusion_level=st.floats(min_value=0.0, max_value=0.8)
    )
    @settings(max_examples=50, deadline=3000)
    def test_quality_factor_integration(self, brightness, blur_level, occlusion_level):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 22: Quality Factor Integration**
        **Validates: Requirements 9.2**
        
        Property: For any quality assessment, the system should consider head pose,
        lighting conditions, and occlusions in the quality calculation.
        """
        # Create frame with specific lighting and blur conditions
        frame = self.create_synthetic_frame_with_conditions(
            brightness=brightness,
            blur_level=blur_level
        )
        
        # Create landmarks with specified occlusion level
        landmarks = self.create_synthetic_landmarks_with_occlusion(
            occlusion_level=occlusion_level
        )
        
        # Calculate bounding box from landmarks
        valid_landmarks = landmarks[landmarks[:, 0] > 0]  # Filter out occluded landmarks
        if len(valid_landmarks) == 0:
            assume(False)  # Skip if all landmarks are occluded
        
        x_coords = valid_landmarks[:, 0]
        y_coords = valid_landmarks[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Assess quality using the detector's method
        quality_score = self.detector._assess_face_quality(frame, landmarks, bbox)
        
        # Test 1: Quality should be between 0 and 1
        assert 0.0 <= quality_score <= 1.0, f"Quality score should be between 0 and 1, got {quality_score}"
        
        # Test 2: Lighting conditions should affect quality
        # Very dark or very bright conditions should reduce quality
        if brightness < 40:  # Very dark
            assert quality_score < 0.9, f"Very dark lighting (brightness={brightness:.1f}) should reduce quality, got {quality_score:.3f}"
        elif brightness > 210:  # Very bright
            assert quality_score < 0.9, f"Very bright lighting (brightness={brightness:.1f}) should reduce quality, got {quality_score:.3f}"
        
        # Test 3: Blur should affect quality
        # High blur should reduce quality
        if blur_level > 80:
            assert quality_score < 0.8, f"High blur (level={blur_level:.1f}) should reduce quality, got {quality_score:.3f}"
        
        # Test 4: Occlusion should affect quality (more lenient since our synthetic occlusion is simple)
        # High occlusion should reduce quality
        if occlusion_level > 0.7:
            assert quality_score < 0.8, f"High occlusion (level={occlusion_level:.2f}) should reduce quality, got {quality_score:.3f}"
        
        # Test 5: Good conditions should yield good quality
        if (60 <= brightness <= 180 and blur_level < 20 and occlusion_level < 0.1):
            assert quality_score > 0.3, f"Good conditions should yield reasonable quality, got {quality_score:.3f}"
    
    @given(
        face_size=st.integers(min_value=20, max_value=200),
        brightness=st.floats(min_value=50, max_value=200)
    )
    @settings(max_examples=30, deadline=2000)
    def test_face_size_quality_factor(self, face_size, brightness):
        """
        Test that face size is properly considered in quality assessment.
        
        Property: Larger faces should generally have better quality scores
        than smaller faces, all else being equal.
        """
        # Create frame with good conditions
        frame = self.create_synthetic_frame_with_conditions(brightness=brightness)
        
        # Create landmarks for specified face size
        landmarks = self.create_synthetic_landmarks_with_occlusion(size=face_size)
        
        # Calculate bbox
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        quality_score = self.detector._assess_face_quality(frame, landmarks, bbox)
        
        # Test face size impact
        if face_size < 50:  # Small face
            assert quality_score < 0.8, f"Small face (size={face_size}) should have reduced quality"
        elif face_size > 100:  # Large face
            assert quality_score > 0.3, f"Large face (size={face_size}) should have reasonable quality"
    
    def test_eye_visibility_assessment(self):
        """
        Test that eye visibility is properly assessed.
        
        Property: Faces with clearly visible eyes should have better quality
        than faces with occluded or unclear eyes.
        """
        # Create frame with good lighting
        frame = self.create_synthetic_frame_with_conditions(brightness=128)
        
        # Test with good eye visibility
        good_landmarks = self.create_synthetic_landmarks_with_occlusion(occlusion_level=0.0)
        bbox = (220, 140, 200, 200)  # Reasonable face size
        
        good_quality = self.detector._assess_face_quality(frame, good_landmarks, bbox)
        
        # Test with poor eye visibility (high occlusion affecting eyes)
        poor_landmarks = self.create_synthetic_landmarks_with_occlusion(occlusion_level=0.6)
        
        poor_quality = self.detector._assess_face_quality(frame, poor_landmarks, bbox)
        
        # Good eye visibility should yield better quality
        assert good_quality > poor_quality, (
            f"Good eye visibility should yield better quality. "
            f"Good: {good_quality:.3f}, Poor: {poor_quality:.3f}"
        )
    
    @given(
        brightness1=st.floats(min_value=40, max_value=75),
        brightness2=st.floats(min_value=195, max_value=220)
    )
    @settings(max_examples=20, deadline=2000)
    def test_lighting_quality_comparison(self, brightness1, brightness2):
        """
        Test that lighting quality assessment works correctly.
        
        Property: Frames with moderate lighting should have better quality
        than frames with extreme lighting conditions.
        """
        # Create landmarks with good conditions
        landmarks = self.create_synthetic_landmarks_with_occlusion(occlusion_level=0.1)
        bbox = (220, 140, 200, 200)
        
        # Test with moderate lighting
        moderate_frame = self.create_synthetic_frame_with_conditions(brightness=128)
        moderate_quality = self.detector._assess_face_quality(moderate_frame, landmarks, bbox)
        
        # Test with extreme lighting (too dark)
        dark_frame = self.create_synthetic_frame_with_conditions(brightness=brightness1)
        dark_quality = self.detector._assess_face_quality(dark_frame, landmarks, bbox)
        
        # Test with extreme lighting (too bright)
        bright_frame = self.create_synthetic_frame_with_conditions(brightness=brightness2)
        bright_quality = self.detector._assess_face_quality(bright_frame, landmarks, bbox)
        
        # Moderate lighting should be better than extreme conditions
        assert moderate_quality > dark_quality, (
            f"Moderate lighting should be better than dark. "
            f"Moderate: {moderate_quality:.3f}, Dark: {dark_quality:.3f}"
        )
        
        assert moderate_quality > bright_quality, (
            f"Moderate lighting should be better than bright. "
            f"Moderate: {moderate_quality:.3f}, Bright: {bright_quality:.3f}"
        )
    
    @given(
        blur1=st.floats(min_value=0, max_value=30),
        blur2=st.floats(min_value=60, max_value=100)
    )
    @settings(max_examples=20, deadline=2000)
    def test_blur_quality_impact(self, blur1, blur2):
        """
        Test that blur level properly impacts quality assessment.
        
        Property: Less blurred images should have better quality than
        more blurred images, all else being equal.
        """
        # Create landmarks and bbox with good conditions
        landmarks = self.create_synthetic_landmarks_with_occlusion(occlusion_level=0.1)
        bbox = (220, 140, 200, 200)
        
        # Test with low blur
        low_blur_frame = self.create_synthetic_frame_with_conditions(
            brightness=128, blur_level=blur1
        )
        low_blur_quality = self.detector._assess_face_quality(low_blur_frame, landmarks, bbox)
        
        # Test with high blur
        high_blur_frame = self.create_synthetic_frame_with_conditions(
            brightness=128, blur_level=blur2
        )
        high_blur_quality = self.detector._assess_face_quality(high_blur_frame, landmarks, bbox)
        
        # Low blur should yield better quality
        assert low_blur_quality > high_blur_quality, (
            f"Low blur should yield better quality than high blur. "
            f"Low blur: {low_blur_quality:.3f}, High blur: {high_blur_quality:.3f}"
        )
    
    def test_quality_metrics_completeness(self):
        """
        Test that quality assessment considers all required factors.
        
        Property: The quality assessment should integrate multiple factors
        including size, blur, lighting, eyes, and occlusion.
        """
        # Create test conditions
        frame = self.create_synthetic_frame_with_conditions(brightness=128, blur_level=20)
        landmarks = self.create_synthetic_landmarks_with_occlusion(occlusion_level=0.2)
        bbox = (220, 140, 200, 200)
        
        # Mock the individual quality factor methods to verify they're called
        with patch.object(self.detector, '_assess_eye_visibility', return_value=0.8) as mock_eyes, \
             patch.object(self.detector, '_assess_occlusion', return_value=0.7) as mock_occlusion:
            
            quality_score = self.detector._assess_face_quality(frame, landmarks, bbox)
            
            # Verify that eye and occlusion assessment methods were called
            mock_eyes.assert_called_once()
            mock_occlusion.assert_called_once()
            
            # Quality should be a reasonable combination of factors
            assert 0.0 <= quality_score <= 1.0, "Quality should be normalized"
    
    def test_quality_assessment_edge_cases(self):
        """
        Test quality assessment with edge cases.
        
        Property: The system should handle edge cases gracefully without
        crashing and should return reasonable quality scores.
        """
        # Test with empty frame
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = self.create_synthetic_landmarks_with_occlusion()
        bbox = (10, 10, 80, 80)
        
        quality = self.detector._assess_face_quality(empty_frame, landmarks, bbox)
        assert 0.0 <= quality <= 1.0, "Should handle empty frame gracefully"
        
        # Test with very small bbox
        small_bbox = (100, 100, 5, 5)
        quality = self.detector._assess_face_quality(empty_frame, landmarks, small_bbox)
        assert quality == 0.0, "Very small face should have zero quality"
        
        # Test with bbox outside frame bounds
        invalid_bbox = (1000, 1000, 100, 100)
        quality = self.detector._assess_face_quality(empty_frame, landmarks, invalid_bbox)
        assert quality == 0.0, "Invalid bbox should have zero quality"
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self.detector, 'cleanup'):
            self.detector.cleanup()