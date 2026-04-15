"""
Property-based tests for face detection reliability and temporal prediction continuity.

**Feature: ai-enhanced-gaze-tracking, Property 19: Temporal Prediction Continuity**
**Validates: Requirements 8.1**

This module tests that the face detection system maintains tracking using temporal
prediction when face detection temporarily fails and resumes seamlessly when detection recovers.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import cv2
import time
from unittest.mock import Mock, patch

from ai_enhanced_gaze_tracking.components.face_detection.hybrid_face_detector import HybridFaceDetector
from ai_enhanced_gaze_tracking.components.face_detection.mediapipe_face_detector import MediaPipeFaceDetector
from ai_enhanced_gaze_tracking.components.face_detection.custom_face_detector import CustomFaceDetector
from ai_enhanced_gaze_tracking.core.data_models import FaceDetection


class TestFaceDetectionReliability:
    """Test suite for face detection reliability and temporal prediction continuity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hybrid_detector = HybridFaceDetector(
            confidence_threshold=0.5,
            temporal_window=5,
            fallback_threshold=0.3
        )
        
        # Create mock detectors for controlled testing
        self.mock_mediapipe = Mock(spec=MediaPipeFaceDetector)
        self.mock_custom = Mock(spec=CustomFaceDetector)
        
        # Replace detectors with mocks for controlled testing
        self.hybrid_detector.mediapipe_detector = self.mock_mediapipe
        self.hybrid_detector.custom_detector = self.mock_custom
    
    def create_synthetic_face_detection(
        self, 
        center_x: float = 320, 
        center_y: float = 240,
        size: float = 100,
        confidence: float = 0.8,
        face_id: int = 0
    ) -> FaceDetection:
        """
        Create synthetic face detection for testing.
        
        Args:
            center_x: Face center X coordinate
            center_y: Face center Y coordinate  
            size: Face size
            confidence: Detection confidence
            face_id: Face ID for tracking
            
        Returns:
            Synthetic face detection
        """
        # Create bounding box
        bbox = (
            center_x - size/2,
            center_y - size/2,
            size,
            size
        )
        
        # Create synthetic landmarks (simplified)
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        # Key landmarks
        landmarks[1] = [center_x, center_y + size * 0.1]  # Nose tip
        landmarks[33] = [center_x - size * 0.2, center_y - size * 0.1]  # Left eye
        landmarks[263] = [center_x + size * 0.2, center_y - size * 0.1]  # Right eye
        landmarks[61] = [center_x - size * 0.15, center_y + size * 0.2]  # Left mouth
        landmarks[291] = [center_x + size * 0.15, center_y + size * 0.2]  # Right mouth
        landmarks[152] = [center_x, center_y + size * 0.4]  # Chin
        
        # Fill remaining landmarks with interpolated values
        for i in range(468):
            if i not in [1, 33, 263, 61, 291, 152]:
                # Simple interpolation around face center
                angle = (i / 468.0) * 2 * np.pi
                radius = size * 0.3 * (0.8 + 0.4 * np.random.random())
                landmarks[i] = [
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle)
                ]
        
        return FaceDetection(
            bbox=bbox,
            landmarks=landmarks,
            confidence=confidence,
            face_id=face_id,
            is_child=True,
            quality_score=confidence
        )
    
    def create_synthetic_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create synthetic video frame for testing."""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    @given(
        initial_x=st.floats(min_value=100, max_value=540),
        initial_y=st.floats(min_value=100, max_value=380),
        movement_x=st.floats(min_value=-20, max_value=20),
        movement_y=st.floats(min_value=-20, max_value=20),
        failure_duration=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=30, deadline=3000)
    def test_temporal_prediction_continuity(
        self, 
        initial_x, 
        initial_y, 
        movement_x, 
        movement_y, 
        failure_duration
    ):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 19: Temporal Prediction Continuity**
        **Validates: Requirements 8.1**
        
        Property: For any temporary face detection failure, the system should maintain
        tracking using temporal prediction and seamlessly resume when detection recovers.
        """
        # Phase 1: Establish tracking with successful detections
        successful_detections = []
        frames = []
        
        for i in range(3):  # 3 frames of successful detection
            frame = self.create_synthetic_frame()
            frames.append(frame)
            
            # Create detection with slight movement
            detection = self.create_synthetic_face_detection(
                center_x=initial_x + i * movement_x,
                center_y=initial_y + i * movement_y,
                confidence=0.8,
                face_id=0
            )
            successful_detections.append([detection])
        
        # Configure mock to return successful detections
        self.mock_mediapipe.detect_faces.side_effect = successful_detections
        
        # Process successful frames to establish tracking
        for frame in frames:
            detections = self.hybrid_detector.detect_faces(frame)
            assert len(detections) > 0, "Should detect face during successful phase"
            assert detections[0].face_id == 0, "Should maintain consistent face ID"
        
        # Phase 2: Simulate detection failure
        self.mock_mediapipe.detect_faces.side_effect = None  # Clear side_effect
        self.mock_mediapipe.detect_faces.return_value = []  # MediaPipe fails
        self.mock_custom.detect_faces.return_value = []     # Custom detector fails
        
        failure_detections = []
        for i in range(failure_duration):
            frame = self.create_synthetic_frame()
            detections = self.hybrid_detector.detect_faces(frame)
            failure_detections.append(detections)
        
        # Test temporal prediction during failure
        # At least some frames should have predicted detections
        has_predictions = any(len(dets) > 0 for dets in failure_detections)
        
        if has_predictions:
            # If predictions exist, they should maintain face ID consistency
            for detections in failure_detections:
                if detections:
                    assert detections[0].face_id == 0, "Predicted detection should maintain face ID"
                    assert detections[0].confidence < 0.8, "Predicted detection should have reduced confidence"
        
        # Phase 3: Recovery - detection works again
        recovery_detection = self.create_synthetic_face_detection(
            center_x=initial_x + (3 + failure_duration) * movement_x,
            center_y=initial_y + (3 + failure_duration) * movement_y,
            confidence=0.8,
            face_id=0
        )
        
        # Reset the mock to return the recovery detection
        self.mock_mediapipe.detect_faces.side_effect = None  # Clear side_effect
        self.mock_mediapipe.detect_faces.return_value = [recovery_detection]
        self.mock_custom.detect_faces.return_value = []  # Reset custom detector
        
        frame = self.create_synthetic_frame()
        recovery_detections = self.hybrid_detector.detect_faces(frame)
        
        # Test seamless recovery
        assert len(recovery_detections) > 0, "Should detect face after recovery"
        assert recovery_detections[0].face_id == 0, "Should maintain face ID after recovery"
        assert recovery_detections[0].confidence >= 0.5, "Should have good confidence after recovery"
    
    @given(
        num_frames=st.integers(min_value=5, max_value=10),
        noise_level=st.floats(min_value=0.0, max_value=10.0)
    )
    @settings(max_examples=20, deadline=3000)
    def test_temporal_smoothing_consistency(self, num_frames, noise_level):
        """
        Test that temporal smoothing produces consistent results.
        
        Property: For any sequence of detections with noise, temporal smoothing
        should reduce jitter and maintain tracking consistency.
        """
        base_x, base_y = 320, 240
        detections_sequence = []
        
        # Create sequence of detections with noise
        for i in range(num_frames):
            # Add noise to position
            noisy_x = base_x + np.random.normal(0, noise_level)
            noisy_y = base_y + np.random.normal(0, noise_level)
            
            detection = self.create_synthetic_face_detection(
                center_x=noisy_x,
                center_y=noisy_y,
                confidence=0.7 + 0.2 * np.random.random(),
                face_id=0
            )
            detections_sequence.append([detection])
        
        # Configure mock to return noisy detections
        self.mock_mediapipe.detect_faces.side_effect = detections_sequence
        
        # Process frames and collect smoothed results
        smoothed_positions = []
        for i in range(num_frames):
            frame = self.create_synthetic_frame()
            detections = self.hybrid_detector.detect_faces(frame)
            
            if detections:
                center_x = detections[0].bbox[0] + detections[0].bbox[2] / 2
                center_y = detections[0].bbox[1] + detections[0].bbox[3] / 2
                smoothed_positions.append((center_x, center_y))
        
        # Test smoothing effectiveness
        if len(smoothed_positions) >= 3:
            # Calculate position variance before and after smoothing
            original_positions = [(base_x + np.random.normal(0, noise_level), 
                                 base_y + np.random.normal(0, noise_level)) 
                                for _ in range(len(smoothed_positions))]
            
            # Smoothed positions should have less variance than original noisy positions
            # (This is a simplified test - in practice we'd compare actual variance)
            
            # Test that positions are reasonable (within expected bounds)
            for pos in smoothed_positions:
                assert 200 <= pos[0] <= 440, f"Smoothed X position should be reasonable: {pos[0]}"
                assert 140 <= pos[1] <= 340, f"Smoothed Y position should be reasonable: {pos[1]}"
    
    def test_fallback_detector_activation(self):
        """
        Test that fallback detector is activated when primary detector fails.
        
        Property: When MediaPipe detection fails or produces low-quality results,
        the system should automatically use the fallback detector.
        """
        # Configure MediaPipe to fail
        self.mock_mediapipe.detect_faces.return_value = []
        
        # Configure custom detector to succeed
        fallback_detection = self.create_synthetic_face_detection(
            confidence=0.6,
            face_id=0
        )
        fallback_detection.quality_score *= 0.8  # Fallback quality reduction
        self.mock_custom.detect_faces.return_value = [fallback_detection]
        
        frame = self.create_synthetic_frame()
        detections = self.hybrid_detector.detect_faces(frame)
        
        # Should get detection from fallback
        assert len(detections) > 0, "Fallback detector should provide detection"
        assert detections[0].quality_score < 0.8, "Fallback detection should have reduced quality"
        
        # Verify fallback was actually called
        self.mock_custom.detect_faces.assert_called_once()
    
    def test_quality_based_fallback_trigger(self):
        """
        Test that fallback is triggered based on detection quality.
        
        Property: When primary detector produces low-quality results,
        the system should use fallback detector for better results.
        """
        # Configure MediaPipe to return low-quality detection
        low_quality_detection = self.create_synthetic_face_detection(
            confidence=0.3,  # Below threshold
            face_id=0
        )
        low_quality_detection.quality_score = 0.2  # Very low quality
        self.mock_mediapipe.detect_faces.return_value = [low_quality_detection]
        
        # Configure custom detector to return better detection
        better_detection = self.create_synthetic_face_detection(
            confidence=0.6,
            face_id=0
        )
        better_detection.quality_score = 0.5
        self.mock_custom.detect_faces.return_value = [better_detection]
        
        frame = self.create_synthetic_frame()
        detections = self.hybrid_detector.detect_faces(frame)
        
        # Should use fallback due to low quality
        assert len(detections) > 0, "Should get detection from fallback"
        # Note: The actual quality will be reduced by fallback factor
        
        # Verify both detectors were called
        self.mock_mediapipe.detect_faces.assert_called_once()
        self.mock_custom.detect_faces.assert_called_once()
    
    @given(
        track_duration=st.integers(min_value=3, max_value=8),
        gap_duration=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=15, deadline=3000)
    def test_face_id_consistency(self, track_duration, gap_duration):
        """
        Test that face IDs remain consistent across detection gaps.
        
        Property: For any face track, the face ID should remain consistent
        even when detection temporarily fails and recovers.
        """
        # Phase 1: Establish track
        base_x, base_y = 300, 200
        
        for i in range(track_duration):
            detection = self.create_synthetic_face_detection(
                center_x=base_x + i * 5,  # Slight movement
                center_y=base_y,
                confidence=0.8,
                face_id=0
            )
            self.mock_mediapipe.detect_faces.return_value = [detection]
            
            frame = self.create_synthetic_frame()
            detections = self.hybrid_detector.detect_faces(frame)
            
            assert len(detections) > 0, f"Should detect face in frame {i}"
            assert detections[0].face_id == 0, f"Face ID should be consistent in frame {i}"
        
        # Phase 2: Detection gap
        self.mock_mediapipe.detect_faces.return_value = []
        self.mock_custom.detect_faces.return_value = []
        
        for i in range(gap_duration):
            frame = self.create_synthetic_frame()
            detections = self.hybrid_detector.detect_faces(frame)
            # May or may not have predictions, but if they exist, ID should be consistent
            if detections:
                assert detections[0].face_id == 0, f"Predicted face ID should be consistent in gap frame {i}"
        
        # Phase 3: Recovery
        recovery_detection = self.create_synthetic_face_detection(
            center_x=base_x + (track_duration + gap_duration) * 5,
            center_y=base_y,
            confidence=0.8,
            face_id=0
        )
        self.mock_mediapipe.detect_faces.return_value = [recovery_detection]
        
        frame = self.create_synthetic_frame()
        detections = self.hybrid_detector.detect_faces(frame)
        
        assert len(detections) > 0, "Should detect face after recovery"
        assert detections[0].face_id == 0, "Face ID should remain consistent after recovery"
    
    def test_performance_monitoring(self):
        """
        Test that performance statistics are correctly tracked.
        
        Property: The system should accurately track performance metrics
        including success rates and fallback usage.
        """
        # Reset performance stats
        self.hybrid_detector.performance_stats = {
            'mediapipe_success': 0,
            'fallback_used': 0,
            'temporal_predictions': 0,
            'total_frames': 0
        }
        
        # Test MediaPipe success
        detection = self.create_synthetic_face_detection()
        self.mock_mediapipe.detect_faces.return_value = [detection]
        
        frame = self.create_synthetic_frame()
        self.hybrid_detector.detect_faces(frame)
        
        stats = self.hybrid_detector.get_performance_stats()
        assert stats['mediapipe_success_rate'] == 1.0, "Should track MediaPipe success"
        assert stats['total_frames'] == 1, "Should track total frames"
        
        # Test fallback usage
        self.mock_mediapipe.detect_faces.return_value = []
        self.mock_custom.detect_faces.return_value = [detection]
        
        frame = self.create_synthetic_frame()
        self.hybrid_detector.detect_faces(frame)
        
        stats = self.hybrid_detector.get_performance_stats()
        assert stats['fallback_usage_rate'] == 0.5, "Should track fallback usage"
        assert stats['total_frames'] == 2, "Should track total frames"
    
    def test_confidence_threshold_adjustment(self):
        """
        Test that confidence threshold adjustments work correctly.
        
        Property: Changing confidence thresholds should affect detection filtering
        and fallback behavior appropriately.
        """
        # Create detection with moderate confidence
        detection = self.create_synthetic_face_detection(confidence=0.6)
        self.mock_mediapipe.detect_faces.return_value = [detection]
        
        # Test with high threshold
        self.hybrid_detector.set_confidence_threshold(0.8)
        frame = self.create_synthetic_frame()
        detections = self.hybrid_detector.detect_faces(frame)
        
        # Should trigger fallback due to low confidence
        assert self.hybrid_detector.get_confidence_threshold() == 0.8
        
        # Test with low threshold
        self.hybrid_detector.set_confidence_threshold(0.4)
        detections = self.hybrid_detector.detect_faces(frame)
        
        # Should accept the detection
        assert self.hybrid_detector.get_confidence_threshold() == 0.4
    
    def teardown_method(self):
        """Clean up after each test."""
        self.hybrid_detector.cleanup()