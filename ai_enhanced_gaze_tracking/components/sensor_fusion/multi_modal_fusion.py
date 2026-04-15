"""
Multi-modal sensor fusion engine for combining gaze estimates.

This module implements weighted averaging, Kalman filtering, and conflict resolution
to fuse multiple gaze estimation sources into a single robust estimate.
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque

from ai_enhanced_gaze_tracking.core.interfaces import SensorFusion
from ai_enhanced_gaze_tracking.core.data_models import GazeEstimate, HeadPose, QualityMetrics


class KalmanFilter:
    """Simple Kalman filter for temporal consistency."""
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = None  # Current state estimate
        self.covariance = None  # State covariance
        
    def predict(self) -> Optional[np.ndarray]:
        """
        Predict next state.
        
        Returns:
            Predicted state or None if not initialized
        """
        if self.state is None:
            return None
            
        # Simple constant velocity model
        # State remains the same, covariance increases
        self.covariance = self.covariance + self.process_noise
        return self.state.copy()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with new measurement.
        
        Args:
            measurement: New measurement vector
            
        Returns:
            Updated state estimate
        """
        if self.state is None:
            # Initialize with first measurement
            self.state = measurement.copy()
            self.covariance = self.measurement_noise
            return self.state.copy()
        
        # Kalman gain
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)
        
        # Update state
        self.state = self.state + kalman_gain * (measurement - self.state)
        
        # Update covariance
        self.covariance = (1 - kalman_gain) * self.covariance
        
        return self.state.copy()
    
    def reset(self):
        """Reset filter state."""
        self.state = None
        self.covariance = None


class MultiModalFusion(SensorFusion):
    """
    Multi-modal sensor fusion engine.
    
    Combines gaze estimates from multiple sources using:
    - Confidence-weighted averaging
    - Kalman filtering for temporal consistency
    - Geometric constraint-based conflict resolution
    - Adaptive weight adjustment for unreliable sources
    """
    
    def __init__(
        self,
        history_size: int = 10,
        min_confidence: float = 0.1,
        conflict_threshold: float = 0.3,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize multi-modal fusion engine.
        
        Args:
            history_size: Number of previous estimates to keep
            min_confidence: Minimum confidence to include estimate
            conflict_threshold: Threshold for detecting conflicts (normalized)
            adaptation_rate: Rate of weight adaptation (0-1)
        """
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.conflict_threshold = conflict_threshold
        self.adaptation_rate = adaptation_rate
        
        # Temporal history
        self.history: deque = deque(maxlen=history_size)
        
        # Kalman filters for 2D and 3D gaze
        self.kalman_2d = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        self.kalman_3d = KalmanFilter(process_noise=0.01, measurement_noise=0.1)
        
        # Source weights (adaptive)
        self.source_weights: Dict[str, float] = {}
        self.default_weight = 1.0
        
    def fuse_estimates(
        self,
        estimates: List[GazeEstimate],
        confidences: List[float]
    ) -> GazeEstimate:
        """
        Fuse multiple gaze estimates into single result.
        
        Args:
            estimates: List of gaze estimates from different sources
            confidences: Confidence scores for each estimate
            
        Returns:
            Fused gaze estimate with combined confidence
        """
        if not estimates:
            raise ValueError("Cannot fuse empty list of estimates")
        
        if len(estimates) != len(confidences):
            raise ValueError("Number of estimates must match number of confidences")
        
        # Filter out low-confidence estimates
        valid_pairs = [
            (est, conf) for est, conf in zip(estimates, confidences)
            if conf >= self.min_confidence
        ]
        
        if not valid_pairs:
            # Return the best estimate even if below threshold
            best_idx = np.argmax(confidences)
            return estimates[best_idx]
        
        valid_estimates, valid_confidences = zip(*valid_pairs)
        valid_estimates = list(valid_estimates)
        valid_confidences = list(valid_confidences)
        
        # Get source weights
        weights = self._get_source_weights(valid_estimates, valid_confidences)
        
        # Combine weights with confidences
        combined_weights = np.array([
            w * c for w, c in zip(weights, valid_confidences)
        ])
        
        # Normalize weights
        weight_sum = np.sum(combined_weights)
        if weight_sum > 0:
            combined_weights = combined_weights / weight_sum
        else:
            combined_weights = np.ones(len(combined_weights)) / len(combined_weights)
        
        # Fuse 3D gaze vectors
        fused_vector_3d = self._fuse_vectors(
            [est.gaze_vector_3d for est in valid_estimates],
            combined_weights
        )
        
        # Fuse 2D gaze points
        fused_point_2d = self._fuse_points_2d(
            [est.gaze_point_2d for est in valid_estimates],
            combined_weights
        )
        
        # Apply Kalman filtering for temporal consistency
        fused_vector_3d = self.kalman_3d.update(fused_vector_3d)
        fused_point_2d = self.kalman_2d.update(np.array(fused_point_2d))
        
        # Calculate combined confidence
        combined_confidence = self._calculate_combined_confidence(
            valid_confidences, combined_weights
        )
        
        # Aggregate source confidences
        source_confidences = self._aggregate_source_confidences(
            valid_estimates, combined_weights
        )
        
        # Use head pose from highest confidence estimate
        best_idx = np.argmax(valid_confidences)
        head_pose = valid_estimates[best_idx].head_pose
        
        # Aggregate quality metrics
        quality_metrics = self._aggregate_quality_metrics(
            valid_estimates, combined_weights
        )
        
        # Create fused estimate
        fused_estimate = GazeEstimate(
            gaze_vector_3d=fused_vector_3d,
            gaze_point_2d=tuple(fused_point_2d),
            confidence=combined_confidence,
            head_pose=head_pose,
            timestamp=valid_estimates[0].timestamp,
            source_confidences=source_confidences,
            quality_metrics=quality_metrics,
            method="multi_modal_fusion"
        )
        
        # Add to history
        self.history.append(fused_estimate)
        
        return fused_estimate
    
    def update_weights(self, source_reliabilities: Dict[str, float]) -> None:
        """
        Update fusion weights based on source reliability.
        
        Args:
            source_reliabilities: Reliability scores for each source (0-1)
        """
        for source, reliability in source_reliabilities.items():
            if source not in self.source_weights:
                self.source_weights[source] = self.default_weight
            
            # Adaptive weight update
            current_weight = self.source_weights[source]
            target_weight = reliability
            
            # Exponential moving average
            new_weight = (
                (1 - self.adaptation_rate) * current_weight +
                self.adaptation_rate * target_weight
            )
            
            # Clamp to reasonable range
            self.source_weights[source] = np.clip(new_weight, 0.1, 2.0)
    
    def resolve_conflicts(
        self,
        estimates: List[GazeEstimate],
        temporal_history: List[GazeEstimate]
    ) -> GazeEstimate:
        """
        Resolve conflicting predictions using temporal consistency.
        
        Args:
            estimates: Conflicting gaze estimates
            temporal_history: Previous gaze estimates for context
            
        Returns:
            Resolved gaze estimate
        """
        if not estimates:
            raise ValueError("Cannot resolve conflicts with empty list")
        
        if len(estimates) == 1:
            return estimates[0]
        
        # Check if there's actually a conflict
        if not self._has_conflict(estimates):
            # No significant conflict, use standard fusion
            confidences = [est.confidence for est in estimates]
            return self.fuse_estimates(estimates, confidences)
        
        # Use temporal consistency to resolve
        if temporal_history:
            # Find estimate most consistent with history
            consistency_scores = [
                self._calculate_temporal_consistency(est, temporal_history)
                for est in estimates
            ]
            
            # Weight by both confidence and consistency
            combined_scores = [
                est.confidence * consistency
                for est, consistency in zip(estimates, consistency_scores)
            ]
            
            best_idx = np.argmax(combined_scores)
            return estimates[best_idx]
        else:
            # No history, use highest confidence
            confidences = [est.confidence for est in estimates]
            best_idx = np.argmax(confidences)
            return estimates[best_idx]
    
    def _get_source_weights(
        self,
        estimates: List[GazeEstimate],
        confidences: List[float]
    ) -> List[float]:
        """Get adaptive weights for each source."""
        weights = []
        for est in estimates:
            source = est.method if hasattr(est, 'method') else 'unknown'
            weight = self.source_weights.get(source, self.default_weight)
            weights.append(weight)
        return weights
    
    def _fuse_vectors(
        self,
        vectors: List[np.ndarray],
        weights: np.ndarray
    ) -> np.ndarray:
        """Fuse 3D vectors using weighted averaging."""
        # Weighted sum
        fused = np.zeros(3)
        for vec, weight in zip(vectors, weights):
            fused += weight * vec
        
        # Normalize to unit vector
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        
        return fused
    
    def _fuse_points_2d(
        self,
        points: List[tuple],
        weights: np.ndarray
    ) -> np.ndarray:
        """Fuse 2D points using weighted averaging."""
        points_array = np.array(points)
        fused = np.sum(points_array * weights[:, np.newaxis], axis=0)
        return fused
    
    def _calculate_combined_confidence(
        self,
        confidences: List[float],
        weights: np.ndarray
    ) -> float:
        """Calculate combined confidence score."""
        # Weighted average of confidences
        combined = np.sum(np.array(confidences) * weights)
        
        # Boost confidence if multiple sources agree
        agreement_bonus = min(0.1 * (len(confidences) - 1), 0.2)
        combined = min(1.0, combined + agreement_bonus)
        
        return float(combined)
    
    def _aggregate_source_confidences(
        self,
        estimates: List[GazeEstimate],
        weights: np.ndarray
    ) -> Dict[str, float]:
        """Aggregate source confidences."""
        source_conf = {}
        for est, weight in zip(estimates, weights):
            for source, conf in est.source_confidences.items():
                if source not in source_conf:
                    source_conf[source] = 0.0
                source_conf[source] += conf * weight
        return source_conf
    
    def _aggregate_quality_metrics(
        self,
        estimates: List[GazeEstimate],
        weights: np.ndarray
    ) -> QualityMetrics:
        """Aggregate quality metrics using weighted averaging."""
        # Weighted average of each metric
        overall_quality = sum(
            est.quality_metrics.overall_quality * w
            for est, w in zip(estimates, weights)
        )
        head_pose_quality = sum(
            est.quality_metrics.head_pose_quality * w
            for est, w in zip(estimates, weights)
        )
        lighting_quality = sum(
            est.quality_metrics.lighting_quality * w
            for est, w in zip(estimates, weights)
        )
        occlusion_level = sum(
            est.quality_metrics.occlusion_level * w
            for est, w in zip(estimates, weights)
        )
        motion_blur = sum(
            est.quality_metrics.motion_blur * w
            for est, w in zip(estimates, weights)
        )
        tracking_stability = sum(
            est.quality_metrics.tracking_stability * w
            for est, w in zip(estimates, weights)
        )
        
        return QualityMetrics(
            overall_quality=overall_quality,
            head_pose_quality=head_pose_quality,
            lighting_quality=lighting_quality,
            occlusion_level=occlusion_level,
            motion_blur=motion_blur,
            tracking_stability=tracking_stability
        )
    
    def _has_conflict(self, estimates: List[GazeEstimate]) -> bool:
        """Check if estimates have significant conflicts."""
        if len(estimates) < 2:
            return False
        
        # Calculate pairwise angular differences
        vectors = [est.gaze_vector_3d for est in estimates]
        max_diff = 0.0
        
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                # Cosine similarity
                dot_product = np.dot(vectors[i], vectors[j])
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                
                # Normalize to [0, 1]
                normalized_diff = angle / np.pi
                max_diff = max(max_diff, normalized_diff)
        
        return max_diff > self.conflict_threshold
    
    def _calculate_temporal_consistency(
        self,
        estimate: GazeEstimate,
        history: List[GazeEstimate]
    ) -> float:
        """Calculate how consistent an estimate is with temporal history."""
        if not history:
            return 1.0
        
        # Compare with recent history
        recent_history = history[-5:] if len(history) > 5 else history
        
        consistencies = []
        for hist_est in recent_history:
            # Angular difference
            dot_product = np.dot(estimate.gaze_vector_3d, hist_est.gaze_vector_3d)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(dot_product)
            
            # Convert to consistency score (0-1)
            consistency = 1.0 - (angle / np.pi)
            consistencies.append(consistency)
        
        # Average consistency
        return np.mean(consistencies)
