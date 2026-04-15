"""
Focus detection system with 3D ray casting and attention tracking.

This module implements improved focus detection logic that:
- Uses 3D ray casting for object intersection detection
- Validates focus based on stability and minimum duration
- Detects wandering behavior for stable gaze without targets
- Tracks attention shifts and sequences
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import time

from ai_enhanced_gaze_tracking.core.interfaces import FocusDetector
from ai_enhanced_gaze_tracking.core.data_models import (
    FocusEvent, GazeEstimate, AttentionType, DetectedObject
)


@dataclass
class GazeStabilityMetrics:
    """Metrics for gaze stability assessment."""
    variance: float  # Position variance
    duration: float  # Duration of stable gaze
    mean_position: Tuple[float, float]  # Mean gaze position
    is_stable: bool  # Whether gaze is considered stable


class ImprovedFocusDetector(FocusDetector):
    """
    Enhanced focus detector with 3D ray casting and attention tracking.
    
    This implementation addresses the requirements:
    - 6.1: Stability-based focus validation with minimum duration
    - 6.2: Non-object focus rejection
    - 6.3: 3D ray casting for object selection
    - 6.4: Wandering behavior detection
    - 6.5: Attention shift tracking
    """
    
    def __init__(
        self,
        min_focus_duration: float = 0.5,  # Minimum duration for focus (seconds)
        stability_threshold: float = 20.0,  # Maximum variance for stable gaze (pixels)
        wandering_stability_threshold: float = 15.0,  # Threshold for wandering detection
        ray_intersection_threshold: float = 50.0,  # Distance threshold for ray-object intersection
        history_size: int = 30  # Number of frames to keep in history
    ):
        """
        Initialize the focus detector.
        
        Args:
            min_focus_duration: Minimum time gaze must remain stable for focus detection
            stability_threshold: Maximum position variance for stable gaze
            wandering_stability_threshold: Variance threshold for wandering detection
            ray_intersection_threshold: Distance for ray-object intersection
            history_size: Number of frames to maintain in history
        """
        self.min_focus_duration = min_focus_duration
        self.stability_threshold = stability_threshold
        self.wandering_stability_threshold = wandering_stability_threshold
        self.ray_intersection_threshold = ray_intersection_threshold
        self.history_size = history_size
        
        # Tracking state
        self.gaze_history: deque = deque(maxlen=history_size)
        self.current_focus: Optional[FocusEvent] = None
        self.focus_history: List[FocusEvent] = []
        self.last_update_time: float = 0.0
        
        # Track when stable gaze started
        self.stable_gaze_start_time: Optional[float] = None
        self.last_stable_position: Optional[Tuple[float, float]] = None
    
    def detect_focus(
        self,
        gaze_vector: np.ndarray,
        tracked_objects: List[Dict[str, Any]],
        stability_metrics: Dict[str, float]
    ) -> Optional[FocusEvent]:
        """
        Detect focus events from gaze data.
        
        Implements Requirements 6.1, 6.2, 6.3, 6.4.
        
        Args:
            gaze_vector: 3D gaze direction vector
            tracked_objects: List of tracked objects in scene
            stability_metrics: Gaze stability measurements
            
        Returns:
            Focus event if detected, None otherwise
        """
        current_time = time.time()
        
        # Update gaze history
        self.gaze_history.append({
            'vector': gaze_vector,
            'timestamp': current_time,
            'objects': tracked_objects
        })
        
        # Calculate stability from recent history
        stability = self._calculate_stability()
        
        # Check if gaze is stable enough
        if not stability.is_stable:
            # Gaze not stable, reset stable gaze tracking
            self.stable_gaze_start_time = None
            self.last_stable_position = None
            
            # End current focus if any
            if self.current_focus is not None:
                self._end_current_focus(current_time)
            return None
        
        # Gaze is stable - track when stable period started
        if self.stable_gaze_start_time is None:
            self.stable_gaze_start_time = current_time
            self.last_stable_position = stability.mean_position
        
        # Calculate how long gaze has been stable
        stable_duration = current_time - self.stable_gaze_start_time
        
        # Check if stability duration meets minimum requirement (Requirement 6.1)
        # Add small tolerance for timing precision (10ms)
        if stable_duration < (self.min_focus_duration - 0.01):
            # Not stable long enough yet
            return None
        
        # Use 3D ray casting to find intersecting objects (Requirement 6.3)
        intersecting_object = self._ray_cast_to_objects(gaze_vector, tracked_objects)
        
        # Determine focus type
        if intersecting_object is not None:
            # Object focus detected
            focus_type = self._classify_object_type(intersecting_object)
            target_id = intersecting_object.get('id', str(intersecting_object.get('class_name', 'unknown')))
            
            # Check if this is a new focus or continuation
            if self.current_focus is None or self.current_focus.target_object_id != target_id:
                # New focus event
                if self.current_focus is not None:
                    self._end_current_focus(current_time)
                
                self.current_focus = FocusEvent(
                    target_object_id=target_id,
                    focus_type=focus_type,
                    start_time=self.stable_gaze_start_time,
                    duration=stable_duration,
                    stability_score=self._calculate_stability_score(stability),
                    confidence=self._calculate_focus_confidence(stability, intersecting_object),
                    target_bbox=intersecting_object.get('bbox'),
                    gaze_trajectory=[stability.mean_position]
                )
            else:
                # Continue existing focus
                self.current_focus.duration = current_time - self.current_focus.start_time
                self.current_focus.stability_score = self._calculate_stability_score(stability)
            
            return self.current_focus
        
        elif len(tracked_objects) == 0:
            # No objects present - check for wandering (Requirement 6.4)
            if stability.variance <= self.wandering_stability_threshold:
                # Stable gaze without target = wandering
                if self.current_focus is None or self.current_focus.focus_type != AttentionType.WANDERING:
                    if self.current_focus is not None:
                        self._end_current_focus(current_time)
                    
                    self.current_focus = FocusEvent(
                        target_object_id=None,
                        focus_type=AttentionType.WANDERING,
                        start_time=self.stable_gaze_start_time,
                        duration=stable_duration,
                        stability_score=self._calculate_stability_score(stability),
                        confidence=0.8,
                        gaze_trajectory=[stability.mean_position]
                    )
                else:
                    # Continue wandering
                    self.current_focus.duration = current_time - self.current_focus.start_time
                
                return self.current_focus
        
        else:
            # Objects present but no intersection (Requirement 6.2)
            # This is NOT classified as object focus
            if self.current_focus is not None:
                self._end_current_focus(current_time)
            return None
        
        return None
    
    def classify_attention_type(
        self,
        gaze_pattern: List[GazeEstimate],
        scene_objects: List[Dict[str, Any]]
    ) -> str:
        """
        Classify type of attention behavior from gaze pattern.
        
        Args:
            gaze_pattern: Sequence of gaze estimates
            scene_objects: Objects present in scene
            
        Returns:
            Attention type: "object", "person", "wandering", "unknown"
        """
        if not gaze_pattern:
            return AttentionType.UNKNOWN.value
        
        # Analyze the gaze pattern
        has_objects = len(scene_objects) > 0
        
        # Calculate overall stability
        positions = [est.gaze_point_2d for est in gaze_pattern]
        variance = self._calculate_position_variance(positions)
        
        # Check for object intersections
        intersection_count = 0
        person_count = 0
        
        for gaze_est in gaze_pattern:
            intersecting_obj = self._ray_cast_to_objects(
                gaze_est.gaze_vector_3d,
                scene_objects
            )
            if intersecting_obj:
                intersection_count += 1
                if intersecting_obj.get('class_name', '').lower() == 'person':
                    person_count += 1
        
        # Classify based on pattern
        intersection_ratio = intersection_count / len(gaze_pattern) if gaze_pattern else 0
        
        if intersection_ratio > 0.6:
            # Mostly looking at objects
            if person_count > intersection_count * 0.5:
                return AttentionType.PERSON.value
            else:
                return AttentionType.OBJECT.value
        elif variance < self.wandering_stability_threshold and not has_objects:
            # Stable gaze without objects
            return AttentionType.WANDERING.value
        else:
            return AttentionType.UNKNOWN.value
    
    def track_attention_shifts(
        self,
        focus_events: List[FocusEvent]
    ) -> Dict[str, Any]:
        """
        Track sequences and patterns of attention shifts.
        
        Implements Requirement 6.5.
        
        Args:
            focus_events: Sequence of focus events
            
        Returns:
            Analysis of attention shift patterns
        """
        if not focus_events:
            return {
                'total_shifts': 0,
                'average_focus_duration': 0.0,
                'attention_span': 0.0,
                'shift_frequency': 0.0,
                'focus_distribution': {}
            }
        
        # Calculate attention shifts
        shifts = []
        for i in range(1, len(focus_events)):
            if focus_events[i].target_object_id != focus_events[i-1].target_object_id:
                shift_time = focus_events[i].start_time - focus_events[i-1].start_time
                shifts.append({
                    'from': focus_events[i-1].target_object_id,
                    'to': focus_events[i].target_object_id,
                    'time': shift_time
                })
        
        # Calculate statistics
        durations = [event.duration for event in focus_events]
        avg_duration = np.mean(durations) if durations else 0.0
        
        # Calculate attention span (longest continuous focus)
        max_duration = max(durations) if durations else 0.0
        
        # Calculate shift frequency
        total_time = focus_events[-1].start_time + focus_events[-1].duration - focus_events[0].start_time
        shift_frequency = len(shifts) / total_time if total_time > 0 else 0.0
        
        # Focus distribution by type
        focus_distribution = {}
        for event in focus_events:
            focus_type = event.focus_type.value if isinstance(event.focus_type, AttentionType) else event.focus_type
            focus_distribution[focus_type] = focus_distribution.get(focus_type, 0) + 1
        
        return {
            'total_shifts': len(shifts),
            'average_focus_duration': avg_duration,
            'attention_span': max_duration,
            'shift_frequency': shift_frequency,
            'focus_distribution': focus_distribution,
            'shift_sequence': shifts
        }
    
    def _calculate_stability(self) -> GazeStabilityMetrics:
        """
        Calculate gaze stability from recent history.
        
        Returns:
            Stability metrics including variance and duration
        """
        if len(self.gaze_history) < 2:
            return GazeStabilityMetrics(
                variance=float('inf'),
                duration=0.0,
                mean_position=(0.0, 0.0),
                is_stable=False
            )
        
        # Extract 2D gaze points (project 3D vectors to 2D for stability calculation)
        positions = []
        for entry in self.gaze_history:
            # Simple projection: use x and y components of normalized vector
            vec = entry['vector']
            if np.linalg.norm(vec) > 0:
                vec_norm = vec / np.linalg.norm(vec)
                # Project to screen coordinates (simplified)
                x = vec_norm[0] * 320 + 320  # Center at 320
                y = -vec_norm[1] * 240 + 240  # Center at 240, flip y
                positions.append((x, y))
        
        if not positions:
            return GazeStabilityMetrics(
                variance=float('inf'),
                duration=0.0,
                mean_position=(0.0, 0.0),
                is_stable=False
            )
        
        # Calculate variance
        variance = self._calculate_position_variance(positions)
        
        # Calculate duration of stable period
        duration = self.gaze_history[-1]['timestamp'] - self.gaze_history[0]['timestamp']
        
        # Calculate mean position
        mean_x = np.mean([p[0] for p in positions])
        mean_y = np.mean([p[1] for p in positions])
        mean_position = (mean_x, mean_y)
        
        # Determine if stable
        is_stable = variance <= self.stability_threshold
        
        return GazeStabilityMetrics(
            variance=variance,
            duration=duration,
            mean_position=mean_position,
            is_stable=is_stable
        )
    
    def _calculate_position_variance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate variance of 2D positions."""
        if not positions:
            return float('inf')
        
        positions_array = np.array(positions)
        mean_pos = np.mean(positions_array, axis=0)
        distances = np.linalg.norm(positions_array - mean_pos, axis=1)
        variance = np.var(distances)
        
        return float(variance)
    
    def _ray_cast_to_objects(
        self,
        gaze_vector: np.ndarray,
        tracked_objects: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Use 3D ray casting to find object intersecting with gaze ray.
        
        Implements Requirement 6.3.
        
        Args:
            gaze_vector: 3D gaze direction vector
            tracked_objects: List of tracked objects
            
        Returns:
            Intersecting object or None
        """
        if not tracked_objects:
            return None
        
        # Normalize gaze vector
        if np.linalg.norm(gaze_vector) == 0:
            return None
        
        gaze_dir = gaze_vector / np.linalg.norm(gaze_vector)
        
        # Ray origin (camera/eye position)
        ray_origin = np.array([0.0, 0.0, 0.0])
        
        # Find closest intersecting object
        # Primary sort: perpendicular distance to ray (must be within threshold)
        # Secondary sort: projection length along ray (prefer nearer objects)
        closest_object = None
        min_distance = float('inf')
        min_projection = float('inf')

        for obj in tracked_objects:
            # Get object bounding box
            bbox = obj.get('bbox')
            if bbox is None:
                continue

            x, y, w, h = bbox
            center_x = x + w / 2
            center_y = y + h / 2

            # Estimate 3D position (simplified - assumes objects at fixed depth)
            depth = obj.get('depth_estimate', 500.0)  # Default depth

            # Convert 2D screen position to 3D world position (simplified)
            # This is a simplified projection - in practice would use camera parameters
            obj_3d_pos = np.array([
                (center_x - 320) / 320 * depth,  # Normalize and scale by depth
                -(center_y - 240) / 240 * depth,  # Flip y and scale
                depth
            ])

            # Calculate distance from ray to object center
            # Using point-to-line distance formula
            ray_to_obj = obj_3d_pos - ray_origin
            projection_length = np.dot(ray_to_obj, gaze_dir)

            if projection_length < 0:
                # Object is behind the ray origin
                continue

            closest_point_on_ray = ray_origin + projection_length * gaze_dir
            distance = np.linalg.norm(obj_3d_pos - closest_point_on_ray)

            # Check if within intersection threshold
            if distance < self.ray_intersection_threshold:
                # Among intersecting objects, prefer the one closest along the ray
                # (nearest object), breaking ties by perpendicular distance
                if projection_length < min_projection or (
                    projection_length == min_projection and distance < min_distance
                ):
                    min_distance = distance
                    min_projection = projection_length
                    closest_object = obj

        return closest_object
    
    def _classify_object_type(self, obj: Dict[str, Any]) -> AttentionType:
        """Classify object type for focus event."""
        class_name = obj.get('class_name', '').lower()
        
        if 'person' in class_name or 'face' in class_name:
            return AttentionType.PERSON
        else:
            return AttentionType.OBJECT
    
    def _calculate_stability_score(self, stability: GazeStabilityMetrics) -> float:
        """Calculate stability score from metrics."""
        # Lower variance = higher stability
        # Normalize variance to 0-1 range
        max_variance = self.stability_threshold * 2
        normalized_variance = min(stability.variance / max_variance, 1.0)
        stability_score = 1.0 - normalized_variance
        
        return float(stability_score)
    
    def _calculate_focus_confidence(
        self,
        stability: GazeStabilityMetrics,
        obj: Dict[str, Any]
    ) -> float:
        """Calculate confidence for focus detection."""
        # Base confidence from stability
        stability_conf = self._calculate_stability_score(stability)
        
        # Object detection confidence
        obj_conf = obj.get('confidence', 0.5)
        
        # Duration factor (longer = more confident)
        duration_factor = min(stability.duration / (self.min_focus_duration * 2), 1.0)
        
        # Combine factors
        confidence = (stability_conf * 0.4 + obj_conf * 0.4 + duration_factor * 0.2)
        
        return float(confidence)
    
    def _end_current_focus(self, end_time: float) -> None:
        """End the current focus event and add to history."""
        if self.current_focus is not None:
            self.current_focus.duration = end_time - self.current_focus.start_time
            self.focus_history.append(self.current_focus)
            self.current_focus = None
