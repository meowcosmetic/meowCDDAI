"""
Personal calibration system for automatic parameter adaptation.

This module implements automatic calibration that adapts to individual children,
stores child-specific parameters, and adjusts to environmental conditions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import time

from ...core.interfaces import CalibrationSystem
from ...core.data_models import FaceDetection, GazeEstimate, HeadPose


logger = logging.getLogger(__name__)


@dataclass
class PersonalCalibrationProfile:
    """Personal calibration profile for an individual child."""
    profile_id: str  # Unique identifier for this profile
    face_characteristics: Dict[str, float] = field(default_factory=dict)  # Face measurements
    gaze_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))  # Personal gaze offset
    head_pose_bias: Dict[str, float] = field(default_factory=dict)  # Head pose biases
    calibration_points: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # Reference points
    accuracy_improvement: float = 0.0  # Measured accuracy improvement
    last_updated: float = field(default_factory=time.time)  # Last update timestamp
    session_count: int = 0  # Number of sessions with this profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        data['gaze_offset'] = self.gaze_offset.tolist()
        data['calibration_points'] = [
            (img_pts.tolist(), world_pts.tolist()) 
            for img_pts, world_pts in self.calibration_points
        ]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalCalibrationProfile':
        """Create profile from dictionary."""
        # Convert lists back to numpy arrays
        data['gaze_offset'] = np.array(data['gaze_offset'])
        data['calibration_points'] = [
            (np.array(img_pts), np.array(world_pts))
            for img_pts, world_pts in data['calibration_points']
        ]
        return cls(**data)


class PersonalCalibrationSystem(CalibrationSystem):
    """
    Automatic calibration system with personal adaptation.
    
    This system provides:
    - Automatic face characteristic detection and parameter adaptation
    - Reference point calibration using known gaze targets
    - Child-specific parameter storage and retrieval
    - Real-time environmental adaptation
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the personal calibration system.
        
        Args:
            storage_path: Path to store calibration profiles (default: ./calibration_profiles)
        """
        self.storage_path = Path(storage_path) if storage_path else Path("./calibration_profiles")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.profiles: Dict[str, PersonalCalibrationProfile] = {}
        self.current_profile: Optional[PersonalCalibrationProfile] = None
        self.baseline_accuracy = 5.0  # Baseline accuracy in degrees
        
        # Environmental adaptation parameters
        self.environmental_factors = {
            'lighting_level': 1.0,
            'distance_factor': 1.0,
            'angle_factor': 1.0,
            'motion_factor': 1.0
        }
        
        # Load existing profiles
        self._load_profiles()
        
    def auto_calibrate(
        self,
        face_characteristics: Dict[str, Any],
        reference_observations: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Automatically calibrate system parameters for an individual.
        
        Args:
            face_characteristics: Detected face characteristics (IPD, face size, etc.)
            reference_observations: List of (gaze_point, target_point) pairs
            
        Returns:
            Calibrated parameters including offsets and accuracy improvement
        """
        try:
            # Extract or create profile ID from face characteristics
            profile_id = self._generate_profile_id(face_characteristics)
            
            # Get or create profile
            if profile_id in self.profiles:
                profile = self.profiles[profile_id]
                logger.info(f"Using existing profile: {profile_id}")
            else:
                profile = PersonalCalibrationProfile(profile_id=profile_id)
                logger.info(f"Creating new profile: {profile_id}")
            
            # Update face characteristics
            profile.face_characteristics = self._extract_face_characteristics(face_characteristics)
            
            # Calculate gaze offset from reference observations
            if reference_observations and len(reference_observations) >= 3:
                gaze_offset = self._calculate_gaze_offset(reference_observations)
                
                # Measure accuracy improvement
                accuracy_improvement = self._measure_accuracy_improvement(
                    reference_observations, gaze_offset
                )
                
                # Only apply calibration if it improves accuracy
                if accuracy_improvement > 0:
                    profile.gaze_offset = gaze_offset
                    profile.calibration_points = reference_observations
                    profile.accuracy_improvement = accuracy_improvement
                    logger.info(f"Calibration improved accuracy by {accuracy_improvement:.2f} degrees")
                else:
                    # Don't apply calibration that makes things worse
                    profile.gaze_offset = np.zeros(2)
                    profile.accuracy_improvement = 0.0
                    logger.info("Calibration did not improve accuracy, using zero offset")
            else:
                logger.warning("Insufficient reference observations for calibration")
                profile.accuracy_improvement = 0.0
            
            # Calculate head pose biases
            profile.head_pose_bias = self._calculate_head_pose_bias(face_characteristics)
            
            # Update profile metadata
            profile.last_updated = time.time()
            profile.session_count += 1
            
            # Store profile
            self.profiles[profile_id] = profile
            self.current_profile = profile
            self._save_profile(profile)
            
            # Return calibrated parameters
            return {
                'profile_id': profile_id,
                'gaze_offset': profile.gaze_offset,
                'head_pose_bias': profile.head_pose_bias,
                'accuracy_improvement': profile.accuracy_improvement,
                'face_characteristics': profile.face_characteristics,
                'calibration_quality': self._assess_calibration_quality(profile)
            }
            
        except Exception as e:
            logger.error(f"Auto-calibration failed: {e}")
            return {
                'profile_id': 'default',
                'gaze_offset': np.zeros(2),
                'head_pose_bias': {},
                'accuracy_improvement': 0.0,
                'face_characteristics': {},
                'calibration_quality': 0.0
            }
    
    def adapt_to_environment(
        self,
        environmental_conditions: Dict[str, Any]
    ) -> None:
        """
        Adapt calibration parameters to environmental conditions.
        
        Args:
            environmental_conditions: Current environment state (lighting, distance, etc.)
        """
        try:
            # Update lighting factor
            if 'lighting_quality' in environmental_conditions:
                lighting = environmental_conditions['lighting_quality']
                self.environmental_factors['lighting_level'] = max(0.5, min(1.5, lighting))
            
            # Update distance factor
            if 'face_distance' in environmental_conditions:
                distance = environmental_conditions['face_distance']
                # Normalize distance (assume optimal distance is 50-70cm)
                optimal_distance = 60.0
                distance_ratio = distance / optimal_distance if distance > 0 else 1.0
                self.environmental_factors['distance_factor'] = 1.0 / max(0.5, min(2.0, distance_ratio))
            
            # Update angle factor
            if 'camera_angle' in environmental_conditions:
                angle = abs(environmental_conditions['camera_angle'])
                # Reduce confidence for larger angles
                self.environmental_factors['angle_factor'] = max(0.5, 1.0 - angle / 60.0)
            
            # Update motion factor
            if 'motion_level' in environmental_conditions:
                motion = environmental_conditions['motion_level']
                self.environmental_factors['motion_factor'] = max(0.5, 1.0 - motion)
            
            logger.debug(f"Environmental factors updated: {self.environmental_factors}")
            
        except Exception as e:
            logger.error(f"Environmental adaptation failed: {e}")
    
    def apply_calibration(
        self,
        gaze_estimate: GazeEstimate,
        profile_id: Optional[str] = None
    ) -> GazeEstimate:
        """
        Apply personal calibration to a gaze estimate.
        
        Args:
            gaze_estimate: Raw gaze estimate
            profile_id: Profile to use (uses current if None)
            
        Returns:
            Calibrated gaze estimate
        """
        # Get profile
        if profile_id and profile_id in self.profiles:
            profile = self.profiles[profile_id]
        elif self.current_profile:
            profile = self.current_profile
        else:
            # No calibration available
            return gaze_estimate
        
        # Apply gaze offset
        calibrated_point = (
            gaze_estimate.gaze_point_2d[0] + profile.gaze_offset[0],
            gaze_estimate.gaze_point_2d[1] + profile.gaze_offset[1]
        )
        
        # Apply head pose bias correction
        if gaze_estimate.head_pose and profile.head_pose_bias:
            # Adjust for head pose biases
            yaw_bias = profile.head_pose_bias.get('yaw', 0.0)
            pitch_bias = profile.head_pose_bias.get('pitch', 0.0)
            
            # Simple linear correction (can be made more sophisticated)
            yaw_correction = yaw_bias * gaze_estimate.head_pose.yaw
            pitch_correction = pitch_bias * gaze_estimate.head_pose.pitch
            
            calibrated_point = (
                calibrated_point[0] + yaw_correction,
                calibrated_point[1] + pitch_correction
            )
        
        # Apply environmental factors
        env_factor = np.mean(list(self.environmental_factors.values()))
        adjusted_confidence = gaze_estimate.confidence * env_factor
        
        # Create calibrated estimate
        calibrated_estimate = GazeEstimate(
            gaze_vector_3d=gaze_estimate.gaze_vector_3d,
            gaze_point_2d=calibrated_point,
            confidence=adjusted_confidence,
            head_pose=gaze_estimate.head_pose,
            timestamp=gaze_estimate.timestamp,
            source_confidences=gaze_estimate.source_confidences,
            quality_metrics=gaze_estimate.quality_metrics,
            method=f"{gaze_estimate.method}_calibrated",
            raw_features=gaze_estimate.raw_features
        )
        
        return calibrated_estimate
    
    def get_profile(self, profile_id: str) -> Optional[PersonalCalibrationProfile]:
        """Get a calibration profile by ID."""
        return self.profiles.get(profile_id)
    
    def list_profiles(self) -> List[str]:
        """List all available profile IDs."""
        return list(self.profiles.keys())
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a calibration profile."""
        if profile_id in self.profiles:
            del self.profiles[profile_id]
            profile_file = self.storage_path / f"{profile_id}.json"
            if profile_file.exists():
                profile_file.unlink()
            logger.info(f"Deleted profile: {profile_id}")
            return True
        return False
    
    def _generate_profile_id(self, face_characteristics: Dict[str, Any]) -> str:
        """Generate a unique profile ID from face characteristics."""
        # Use face measurements to create a consistent ID
        # In a real system, this might use face recognition
        ipd = face_characteristics.get('interpupillary_distance', 0)
        face_width = face_characteristics.get('face_width', 0)
        face_height = face_characteristics.get('face_height', 0)
        
        # Create a simple hash-based ID
        id_string = f"{ipd:.2f}_{face_width:.2f}_{face_height:.2f}"
        profile_id = f"profile_{hash(id_string) % 10000:04d}"
        
        return profile_id
    
    def _extract_face_characteristics(self, face_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant face characteristics for calibration."""
        characteristics = {}
        
        # Extract interpupillary distance (IPD)
        if 'interpupillary_distance' in face_data:
            characteristics['ipd'] = float(face_data['interpupillary_distance'])
        
        # Extract face dimensions
        if 'face_width' in face_data:
            characteristics['face_width'] = float(face_data['face_width'])
        if 'face_height' in face_data:
            characteristics['face_height'] = float(face_data['face_height'])
        
        # Extract eye characteristics
        if 'eye_aspect_ratio' in face_data:
            characteristics['eye_aspect_ratio'] = float(face_data['eye_aspect_ratio'])
        
        # Extract age-related characteristics (for child vs adult)
        if 'is_child' in face_data:
            characteristics['is_child'] = 1.0 if face_data['is_child'] else 0.0
        
        return characteristics
    
    def _calculate_gaze_offset(
        self,
        reference_observations: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Calculate personal gaze offset from reference observations using weighted mean.
        
        Points are weighted by their consistency with the overall error pattern,
        making the calibration robust to outliers while still being sensitive to
        systematic errors.
        
        Args:
            reference_observations: List of (gaze_point, target_point) pairs
            
        Returns:
            2D gaze offset vector
        """
        if not reference_observations:
            return np.zeros(2)
        
        offsets = []
        for gaze_point, target_point in reference_observations:
            # Ensure points are 2D
            if gaze_point.size >= 2 and target_point.size >= 2:
                gaze_2d = gaze_point.flatten()[:2]
                target_2d = target_point.flatten()[:2]
                offset = target_2d - gaze_2d
                offsets.append(offset)
        
        if not offsets:
            return np.zeros(2)
        
        offsets_array = np.array(offsets)
        
        # Calculate initial mean offset
        mean_offset = np.mean(offsets_array, axis=0)
        
        # Calculate consistency weights based on distance from mean
        # Points closer to the mean get higher weight
        distances = np.linalg.norm(offsets_array - mean_offset, axis=1)
        
        # Use inverse distance for weighting (with small epsilon to avoid division by zero)
        # Points with zero distance get maximum weight
        epsilon = 0.1
        weights = 1.0 / (distances + epsilon)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate weighted mean offset
        weighted_offset = np.average(offsets_array, axis=0, weights=weights)
        
        return weighted_offset
    
    def _calculate_head_pose_bias(self, face_characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate head pose biases based on face characteristics."""
        biases = {}
        
        # Children may have different head pose patterns
        is_child = face_characteristics.get('is_child', False)
        
        if is_child:
            # Children tend to look slightly upward at screens
            biases['pitch'] = 0.1  # Small upward bias correction
            biases['yaw'] = 0.0
            biases['roll'] = 0.0
        else:
            # Adults typically have neutral biases
            biases['pitch'] = 0.0
            biases['yaw'] = 0.0
            biases['roll'] = 0.0
        
        return biases
    
    def _measure_accuracy_improvement(
        self,
        reference_observations: List[Tuple[np.ndarray, np.ndarray]],
        gaze_offset: np.ndarray
    ) -> float:
        """
        Measure accuracy improvement from calibration.
        
        Args:
            reference_observations: Reference gaze-target pairs
            gaze_offset: Calculated gaze offset
            
        Returns:
            Accuracy improvement in degrees
        """
        if not reference_observations:
            return 0.0
        
        errors_before = []
        errors_after = []
        
        for gaze_point, target_point in reference_observations:
            if gaze_point.size >= 2 and target_point.size >= 2:
                gaze_2d = gaze_point.flatten()[:2]
                target_2d = target_point.flatten()[:2]
                
                # Error before calibration
                error_before = np.linalg.norm(target_2d - gaze_2d)
                errors_before.append(error_before)
                
                # Error after calibration
                calibrated_gaze = gaze_2d + gaze_offset
                error_after = np.linalg.norm(target_2d - calibrated_gaze)
                errors_after.append(error_after)
        
        if errors_before and errors_after:
            mean_error_before = np.mean(errors_before)
            mean_error_after = np.mean(errors_after)
            improvement = mean_error_before - mean_error_after
            return float(improvement)
        else:
            return 0.0
    
    def _assess_calibration_quality(self, profile: PersonalCalibrationProfile) -> float:
        """Assess the quality of a calibration profile."""
        quality_factors = []
        
        # Factor 1: Number of calibration points
        num_points = len(profile.calibration_points)
        point_quality = min(1.0, num_points / 9.0)  # 9 points is ideal
        quality_factors.append(point_quality)
        
        # Factor 2: Accuracy improvement
        if profile.accuracy_improvement > 0:
            improvement_quality = min(1.0, profile.accuracy_improvement / self.baseline_accuracy)
            quality_factors.append(improvement_quality)
        else:
            quality_factors.append(0.0)
        
        # Factor 3: Session count (more sessions = more reliable)
        session_quality = min(1.0, profile.session_count / 5.0)
        quality_factors.append(session_quality)
        
        # Factor 4: Recency (recent calibrations are more reliable)
        time_since_update = time.time() - profile.last_updated
        days_since_update = time_since_update / (24 * 3600)
        recency_quality = max(0.0, 1.0 - days_since_update / 30.0)  # Decay over 30 days
        quality_factors.append(recency_quality)
        
        # Overall quality is weighted average
        weights = [0.3, 0.4, 0.2, 0.1]  # Prioritize accuracy improvement
        overall_quality = sum(q * w for q, w in zip(quality_factors, weights))
        
        return overall_quality
    
    def _save_profile(self, profile: PersonalCalibrationProfile) -> None:
        """Save a calibration profile to disk."""
        try:
            profile_file = self.storage_path / f"{profile.profile_id}.json"
            with open(profile_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            logger.debug(f"Saved profile: {profile.profile_id}")
        except Exception as e:
            logger.error(f"Failed to save profile {profile.profile_id}: {e}")
    
    def _load_profiles(self) -> None:
        """Load all calibration profiles from disk."""
        try:
            for profile_file in self.storage_path.glob("*.json"):
                try:
                    with open(profile_file, 'r') as f:
                        data = json.load(f)
                    profile = PersonalCalibrationProfile.from_dict(data)
                    self.profiles[profile.profile_id] = profile
                    logger.debug(f"Loaded profile: {profile.profile_id}")
                except Exception as e:
                    logger.error(f"Failed to load profile {profile_file}: {e}")
            
            logger.info(f"Loaded {len(self.profiles)} calibration profiles")
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
