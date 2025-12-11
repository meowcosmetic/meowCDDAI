"""
3D Gaze Estimation - Tính toán chính xác hướng nhìn với head pose estimation
"""
import logging
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class GazeEstimator3D:
    """3D Gaze Estimation với head pose và ray casting"""
    
    def __init__(self, image_width: int = 640, image_height: int = 480):
        """
        Args:
            image_width: Width của frame
            image_height: Height của frame
        """
        self.image_width = image_width
        self.image_height = image_height
        
        # Camera intrinsic parameters (có thể calibrate sau)
        self.camera_matrix = self._get_camera_matrix(image_width, image_height)
        self.dist_coeffs = np.zeros((4, 1))  # Giả định không có distortion
        
        # 3D face model points (dựa trên MediaPipe Face Mesh)
        self.model_points_3d = self._get_3d_face_model()
        
        # MediaPipe landmark indices
        self.LEFT_PUPIL_IDX = 468
        self.RIGHT_PUPIL_IDX = 473
        self.NOSE_TIP_IDX = 4
        self.CHIN_IDX = 175
        self.LEFT_EYE_CORNER_IDX = 33
        self.RIGHT_EYE_CORNER_IDX = 362
    
    def _get_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """
        Tạo camera matrix với giả định reasonable focal length
        """
        focal_length = width
        center_x = width / 2.0
        center_y = height / 2.0
        
        return np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _get_3d_face_model(self) -> np.ndarray:
        """
        3D face model points (tọa độ thực tế, đơn vị: mm hoặc normalized)
        Dựa trên average face dimensions
        """
        # MediaPipe face mesh 3D coordinates (normalized)
        # Chuyển sang metric space (giả định face width ~140mm)
        face_width_mm = 140.0
        
        return np.array([
            # Nose tip
            [0.0, 0.0, 0.0],
            # Chin
            [0.0, -33.0, -65.0],
            # Left eye corner
            [-43.0, 32.5, -26.0],
            # Right eye corner
            [43.0, 32.5, -26.0],
            # Left mouth corner
            [-28.0, -28.0, -5.0],
            # Right mouth corner
            [28.0, -28.0, -5.0],
        ], dtype=np.float32) * (face_width_mm / 100.0)
    
    def estimate_head_pose(self, face_landmarks) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate head pose (6DoF) sử dụng solvePnP
        
        Args:
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            (success, rotation_vec, translation_vec)
        """
        try:
            # Extract 2D image points từ landmarks
            image_points = self._get_2d_landmarks(face_landmarks)
            
            if image_points is None or len(image_points) < 4:
                return False, None, None
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points_3d,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            return success, rotation_vec, translation_vec
        except Exception as e:
            logger.warning(f"[Gaze3D] Head pose estimation error: {str(e)}")
            return False, None, None
    
    def _get_2d_landmarks(self, face_landmarks) -> Optional[np.ndarray]:
        """
        Extract 2D image points từ MediaPipe landmarks
        """
        try:
            landmark = face_landmarks.landmark
            
            # Map 3D model points to 2D image points
            # Nose tip
            nose_tip = landmark[self.NOSE_TIP_IDX]
            # Chin
            chin = landmark[self.CHIN_IDX]
            # Left eye corner
            left_eye = landmark[self.LEFT_EYE_CORNER_IDX]
            # Right eye corner
            right_eye = landmark[self.RIGHT_EYE_CORNER_IDX]
            # Left mouth corner (approximate)
            left_mouth = landmark[61]  # MediaPipe mouth corner
            # Right mouth corner
            right_mouth = landmark[291]
            
            image_points = np.array([
                [nose_tip.x * self.image_width, nose_tip.y * self.image_height],
                [chin.x * self.image_width, chin.y * self.image_height],
                [left_eye.x * self.image_width, left_eye.y * self.image_height],
                [right_eye.x * self.image_width, right_eye.y * self.image_height],
                [left_mouth.x * self.image_width, left_mouth.y * self.image_height],
                [right_mouth.x * self.image_width, right_mouth.y * self.image_height],
            ], dtype=np.float32)
            
            return image_points
        except Exception as e:
            logger.warning(f"[Gaze3D] Landmark extraction error: {str(e)}")
            return None
    
    def calculate_eye_direction(self, face_landmarks) -> Optional[np.ndarray]:
        """
        Tính eye gaze direction trong head coordinate system
        
        Args:
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            Eye direction vector (normalized) trong head coordinate
        """
        try:
            landmark = face_landmarks.landmark
            
            # Get pupil positions
            left_pupil = landmark[self.LEFT_PUPIL_IDX]
            right_pupil = landmark[self.RIGHT_PUPIL_IDX]
            
            # Get eye corners for reference
            left_eye_corner = landmark[self.LEFT_EYE_CORNER_IDX]
            right_eye_corner = landmark[self.RIGHT_EYE_CORNER_IDX]
            
            # Calculate eye center
            left_eye_center_x = (left_eye_corner.x + landmark[7].x) / 2
            left_eye_center_y = (left_eye_corner.y + landmark[7].y) / 2
            
            right_eye_center_x = (right_eye_corner.x + landmark[263].x) / 2
            right_eye_center_y = (right_eye_corner.y + landmark[263].y) / 2
            
            # Calculate pupil offset from eye center (normalized)
            left_offset_x = left_pupil.x - left_eye_center_x
            left_offset_y = left_pupil.y - left_eye_center_y
            
            right_offset_x = right_pupil.x - right_eye_center_x
            right_offset_y = right_pupil.y - right_eye_center_y
            
            # Average offset
            avg_offset_x = (left_offset_x + right_offset_x) / 2
            avg_offset_y = (left_offset_y + right_offset_y) / 2
            
            # Convert to 3D direction (giả định depth = 1)
            # Scale based on eye size
            eye_width = abs(left_eye_corner.x - right_eye_corner.x) * self.image_width
            scale_factor = eye_width / 60.0  # Normalize
            
            eye_direction = np.array([
                avg_offset_x * scale_factor,
                avg_offset_y * scale_factor,
                1.0  # Forward direction
            ], dtype=np.float32)
            
            # Normalize
            norm = np.linalg.norm(eye_direction)
            if norm > 0:
                eye_direction = eye_direction / norm
            
            return eye_direction
        except Exception as e:
            logger.warning(f"[Gaze3D] Eye direction calculation error: {str(e)}")
            return None
    
    def estimate_3d_gaze(self, 
                        face_landmarks,
                        tracked_objects: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
        """
        Estimate 3D gaze và tìm object đang được nhìn
        
        Args:
            face_landmarks: MediaPipe face landmarks
            tracked_objects: List of tracked objects với bbox
        
        Returns:
            (object_id, confidence_score)
            object_id: "class_track_id" hoặc None
            confidence: 0.0 - 1.0
        """
        try:
            # 1. Head Pose Estimation
            success, rotation_vec, translation_vec = self.estimate_head_pose(face_landmarks)
            
            if not success or rotation_vec is None:
                return None, 0.0
            
            # 2. Eye Gaze Direction trong head coordinate
            eye_direction_head = self.calculate_eye_direction(face_landmarks)
            
            if eye_direction_head is None:
                return None, 0.0
            
            # 3. Transform to world coordinate
            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
            gaze_direction_world = rotation_matrix @ eye_direction_head
            
            # 4. Gaze origin (face position in 3D)
            gaze_origin_3d = translation_vec.flatten()
            
            # 5. Ray casting - tìm object intersect với gaze ray
            best_object = None
            best_confidence = 0.0
            min_distance = float('inf')
            
            for obj in tracked_objects:
                bbox = obj.get('bbox', [])
                if len(bbox) < 4:
                    continue
                
                # Convert 2D bbox to 3D (giả định depth)
                obj_3d_info = self._bbox_to_3d(bbox, translation_vec[2, 0])
                
                # Check ray intersection
                intersects, distance = self._ray_intersects_bbox(
                    gaze_origin_3d,
                    gaze_direction_world,
                    obj_3d_info
                )
                
                if intersects:
                    # Calculate confidence dựa trên distance
                    confidence = self._calculate_confidence(distance, obj_3d_info['size'])
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_object = obj
                        min_distance = distance
            
            if best_object:
                # Tạo object_id
                class_name = best_object.get('class', 'unknown')
                track_id = best_object.get('track_id')
                
                if track_id is not None:
                    object_id = f"{class_name}_{track_id}"
                else:
                    object_id = f"{class_name}_unknown"
                
                return object_id, best_confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.warning(f"[Gaze3D] 3D gaze estimation error: {str(e)}")
            return None, 0.0
    
    def _bbox_to_3d(self, bbox: List[float], depth: float) -> Dict[str, Any]:
        """
        Convert 2D bbox sang 3D bounding box
        
        Args:
            bbox: [x, y, w, h]
            depth: Estimated depth (z coordinate)
        
        Returns:
            Dict với 3D bbox info
        """
        x, y, w, h = bbox
        
        # Project 2D to 3D
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Unproject to 3D
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 3D center
        center_3d = np.array([
            (center_x - cx) * depth / fx,
            (center_y - cy) * depth / fy,
            depth
        ], dtype=np.float32)
        
        # 3D size (approximate)
        size_3d = np.array([
            w * depth / fx,
            h * depth / fy,
            depth * 0.1  # Giả định object depth
        ], dtype=np.float32)
        
        return {
            'center': center_3d,
            'size': size_3d,
            'bbox_3d': {
                'min': center_3d - size_3d / 2,
                'max': center_3d + size_3d / 2
            }
        }
    
    def _ray_intersects_bbox(self,
                            ray_origin: np.ndarray,
                            ray_direction: np.ndarray,
                            bbox_3d: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Kiểm tra xem ray có intersect với 3D bounding box không
        
        Args:
            ray_origin: 3D point (gaze origin)
            ray_direction: 3D direction vector (normalized)
            bbox_3d: 3D bounding box info
        
        Returns:
            (intersects, distance)
        """
        try:
            bbox_min = bbox_3d['bbox_3d']['min']
            bbox_max = bbox_3d['bbox_3d']['max']
            
            # Ray-box intersection algorithm (slab method)
            t_min = 0.0
            t_max = float('inf')
            
            for i in range(3):
                if abs(ray_direction[i]) < 1e-6:
                    # Ray parallel to plane
                    if ray_origin[i] < bbox_min[i] or ray_origin[i] > bbox_max[i]:
                        return False, float('inf')
                else:
                    inv_dir = 1.0 / ray_direction[i]
                    t1 = (bbox_min[i] - ray_origin[i]) * inv_dir
                    t2 = (bbox_max[i] - ray_origin[i]) * inv_dir
                    
                    if t1 > t2:
                        t1, t2 = t2, t1
                    
                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)
                    
                    if t_min > t_max:
                        return False, float('inf')
            
            if t_min > 0:
                # Intersection point
                intersection = ray_origin + ray_direction * t_min
                distance = np.linalg.norm(intersection - ray_origin)
                return True, distance
            
            return False, float('inf')
            
        except Exception as e:
            logger.warning(f"[Gaze3D] Ray intersection error: {str(e)}")
            return False, float('inf')
    
    def _calculate_confidence(self, distance: float, object_size: np.ndarray) -> float:
        """
        Tính confidence score dựa trên distance và object size
        
        Args:
            distance: Distance từ gaze origin đến object
            object_size: Size của object trong 3D
        
        Returns:
            Confidence score (0.0 - 1.0)
        """
        # Object size (diagonal)
        obj_diagonal = np.linalg.norm(object_size)
        
        # Normalize distance by object size
        normalized_distance = distance / (obj_diagonal + 1e-6)
        
        # Confidence: closer = higher confidence
        # Exponential decay
        confidence = np.exp(-normalized_distance * 2.0)
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return float(confidence)
    
    def update_camera_params(self, width: int, height: int) -> None:
        """Update camera parameters khi frame size thay đổi"""
        self.image_width = width
        self.image_height = height
        self.camera_matrix = self._get_camera_matrix(width, height)

