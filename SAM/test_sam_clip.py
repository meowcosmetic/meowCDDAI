"""
SAM + CLIP Object Detection Test
Sử dụng Segment Anything Model (SAM) + CLIP để detect objects dựa trên image embedding

Cách hoạt động:
1. Load ảnh mẫu (reference image) của object cần tìm (ví dụ: cây bút)
2. Tính CLIP embedding của ảnh mẫu
3. Segment ảnh target với SAM để tìm tất cả các objects
4. Tính CLIP embedding của mỗi segment
5. So sánh similarity giữa reference embedding và segment embeddings
6. Return các segments có similarity cao (match với object mẫu)

Ưu điểm:
- Detect được objects ngay cả khi bị che khuất một phần
- Không cần dataset training
- Chỉ cần ảnh mẫu của object cần tìm
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Lazy imports
try:
    from ultralytics import SAM, FastSAM
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("[SAM+CLIP] Ultralytics không được cài đặt. Cài: pip install ultralytics>=8.0.0")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("[SAM+CLIP] CLIP không được cài đặt. Cài: pip install git+https://github.com/openai/CLIP.git")

try:
    from ultralytics.nn.text_model import CLIP as UltralyticsCLIP
    ULTRALYTICS_CLIP_AVAILABLE = True
except ImportError:
    ULTRALYTICS_CLIP_AVAILABLE = False


class SAMCLIPDetector:
    """
    SAM + CLIP Object Detector
    
    Detect objects trong ảnh bằng cách:
    1. Segment với SAM
    2. Match với reference image embedding bằng CLIP
    """
    
    def __init__(
        self,
        sam_model: str = "sam_b.pt",
        clip_model: str = "ViT-B/32",
        use_fastsam: bool = False,
        device: Optional[str] = None,
        similarity_threshold: float = 0.25
    ):
        """
        Args:
            sam_model: SAM model name hoặc path (sam_b.pt, sam_l.pt, sam_x.pt)
            clip_model: CLIP model name (ViT-B/32, ViT-L/14, etc.)
            use_fastsam: Sử dụng FastSAM thay vì SAM (nhanh hơn nhưng kém chính xác hơn)
            device: Device để chạy model ('cuda', 'cpu', hoặc None để auto-detect)
            similarity_threshold: Ngưỡng similarity để coi là match (0-1)
        """
        self.sam_model_name = sam_model
        self.clip_model_name = clip_model
        self.use_fastsam = use_fastsam
        self.similarity_threshold = similarity_threshold
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"[SAM+CLIP] 🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.info(f"[SAM+CLIP] ⚠️  GPU không available, sử dụng CPU (sẽ chậm hơn)")
        
        logger.info(f"[SAM+CLIP] Initializing on device: {self.device}")
        
        # Initialize SAM
        self.sam = None
        if SAM_AVAILABLE:
            try:
                # Ultralytics SAM/FastSAM tự động detect GPU, nhưng có thể chỉ định device
                # Thử set device nếu có GPU
                if use_fastsam:
                    # FastSAM có thể nhận device parameter
                    try:
                        self.sam = FastSAM(sam_model)
                        # Set device nếu có thể
                        if hasattr(self.sam, 'to') and self.device != "cpu":
                            try:
                                self.sam.to(self.device)
                            except:
                                pass
                    except:
                        self.sam = FastSAM(sam_model)
                    
                    # Check SAM device
                    sam_device = "cpu"
                    if hasattr(self.sam, 'device'):
                        sam_device = str(self.sam.device)
                    elif hasattr(self.sam, 'model'):
                        if hasattr(self.sam.model, 'device'):
                            sam_device = str(self.sam.model.device)
                        elif hasattr(self.sam.model, 'parameters'):
                            sam_device = str(next(self.sam.model.parameters()).device)
                    logger.info(f"[SAM+CLIP] ✅ FastSAM loaded: {sam_model}")
                    logger.info(f"[SAM+CLIP]    Device: {sam_device}")
                else:
                    self.sam = SAM(sam_model)
                    # Set device nếu có thể
                    if hasattr(self.sam, 'to') and self.device != "cpu":
                        try:
                            self.sam.to(self.device)
                        except:
                            pass
                    
                    # Check SAM device
                    sam_device = "cpu"
                    if hasattr(self.sam, 'device'):
                        sam_device = str(self.sam.device)
                    elif hasattr(self.sam, 'model'):
                        if hasattr(self.sam.model, 'device'):
                            sam_device = str(self.sam.model.device)
                        elif hasattr(self.sam.model, 'parameters'):
                            sam_device = str(next(self.sam.model.parameters()).device)
                    logger.info(f"[SAM+CLIP] ✅ SAM loaded: {sam_model}")
                    logger.info(f"[SAM+CLIP]    Device: {sam_device}")
            except Exception as e:
                logger.error(f"[SAM+CLIP] ❌ Không thể load SAM model: {str(e)}")
                raise
        
        # Initialize CLIP
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_type = None  # 'openai' or 'ultralytics'
        
        # Thử load Ultralytics CLIP trước (thường có sẵn với ultralytics)
        if ULTRALYTICS_CLIP_AVAILABLE:
            try:
                self.clip_model = UltralyticsCLIP(clip_model, device=self.device)
                self.clip_type = 'ultralytics'
                logger.info(f"[SAM+CLIP] ✅ Ultralytics CLIP loaded: {clip_model} (device: {self.device})")
            except Exception as e:
                logger.warning(f"[SAM+CLIP] Không thể load Ultralytics CLIP: {str(e)}")
        
        # Nếu Ultralytics CLIP không available, thử OpenAI CLIP
        if self.clip_model is None and CLIP_AVAILABLE:
            try:
                self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
                self.clip_model.eval()
                self.clip_type = 'openai'
                # Check CLIP device
                clip_device = next(self.clip_model.parameters()).device if hasattr(self.clip_model, 'parameters') else self.device
                logger.info(f"[SAM+CLIP] ✅ OpenAI CLIP loaded: {clip_model} (device: {clip_device})")
            except Exception as e:
                logger.warning(f"[SAM+CLIP] Không thể load OpenAI CLIP: {str(e)}")
        
        # Nếu cả hai đều không available
        if self.clip_model is None:
            logger.error("[SAM+CLIP] ❌ CLIP không available!")
            logger.error("[SAM+CLIP]    Để sử dụng classification, cài đặt một trong các cách sau:")
            logger.error("[SAM+CLIP]    1. pip install git+https://github.com/openai/CLIP.git")
            logger.error("[SAM+CLIP]    2. Hoặc đảm bảo ultralytics>=8.0.0 đã được cài đặt")
            logger.warning("[SAM+CLIP]    Classification sẽ bị tắt, chỉ có thể detect objects không có tên")
        
        # Cache để lưu reference embeddings
        self.reference_embeddings: Dict[str, torch.Tensor] = {}
        
        # Common object classes để phân loại
        self.common_classes = [
            "pen", "pencil", "book", "cup", "bottle", "glass", "bowl", "plate",
            "phone", "cell phone", "mobile phone", "laptop", "computer", "tablet",
            "mouse", "keyboard", "remote control",
            "toy", "doll", "teddy bear", "ball", "building blocks",
            "chair", "table", "desk", "bed", "couch", "sofa",
            "scissors", "knife", "fork", "spoon",
            "clock", "watch", "vase", "flower", "plant",
            "apple", "banana", "orange", "sandwich", "pizza", "food",
            "shirt", "pants", "shoe", "hat", "dress",
            "car", "bicycle", "motorcycle", "vehicle",
            "dog", "cat", "bird", "horse", "animal",
            "person", "child", "adult", "baby",
            "hand", "finger", "arm", "leg"
        ]
    
    def _load_image(self, image_path: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Image.Image]:
        """
        Load image từ path hoặc numpy array
        
        Returns:
            (numpy_image, PIL_image)
        """
        if isinstance(image_path, (str, Path)):
            # Convert to Path object
            image_path_orig = Path(image_path)
            image_path = image_path_orig
            
            # Nếu path không tồn tại và là relative path, thử tìm trong các thư mục có thể
            if not image_path.exists() and not image_path.is_absolute():
                # Danh sách các thư mục để thử tìm
                search_dirs = []
                
                # 1. Thư mục "test SAM" (từ vị trí file hiện tại)
                try:
                    if '__file__' in globals():
                        test_sam_dir = Path(__file__).parent
                        search_dirs.append(test_sam_dir)
                except:
                    pass
                
                # 2. Thư mục "test SAM" từ working directory
                test_sam_dir_cwd = Path.cwd() / "test SAM"
                if test_sam_dir_cwd.exists():
                    search_dirs.append(test_sam_dir_cwd)
                
                # 3. Thư mục hiện tại
                search_dirs.append(Path.cwd())
                
                # Thử tìm trong các thư mục
                found = False
                for search_dir in search_dirs:
                    test_path = search_dir / image_path_orig
                    if test_path.exists():
                        image_path = test_path
                        logger.info(f"[SAM+CLIP] Found image: {image_path}")
                        found = True
                        break
                
                if not found:
                    searched_paths = [str(d / image_path_orig) for d in search_dirs]
                    raise FileNotFoundError(
                        f"Không tìm thấy ảnh: {image_path_orig}\n"
                        f"Đã thử tìm trong:\n" + "\n".join(f"  - {p}" for p in searched_paths)
                    )
            
            # Load image
            numpy_image = cv2.imread(str(image_path))
            if numpy_image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")
            pil_image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            numpy_image = image_path
            if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
                # BGR to RGB
                pil_image = Image.fromarray(cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(numpy_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
        
        return numpy_image, pil_image
    
    def compute_clip_embedding(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Tính CLIP embedding của ảnh
        
        Args:
            image: Image path, numpy array, hoặc PIL Image
        
        Returns:
            CLIP embedding tensor (normalized)
        """
        # Convert to PIL Image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Compute embedding
        if hasattr(self.clip_model, 'encode_image'):
            # Ultralytics CLIP
            image_tensor = self.clip_model.image_preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_tensor)
        else:
            # OpenAI CLIP
            image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_tensor)
        
        # Normalize embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze(0)
    
    def register_reference_image(
        self,
        image_path: Union[str, Path, np.ndarray, Image.Image],
        object_name: str = "object"
    ) -> None:
        """
        Đăng ký ảnh mẫu (reference image) để tìm kiếm
        
        Args:
            image_path: Path đến ảnh mẫu hoặc image array
            object_name: Tên object (để cache embedding)
        """
        logger.info(f"[SAM+CLIP] Registering reference image: {object_name}")
        embedding = self.compute_clip_embedding(image_path)
        self.reference_embeddings[object_name] = embedding
        logger.info(f"[SAM+CLIP] ✅ Reference embedding computed for: {object_name}")
    
    def segment_image(self, image: Union[str, Path, np.ndarray]) -> List[Dict]:
        """
        Segment ảnh với SAM để tìm tất cả các objects
        
        Args:
            image: Image path hoặc numpy array
        
        Returns:
            List of segments với format:
            {
                'mask': np.ndarray,  # Binary mask
                'bbox': [x, y, w, h],  # Bounding box
                'area': int,  # Pixel area
                'confidence': float  # Confidence score
            }
        """
        if self.sam is None:
            raise RuntimeError("SAM model chưa được khởi tạo")
        
        # Load image
        numpy_image, pil_image = self._load_image(image)
        
        # Run SAM segmentation
        # FastSAM tự động segment tất cả objects
        # SAM cần prompts, nhưng có thể dùng auto-annotation
        if self.use_fastsam:
            # FastSAM: predict without prompts để segment tất cả
            results = self.sam.predict(pil_image, imgsz=1024)
        else:
            # SAM: cần prompts, nhưng có thể dùng grid points
            # Tạo grid points để segment nhiều objects
            h, w = numpy_image.shape[:2]
            grid_points = []
            step = 100  # Grid step size
            for y in range(step, h, step):
                for x in range(step, w, step):
                    grid_points.append([x, y])
            
            results = self.sam.predict(pil_image, points=grid_points, imgsz=1024)
        
        # Extract segments
        segments = []
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy() if hasattr(result.masks, 'data') else result.masks
                
                for i, mask in enumerate(masks):
                    # Convert mask to binary
                    if mask.dtype != np.uint8:
                        mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Get bounding box
                    y_indices, x_indices = np.where(mask > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min, x_max = int(x_indices.min()), int(x_indices.max())
                        y_min, y_max = int(y_indices.min()), int(y_indices.max())
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        
                        # Calculate area
                        area = int(np.sum(mask > 0))
                        
                        # Filter out very small segments (noise)
                        min_area = (h * w) * 0.001  # Tối thiểu 0.1% diện tích ảnh
                        if area >= min_area:
                            segments.append({
                                'mask': mask,
                                'bbox': bbox,
                                'area': area,
                                'confidence': 1.0,  # SAM không có confidence score
                                'segment_id': i
                            })
        
        logger.info(f"[SAM+CLIP] Segmented {len(segments)} objects")
        return segments
    
    def classify_object(
        self,
        image_crop: Union[np.ndarray, Image.Image],
        class_names: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Phân loại object trong image crop bằng CLIP
        
        Args:
            image_crop: Cropped image của object
            class_names: List các class names để phân loại, None = dùng common_classes
        
        Returns:
            (class_name, confidence): Tên class và confidence score
        """
        if self.clip_model is None:
            logger.warning("[SAM+CLIP] CLIP model is None, cannot classify")
            return "unknown", 0.0
        
        if class_names is None:
            class_names = self.common_classes
        
        try:
            # Convert to PIL Image nếu cần
            if isinstance(image_crop, np.ndarray):
                # Đảm bảo có kích thước hợp lệ
                if image_crop.size == 0 or len(image_crop.shape) < 2:
                    return "unknown", 0.0
                
                if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image_crop)
            else:
                pil_image = image_crop.convert("RGB")
            
            # Resize nếu quá nhỏ hoặc quá lớn
            min_size = 32
            max_size = 512
            w, h = pil_image.size
            if min(w, h) < min_size:
                scale = min_size / min(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            elif max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Compute image embedding
            image_embedding = self.compute_clip_embedding(pil_image)
            
            # Compute text embeddings cho các classes
            if self.clip_type == 'ultralytics':
                # Ultralytics CLIP
                text_tokens = self.clip_model.tokenize(class_names)
                text_embeddings = self.clip_model.encode_text(text_tokens)
            elif self.clip_type == 'openai' and CLIP_AVAILABLE:
                # OpenAI CLIP
                text_tokens = clip.tokenize(class_names).to(self.device)
                with torch.no_grad():
                    text_embeddings = self.clip_model.encode_text(text_tokens)
            else:
                logger.error("[SAM+CLIP] CLIP không available để encode text")
                return "unknown", 0.0
            
            # Normalize text embeddings
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(
                image_embedding.unsqueeze(0),
                text_embeddings
            )
            
            # Find best match
            best_idx = similarities.argmax().item()
            best_class = class_names[best_idx]
            best_confidence = float(similarities[best_idx].item())
            
            # Chỉ return nếu confidence đủ cao
            if best_confidence < 0.15:  # Threshold tối thiểu
                return "unknown", best_confidence
            
            return best_class, best_confidence
            
        except Exception as e:
            logger.warning(f"[SAM+CLIP] Error classifying object: {str(e)}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0
    
    def detect_all_objects(
        self,
        target_image: Union[str, Path, np.ndarray],
        min_area: Optional[int] = None,
        max_objects: Optional[int] = None,
        classify_objects: bool = True,
        custom_classes: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Detect TẤT CẢ objects trong ảnh (không cần reference image)
        
        Args:
            target_image: Ảnh cần detect
            min_area: Diện tích tối thiểu của object (pixels), None = auto
            max_objects: Số lượng objects tối đa để return, None = không giới hạn
            classify_objects: Có phân loại và đặt tên objects không
            custom_classes: List các class names tùy chỉnh để phân loại
        
        Returns:
            List of detected objects với format:
            {
                'mask': np.ndarray,
                'bbox': [x, y, w, h],
                'area': int,
                'confidence': float,
                'segment_id': int,
                'class_name': str,  # Tên object (nếu classify_objects=True)
                'class_confidence': float  # Confidence của classification (nếu classify_objects=True)
            }
        """
        # Segment tất cả objects
        segments = self.segment_image(target_image)
        
        if len(segments) == 0:
            logger.warning("[SAM+CLIP] Không tìm thấy objects nào")
            return []
        
        # Filter by min_area nếu có
        if min_area is not None:
            segments = [s for s in segments if s['area'] >= min_area]
        
        # Sort by area (largest first)
        segments.sort(key=lambda x: x['area'], reverse=True)
        
        # Limit số lượng nếu có
        if max_objects is not None:
            segments = segments[:max_objects]
        
        # Classify objects nếu được yêu cầu
        if classify_objects:
            if self.clip_model is None:
                logger.warning("[SAM+CLIP] CLIP model không available, không thể phân loại objects")
                logger.warning("[SAM+CLIP] Set classify_objects=False hoặc cài đặt CLIP")
                for segment in segments:
                    segment['class_name'] = "object"
                    segment['class_confidence'] = 0.0
            else:
                # Load image để crop segments
                numpy_image, _ = self._load_image(target_image)
                
                logger.info(f"[SAM+CLIP] Classifying {len(segments)} objects...")
                classified_count = 0
                for i, segment in enumerate(segments):
                    try:
                        # Crop segment từ image
                        x, y, w, h = segment['bbox']
                        x_max = min(x + w, numpy_image.shape[1])
                        y_max = min(y + h, numpy_image.shape[0])
                        
                        # Đảm bảo có kích thước hợp lệ
                        if x_max <= x or y_max <= y:
                            segment['class_name'] = "unknown"
                            segment['class_confidence'] = 0.0
                            continue
                        
                        segment_crop = numpy_image[y:y_max, x:x_max]
                        
                        if segment_crop.size == 0:
                            segment['class_name'] = "unknown"
                            segment['class_confidence'] = 0.0
                            continue
                        
                        # Classify object
                        class_name, class_confidence = self.classify_object(
                            segment_crop,
                            class_names=custom_classes
                        )
                        
                        segment['class_name'] = class_name
                        segment['class_confidence'] = class_confidence
                        
                        if class_name != "unknown":
                            classified_count += 1
                            logger.debug(f"  Object {i+1}: {class_name} (confidence: {class_confidence:.2f})")
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"  Classified {i + 1}/{len(segments)} objects...")
                            
                    except Exception as e:
                        logger.warning(f"[SAM+CLIP] Error classifying segment {i}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        segment['class_name'] = "unknown"
                        segment['class_confidence'] = 0.0
                
                logger.info(f"[SAM+CLIP] ✅ Successfully classified {classified_count}/{len(segments)} objects")
        else:
            # Không classify, set default values
            logger.info("[SAM+CLIP] Classification disabled")
            for segment in segments:
                segment['class_name'] = "object"
                segment['class_confidence'] = 0.0
        
        logger.info(f"[SAM+CLIP] Detected {len(segments)} objects (all objects mode)")
        
        # Log summary của classification
        if classify_objects:
            class_counts = {}
            for seg in segments:
                class_name = seg.get('class_name', 'unknown')
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            logger.info(f"[SAM+CLIP] Classification summary:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {class_name}: {count}")
        
        return segments
    
    def detect_objects(
        self,
        target_image: Union[str, Path, np.ndarray],
        reference_image: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        reference_name: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect objects trong target image dựa trên reference image
        
        Args:
            target_image: Ảnh cần detect objects
            reference_image: Ảnh mẫu của object cần tìm (nếu chưa register)
            reference_name: Tên reference đã register (nếu đã register trước)
            similarity_threshold: Ngưỡng similarity (override default)
        
        Returns:
            List of detected objects với format:
            {
                'mask': np.ndarray,
                'bbox': [x, y, w, h],
                'area': int,
                'similarity': float,  # Similarity với reference
                'confidence': float
            }
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        # Get reference embedding
        if reference_name and reference_name in self.reference_embeddings:
            reference_embedding = self.reference_embeddings[reference_name]
        elif reference_image is not None:
            reference_embedding = self.compute_clip_embedding(reference_image)
        else:
            raise ValueError("Cần cung cấp reference_image hoặc reference_name đã register")
        
        # Segment target image
        segments = self.segment_image(target_image)
        
        if len(segments) == 0:
            logger.warning("[SAM+CLIP] Không tìm thấy segments nào")
            return []
        
        # Load target image để crop segments
        numpy_image, pil_image = self._load_image(target_image)
        
        # Compute embeddings cho mỗi segment và match với reference
        detected_objects = []
        for segment in segments:
            try:
                # Crop segment từ image
                x, y, w, h = segment['bbox']
                x_max = min(x + w, numpy_image.shape[1])
                y_max = min(y + h, numpy_image.shape[0])
                
                segment_crop = numpy_image[y:y_max, x:x_max]
                
                if segment_crop.size == 0:
                    continue
                
                # Compute CLIP embedding của segment
                segment_embedding = self.compute_clip_embedding(segment_crop)
                
                # Calculate cosine similarity
                similarity = float(torch.cosine_similarity(
                    reference_embedding.unsqueeze(0),
                    segment_embedding.unsqueeze(0)
                ).item())
                
                # Filter by threshold
                if similarity >= similarity_threshold:
                    detected_objects.append({
                        'mask': segment['mask'],
                        'bbox': segment['bbox'],
                        'area': segment['area'],
                        'similarity': similarity,
                        'confidence': similarity,  # Use similarity as confidence
                        'segment_id': segment.get('segment_id', 0)
                    })
            
            except Exception as e:
                logger.warning(f"[SAM+CLIP] Error processing segment: {str(e)}")
                continue
        
        # Sort by similarity (highest first)
        detected_objects.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"[SAM+CLIP] Detected {len(detected_objects)} objects với similarity >= {similarity_threshold}")
        
        return detected_objects
    
    def visualize_detections(
        self,
        image: Union[str, Path, np.ndarray],
        detections: List[Dict],
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        show_similarity: bool = True
    ) -> np.ndarray:
        """
        Vẽ detections lên ảnh
        
        Args:
            image: Original image
            detections: List of detections từ detect_objects() hoặc detect_all_objects()
            output_path: Path để save ảnh (optional)
            show: Có hiển thị ảnh không
            show_similarity: Có hiển thị similarity score không (chỉ khi có)
        
        Returns:
            Annotated image (numpy array)
        """
        numpy_image, _ = self._load_image(image)
        annotated = numpy_image.copy()
        
        # Generate colors for different objects
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw each detection
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            color = colors[i % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw mask overlay (semi-transparent)
            mask = det['mask']
            if mask.shape[:2] == annotated.shape[:2]:
                mask_resized = cv2.resize(mask, (w, h))
                mask_binary = (mask_resized > 0).astype(np.uint8)
                overlay = annotated.copy()
                overlay[y:y+h, x:x+w][mask_binary > 0] = color
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
            
            # Draw label - ưu tiên hiển thị class_name
            label = f"Object {i+1}"
            if 'class_name' in det:
                class_name = det['class_name']
                class_conf = det.get('class_confidence', 0.0)
                if class_name != "unknown" and class_name != "object" and class_conf > 0.15:
                    # Hiển thị tên class nếu có và confidence đủ cao
                    label = f"{class_name} ({class_conf:.2f})"
                elif show_similarity and 'similarity' in det:
                    label = f"Obj {i+1}: {det['similarity']:.2f}"
            elif show_similarity and 'similarity' in det:
                label = f"Obj {i+1}: {det['similarity']:.2f}"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x, y - text_height - 10), (x + text_width, y), color, -1)
            cv2.putText(annotated, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), annotated)
            logger.info(f"[SAM+CLIP] Saved visualization to: {output_path}")
        
        # Show if requested
        if show:
            cv2.imshow("SAM+CLIP Detections", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        classify_objects: bool = True,
        custom_classes: Optional[List[str]] = None,
        frame_skip: int = 1,
        show_video: bool = True,
        save_video: bool = True
    ) -> Dict:
        """
        Xử lý video để detect và classify objects trong từng frame
        
        Args:
            video_path: Path đến video file
            output_path: Path để save video output (optional)
            classify_objects: Có phân loại objects không
            custom_classes: List các class names tùy chỉnh
            frame_skip: Xử lý mỗi N frames (1 = tất cả frames, 2 = mỗi 2 frames, ...)
            show_video: Có hiển thị video trong quá trình xử lý không
            save_video: Có save video output không
        
        Returns:
            Dict với thông tin:
            {
                'total_frames': int,
                'processed_frames': int,
                'detections_per_frame': List[List[Dict]],
                'object_counts': Dict[str, int],  # Số lượng mỗi loại object
                'output_path': str
            }
        """
        import cv2
        
        # Load video
        video_path = Path(video_path)
        if not video_path.exists():
            # Thử tìm trong test SAM directory
            test_sam_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / "test SAM"
            test_path = test_sam_dir / video_path
            if test_path.exists():
                video_path = test_path
            else:
                raise FileNotFoundError(f"Không tìm thấy video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"[SAM+CLIP] Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer nếu cần save
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            logger.info(f"[SAM+CLIP] Will save output to: {output_path}")
        
        # Statistics
        frame_count = 0
        processed_count = 0
        detections_per_frame = []
        object_counts = {}
        
        logger.info(f"[SAM+CLIP] Bắt đầu xử lý video (frame_skip={frame_skip})...")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames nếu cần
                if frame_count % frame_skip != 0:
                    # Vẫn save frame gốc nếu không process
                    if video_writer:
                        video_writer.write(frame)
                    if show_video:
                        cv2.imshow("Video Processing", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue
                
                processed_count += 1
                
                # Detect objects trong frame
                detections = self.detect_all_objects(
                    target_image=frame,
                    classify_objects=classify_objects,
                    custom_classes=custom_classes,
                    min_area=1000,  # Filter small objects
                    max_objects=50  # Limit số lượng objects
                )
                
                # Update statistics
                detections_per_frame.append(detections)
                for det in detections:
                    class_name = det.get('class_name', 'unknown')
                    if class_name not in object_counts:
                        object_counts[class_name] = 0
                    object_counts[class_name] += 1
                
                # Visualize detections
                annotated_frame = self.visualize_detections(
                    image=frame,
                    detections=detections,
                    show=False,
                    show_similarity=False
                )
                
                # Save frame
                if video_writer:
                    video_writer.write(annotated_frame)
                
                # Show frame
                if show_video:
                    cv2.imshow("Video Processing", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        logger.info("[SAM+CLIP] Người dùng dừng xử lý")
                        break
                
                # Log progress
                if processed_count % 30 == 0:
                    logger.info(f"[SAM+CLIP] Processed {processed_count} frames ({frame_count}/{total_frames})...")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if show_video:
                cv2.destroyAllWindows()
        
        logger.info(f"[SAM+CLIP] ✅ Hoàn thành xử lý video")
        logger.info(f"[SAM+CLIP]    Processed {processed_count}/{total_frames} frames")
        logger.info(f"[SAM+CLIP]    Object counts:")
        for class_name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"      - {class_name}: {count}")
        
        return {
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'detections_per_frame': detections_per_frame,
            'object_counts': object_counts,
            'output_path': str(output_path) if output_path else None
        }


def process_video_sam_clip(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sam_model: str = "sam_b.pt",
    use_fastsam: bool = True,
    classify_objects: bool = True,
    custom_classes: Optional[List[str]] = None,
    frame_skip: int = 5,
    show_video: bool = True,
    save_video: bool = True,
    device: Optional[str] = None
) -> Dict:
    """
    Xử lý video để detect và classify objects
    
    Args:
        video_path: Path đến video file
        output_path: Path để save video output
        sam_model: SAM model name
        use_fastsam: Sử dụng FastSAM (nhanh hơn)
        classify_objects: Có phân loại objects không
        custom_classes: List các class names tùy chỉnh
        frame_skip: Xử lý mỗi N frames (1 = tất cả, 5 = mỗi 5 frames)
        show_video: Có hiển thị video không
        save_video: Có save video output không
        device: Device ('cuda', 'cpu', hoặc None)
    
    Returns:
        Dict với thông tin xử lý
    
    Example:
        >>> result = process_video_sam_clip(
        ...     video_path="test_video.mp4",
        ...     output_path="result.mp4",
        ...     frame_skip=5
        ... )
        >>> print(f"Processed {result['processed_frames']} frames")
    """
    # Initialize detector
    detector = SAMCLIPDetector(
        sam_model=sam_model,
        use_fastsam=use_fastsam,
        device=device
    )
    
    # Process video
    result = detector.process_video(
        video_path=video_path,
        output_path=output_path,
        classify_objects=classify_objects,
        custom_classes=custom_classes,
        frame_skip=frame_skip,
        show_video=show_video,
        save_video=save_video
    )
    
    return result


def detect_all_objects_sam(
    target_image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sam_model: str = "sam_b.pt",
    use_fastsam: bool = False,
    min_area: Optional[int] = None,
    max_objects: Optional[int] = None,
    classify_objects: bool = True,
    custom_classes: Optional[List[str]] = None,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Detect TẤT CẢ objects trong ảnh với SAM (không cần reference image)
    
    Args:
        target_image_path: Path đến ảnh cần detect
        output_path: Path để save ảnh kết quả (optional)
        sam_model: SAM model name
        use_fastsam: Sử dụng FastSAM
        min_area: Diện tích tối thiểu (pixels)
        max_objects: Số lượng objects tối đa
        classify_objects: Có phân loại và đặt tên objects không
        custom_classes: List các class names tùy chỉnh
        device: Device ('cuda', 'cpu', hoặc None)
    
    Returns:
        List of detected objects với class_name và class_confidence
    
    Example:
        >>> detections = detect_all_objects_sam(
        ...     target_image_path="test_image.jpg",
        ...     output_path="result.jpg",
        ...     classify_objects=True
        ... )
        >>> print(f"Found {len(detections)} objects")
        >>> for det in detections:
        ...     print(f"  - {det['class_name']}: {det['class_confidence']:.2f}")
    """
    # Initialize detector
    detector = SAMCLIPDetector(
        sam_model=sam_model,
        use_fastsam=use_fastsam,
        device=device
    )
    
    # Detect all objects
    detections = detector.detect_all_objects(
        target_image=target_image_path,
        min_area=min_area,
        max_objects=max_objects,
        classify_objects=classify_objects,
        custom_classes=custom_classes
    )
    
    # Visualize
    if len(detections) > 0:
        detector.visualize_detections(
            image=target_image_path,
            detections=detections,
            output_path=output_path,
            show=True
        )
    else:
        logger.warning("[SAM+CLIP] Không tìm thấy objects nào")
    
    return detections


def test_sam_clip_detection(
    reference_image_path: Union[str, Path],
    target_image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sam_model: str = "sam_b.pt",
    use_fastsam: bool = False,
    similarity_threshold: float = 0.25,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Hàm test đơn giản để detect objects với SAM + CLIP
    
    Args:
        reference_image_path: Path đến ảnh mẫu (ví dụ: ảnh cây bút)
        target_image_path: Path đến ảnh cần detect
        output_path: Path để save ảnh kết quả (optional)
        sam_model: SAM model name
        use_fastsam: Sử dụng FastSAM
        similarity_threshold: Ngưỡng similarity
        device: Device ('cuda', 'cpu', hoặc None)
    
    Returns:
        List of detected objects
    
    Example:
        >>> detections = test_sam_clip_detection(
        ...     reference_image_path="pen_sample.jpg",
        ...     target_image_path="test_image.jpg",
        ...     output_path="result.jpg"
        ... )
        >>> print(f"Found {len(detections)} matches")
    """
    # Initialize detector
    detector = SAMCLIPDetector(
        sam_model=sam_model,
        use_fastsam=use_fastsam,
        similarity_threshold=similarity_threshold,
        device=device
    )
    
    # Detect objects
    detections = detector.detect_objects(
        target_image=target_image_path,
        reference_image=reference_image_path
    )
    
    # Visualize
    if len(detections) > 0:
        detector.visualize_detections(
            image=target_image_path,
            detections=detections,
            output_path=output_path,
            show=True
        )
    else:
        logger.warning("[SAM+CLIP] Không tìm thấy objects nào")
    
    return detections


if __name__ == "__main__":
    """
    Script để test trực tiếp
    
    Usage:
        python test_sam_clip.py --reference pen_sample.jpg --target test_image.jpg --output result.jpg
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM + CLIP Object Detection")
    parser.add_argument("--reference", type=str, default=None, help="Path to reference image (optional, nếu không có sẽ detect tất cả)")
    parser.add_argument("--target", type=str, required=True, help="Path to target image")
    parser.add_argument("--output", type=str, default=None, help="Path to save result image")
    parser.add_argument("--sam-model", type=str, default="sam_b.pt", help="SAM model name")
    parser.add_argument("--fastsam", action="store_true", help="Use FastSAM instead of SAM")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold (chỉ dùng khi có --reference)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum area in pixels (chỉ dùng khi detect all)")
    parser.add_argument("--max-objects", type=int, default=None, help="Maximum number of objects (chỉ dùng khi detect all)")
    parser.add_argument("--all", action="store_true", help="Detect all objects (không cần reference image)")
    parser.add_argument("--no-classify", action="store_true", help="Không phân loại objects (nhanh hơn nhưng không có tên)")
    parser.add_argument("--classes", type=str, default=None, help="Custom classes (comma-separated), ví dụ: 'pen,book,cup'")
    parser.add_argument("--video", action="store_true", help="Xử lý video thay vì ảnh")
    parser.add_argument("--frame-skip", type=int, default=5, help="Xử lý mỗi N frames khi xử lý video (default: 5)")
    parser.add_argument("--no-show", action="store_true", help="Không hiển thị video trong quá trình xử lý")
    parser.add_argument("--no-save", action="store_true", help="Không save video output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if processing video
    if args.video:
        logger.info("🎥 Processing VIDEO mode")
        result = process_video_sam_clip(
            video_path=args.target,
            output_path=args.output,
            sam_model=args.sam_model,
            use_fastsam=args.fastsam,
            classify_objects=not args.no_classify,
            custom_classes=[c.strip() for c in args.classes.split(',')] if args.classes else None,
            frame_skip=args.frame_skip,
            show_video=not args.no_show,
            save_video=not args.no_save,
            device=args.device
        )
        
        print(f"\n✅ Video processing completed!")
        print(f"  Processed: {result['processed_frames']}/{result['total_frames']} frames")
        print(f"  Object counts:")
        for class_name, count in sorted(result['object_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {class_name}: {count}")
        if result['output_path']:
            print(f"  Output saved to: {result['output_path']}")
        import sys
        sys.exit(0)
    
    # Run test for images
    if args.all or args.reference is None:
        # Detect all objects mode
        logger.info("🔍 Detecting ALL objects (no reference image needed)")
        
        # Parse custom classes nếu có
        custom_classes = None
        if args.classes:
            custom_classes = [c.strip() for c in args.classes.split(',')]
            logger.info(f"Using custom classes: {custom_classes}")
        
        detections = detect_all_objects_sam(
            target_image_path=args.target,
            output_path=args.output,
            sam_model=args.sam_model,
            use_fastsam=args.fastsam,
            min_area=args.min_area,
            max_objects=args.max_objects,
            classify_objects=not args.no_classify,
            custom_classes=custom_classes,
            device=args.device
        )
        
        print(f"\n✅ Detected {len(detections)} objects")
        for i, det in enumerate(detections):
            if 'class_name' in det and det['class_name'] != "unknown" and det['class_name'] != "object":
                print(f"  {i+1}. {det['class_name']} (confidence: {det.get('class_confidence', 0):.2f}), Area: {det['area']} pixels")
            else:
                print(f"  {i+1}. Area: {det['area']} pixels, BBox: {det['bbox']}")
    else:
        # Reference-based detection mode
        logger.info("🔍 Detecting objects matching reference image")
        detections = test_sam_clip_detection(
            reference_image_path=args.reference,
            target_image_path=args.target,
            output_path=args.output,
            sam_model=args.sam_model,
            use_fastsam=args.fastsam,
            similarity_threshold=args.threshold,
            device=args.device
        )
        
        print(f"\n✅ Detected {len(detections)} objects")
        for i, det in enumerate(detections):
            print(f"  {i+1}. Similarity: {det['similarity']:.3f}, Area: {det['area']} pixels")

