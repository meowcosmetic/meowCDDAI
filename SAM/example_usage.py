"""
Example usage của SAM + CLIP detector

Ví dụ đơn giản để test nhanh
"""

import sys
from pathlib import Path

# Add parent directory to path để import
sys.path.append(str(Path(__file__).parent.parent))

from test_sam_clip import SAMCLIPDetector, test_sam_clip_detection, detect_all_objects_sam, process_video_sam_clip
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_detection():
    """
    Ví dụ 1: Detect đơn giản với hàm test_sam_clip_detection
    """
    logger.info("=== Example 1: Simple Detection ===")
    
    # Thay đổi paths theo ảnh của bạn
    reference_image = "pen_sample.jpg"  # Ảnh mẫu cây bút
    target_image = "test_image.jpg"     # Ảnh cần detect
    
    if not Path(reference_image).exists() or not Path(target_image).exists():
        logger.warning(f"⚠️  Không tìm thấy ảnh. Vui lòng thay đổi paths trong code.")
        logger.info("   Sử dụng: python example_usage.py --reference <path> --target <path>")
        return
    
    detections = test_sam_clip_detection(
        reference_image_path=reference_image,
        target_image_path=target_image,
        output_path="result_example1.jpg",
        similarity_threshold=0.25
    )
    
    logger.info(f"✅ Found {len(detections)} matches")
    for i, det in enumerate(detections):
        logger.info(f"   {i+1}. Similarity: {det['similarity']:.3f}, Area: {det['area']} pixels")


def example_2_multiple_references():
    """
    Ví dụ 2: Register nhiều reference images và detect từng loại
    """
    logger.info("=== Example 2: Multiple References ===")
    
    detector = SAMCLIPDetector(
        sam_model="sam_b.pt",
        similarity_threshold=0.25
    )
    
    # Register nhiều objects
    references = {
        "pen": "pen_sample.jpg",
        "book": "book_sample.jpg",
        "cup": "cup_sample.jpg"
    }
    
    target_image = "test_image.jpg"
    
    # Check files exist
    if not Path(target_image).exists():
        logger.warning(f"⚠️  Không tìm thấy target image: {target_image}")
        return
    
    for obj_name, ref_path in references.items():
        if Path(ref_path).exists():
            detector.register_reference_image(ref_path, object_name=obj_name)
        else:
            logger.warning(f"⚠️  Không tìm thấy reference: {ref_path}")
    
    # Detect từng loại
    for obj_name in references.keys():
        if obj_name in detector.reference_embeddings:
            logger.info(f"Detecting {obj_name}...")
            detections = detector.detect_objects(
                target_image=target_image,
                reference_name=obj_name
            )
            logger.info(f"   Found {len(detections)} {obj_name}(s)")
            
            # Visualize
            if len(detections) > 0:
                detector.visualize_detections(
                    image=target_image,
                    detections=detections,
                    output_path=f"result_{obj_name}.jpg",
                    show=False
                )


def example_3_video_processing():
    """
    Ví dụ 3: Process video để detect và classify TẤT CẢ objects
    """
    logger.info("=== Example 3: Video Processing ===")
    
    video_path = "test_video.mp4"
    
    if not Path(video_path).exists():
        logger.warning(f"⚠️  Không tìm thấy video: {video_path}")
        logger.info("   Sử dụng: python example_usage.py --example 3 --video <path>")
        return
    
    # Process video với SAM+CLIP
    result = process_video_sam_clip(
        video_path=video_path,
        output_path="result_video.mp4",
        use_fastsam=True,  # Dùng FastSAM để nhanh hơn
        classify_objects=True,  # Phân loại objects
        frame_skip=5,  # Xử lý mỗi 5 frames để tăng tốc độ
        show_video=True,  # Hiển thị video
        save_video=True  # Save video output
    )
    
    logger.info(f"✅ Processed {result['processed_frames']} frames")
    logger.info(f"   Object counts:")
    for class_name, count in sorted(result['object_counts'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"     - {class_name}: {count}")


def example_0_detect_all():
    """
    Ví dụ 0: Detect TẤT CẢ objects trong ảnh (không cần reference image)
    """
    logger.info("=== Example 0: Detect ALL Objects ===")
    
    target_image = "test_image.jpg"
    
    if not Path(target_image).exists():
        logger.warning(f"⚠️  Không tìm thấy ảnh: {target_image}")
        return
    
    # Detect tất cả objects
    detections = detect_all_objects_sam(
        target_image_path=target_image,
        output_path="result_all_objects.jpg",
        use_fastsam=True,  # Dùng FastSAM để nhanh hơn
        min_area=1000,  # Chỉ lấy objects có diện tích >= 1000 pixels
        max_objects=20   # Tối đa 20 objects
    )
    
    logger.info(f"✅ Found {len(detections)} objects")
    for i, det in enumerate(detections):
        logger.info(f"   {i+1}. Area: {det['area']} pixels, BBox: {det['bbox']}")


def example_4_custom_threshold():
    """
    Ví dụ 4: Điều chỉnh threshold để tăng/giảm độ nhạy
    """
    logger.info("=== Example 4: Custom Threshold ===")
    
    reference_image = "pen_sample.jpg"
    target_image = "test_image.jpg"
    
    if not Path(reference_image).exists() or not Path(target_image).exists():
        logger.warning("⚠️  Không tìm thấy ảnh")
        return
    
    # Test với các threshold khác nhau
    thresholds = [0.15, 0.25, 0.35]
    
    for threshold in thresholds:
        logger.info(f"Testing với threshold: {threshold}")
        detections = test_sam_clip_detection(
            reference_image_path=reference_image,
            target_image_path=target_image,
            output_path=f"result_threshold_{threshold}.jpg",
            similarity_threshold=threshold
        )
        logger.info(f"   Found {len(detections)} matches với threshold {threshold}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM + CLIP Example Usage")
    parser.add_argument("--reference", type=str, help="Path to reference image")
    parser.add_argument("--target", type=str, help="Path to target image")
    parser.add_argument("--example", type=int, default=0, choices=[0, 1, 2, 3, 4],
                       help="Example number to run (0-4, 0=detect all)")
    parser.add_argument("--video", type=str, help="Path to video file (for example 3)")
    
    args = parser.parse_args()
    
    # Override paths nếu có
    if args.reference and args.target:
        import test_sam_clip
        test_sam_clip.test_sam_clip_detection(
            reference_image_path=args.reference,
            target_image_path=args.target,
            output_path="result.jpg"
        )
    else:
        # Run examples
        examples = {
            0: example_0_detect_all,
            1: example_1_simple_detection,
            2: example_2_multiple_references,
            3: example_3_video_processing,
            4: example_4_custom_threshold
        }
        
        # Override video path nếu có
        if args.video and args.example == 3:
            import test_sam_clip
            result = test_sam_clip.process_video_sam_clip(
                video_path=args.video,
                output_path="result_video.mp4",
                use_fastsam=True,
                classify_objects=True,
                frame_skip=5
            )
            logger.info(f"✅ Processed {result['processed_frames']} frames")
        elif args.example in examples:
            examples[args.example]()
        else:
            logger.info("Available examples:")
            logger.info("  0. Detect ALL objects (không cần reference)")
            logger.info("  1. Simple detection (với reference)")
            logger.info("  2. Multiple references")
            logger.info("  3. Video processing")
            logger.info("  4. Custom threshold")
            logger.info("\nUsage: python example_usage.py --example <0-4>")
            logger.info("Or: python example_usage.py --reference <path> --target <path>")
            logger.info("Or: python example_usage.py --target <path> --all (detect all)")

