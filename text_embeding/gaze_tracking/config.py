"""
Configuration và Constants cho Gaze Tracking

File này chứa tất cả các tham số cấu hình cho hệ thống Gaze Tracking.
Các giá trị này có thể được điều chỉnh để tối ưu hóa độ chính xác và hiệu suất.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class GazeConfig:
    """
    Configuration class cho Gaze Tracking - loại bỏ magic numbers
    
    Tất cả các tham số quan trọng được tập trung ở đây để dễ dàng điều chỉnh
    mà không cần sửa code ở nhiều nơi.
    """
    
    # ========================================================================
    # GAZE ANALYSIS THRESHOLDS - Ngưỡng phân tích hướng nhìn
    # ========================================================================
    
    MIN_FOCUSING_DURATION: float = 5.0
    """
    Thời gian tối thiểu (giây) để coi là trẻ đang "focusing" vào một đối tượng.
    
    Giải thích:
    - Nếu trẻ nhìn vào một object trong thời gian >= giá trị này → được tính là "focusing"
    - Giá trị cao hơn → chỉ tính những lúc focus lâu, bỏ qua những cái nhìn ngắn
    - Giá trị thấp hơn → nhạy cảm hơn, tính cả những cái nhìn ngắn
    
    Ví dụ:
    - 0.5 giây: Nhạy cảm, tính cả những cái nhìn rất ngắn
    - 2.0 giây: Vừa phải, chỉ tính những cái nhìn có chủ ý
    - 5.0 giây: Nghiêm ngặt, chỉ tính những lúc focus lâu (phù hợp cho screening ASD)
    
    Khuyến nghị: 3-5 giây cho screening ASD (cần focus lâu mới có ý nghĩa)
    """
    
    # ========================================================================
    # IMPROVED GAZE STABILITY - Cải thiện tính toán gaze stability
    # ========================================================================
    
    GAZE_STABILITY_USE_IMPROVED: bool = True
    """
    Có sử dụng improved gaze stability calculation không.
    
    Improved calculation bao gồm:
    - Normalization by interocular distance
    - Head motion compensation
    - Outlier removal & smoothing
    - RMS distance metric (dễ hiểu hơn variance)
    - Adaptive threshold
    
    Khuyến nghị: True (bật để có kết quả chính xác hơn)
    """
    
    GAZE_STABILITY_WINDOW_MS: float = 200.0
    """
    Window size tính bằng milliseconds để tính gaze stability.
    
    Giải thích:
    - Window size 100-300ms được khuyến nghị (trade-off giữa noise và phản ứng)
    - Window nhỏ hơn → phản ứng nhanh nhưng dễ bị nhiễu
    - Window lớn hơn → ổn định hơn nhưng phản ứng chậm
    
    Ví dụ với FPS = 30:
    - 100ms = 3 frames (phản ứng nhanh, dễ nhiễu)
    - 200ms = 6 frames (cân bằng tốt - mặc định)
    - 300ms = 9 frames (ổn định, phản ứng chậm)
    
    Khuyến nghị: 200ms cho hầu hết trường hợp
    """
    
    GAZE_STABILITY_RMS_THRESHOLD: float = 0.02
    """
    RMS (Root Mean Square) distance threshold để xác định gaze ổn định.
    
    Giải thích:
    - RMS distance là "bán kính" dispersion - dễ hiểu hơn variance
    - Đơn vị: normalized by interocular distance
    - Nếu RMS < threshold → gaze ổn định (đang focus)
    - Nếu RMS >= threshold → gaze không ổn định (đang di chuyển)
    
    Công thức:
    - RMS = sqrt(mean(distances_from_center²))
    - distances_from_center = khoảng cách từ mỗi gaze position đến center
    
    Ví dụ:
    - 0.01: Rất nghiêm ngặt, chỉ tính khi gaze cực kỳ ổn định
    - 0.02: Vừa phải (mặc định)
    - 0.05: Dễ dãi hơn, chấp nhận dao động lớn hơn
    
    Khuyến nghị: 0.02 cho hầu hết trường hợp
    """
    
    GAZE_STABILITY_USE_HEAD_COMPENSATION: bool = True
    """
    Có bù trừ chuyển động đầu (head motion) không.
    
    Giải thích:
    - Đầu dịch chuyển làm eye-gaze pixel thay đổi dù mắt có thể đang focus
    - Bù trừ head motion giúp tách riêng eye movement và head movement
    - Cần head pose estimation (yaw, pitch, roll)
    
    Khuyến nghị: True (bật để chính xác hơn)
    """
    
    GAZE_STABILITY_USE_OUTLIER_REMOVAL: bool = True
    """
    Có loại bỏ outliers (giật mắt, blink, missing data) không.
    
    Giải thích:
    - Saccade (giật mắt) / blink / missing data làm variance tăng đột ngột
    - Loại bỏ outliers giúp tính toán ổn định hơn
    - Sử dụng Z-score method (mặc định: z_threshold = 2.5)
    
    Khuyến nghị: True (bật để ổn định hơn)
    """
    
    GAZE_STABILITY_USE_SMOOTHING: bool = True
    """
    Có làm mượt (smoothing) gaze positions không.
    
    Giải thích:
    - Smoothing giúp giảm nhiễu và làm mượt dữ liệu
    - Sử dụng moving average (mặc định: window_size = 3)
    
    Khuyến nghị: True (bật để ổn định hơn)
    """
    
    GAZE_STABILITY_Z_THRESHOLD: float = 2.5
    """
    Z-score threshold cho outlier removal.
    
    Giải thích:
    - Z-score = |value - mean| / std
    - Nếu Z-score > threshold → coi là outlier và loại bỏ
    - Giá trị cao hơn → ít loại bỏ hơn (dễ dãi)
    - Giá trị thấp hơn → nhiều loại bỏ hơn (nghiêm ngặt)
    
    Ví dụ:
    - 2.0: Nghiêm ngặt, loại bỏ nhiều
    - 2.5: Vừa phải (mặc định)
    - 3.0: Dễ dãi, ít loại bỏ
    
    Khuyến nghị: 2.5 cho hầu hết trường hợp
    """
    
    GAZE_STABILITY_SMOOTHING_WINDOW: int = 3
    """
    Window size cho smoothing (moving average).
    
    Giải thích:
    - Window càng lớn → mượt hơn nhưng phản ứng chậm hơn
    - Window càng nhỏ → phản ứng nhanh hơn nhưng ít mượt hơn
    
    Ví dụ:
    - 2: Phản ứng nhanh
    - 3: Vừa phải (mặc định)
    - 5: Mượt hơn, phản ứng chậm hơn
    
    Khuyến nghị: 3 cho hầu hết trường hợp
    """
    
    GAZE_STABILITY_ADAPTIVE_THRESHOLD: bool = False
    """
    Có dùng adaptive threshold không.
    
    Giải thích:
    - Adaptive threshold tự điều chỉnh dựa trên history
    - Phù hợp khi môi trường/camera thay đổi
    - Threshold = mean_rms + 1.5 * std_rms
    
    Khuyến nghị: False (tắt nếu muốn threshold cố định)
    """
    
    # Legacy configs (giữ lại để backward compatibility)
    GAZE_STABILITY_THRESHOLD: float = 0.05
    """
    [LEGACY] Ngưỡng ổn định của hướng nhìn (variance threshold).
    Chỉ dùng nếu GAZE_STABILITY_USE_IMPROVED = False.
    
    Khuyến nghị: Dùng GAZE_STABILITY_RMS_THRESHOLD thay vì config này
    """
    
    GAZE_WINDOW_SIZE: int = 300
    """
    [LEGACY] Số frames trong sliding window để tính gaze stability.
    Chỉ dùng nếu GAZE_STABILITY_USE_IMPROVED = False.
    
    Khuyến nghị: Dùng GAZE_STABILITY_WINDOW_MS thay vì config này
    """
    
    # ========================================================================
    # OBJECT DETECTION - Phát hiện đối tượng
    # ========================================================================
    
    OBJECT_DETECTION_INTERVAL: int = 5
    """
    Khoảng cách giữa các lần detect objects (mỗi N frames).
    
    Giải thích:
    - Object detection tốn nhiều tài nguyên, không cần detect mỗi frame
    - Detect mỗi N frames và dùng tracking để theo dõi giữa các lần detect
    - Giá trị nhỏ hơn → chính xác hơn nhưng chậm hơn
    - Giá trị lớn hơn → nhanh hơn nhưng có thể miss objects mới xuất hiện
    
    Ví dụ:
    - 1: Detect mỗi frame (chậm nhất, chính xác nhất)
    - 5: Detect mỗi 5 frames (cân bằng tốt - mặc định)
    - 10: Detect mỗi 10 frames (nhanh hơn, có thể miss objects nhỏ)
    
    Lưu ý: DeepSort tracking sẽ theo dõi objects giữa các lần detect
    Khuyến nghị: 5 frames cho hầu hết trường hợp
    """
    
    OBJECT_CONFIDENCE_THRESHOLD: float = 0.5
    """
    Ngưỡng confidence tối thiểu để chấp nhận một object detection.
    
    Giải thích:
    - YOLO trả về confidence score (0-1) cho mỗi detection
    - Chỉ chấp nhận detections có confidence >= threshold
    - Giá trị cao hơn → ít false positives nhưng có thể miss objects
    - Giá trị thấp hơn → detect nhiều hơn nhưng có thể có false positives
    
    Ví dụ:
    - 0.3: Dễ dãi, chấp nhận nhiều detections (có thể có false positives)
    - 0.5: Vừa phải (mặc định)
    - 0.7: Nghiêm ngặt, chỉ chấp nhận detections rất chắc chắn
    
    Khuyến nghị: 0.5 cho hầu hết trường hợp
    """
    
    LOOKING_AT_OBJECT_THRESHOLD: float = 0.6
    """
    Ngưỡng để xác định trẻ có đang nhìn vào một object không.
    
    Giải thích:
    - Tính khoảng cách giữa gaze position và object center
    - Nếu khoảng cách < threshold → đang nhìn vào object
    - Đơn vị: normalized (0-1), với 0 = nhìn chính xác vào center, 1 = nhìn rất xa
    
    Công thức:
    - distance = sqrt((gaze_x - object_center_x)^2 + (gaze_y - object_center_y)^2)
    - Nếu distance < threshold → is_looking_at_object = True
    
    Ví dụ:
    - 0.4: Nghiêm ngặt, chỉ tính khi nhìn rất gần center
    - 0.6: Vừa phải (mặc định)
    - 0.8: Dễ dãi, tính cả khi nhìn gần object
    
    Khuyến nghị: 0.6 cho hầu hết trường hợp
    """
    
    OID_MODEL_SIZE: str = 'l'
    """
    Kích thước model YOLOv8 OID (Open Images Dataset).
    
    Giải thích:
    - YOLOv8 có nhiều kích thước model khác nhau
    - Model lớn hơn → chính xác hơn nhưng chậm hơn và tốn nhiều RAM hơn
    - Model nhỏ hơn → nhanh hơn nhưng kém chính xác hơn
    
    Các lựa chọn:
    - 'n' (nano):   ~6MB,  nhanh nhất,  kém chính xác nhất
    - 's' (small):  ~22MB, nhanh,       chính xác trung bình
    - 'm' (medium): ~52MB, vừa phải,    chính xác tốt (mặc định)
    - 'l' (large):  ~87MB, chậm,        chính xác rất tốt
    - 'x' (xlarge): ~136MB, chậm nhất,  chính xác nhất
    
    Khuyến nghị:
    - CPU: 'n' hoặc 's'
    - GPU: 'm' hoặc 'l'
    - High-end GPU: 'x'
    
    Lưu ý: Model sẽ được download tự động lần đầu tiên sử dụng
    """
    
    # ========================================================================
    # FACE DETECTION - Phát hiện khuôn mặt
    # ========================================================================
    
    ADULT_FACE_SIZE_THRESHOLD: float = 0.15
    """
    Ngưỡng kích thước tối thiểu để coi là face người lớn (tỷ lệ so với frame).
    
    Giải thích:
    - Người lớn thường ngồi xa camera hơn trẻ → face nhỏ hơn
    - Nhưng nếu người lớn ngồi gần → face lớn hơn
    - Dùng kích thước face để phân biệt child vs adult
    
    Công thức:
    - face_size = (face_width * face_height) / (frame_width * frame_height)
    - Nếu face_size >= threshold → có thể là adult
    
    Ví dụ:
    - 0.10: Dễ dãi, coi nhiều face là adult
    - 0.15: Vừa phải (mặc định)
    - 0.20: Nghiêm ngặt, chỉ coi face rất lớn là adult
    
    Lưu ý: Kết hợp với các yếu tố khác (vị trí, tỷ lệ) để phân biệt chính xác hơn
    """
    
    CHILD_FACE_SIZE_THRESHOLD: float = 0.10
    """
    Ngưỡng kích thước tối thiểu để coi là face trẻ em (tỷ lệ so với frame).
    
    Giải thích:
    - Trẻ em thường ngồi gần camera hơn → face lớn hơn
    - Dùng kích thước face để phân biệt child vs adult
    
    Công thức:
    - face_size = (face_width * face_height) / (frame_width * frame_height)
    - Nếu face_size >= threshold → có thể là child
    
    Ví dụ:
    - 0.08: Dễ dãi, coi nhiều face là child
    - 0.10: Vừa phải (mặc định)
    - 0.15: Nghiêm ngặt, chỉ coi face rất lớn là child
    
    Lưu ý: Thường child face lớn hơn adult face trong cùng một frame
    """
    
    LOOKING_AT_ADULT_RATIO: float = 0.5
    """
    Tỷ lệ tối thiểu trong window để coi là đang nhìn vào adult.
    
    Giải thích:
    - Tính tỷ lệ số frames nhìn vào adult trong một window
    - Nếu tỷ lệ >= threshold → đang nhìn vào adult
    
    Công thức:
    - ratio = (số frames nhìn vào adult) / (tổng số frames trong window)
    - Nếu ratio >= LOOKING_AT_ADULT_RATIO → is_looking_at_adult = True
    
    Ví dụ:
    - 0.3: Dễ dãi, chỉ cần nhìn vào adult 30% thời gian
    - 0.5: Vừa phải (mặc định) - nhìn vào adult ít nhất 50% thời gian
    - 0.7: Nghiêm ngặt, phải nhìn vào adult ít nhất 70% thời gian
    
    Khuyến nghị: 0.5 cho hầu hết trường hợp
    """
    
    # ========================================================================
    # FOCUS DETECTION - Phát hiện tập trung
    # ========================================================================
    
    REQUIRE_OBJECT_FOCUS: bool = True
    """
    Có bắt buộc phải focus vào object/adult cụ thể không.
    
    Giải thích:
    - True: Chỉ tính focus khi có object/adult cụ thể được nhìn vào
    - False: Tính cả khi chỉ nhìn về camera (không có object cụ thể)
    
    Ví dụ:
    - True: Trẻ nhìn vào sách → tính focus
    - True: Trẻ nhìn vào camera nhưng không có object → KHÔNG tính focus
    - False: Trẻ nhìn vào camera → tính focus (ngay cả khi không có object)
    
    Khuyến nghị: True cho screening ASD (cần focus vào object cụ thể)
    """
    
    MIN_OBJECT_FOCUS_RATIO: float = 0.5
    """
    Tỷ lệ tối thiểu nhìn vào object trong window để coi là focus.
    
    Giải thích:
    - Tính tỷ lệ số frames nhìn vào object trong một window
    - Nếu tỷ lệ >= threshold → đang focus vào object
    
    Công thức:
    - ratio = (số frames nhìn vào object) / (tổng số frames trong window)
    - Nếu ratio >= MIN_OBJECT_FOCUS_RATIO → is_focusing = True
    
    Ví dụ:
    - 0.3: Dễ dãi, chỉ cần nhìn vào object 30% thời gian
    - 0.5: Vừa phải (mặc định) - nhìn vào object ít nhất 50% thời gian
    - 0.7: Nghiêm ngặt, phải nhìn vào object ít nhất 70% thời gian
    
    Khuyến nghị: 0.5 cho hầu hết trường hợp
    """
    
    ALLOW_CAMERA_FOCUS_WITH_ADULT: bool = True
    """
    Cho phép tính focus khi nhìn về camera nếu có adult trong frame.
    
    Giải thích:
    - Nếu True: Khi có adult trong frame và trẻ nhìn về camera → tính focus
    - Logic: Adult có thể ngồi kế camera, nên nhìn về camera = nhìn vào adult
    - Nếu False: Chỉ tính focus khi nhìn trực tiếp vào adult face
    
    Ví dụ:
    - True: Adult ngồi kế camera, trẻ nhìn về camera → tính focus vào adult
    - False: Trẻ nhìn về camera → KHÔNG tính focus (ngay cả khi có adult)
    
    Khuyến nghị: True cho hầu hết trường hợp (thực tế adult thường ngồi kế camera)
    """
    
    CAMERA_FOCUS_THRESHOLD: float = 0.2
    """
    Ngưỡng gaze offset khi nhìn về camera (normalized).
    
    Giải thích:
    - Khi nhìn về camera, gaze position gần center của frame
    - Nếu offset từ center < threshold → đang nhìn về camera
    
    Công thức:
    - offset = sqrt((gaze_x - center_x)^2 + (gaze_y - center_y)^2)
    - Nếu offset < CAMERA_FOCUS_THRESHOLD → is_looking_at_camera = True
    
    Ví dụ:
    - 0.1: Nghiêm ngặt, chỉ tính khi nhìn rất gần center
    - 0.2: Vừa phải (mặc định)
    - 0.3: Dễ dãi, tính cả khi nhìn gần camera
    
    Khuyến nghị: 0.2 cho hầu hết trường hợp
    """
    
    # ========================================================================
    # 3D GAZE ESTIMATION - Ước lượng hướng nhìn 3D
    # ========================================================================
    
    USE_3D_GAZE_CONFIDENCE: bool = True
    """
    Có sử dụng confidence từ 3D gaze estimation không.
    
    Giải thích:
    - 3D gaze estimation tính chính xác hướng nhìn trong không gian 3D
    - Có thể tính được confidence score cho mỗi gaze estimation
    - Nếu True: Ưu tiên dùng confidence từ 3D gaze
    - Nếu False: Dùng logic 2D đơn giản hơn
    
    Lợi ích khi True:
    - Chính xác hơn, đặc biệt khi đầu nghiêng/xoay
    - Có confidence score để đánh giá độ tin cậy
    - Giảm false positives
    
    Khuyến nghị: True cho hầu hết trường hợp (nếu có MediaPipe)
    """
    
    MIN_3D_GAZE_CONFIDENCE: float = 0.5
    """
    Ngưỡng confidence tối thiểu từ 3D gaze để chấp nhận.
    
    Giải thích:
    - 3D gaze estimation trả về confidence score (0-1)
    - Chỉ chấp nhận estimations có confidence >= threshold
    
    Ví dụ:
    - 0.3: Dễ dãi, chấp nhận nhiều estimations
    - 0.5: Vừa phải (mặc định)
    - 0.7: Nghiêm ngặt, chỉ chấp nhận estimations rất chắc chắn
    
    Khuyến nghị: 0.5 cho hầu hết trường hợp
    """
    
    # ========================================================================
    # GAZE WANDERING DETECTION - Phát hiện nhìn vô định
    # ========================================================================
    
    ENABLE_WANDERING_DETECTION: bool = True
    """
    Có bật detection "nhìn vô định" (gaze wandering) không.
    
    Giải thích:
    - "Nhìn vô định" = gaze ổn định nhưng không nhìn vào object/adult cụ thể
    - Đây là một dấu hiệu quan trọng trong screening ASD
    - Nếu True: Tính gaze_wandering_score và gaze_wandering_percentage
    
    Ví dụ:
    - True: Phát hiện khi trẻ "nhìn vào khoảng không"
    - False: Không tính wandering score
    
    Khuyến nghị: True cho screening ASD
    """
    
    WANDERING_OBJECT_RATIO_THRESHOLD: float = 0.1
    """
    Tỷ lệ tối đa nhìn vào object để coi là wandering.
    
    Giải thích:
    - Nếu tỷ lệ nhìn vào object < threshold → có thể là wandering
    - Kết hợp với các điều kiện khác (gaze ổn định, không nhìn adult)
    
    Ví dụ:
    - 0.1: Nghiêm ngặt, chỉ coi là wandering khi hầu như không nhìn object
    - 0.2: Vừa phải (mặc định)
    - 0.3: Dễ dãi, coi là wandering ngay cả khi nhìn object một chút
    
    Khuyến nghị: 0.2 cho hầu hết trường hợp
    """
    
    WANDERING_ADULT_RATIO_THRESHOLD: float = 0.2
    """
    Tỷ lệ tối đa nhìn vào adult để coi là wandering.
    
    Giải thích:
    - Nếu tỷ lệ nhìn vào adult < threshold → có thể là wandering
    - Kết hợp với các điều kiện khác
    
    Ví dụ:
    - 0.1: Nghiêm ngặt
    - 0.2: Vừa phải (mặc định)
    - 0.3: Dễ dãi
    
    Khuyến nghị: 0.2 cho hầu hết trường hợp
    """
    
    WANDERING_GAZE_OFFSET_THRESHOLD: float = 0.1
    """
    Ngưỡng gaze offset để coi là nhìn về center (camera).
    
    Giải thích:
    - Khi wandering, gaze thường ở gần center (nhìn về camera nhưng không focus)
    - Nếu offset < threshold → gaze ở gần center
    
    Ví dụ:
    - 0.1: Nghiêm ngặt, chỉ tính khi gaze rất gần center
    - 0.2: Vừa phải (mặc định)
    - 0.3: Dễ dãi
    
    Khuyến nghị: 0.2 cho hầu hết trường hợp
    """
    
    WANDERING_WINDOW_SIZE: int = 30
    """
    Số frames trong window để tính wandering.
    
    Giải thích:
    - Tương tự GAZE_WINDOW_SIZE
    - Window càng lớn → tính toán ổn định hơn
    
    Khuyến nghị: 30 frames (1 giây với FPS 30)
    """
    
    # ========================================================================
    # BOOK DETECTION - Phát hiện sách
    # ========================================================================
    
    BOOK_FOCUSING_SCORE_THRESHOLD: float = 0.7
    """
    Ngưỡng focusing score tối thiểu để coi là focus tốt vào sách.
    
    Giải thích:
    - Book focusing score được tính dựa trên:
      + Khoảng cách từ gaze đến book center
      + Kích thước book trong frame
      + Thời gian focus
    - Score cao hơn → focus tốt hơn
    
    Ví dụ:
    - 0.5: Dễ dãi, coi là focus tốt ngay cả khi score thấp
    - 0.7: Vừa phải (mặc định)
    - 0.9: Nghiêm ngặt, chỉ coi là focus tốt khi score rất cao
    
    Khuyến nghị: 0.7 cho hầu hết trường hợp
    """
    
    # ========================================================================
    # VIDEO PROCESSING - Xử lý video
    # ========================================================================
    
    MAX_FRAME_WIDTH: int = 1280
    """
    Chiều rộng tối đa của frame (pixels) để hiển thị. Frame lớn hơn sẽ được resize.
    
    Giải thích:
    - Resize frame để vừa với màn hình và tăng tốc độ xử lý
    - Giữ nguyên tỷ lệ khung hình (aspect ratio)
    - Resize nếu frame lớn hơn MAX_FRAME_WIDTH HOẶC MAX_FRAME_HEIGHT
    
    Ví dụ:
    - 640: Nhỏ, phù hợp màn hình nhỏ
    - 1280: Vừa phải (mặc định) - phù hợp màn hình 1920x1080
    - 1920: Lớn, có thể tràn màn hình nhỏ
    
    Khuyến nghị: 1280 cho màn hình 1920x1080 (để lại margin)
    """
    
    MAX_FRAME_HEIGHT: int = 720
    """
    Chiều cao tối đa của frame (pixels) để hiển thị. Frame lớn hơn sẽ được resize.
    
    Giải thích:
    - Resize frame để vừa với màn hình
    - Giữ nguyên tỷ lệ khung hình (aspect ratio)
    - Resize nếu frame lớn hơn MAX_FRAME_WIDTH HOẶC MAX_FRAME_HEIGHT
    
    Ví dụ:
    - 480: Nhỏ, phù hợp màn hình nhỏ
    - 720: Vừa phải (mặc định) - phù hợp màn hình 1920x1080
    - 1080: Lớn, có thể tràn màn hình nhỏ
    
    Khuyến nghị: 720 cho màn hình 1920x1080 (để lại margin)
    """
    
    FPS_DEFAULT: int = 30
    """
    FPS mặc định nếu không detect được từ video.
    
    Giải thích:
    - Một số video không có metadata FPS
    - Dùng giá trị này làm fallback
    
    Ví dụ:
    - 24: FPS phim
    - 30: FPS video thông thường (mặc định)
    - 60: FPS video chất lượng cao
    
    Khuyến nghị: 30 cho hầu hết trường hợp
    """
    
    # Object Classes - OID có 600 classes, không cần map như COCO
    
    @classmethod
    def from_config(cls, config_obj=None):
        """
        Create config from external config object
        
        Cho phép tạo GazeConfig từ một config object bên ngoài,
        hữu ích khi muốn override một số giá trị từ file config chính.
        """
        instance = cls()
        
        if config_obj and hasattr(config_obj, 'GAZE_MIN_FOCUSING_DURATION'):
            instance.MIN_FOCUSING_DURATION = config_obj.GAZE_MIN_FOCUSING_DURATION
        if config_obj and hasattr(config_obj, 'GAZE_STABILITY_THRESHOLD'):
            instance.GAZE_STABILITY_THRESHOLD = config_obj.GAZE_STABILITY_THRESHOLD
        
        return instance
