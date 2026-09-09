import cv2
import numpy as np
import io
from PIL import Image
from typing import Dict, Any, List, Optional
import math
import logging

logger = logging.getLogger(__name__)

class WorksheetAnalysisService:
    """
    Computer-Assisted Worksheet & Drawing Analysis Engine
    
    Định vị khoa học: Công cụ hỗ trợ đo lường các thuộc tính vật lý / thị giác khách quan
    (Độ ổn định nét vẽ, độ tuân thủ đường biên, tỷ lệ hoàn thành bài tập) phục vụ
    rèn luyện vận động tinh (Fine Motor Skills). Tuyệt đối không phán đoán cảm xúc/tâm lý võ đoán.
    """

    DISCLAIMER = (
        "Kết quả phân tích mang tính chất hỗ trợ kỹ thuật số định lượng về vận động tinh "
        "(độ rung nét, độ tràn viền, độ phủ), nhận định can thiệp sư phạm/lâm sàng cuối cùng "
        "thuộc về giáo viên và chuyên viên phụ trách."
    )

    def analyze(self, image_bytes: bytes, worksheet_type: str = "TRACING") -> Dict[str, Any]:
        """
        Phân tích hình ảnh bài tập/nét vẽ của trẻ bằng OpenCV.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Dữ liệu hình ảnh bài tập không được để trống hoặc rỗng")

        # 1. Đọc ảnh từ bytes
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Định dạng file không hợp lệ hoặc ảnh bị lỗi/hỏng")

        h, w, c = img.shape
        total_pixels = h * w

        # 2. Tiền xử lý: Grayscale, làm mờ giảm nhiễu hạt
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Phát hiện nét vẽ bằng Adaptive Threshold & Canny
        # Nét vẽ (thường đậm màu hơn giấy)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Canny edge để bắt cạnh
        edges = cv2.Canny(blurred, 50, 150)

        # 4. Tính toán Stroke Stability (Độ ổn định nét vẽ) qua Contour Analysis
        stability_score, tremor_level = self._compute_stroke_stability(thresh)

        # 5. Tính toán Boundary Compliance (Độ tuân thủ đường biên / tràn viền)
        boundary_score = self._compute_boundary_compliance(thresh, edges, worksheet_type)

        # 6. Tính toán Worksheet Coverage (Tỷ lệ diện tích bài làm)
        drawing_pixels = cv2.countNonZero(thresh)
        coverage_ratio = min(1.0, drawing_pixels / (total_pixels * 0.45))
        completion_rate = round(coverage_ratio * 100.0, 1)

        # 7. Phân tích màu sắc khách quan (Color diversity)
        color_count, dominant_hex = self._analyze_color_palette(img, thresh)

        # 8. Tính điểm vận động tinh tổng hợp (Overall Motor Score)
        overall_score = round(
            (stability_score * 0.4) + (boundary_score * 0.4) + (completion_rate * 0.2), 1
        )
        overall_score = max(10.0, min(100.0, overall_score))

        # 9. Tạo các nhận xét kỹ thuật khách quan và gợi ý sư phạm
        technical_obs = self._generate_technical_observations(
            stability_score, tremor_level, boundary_score, completion_rate, color_count
        )
        pedagogical_sugg = self._generate_pedagogical_suggestions(
            stability_score, boundary_score, completion_rate, worksheet_type
        )

        return {
            "status": "success",
            "worksheet_type": worksheet_type.upper(),
            "metrics": {
                "overall_motor_score": overall_score,
                "stability_score": stability_score,
                "boundary_compliance_score": boundary_score,
                "completion_rate": completion_rate,
                "stroke_tremor_level": tremor_level,
                "detected_colors_count": color_count,
                "dominant_color_hex": dominant_hex,
                "image_resolution": f"{w}x{h}"
            },
            "technical_observations": technical_obs,
            "pedagogical_suggestions": pedagogical_sugg,
            "disclaimer": self.DISCLAIMER
        }

    def _compute_stroke_stability(self, binary_img: np.ndarray) -> tuple[float, str]:
        """
        Đo độ rung rẩy/gập ghềnh của nét bút dựa trên phương sai độ cong của các contour nét vẽ.
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 75.0, "MODERATE"

        # Lọc các contour có độ dài hợp lý (loại bỏ hạt bụi nhỏ)
        valid_contours = [c for c in contours if cv2.arcLength(c, False) > 40]

        if not valid_contours:
            return 75.0, "MODERATE"

        total_roughness = []
        for c in valid_contours[:15]: # Lấy tối đa 15 nét dài nhất
            peri = cv2.arcLength(c, False)
            approx = cv2.approxPolyDP(c, 0.02 * peri, False)
            # Tỷ lệ giữa chu vi thực và chu vi đa giác xấp xỉ thể hiện độ run
            approx_peri = cv2.arcLength(approx, False)
            if approx_peri > 0:
                roughness = peri / approx_peri
                total_roughness.append(roughness)

        if not total_roughness:
            return 80.0, "LOW"

        avg_roughness = float(np.mean(total_roughness))

        # Điểm mượt: roughness càng gần 1.0 nét càng thẳng/mượt, > 1.5 nét bắt đầu run
        if avg_roughness <= 1.25:
            score = 90.0 - (avg_roughness - 1.0) * 40
            tremor = "LOW"
        elif avg_roughness <= 1.55:
            score = 80.0 - (avg_roughness - 1.25) * 50
            tremor = "MODERATE"
        else:
            score = max(40.0, 65.0 - (avg_roughness - 1.55) * 30)
            tremor = "HIGH"

        return round(float(score), 1), tremor

    def _compute_boundary_compliance(self, thresh: np.ndarray, edges: np.ndarray, worksheet_type: str) -> float:
        """
        Tính toán độ tuân thủ viền: Nét vẽ có nằm gọn gàng bên trong khuôn viền hay lem viền.
        """
        # Nếu là bài tập tô màu (COLORING)
        # Sử dụng biến đổi hình thái học để tìm tỷ lệ màu tràn ra ngoài đường biên mẫu
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)

        # Giao điểm giữa nét vẽ và vùng đệm biên
        overlap = cv2.bitwise_and(thresh, dilated_edges)
        overlap_count = cv2.countNonZero(overlap)
        total_drawn = cv2.countNonZero(thresh)

        if total_drawn == 0:
            return 80.0

        # Nếu nét vẽ đè viền hoặc ra ngoài viền quá nhiều
        spill_ratio = overlap_count / total_drawn
        if spill_ratio < 0.25:
            compliance = 92.0 - spill_ratio * 40
        elif spill_ratio < 0.50:
            compliance = 82.0 - (spill_ratio - 0.25) * 60
        else:
            compliance = max(45.0, 67.0 - (spill_ratio - 0.50) * 40)

        return round(float(compliance), 1)

    def _analyze_color_palette(self, img: np.ndarray, thresh: np.ndarray) -> tuple[int, str]:
        """
        Nhận diện số lượng màu và mã HEX chủ đạo của vùng trẻ vẽ/tô.
        """
        # Áp mask vùng có nét vẽ
        masked = cv2.bitwise_and(img, img, mask=thresh)
        non_zero_coords = np.where(thresh > 0)

        if len(non_zero_coords[0]) == 0:
            return 1, "#1E293B"

        pixels = masked[non_zero_coords]

        # Chuyển sang RGB
        rgb_pixels = pixels[:, [2, 1, 0]]
        
        # Tính màu trung bình
        mean_r = int(np.mean(rgb_pixels[:, 0]))
        mean_g = int(np.mean(rgb_pixels[:, 1]))
        mean_b = int(np.mean(rgb_pixels[:, 2]))
        dominant_hex = f"#{mean_r:02X}{mean_g:02X}{mean_b:02X}"

        # Đếm số lượng màu phân biệt bằng k-means giản lược hoặc binning
        bins = np.histogram2d(pixels[:, 0], pixels[:, 1], bins=4)[0]
        distinct_colors = int(np.count_nonzero(bins > (len(pixels) * 0.05)))
        distinct_colors = max(1, min(10, distinct_colors))

        return distinct_colors, dominant_hex

    def _generate_technical_observations(
        self, stability: float, tremor: str, boundary: float, completion: float, colors: int
    ) -> List[str]:
        obs = []
        if tremor == "LOW":
            obs.append("Lực ấn bút và điều khiển nét tương đối đều đặn, ít độ rung giật.")
        elif tremor == "MODERATE":
            obs.append("Nét vẽ có độ dao động vừa phải, xuất hiện dấu hiệu rung nhẹ ở các đoạn uốn cong.")
        else:
            obs.append("Ghi nhận độ rung lắc nét bút cao, trẻ có thể mỏi cơ tay hoặc lực cầm bút chưa vững.")

        if boundary >= 85.0:
            obs.append("Độ kiểm soát viền tốt: Nét vẽ tuân thủ khuôn viền mẫu với tỷ lệ lem viền thấp.")
        elif boundary >= 70.0:
            obs.append("Độ tuân thủ viền ở mức trung bình, có một số đoạn nét vượt nhẹ ra ngoài ranh giới.")
        else:
            obs.append("Tỷ lệ nét lem ngoài ranh giới cao, trẻ cần thêm bài tập định vị thị giác - vận động.")

        obs.append(f"Mức độ phủ nét trên bài tập đạt {completion}%, sử dụng khoảng {colors} tông màu.")
        return obs

    def _generate_pedagogical_suggestions(
        self, stability: float, boundary: float, completion: float, worksheet_type: str
    ) -> List[str]:
        suggestions = []
        if stability < 70.0:
            suggestions.append("Khuyến khích cho bé dùng bút sáp hình tam giác to hoặc bút lông ngòi to để dễ cầm nắm và trợ lực.")
            suggestions.append("Tập thêm các bài tập vận động thô và trung gian: Lăn bóng bằng hai tay, bóp bóng cao su mềm để tăng cơ lực bàn tay.")
        else:
            suggestions.append("Bé kiểm soát nét tốt, có thể nâng dần độ khó với bài tập đồ nét zíc-zắc, xoắn ốc hoặc nét đứt đoạn.")

        if boundary < 75.0:
            suggestions.append("Áp dụng kỹ thuật viền nổi xúc giác: Dán dây len hoặc viền đất nặn quanh hình vẽ để bé cảm nhận ranh giới vật lý khi tô.")
        
        if completion < 50.0:
            suggestions.append("Chia nhỏ trang bài tập thành từng góc nhỏ để bé không bị ngợp và duy trì sự chú ý tốt hơn.")

        return suggestions
