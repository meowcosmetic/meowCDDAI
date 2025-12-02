from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config
import json


class BaseAgent:
    """Base class cho tất cả AI agents"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=Config.GOOGLE_AI_API_KEY,
            temperature=0.7
        )
    
    def process(self, input_data: str, context: Dict[str, Any] = None) -> str:
        """Xử lý input và trả về kết quả"""
        raise NotImplementedError


class ExpertAgent(BaseAgent):
    """Agent chuyên gia phân tích chủ đề và tạo khung lý thuyết"""
    
    def process(self, intervention_goal: str, context: Dict[str, Any] = None) -> str:
        system_prompt = """
        Bạn là một chuyên gia tâm lý học và giáo dục đặc biệt với hơn 15 năm kinh nghiệm. 
        Nhiệm vụ của bạn là phân tích mục tiêu can thiệp cho trẻ đặc biệt và tạo ra khung lý thuyết khoa học.
        
        Hãy phân tích mục tiêu can thiệp được cung cấp và tạo ra:
        1. Phân tích chuyên sâu về chủ đề
        2. Khung lý thuyết khoa học đằng sau mục tiêu
        3. Các nguyên tắc tâm lý học áp dụng
        4. Cơ sở khoa học cho việc can thiệp
        
        Trả lời bằng tiếng Việt, sử dụng ngôn ngữ chuyên môn nhưng dễ hiểu.
        """
        
        # Tạo context từ book_content nếu có
        book_context = ""
        if context and "book_content" in context:
            book_content = context["book_content"]
            if isinstance(book_content, list) and len(book_content) > 0:
                combined_content = "\n\n".join(book_content)
                book_context = f"""
                
                Nội dung sách liên quan để tham khảo:
                {combined_content[:1500]}...
                """
        
        human_prompt = f"""
        Mục tiêu can thiệp: {intervention_goal}{book_context}
        
        Hãy phân tích và tạo khung lý thuyết cho mục tiêu này. Sử dụng thông tin từ nội dung sách nếu có để làm phân tích chính xác và chuyên sâu hơn.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class EditorAgent(BaseAgent):
    """Agent biên tập viên tổng hợp và format nội dung"""
    
    def process(self, content: str, context: Dict[str, Any] = None) -> str:
        system_prompt = """
        Bạn là một biên tập viên chuyên nghiệp và chuyên gia về HTML/Markdown với kinh nghiệm trong lĩnh vực giáo dục đặc biệt.
        Nhiệm vụ của bạn là tổng hợp, biên tập và format nội dung thành một bài viết hoàn chỉnh.
        
        Hãy:
        1. Tổng hợp tất cả nội dung từ các bước trước
        2. Biên tập thành một bài viết mạch lạc, dễ hiểu
        3. Tổ chức cấu trúc logic với tiêu đề, đoạn văn rõ ràng
        4. Format theo HTML5 semantic với CSS inline đẹp mắt
        5. Sử dụng typography đẹp (h1, h2, h3, p, ul, ol, strong, em)
        6. Tạo responsive design và màu sắc phù hợp
        7. Highlight các điểm quan trọng
        
        Trả lời bằng HTML5 hoàn chỉnh, sẵn sàng để hiển thị trên web.
        """
        
        human_prompt = f"""
        Nội dung cần tổng hợp và biên tập: {content}
        
        Hãy tổng hợp tất cả nội dung trên thành một bài viết hoàn chỉnh, dễ hiểu và format theo HTML5 đẹp mắt.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class PracticalAgent(BaseAgent):
    """Agent thực tiễn thêm ví dụ và checklist"""
    
    def process(self, edited_content: str, intervention_goal: str, context: Dict[str, Any] = None) -> str:
        system_prompt = """
        Bạn là một chuyên gia thực hành trong lĩnh vực can thiệp sớm cho trẻ đặc biệt.
        Nhiệm vụ của bạn là bổ sung các yếu tố thực tiễn vào nội dung lý thuyết.
        
        Hãy thêm vào:
        1. Ví dụ cụ thể, tình huống thực tế
        2. Checklist các bước thực hiện
        3. Lưu ý quan trọng khi áp dụng
        4. Các tình huống có thể gặp phải và cách xử lý
        5. Mẹo thực hành hữu ích
        
        Trả lời bằng tiếng Việt, tập trung vào tính thực tiễn và khả năng áp dụng.
        """
        
        human_prompt = f"""
        Mục tiêu can thiệp gốc: {intervention_goal}
        
        Nội dung đã biên tập: {edited_content}
        
        Hãy bổ sung các yếu tố thực tiễn, ví dụ và checklist cho nội dung này.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class VerifierAgent(BaseAgent):
    """Agent kiểm chứng và thêm nguồn tham khảo"""
    
    def process(self, practical_content: str, context: Dict[str, Any] = None) -> str:
        system_prompt = """
        Bạn là một chuyên gia kiểm chứng và đánh giá chất lượng nội dung khoa học.
        Nhiệm vụ của bạn là kiểm tra tính chính xác và bổ sung nguồn tham khảo.
        
        Hãy:
        1. Kiểm tra tính chính xác của thông tin
        2. Thêm nguồn tham khảo khoa học uy tín
        3. Đánh giá mức độ tin cậy của nội dung
        4. Đề xuất cải thiện nếu cần
        5. Tóm tắt các điểm chính và kết luận
        
        Trả lời bằng tiếng Việt, đảm bảo tính khoa học và độ tin cậy.
        """
        
        human_prompt = f"""
        Nội dung cần kiểm chứng: {practical_content}
        
        Hãy kiểm chứng và bổ sung nguồn tham khảo cho nội dung này.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class HTMLFormatterAgent(BaseAgent):
    """Agent format nội dung thành HTML đẹp để hiển thị"""
    
    def process(self, verified_content: str, context: Dict[str, Any] = None) -> str:
        system_prompt = """
        Bạn là một chuyên gia về HTML và CSS với kinh nghiệm thiết kế web.
        Nhiệm vụ của bạn là chuyển đổi nội dung văn bản thành HTML5 đẹp mắt và dễ đọc.
        
        Hãy format nội dung thành HTML với:
        1. Cấu trúc HTML5 semantic (article, section, header, etc.)
        2. Typography đẹp với các thẻ h1, h2, h3, p, ul, ol
        3. Highlight các từ khóa quan trọng
        4. Tạo danh sách đẹp cho các điểm chính
        5. Thêm CSS inline cơ bản để styling
        6. Responsive design
        7. Màu sắc và spacing phù hợp
        
        Trả lời bằng HTML5 hoàn chỉnh, sẵn sàng để hiển thị trên web.
        """
        
        human_prompt = f"""
        Nội dung cần format thành HTML: {verified_content}
        
        Hãy chuyển đổi nội dung này thành HTML5 đẹp mắt, dễ đọc và chuyên nghiệp.
        Sử dụng CSS inline để tạo styling đẹp.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class InterventionProcessor:
    """Class chính để xử lý mục tiêu can thiệp qua 4 agents"""
    
    def __init__(self):
        self.expert_agent = ExpertAgent()
        self.editor_agent = EditorAgent()
        self.practical_agent = PracticalAgent()
        self.verifier_agent = VerifierAgent()
        self.html_formatter_agent = HTMLFormatterAgent()
    
    def process_intervention_goal(self, intervention_goal: str, context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Xử lý mục tiêu can thiệp qua 4 bước theo workflow mới
        
        Args:
            intervention_goal: Mục tiêu can thiệp cần xử lý
            context: Context bổ sung (có thể chứa book_content)
            
        Returns:
            Dict chứa kết quả từ từng agent
        """
        try:
            # Bước 1: Expert Agent phân tích và tạo khung lý thuyết
            expert_result = self.expert_agent.process(intervention_goal, context)
            
            # Bước 2: Practical Agent thêm ví dụ và checklist thực tế
            practical_result = self.practical_agent.process(expert_result, intervention_goal)
            
            # Bước 3: Verifier Agent kiểm chứng và bổ sung nguồn tham khảo (trên nội dung từ bước 1 + 2)
            verifier_result = self.verifier_agent.process(practical_result)
            
            # Bước 4: Editor Agent (duy nhất) - Gom tất cả nội dung và biên tập + format
            final_content = self.editor_agent.process(
                f"""Tổng hợp và biên tập tất cả nội dung:
                
1. EXPERT ANALYSIS:
{expert_result}

2. PRACTICAL CONTENT:
{practical_result}

3. VERIFIED CONTENT:
{verifier_result}

Hãy gom tất cả nội dung trên, biên tập thành một bài viết hoàn chỉnh, dễ hiểu và format theo HTML."""
            )
            
            return {
                "original_goal": intervention_goal,
                "expert_analysis": expert_result,
                "practical_content": practical_result,
                "verified_content": verifier_result,
                "final_content": final_content,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "original_goal": intervention_goal,
                "error": str(e),
                "status": "error"
            }
