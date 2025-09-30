"""
Test script để kiểm tra hệ thống 4 AI agents xử lý mục tiêu can thiệp
"""

import requests
import json
from ai_agents import InterventionProcessor

def test_direct_processing():
    """Test trực tiếp qua InterventionProcessor"""
    print("=== TEST TRỰC TIẾP QUA INTERVENTION PROCESSOR ===")
    
    # Khởi tạo processor
    processor = InterventionProcessor()
    
    # Mục tiêu can thiệp mẫu
    intervention_goal = "Trẻ quay đầu và nhìn về phía âm thanh khi phát gần trước mặt (0.5–1m). Thực hiện ≥70% số lần trong 5–6 cơ hội mỗi buổi."
    
    print(f"Mục tiêu can thiệp: {intervention_goal}")
    print("\n" + "="*80)
    
    try:
        # Xử lý qua 4 agents
        result = processor.process_intervention_goal(intervention_goal)
        
        if result["status"] == "success":
            print("✅ Xử lý thành công!")
            print(f"\n📋 KẾT QUẢ TỪNG AGENT:")
            
            print(f"\n🔬 EXPERT AGENT (Phân tích lý thuyết):")
            print("-" * 50)
            print(result["expert_analysis"])
            
            print(f"\n✏️ EDITOR AGENT (Biên tập dễ hiểu):")
            print("-" * 50)
            print(result["edited_content"])
            
            print(f"\n🛠️ PRACTICAL AGENT (Ví dụ & Checklist):")
            print("-" * 50)
            print(result["practical_content"])
            
            print(f"\n✅ VERIFIER AGENT (Kiểm chứng & Nguồn):")
            print("-" * 50)
            print(result["verified_content"])
            
        else:
            print(f"❌ Lỗi: {result['error']}")
            
    except Exception as e:
        print(f"❌ Lỗi khi test: {str(e)}")


def test_api_endpoint():
    """Test qua API endpoint"""
    print("\n\n=== TEST QUA API ENDPOINT ===")
    
    # URL của API
    api_url = "http://localhost:8102/process-intervention-goal"
    
    # Dữ liệu test
    test_data = {
        "intervention_goal": "Trẻ quay đầu và nhìn về phía âm thanh khi phát gần trước mặt (0.5–1m). Thực hiện ≥70% số lần trong 5–6 cơ hội mỗi buổi.",
        "title": "Can thiệp phản ứng âm thanh cho trẻ đặc biệt"
    }
    
    try:
        print(f"Gửi request đến: {api_url}")
        print(f"Dữ liệu: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
        
        response = requests.post(api_url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call thành công!")
            print(f"\n📋 KẾT QUẢ API:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"❌ API call thất bại: {response.status_code}")
            print(f"Chi tiết lỗi: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Không thể kết nối đến API. Hãy đảm bảo server đang chạy trên port 8102")
    except Exception as e:
        print(f"❌ Lỗi khi test API: {str(e)}")


if __name__ == "__main__":
    print("🚀 BẮT ĐẦU TEST HỆ THỐNG 4 AI AGENTS")
    print("="*80)
    
    # Test trực tiếp
    test_direct_processing()
    
    # Test API
    test_api_endpoint()
    
    print("\n" + "="*80)
    print("🏁 HOÀN THÀNH TEST")
