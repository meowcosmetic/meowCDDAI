"""
Test đơn giản chỉ 4 AI agents mà không cần server
"""

from ai_agents import InterventionProcessor

def test_agents_only():
    """Test chỉ 4 AI agents"""
    print("🚀 TEST 4 AI AGENTS - XỬ LÝ MỤC TIÊU CAN THIỆP")
    print("="*80)
    
    # Khởi tạo processor
    processor = InterventionProcessor()
    
    # Mục tiêu can thiệp mẫu
    intervention_goal = "Trẻ quay đầu và nhìn về phía âm thanh khi phát gần trước mặt (0.5–1m). Thực hiện ≥70% số lần trong 5–6 cơ hội mỗi buổi."
    
    print(f"📝 Mục tiêu can thiệp: {intervention_goal}")
    print("\n" + "="*80)
    
    try:
        # Xử lý qua 4 agents
        result = processor.process_intervention_goal(intervention_goal)
        
        if result["status"] == "success":
            print("✅ Xử lý thành công qua 4 AI agents!")
            print(f"\n📋 TÓM TẮT KẾT QUẢ:")
            
            print(f"\n🔬 EXPERT AGENT:")
            print("-" * 50)
            print(result["expert_analysis"][:200] + "...")
            
            print(f"\n✏️ EDITOR AGENT:")
            print("-" * 50)
            print(result["edited_content"][:200] + "...")
            
            print(f"\n🛠️ PRACTICAL AGENT:")
            print("-" * 50)
            print(result["practical_content"][:200] + "...")
            
            print(f"\n✅ VERIFIER AGENT:")
            print("-" * 50)
            print(result["verified_content"][:200] + "...")
            
            print(f"\n🎯 WORKFLOW HOÀN THÀNH:")
            print("1. ExpertAgent đã phân tích và tạo khung lý thuyết")
            print("2. EditorAgent đã biên tập và diễn đạt dễ hiểu")
            print("3. PracticalAgent đã thêm ví dụ và checklist")
            print("4. VerifierAgent đã kiểm chứng và thêm nguồn tham khảo")
            
        else:
            print(f"❌ Lỗi: {result['error']}")
            
    except Exception as e:
        print(f"❌ Lỗi khi test: {str(e)}")

if __name__ == "__main__":
    test_agents_only()


