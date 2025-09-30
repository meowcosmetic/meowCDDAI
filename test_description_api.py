#!/usr/bin/env python3
"""
Test script cho API generate description
"""

import requests
import json

# URL của API server
BASE_URL = "http://localhost:8102"

def test_generate_description():
    """Test API generate description"""
    
    # Test data
    test_cases = [
        "Trẻ có thể ngồi vững trong 30 giây",
        "Trẻ có thể giao tiếp bằng mắt khi được gọi tên",
        "Trẻ có thể cầm bút chì và vẽ nét thẳng",
        "Trẻ có thể nhận biết 5 màu cơ bản",
        "Trẻ có thể tự mặc áo sơ mi"
    ]
    
    print("🧪 Testing Description Generation API")
    print("=" * 50)
    
    for i, goal in enumerate(test_cases, 1):
        print(f"\n📝 Test case {i}: {goal}")
        
        # Prepare request data
        request_data = {
            "intervention_goal": goal
        }
        
        try:
            # Make API request
            response = requests.post(
                f"{BASE_URL}/generate-description",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"   Vietnamese: {result['description']['vi']}")
                print(f"   English: {result['description']['en']}")
                print(f"   Original: {result['original_goal']}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Server không chạy. Hãy start server trước!")
            break
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")

def test_api_info():
    """Test API info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API Documentation available at: http://localhost:8102/docs")
        else:
            print("❌ API Documentation not available")
    except:
        print("❌ Cannot connect to API server")

if __name__ == "__main__":
    print("🚀 Starting API Tests...")
    test_api_info()
    test_generate_description()
