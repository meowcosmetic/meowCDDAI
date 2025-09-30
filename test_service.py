import requests
import json

# Base URL của service
BASE_URL = "http://localhost:8102"

def test_health():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_upload_book():
    """Test upload book endpoint"""
    print("Testing upload book...")
    
    book_data = {
        "book_id": "python_basic_001",
        "title": "Lập trình Python cơ bản",
        "author": "Nguyễn Văn A",
        "year": 2021,
        "content": """
        Python là ngôn ngữ lập trình bậc cao, được tạo ra bởi Guido van Rossum và được phát hành lần đầu vào năm 1991.
        
        Python có cú pháp đơn giản và dễ đọc, phù hợp cho người mới bắt đầu học lập trình.
        
        Danh sách trong Python là một cấu trúc dữ liệu cơ bản, cho phép lưu trữ nhiều giá trị trong một biến.
        
        Từ điển trong Python là một cấu trúc dữ liệu key-value, rất hữu ích cho việc lưu trữ dữ liệu có cấu trúc.
        
        Hàm trong Python cho phép bạn tái sử dụng code và tổ chức chương trình một cách hiệu quả.
        
        Lập trình hướng đối tượng trong Python giúp tạo ra code có cấu trúc và dễ bảo trì.
        """,
        "tags": ["Python", "lập trình", "cơ bản"],
        "language": "vi",
        "category": "Lập trình"
    }
    
    response = requests.post(f"{BASE_URL}/upload-book", json=book_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_upload_second_book():
    """Test upload second book with different tags"""
    print("Testing upload second book...")
    
    book_data = {
        "book_id": "java_advanced_002",
        "title": "Lập trình Java nâng cao",
        "author": "Trần Thị B",
        "year": 2022,
        "content": """
        Java là ngôn ngữ lập trình hướng đối tượng mạnh mẽ, được phát triển bởi Sun Microsystems.
        
        Java có tính bảo mật cao và khả năng chạy trên nhiều nền tảng khác nhau.
        
        Collections Framework trong Java cung cấp các cấu trúc dữ liệu tiên tiến như ArrayList, HashMap, HashSet.
        
        Multithreading trong Java cho phép thực hiện nhiều tác vụ đồng thời.
        
        Spring Framework là framework phổ biến cho phát triển ứng dụng Java enterprise.
        
        Design patterns trong Java giúp tạo ra code có cấu trúc và dễ bảo trì.
        """,
        "tags": ["Java", "lập trình", "nâng cao", "Spring"],
        "language": "vi",
        "category": "Lập trình"
    }
    
    response = requests.post(f"{BASE_URL}/upload-book", json=book_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_embedding_search():
    """Test embedding search endpoint"""
    print("Testing embedding search...")
    
    search_data = {
        "query": "cấu trúc dữ liệu Python",
        "limit": 5,
        "score_threshold": 0.5
    }
    
    response = requests.post(f"{BASE_URL}/search", json=search_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Embedding Score: {result.get('embedding_score', 'N/A')}")
            print(f"   Book: {result['payload']['title']}")
            print(f"   Author: {result['payload']['author']}")
            print(f"   Content: {result['payload']['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
    print()

def test_keyword_search():
    """Test keyword search endpoint"""
    print("Testing keyword search...")
    
    search_data = {
        "query": "Python lập trình",
        "limit": 5,
        "score_threshold": 0.5
    }
    
    response = requests.post(f"{BASE_URL}/search-keywords", json=search_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Keyword Score: {result.get('keyword_score', 'N/A')}")
            print(f"   Book: {result['payload']['title']}")
            print(f"   Author: {result['payload']['author']}")
            print(f"   Content: {result['payload']['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
    print()

def test_hybrid_search():
    """Test hybrid search endpoint"""
    print("Testing hybrid search...")
    
    search_data = {
        "query": "cấu trúc dữ liệu Python",
        "limit": 5,
        "score_threshold": 0.3,
        "alpha": 0.7,  # Weight for embedding
        "beta": 0.3,   # Weight for keyword
        "keyword_fields": ["content", "title", "tags"]
    }
    
    response = requests.post(f"{BASE_URL}/search-hybrid", json=search_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Hybrid Score: {result['hybrid_score']:.3f}")
            print(f"   Embedding Score: {result['embedding_score']:.3f}")
            print(f"   Keyword Score: {result['keyword_score']:.3f}")
            print(f"   Book: {result['payload']['title']}")
            print(f"   Author: {result['payload']['author']}")
            print(f"   Content: {result['payload']['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
    print()

def test_search_by_book_id():
    """Test search by book_id endpoint"""
    print("Testing search by book_id...")
    
    book_id = "python_basic_001"
    response = requests.get(f"{BASE_URL}/search-by-book-id/{book_id}?limit=3")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results for book_id: {book_id}")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Book: {result['payload']['title']}")
            print(f"   Chapter: {result['payload']['chapter']}")
            print(f"   Page: {result['payload']['page']}")
            print(f"   Content: {result['payload']['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
    print()

def test_search_by_tags():
    """Test search by tags endpoint"""
    print("Testing search by tags...")
    
    # Test with single tag
    tags = "Python"
    response = requests.get(f"{BASE_URL}/search-by-tags?tags={tags}&limit=3")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results for tags: {tags}")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Book: {result['payload']['title']}")
            print(f"   Tags: {result['payload']['tags']}")
            print(f"   Content: {result['payload']['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
    print()

def test_search_by_multiple_tags():
    """Test search by multiple tags"""
    print("Testing search by multiple tags...")
    
    # Test with multiple tags (OR logic)
    tags = "Python,Java"
    response = requests.get(f"{BASE_URL}/search-by-tags?tags={tags}&limit=5&match_all=false")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results for tags: {tags} (OR logic)")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Book: {result['payload']['title']}")
            print(f"   Tags: {result['payload']['tags']}")
            print(f"   Content: {result['payload']['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
    print()

def test_search_by_tags_match_all():
    """Test search by tags with match_all=True"""
    print("Testing search by tags (match all)...")
    
    # Test with multiple tags (AND logic)
    tags = "lập trình,cơ bản"
    response = requests.get(f"{BASE_URL}/search-by-tags?tags={tags}&limit=5&match_all=true")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results)} results for tags: {tags} (AND logic)")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Book: {result['payload']['title']}")
            print(f"   Tags: {result['payload']['tags']}")
            print(f"   Content: {result['payload']['content'][:100]}...")
    else:
        print(f"Error: {response.json()}")
    print()

def test_collection_info():
    """Test collection info endpoint"""
    print("Testing collection info...")
    response = requests.get(f"{BASE_URL}/collection-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_rebuild_index():
    """Test rebuild keyword index endpoint"""
    print("Testing rebuild keyword index...")
    response = requests.post(f"{BASE_URL}/rebuild-keyword-index")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_delete_book():
    """Test delete book endpoint"""
    print("Testing delete book...")
    book_id = "python_basic_001"
    response = requests.delete(f"{BASE_URL}/delete-book/{book_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("=== Book Vector Service với Hybrid Search Test ===\n")
    
    # Test các endpoint
    test_health()
    test_upload_book()
    test_upload_second_book()
    test_embedding_search()
    test_keyword_search()
    test_hybrid_search()
    test_search_by_book_id()
    test_search_by_tags()
    test_search_by_multiple_tags()
    test_search_by_tags_match_all()
    test_collection_info()
    test_rebuild_index()
    
    # Uncomment để test delete
    # test_delete_book()
    
    print("=== Test completed ===")
