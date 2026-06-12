"""
Script kiểm tra dữ liệu rác trong Qdrant collection 'books'.
Chạy: python tools/check_qdrant_data.py
"""
import sys
sys.path.insert(0, '.')

from qdrant_client import QdrantClient
from config import Config

client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
collection_name = Config.COLLECTION_NAME

# Get collection info
info = client.get_collection(collection_name)
print(f"=== Collection: {collection_name} ===")
print(f"Total points: {info.points_count}")
print()

# Scroll all points
all_points = []
offset = None
while True:
    batch, next_offset = client.scroll(
        collection_name=collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    if not batch:
        break
    all_points.extend(batch)
    if next_offset is None:
        break
    offset = next_offset

print(f"Fetched {len(all_points)} points total")
print()

# Analyze
empty_content = []
no_book_name = []
unknown_book_id = []
valid = []

for p in all_points:
    payload = p.payload or {}
    content = payload.get("content", "")
    book_name = payload.get("book_name")
    book_id = payload.get("book_id", "")

    is_bad = False
    if not content or not content.strip():
        empty_content.append(p)
        is_bad = True
    if not book_name:
        no_book_name.append(p)
        is_bad = True
    if book_id == "unknown":
        unknown_book_id.append(p)
        is_bad = True
    if not is_bad:
        valid.append(p)

print(f"=== Kết quả phân tích ===")
print(f"✅ Valid points (có content + book_name): {len(valid)}")
print(f"❌ Empty content: {len(empty_content)}")
print(f"❌ No book_name (null): {len(no_book_name)}")
print(f"❌ book_id = 'unknown': {len(unknown_book_id)}")
print()

# Show sample of bad data
all_bad = set()
for p in empty_content + no_book_name + unknown_book_id:
    all_bad.add(p.id)

bad_points = [p for p in all_points if p.id in all_bad]
print(f"=== Tổng số points rác (unique): {len(bad_points)} ===")
print()

if bad_points:
    print("--- Mẫu dữ liệu rác (tối đa 10) ---")
    for p in bad_points[:10]:
        payload = p.payload or {}
        print(f"  ID: {p.id}")
        print(f"    book_id: {payload.get('book_id')}")
        print(f"    book_name: {payload.get('book_name')}")
        print(f"    content length: {len(payload.get('content', ''))}")
        print(f"    chapter: {payload.get('chapter')}")
        print(f"    page: {payload.get('page')}")
        print(f"    postgres_id: {payload.get('postgres_id')}")
        print(f"    summary: {str(payload.get('summary', ''))[:80]}")
        print()

# Summary by book_id
print("=== Phân bố theo book_id ===")
book_id_counts = {}
for p in all_points:
    bid = (p.payload or {}).get("book_id", "(none)")
    book_id_counts[bid] = book_id_counts.get(bid, 0) + 1

for bid, count in sorted(book_id_counts.items(), key=lambda x: -x[1]):
    marker = "❌" if bid in ("unknown", "(none)", "") else "  "
    print(f"  {marker} book_id='{bid}': {count} points")
