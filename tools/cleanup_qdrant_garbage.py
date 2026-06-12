"""
Script xóa dữ liệu rác (content rỗng) trong Qdrant collection 'books'.
Chạy: python tools/cleanup_qdrant_garbage.py
"""
import sys
sys.path.insert(0, '.')

from qdrant_client import QdrantClient
from config import Config

client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
collection_name = Config.COLLECTION_NAME

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

# Find garbage points (empty content or no book_name)
garbage_ids = []
for p in all_points:
    payload = p.payload or {}
    content = payload.get("content", "")
    book_name = payload.get("book_name")

    if not content or not content.strip():
        garbage_ids.append(p.id)

print(f"Found {len(garbage_ids)} garbage points to delete")

if garbage_ids:
    print("Deleting...")
    client.delete(
        collection_name=collection_name,
        points_selector=garbage_ids,
    )
    print(f"✅ Deleted {len(garbage_ids)} garbage points")
else:
    print("No garbage found, collection is clean!")
