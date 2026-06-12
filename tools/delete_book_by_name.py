"""
Xóa sách theo tên khỏi Qdrant và Postgres.
Chạy: python tools/delete_book_by_name.py
"""
import sys
sys.path.insert(0, '.')

from qdrant_client import QdrantClient
from config import Config
import psycopg2

BOOK_NAME = "Giảng dạy ngôn ngữ cho trẻ em mắc bệnh tự kỷ"

# === QDRANT ===
client = QdrantClient(url="http://localhost:6333", check_compatibility=False)
collection_name = Config.COLLECTION_NAME

print(f"=== Xóa sách: '{BOOK_NAME}' ===")
print()

# Scroll all points and find matching ones
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

# Find points matching book_name
matching_ids = []
matching_book_ids = set()
for p in all_points:
    payload = p.payload or {}
    book_name = payload.get("book_name", "")
    if book_name == BOOK_NAME:
        matching_ids.append(p.id)
        matching_book_ids.add(payload.get("book_id", ""))

print(f"[Qdrant] Found {len(matching_ids)} points with book_name='{BOOK_NAME}'")
print(f"[Qdrant] book_id(s): {matching_book_ids}")

if matching_ids:
    client.delete(
        collection_name=collection_name,
        points_selector=matching_ids,
    )
    print(f"[Qdrant] ✅ Deleted {len(matching_ids)} points")
else:
    print("[Qdrant] No points found to delete")

print()

# === POSTGRES ===
print("[Postgres] Connecting...")
params = Config.get_postgres_params()
try:
    # Thử localhost trước
    conn = psycopg2.connect(
        host="localhost",
        port=params["port"],
        dbname=params["database"],
        user=params["user"],
        password=params["password"],
    )
    cur = conn.cursor()

    # Count first
    cur.execute("SELECT COUNT(*) FROM book_contents WHERE book_name = %s", (BOOK_NAME,))
    count = cur.fetchone()[0]
    print(f"[Postgres] Found {count} rows with book_name='{BOOK_NAME}'")

    if count > 0:
        cur.execute("DELETE FROM book_contents WHERE book_name = %s", (BOOK_NAME,))
        conn.commit()
        print(f"[Postgres] ✅ Deleted {count} rows")
    else:
        print("[Postgres] No rows found to delete")

    cur.close()
    conn.close()
except Exception as e:
    print(f"[Postgres] Error: {e}")

print()
print("Done!")
