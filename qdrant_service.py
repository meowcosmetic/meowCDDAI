from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import uuid
import logging
import time
from datetime import datetime
from config import Config

# Setup logger
logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self, max_retries=5, retry_delay=2):
        logger.info(f"[QDRANT] Khởi tạo QdrantService")
        logger.info(f"[QDRANT] URL: {Config.QDRANT_URL}")
        logger.info(f"[QDRANT] Collection: {Config.COLLECTION_NAME}")
        
        # Initialize client with check_compatibility=False to avoid version check warnings
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=(Config.QDRANT_API_KEY or None),
            check_compatibility=False
        )
        self.collection_name = Config.COLLECTION_NAME
        
        # Retry connection with exponential backoff
        for attempt in range(max_retries):
            try:
                self._ensure_collection_exists()
                logger.info(f"[QDRANT] ✅ Kết nối thành công đến Qdrant")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"[QDRANT] ⚠️ Lỗi kết nối (lần thử {attempt + 1}/{max_retries}): {str(e)}")
                    logger.info(f"[QDRANT] Đợi {wait_time}s trước khi thử lại...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[QDRANT] ❌ Không thể kết nối đến Qdrant sau {max_retries} lần thử")
                    raise
    
    def _ensure_collection_exists(self):
        """
        Ensure the collection exists with proper configuration
        """
        try:
            logger.info(f"[QDRANT] Kiểm tra collection: {self.collection_name}")
            exists = False
            try:
                self.client.get_collection(self.collection_name)
                exists = True
                logger.info(f"[QDRANT] Collection {self.collection_name} đã tồn tại")
            except Exception:
                exists = False
            
            if not exists:
                # Create collection with single "content" vector (1024D)
                logger.info(f"[QDRANT] Đang tạo mới collection: {self.collection_name}")
                try:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            "content": VectorParams(size=1024, distance=Distance.COSINE),
                        },
                    )
                except Exception as e:
                    if "409" in str(e) or "already exists" in str(e).lower():
                        logger.info(f"[QDRANT] Collection {self.collection_name} đã được tạo bởi tiến trình khác")
                    else:
                        raise e
                logger.info(f"[QDRANT] ✅ Đã tạo collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Lỗi khi tạo/kiểm tra collection: {str(e)}")
            # Không raise ở đây để tránh crash app nếu Qdrant chưa sẵn sàng lúc khởi động
            # Nhưng sẽ check lại lúc thực hiện thao tác

    def search_similar(self, query_vector: List[float], limit: int = 10, score_threshold: float = 0.7):
        """
        Search for similar vectors
        """
        try:
            if hasattr(self.client, "query_points"):
                search_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="content",
                    limit=limit,
                    score_threshold=score_threshold
                ).points
            else:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector={"name": "content", "vector": query_vector},
                    limit=limit,
                    score_threshold=score_threshold
                )
            
            return search_result
        except Exception as e:
            print(f"Error searching similar vectors: {e}")
            return []

    def scroll_by_filter(self, filters: dict, limit: int = 50):
        """Scroll points matching payload filters (server-side, không load all)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        try:
            must_conditions = [
                FieldCondition(key=key, match=MatchValue(value=val))
                for key, val in filters.items()
            ]
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must_conditions),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            return points or []
        except Exception as e:
            logger.error(f"[QDRANT] Error scrolling by filter: {e}")
            return []

    def upsert_named_points(self, points: List[PointStruct]) -> List[str]:
        """
        Upsert points that use named vectors (e.g., summary, content)
        """
        if not points:
            logger.warning("[QDRANT] Không có points để upsert")
            return []
        
        logger.info(f"[QDRANT] Bắt đầu upsert {len(points)} points...")
        start_time = datetime.now()
        
        try:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
            except Exception as e:
                # Nếu lỗi 404 (Not Found), thử tạo lại collection và upsert lại
                if "Not Found" in str(e) or "doesn't exist" in str(e).lower():
                    logger.warning(f"[QDRANT] Collection rỗng hoặc bị mất, đang tạo lại...")
                    self._ensure_collection_exists()
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                    )
                else:
                    raise e
            
            elapsed = (datetime.now() - start_time).total_seconds()
            point_ids = [str(p.id) for p in points]
            logger.info(f"[QDRANT] ✅ Upsert thành công {len(points)} points ({elapsed:.2f}s)")
            return point_ids
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Lỗi khi upsert points: {str(e)}", exc_info=True)
            raise
    
    def get_all_vectors(self, limit: int = 10000):
        """
        Get all vectors from the collection for building keyword index
        """
        try:
            # Get collection info to know total count
            try:
                collection_info = self.get_collection_info()
            except Exception as e:
                if "Not Found" in str(e) or "doesn't exist" in str(e).lower():
                    logger.warning(f"[QDRANT] Collection {self.collection_name} không tồn tại khi get_all_vectors")
                    return []
                raise e
                
            total_count = collection_info.vectors_count
            
            if total_count == 0:
                return []
            
            # Scroll through all points
            all_points = []
            offset = 0
            batch_size = 1000
            
            while offset < total_count and len(all_points) < limit:
                batch = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # We don't need vectors for keyword search
                )
                
                if not batch[0]:  # No more points
                    break
                
                all_points.extend(batch[0])
                offset += batch_size
            
            return all_points
            
        except Exception as e:
            print(f"Error getting all vectors: {e}")
            return []
    
    def delete_book(self, book_id: str):
        """
        Delete all vectors for a specific book
        """
        try:
            # Get all vectors first
            all_points = self.get_all_vectors()
            
            # Find points with matching book_id
            points_to_delete = []
            for point in all_points:
                if point.payload.get('book_id') == book_id:
                    points_to_delete.append(point.id)
            
            if not points_to_delete:
                print(f"No points found for book_id: {book_id}")
                return
            
            # Delete points by their IDs
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=points_to_delete
            )
            
            print(f"Deleted {len(points_to_delete)} points for book_id: {book_id}")
            
        except Exception as e:
            print(f"Error deleting book: {e}")
            raise
    
    def get_collection_info(self):
        """
        Get collection information
        """
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            # Nếu đang kiểm tra tại init thì lỗi này sẽ được bắt
            raise e
