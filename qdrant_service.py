from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import uuid
import logging
import time
from datetime import datetime
from config import Config
from models import BookPayload, BookVector

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
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with 1024 dimensions for two named vectors
                logger.info(f"[QDRANT] Đang tạo collection: {self.collection_name}")
                logger.info(f"[QDRANT] Named vectors: summary (1024 dim), content (1024 dim)")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "summary": VectorParams(size=1024, distance=Distance.COSINE),
                        "content": VectorParams(size=1024, distance=Distance.COSINE),
                    },
                )
                logger.info(f"[QDRANT] ✅ Đã tạo collection: {self.collection_name}")
            else:
                logger.info(f"[QDRANT] Collection {self.collection_name} đã tồn tại")
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Lỗi khi tạo/kiểm tra collection: {str(e)}", exc_info=True)
            raise
    
    def add_book_vectors(self, book_vectors: List[BookVector]) -> List[str]:
        """
        Add multiple book vectors to the collection
        """
        if not book_vectors:
            return []
            
        try:
            points = []
            for book_vector in book_vectors:
                point = PointStruct(
                    id=book_vector.id,
                    vector=book_vector.vector,
                    payload=book_vector.payload.dict()
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return [bv.id for bv in book_vectors]
        except Exception as e:
            print(f"Error adding book vectors: {e}")
            raise
    
    def add_single_book_vector(self, book_vector: BookVector) -> str:
        """
        Add a single book vector to the collection
        """
        return self.add_book_vectors([book_vector])[0]
    
    def search_similar(self, query_vector: List[float], limit: int = 10, score_threshold: float = 0.7):
        """
        Search for similar vectors
        """
        try:
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
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            point_ids = [str(p.id) for p in points]
            logger.info(f"[QDRANT] ✅ Upsert thành công {len(points)} points ({elapsed:.2f}s)")
            logger.debug(f"[QDRANT] Point IDs: {point_ids[:5]}{'...' if len(point_ids) > 5 else ''}")
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
            collection_info = self.get_collection_info()
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
            print(f"Error getting collection info: {e}")
            raise
