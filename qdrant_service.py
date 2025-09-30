from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import uuid
from config import Config
from models import BookPayload, BookVector

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY
        )
        self.collection_name = Config.COLLECTION_NAME
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """
        Ensure the collection exists with proper configuration
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with 1024 dimensions (multilingual-e5-large)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1024,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
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
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            return search_result
        except Exception as e:
            print(f"Error searching similar vectors: {e}")
            return []
    
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
