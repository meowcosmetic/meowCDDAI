from typing import List

from embedding_service import EmbeddingService
from qdrant_service import QdrantService
from text_processor import TextProcessor
from hybrid_search_service import HybridSearchService


# Initialize shared services (simple singletons for app lifetime)
embedding_service = EmbeddingService()
qdrant_service = QdrantService()
text_processor = TextProcessor()
hybrid_search_service = HybridSearchService(embedding_service, qdrant_service)


# Global flag to track if keyword index needs rebuilding
keyword_index_needs_rebuild: bool = True


def ensure_keyword_index_if_needed() -> None:
    """Ensure keyword index is built when needed.

    This reads all points from Qdrant, adapts them to the hybrid search service's
    expected document format, and updates the in-memory keyword index.
    """
    global keyword_index_needs_rebuild
    if not keyword_index_needs_rebuild:
        return

    try:
        all_points = qdrant_service.get_all_vectors()
        if not all_points:
            return

        # Convert points to book vectors format for hybrid search
        all_book_vectors: List[object] = []
        for point in all_points:
            book_vector = type("BookVector", (), {
                "id": point.id,
                "payload": type("Payload", (), point.payload)(),
            })()
            all_book_vectors.append(book_vector)

        hybrid_search_service.update_index(all_book_vectors)
        keyword_index_needs_rebuild = False
        print(f"Keyword index rebuilt with {len(all_book_vectors)} documents")
    except Exception as exc:
        print(f"Warning: Could not rebuild keyword index: {exc}")


