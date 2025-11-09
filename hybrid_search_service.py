from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models import BookPayload, HybridSearchResponse
from embedding_service import EmbeddingService
from qdrant_service import QdrantService

class HybridSearchService:
    def __init__(self, embedding_service: EmbeddingService, qdrant_service: QdrantService):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.bm25 = None
        self.documents = []
        self.document_ids = []
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # Keep stop words for Vietnamese
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_features=10000
        )
        
    def build_keyword_index(self, book_vectors: List[Any]):
        """
        Build BM25 and TF-IDF index from book vectors
        """
        if not book_vectors:
            return
            
        self.documents = []
        self.document_ids = []
        
        for book_vector in book_vectors:
            # Combine content, title, and tags for keyword search
            content = book_vector.payload.content
            title = book_vector.payload.title
            tags = " ".join(book_vector.payload.tags)
            
            # Create searchable text
            searchable_text = f"{title} {content} {tags}"
            
            # Tokenize for BM25 (simple word splitting for Vietnamese)
            tokens = self._tokenize_vietnamese(searchable_text)
            
            if tokens:  # Only add if we have tokens
                self.documents.append(tokens)
                self.document_ids.append(book_vector.id)
        
        # Build BM25 index
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
    
    def _tokenize_vietnamese(self, text: str) -> List[str]:
        """
        Simple tokenization for Vietnamese text
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split by whitespace and filter out empty tokens
        tokens = [token.strip() for token in text.split() if token.strip() and len(token.strip()) > 1]
        return tokens
    
    def search_keywords(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25 keyword matching
        """
        if not self.bm25 or not query.strip():
            return []
        
        # Tokenize query
        query_tokens = self._tokenize_vietnamese(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Create list of (doc_id, score) tuples
        doc_scores = list(zip(self.document_ids, scores))
        
        # Sort by score (descending) and return top results
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out zero scores
        doc_scores = [(doc_id, score) for doc_id, score in doc_scores if score > 0]
        
        return doc_scores[:limit]
    
    def search_embeddings(self, query: str, limit: int = 10, score_threshold: float = 0.5) -> List[Any]:
        """
        Search using embedding similarity
        """
        if not query.strip():
            return []
            
        # Generate query embedding
        query_vector = self.embedding_service.encode_single_text(query)
        
        # Search in Qdrant
        search_results = self.qdrant_service.search_similar(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return search_results
    
    def hybrid_search(self, query: str, limit: int = 10, alpha: float = 0.7, 
                     beta: float = 0.3, score_threshold: float = 0.5) -> List[HybridSearchResponse]:
        """
        Perform hybrid search combining keyword and embedding search
        """
        if not query.strip():
            return []
        
        # Get keyword search results
        keyword_results = self.search_keywords(query, limit * 2)  # Get more results for better combination
        
        # Get embedding search results
        embedding_results = self.search_embeddings(query, limit * 2, score_threshold)
        
        # If no results from either method, return empty
        if not keyword_results and not embedding_results:
            return []
        
        # Create mapping for quick lookup
        keyword_scores = {doc_id: score for doc_id, score in keyword_results}
        embedding_scores = {result.id: result.score for result in embedding_results}
        
        # Combine results
        all_doc_ids = set(keyword_scores.keys()) | set(embedding_scores.keys())
        
        hybrid_results = []
        
        for doc_id in all_doc_ids:
            # Get scores (default to 0 if not found)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            embedding_score = embedding_scores.get(doc_id, 0.0)
            
            # Normalize scores to [0, 1] range
            keyword_score = min(keyword_score / 10.0, 1.0) if keyword_score > 0 else 0.0
            embedding_score = max(embedding_score, 0.0)  # Already normalized by Qdrant
            
            # Calculate hybrid score
            hybrid_score = alpha * embedding_score + beta * keyword_score
            
            # Only include results above threshold
            if hybrid_score >= score_threshold:
                # Find the corresponding payload
                payload = None
                for result in embedding_results:
                    if result.id == doc_id:
                        # Parse payload format mới
                        payload = BookPayload(
                            book_id=result.payload.get("book_id", "unknown"),
                            summary=result.payload.get("summary"),
                            content=result.payload.get("content", ""),
                        )
                        break
                
                if payload:
                    hybrid_results.append(HybridSearchResponse(
                        id=doc_id,
                        hybrid_score=hybrid_score,
                        embedding_score=embedding_score,
                        keyword_score=keyword_score,
                        payload=payload
                    ))
        
        # Sort by hybrid score and return top results
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return hybrid_results[:limit]
    
    def update_index(self, book_vectors: List[Any]):
        """
        Update the keyword index when new books are added
        """
        self.build_keyword_index(book_vectors)
