import re
from typing import List, Dict, Any
from models import BookPayload, BookVector
import uuid

class TextProcessor:
    def __init__(self):
        self.min_paragraph_length = 50  # Minimum characters for a paragraph
        self.max_paragraph_length = 1000  # Maximum characters for a paragraph
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into meaningful paragraphs
        """
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        # Further split long paragraphs
        final_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < self.min_paragraph_length:
                continue
                
            if len(paragraph) <= self.max_paragraph_length:
                final_paragraphs.append(paragraph)
            else:
                # Split long paragraphs by sentences
                sentences = re.split(r'[.!?]+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len(current_chunk) + len(sentence) <= self.max_paragraph_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            final_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    final_paragraphs.append(current_chunk.strip())
        
        return final_paragraphs
    
    def create_book_vectors(self, book_data: Dict[str, Any], paragraphs: List[str]) -> List[BookVector]:
        """
        Create BookVector objects from book data and paragraphs (format mới)
        """
        book_vectors = []
        
        for i, paragraph in enumerate(paragraphs):
            payload = BookPayload(
                book_id=book_data["book_id"],
                content=paragraph
            )
            
            book_vector = BookVector(
                id=str(uuid.uuid4()),
                vector=[],  # Will be filled by embedding service
                payload=payload
            )
            
            book_vectors.append(book_vector)
        
        return book_vectors
