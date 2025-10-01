"""
Document processing module for extracting and chunking text from PDFs
"""

import os
import logging
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
import tiktoken

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and text extraction"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def process_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing document metadata and text
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from all pages
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            # Get metadata
            metadata = {
                'source': os.path.basename(pdf_path),
                'path': pdf_path,
                'pages': len(reader.pages),
                'total_chars': len(text)
            }
            
            return {
                'text': text,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def process_directory(self, directory: str) -> List[Dict[str, any]]:
        """
        Process all PDF files in a directory
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of processed documents
        """
        documents = []
        pdf_files = list(Path(directory).glob("**/*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_path in pdf_files:
            try:
                doc = self.process_pdf(str(pdf_path))
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipping {pdf_path}: {e}")
                continue
        
        return documents
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of text chunks with metadata
        """
        # Tokenize the text
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            # Get chunk of tokens
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk = {
                'text': chunk_text.strip(),
                'metadata': metadata or {},
                'chunk_index': len(chunks),
                'token_count': len(chunk_tokens)
            }
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def process_and_chunk_documents(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Process and chunk multiple documents
        
        Args:
            documents: List of documents with text and metadata
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            text = doc['text']
            metadata = doc['metadata']
            
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        # Add more cleaning rules as needed
        
        return text.strip()
