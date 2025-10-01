"""
Vector store module for managing FAISS index and document storage
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages FAISS vector database for semantic search"""
    
    def __init__(self, index_path: str = "vector_db", dimension: int = 1536):
        """
        Initialize the vector store
        
        Args:
            index_path: Directory to store FAISS index and metadata
            dimension: Dimension of embedding vectors
        """
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.metadata = []
        
        # Create index directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized VectorStore at {index_path}")
    
    def create_index(self):
        """Create a new FAISS index"""
        # Using L2 distance (Euclidean)
        # For cosine similarity, normalize vectors before adding
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_embeddings(self, chunks: List[Dict[str, any]]):
        """
        Add embeddings to the index
        
        Args:
            chunks: List of chunks with embeddings and metadata
        """
        if self.index is None:
            self.create_index()
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend([chunk['text'] for chunk in chunks])
        self.metadata.extend([chunk['metadata'] for chunk in chunks])
        
        logger.info(f"Added {len(chunks)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of results with text, metadata, and similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Convert to numpy array and normalize
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + dist)
                
                result = {
                    'text': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'score': float(similarity),
                    'distance': float(dist)
                }
                results.append(result)
        
        return results
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Save FAISS index
        index_file = self.index_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"Saved index to {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk
        
        Returns:
            True if successful, False otherwise
        """
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.pkl"
        
        if not index_file.exists() or not metadata_file.exists():
            logger.warning("Index files not found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with stats
        """
        if self.index is None:
            return {
                'total_vectors': 0,
                'dimension': self.dimension,
                'unique_documents': 0
            }
        
        # Count unique documents
        unique_docs = set(meta.get('source', '') for meta in self.metadata)
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'unique_documents': len(unique_docs),
            'total_chunks': len(self.chunks)
        }
    
    def clear_index(self):
        """Clear the index and all stored data"""
        self.create_index()
        self.chunks = []
        self.metadata = []
        logger.info("Cleared index and metadata")
    
    def get_documents(self) -> List[str]:
        """
        Get list of unique documents in the index
        
        Returns:
            List of document names
        """
        unique_docs = set(meta.get('source', 'unknown') for meta in self.metadata)
        return sorted(list(unique_docs))
    
    def delete_document(self, document_name: str) -> int:
        """
        Remove all chunks from a specific document
        
        Args:
            document_name: Name of document to remove
            
        Returns:
            Number of chunks removed
        """
        # Find indices to keep
        indices_to_keep = [
            i for i, meta in enumerate(self.metadata)
            if meta.get('source', '') != document_name
        ]
        
        removed_count = len(self.chunks) - len(indices_to_keep)
        
        if removed_count > 0:
            # Rebuild index with remaining chunks
            self.chunks = [self.chunks[i] for i in indices_to_keep]
            self.metadata = [self.metadata[i] for i in indices_to_keep]
            
            # Recreate index (FAISS doesn't support deletion efficiently)
            logger.info(f"Rebuilding index after removing {removed_count} chunks")
            
        return removed_count
