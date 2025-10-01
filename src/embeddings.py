"""
Embeddings module for generating vector representations using OpenAI
"""

import os
import logging
import time
from typing import List, Dict
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles generation of embeddings using OpenAI API"""
    
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        """
        Initialize the embedding generator
        
        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in each batch
        """
        self.model = model
        self.batch_size = batch_size
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized EmbeddingGenerator with model: {model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Clean text
            text = text.replace("\n", " ").strip()
            
            if not text:
                logger.warning("Empty text provided for embedding")
                return []
            
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                # Clean texts
                cleaned_batch = [text.replace("\n", " ").strip() for text in batch]
                
                # Generate embeddings for batch
                response = self.client.embeddings.create(
                    model=self.model,
                    input=cleaned_batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//self.batch_size + 1}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch {i//self.batch_size + 1}: {e}")
                # Continue with remaining batches
                continue
        
        return all_embeddings
    
    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of text chunks with metadata
            
        Returns:
            Chunks with added embedding vectors
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        logger.info("Successfully generated all embeddings")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model
        
        Returns:
            Embedding dimension
        """
        # Generate a test embedding
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
