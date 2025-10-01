"""
Question Answering Engine - Combines retrieval and generation
"""

import os
import logging
from typing import List, Dict
from openai import OpenAI
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

class QAEngine:
    """Main QA engine that orchestrates RAG pipeline"""
    
    def __init__(self, 
                 vector_db_path: str = "vector_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 model: str = "gpt-4o-mini"):
        """
        Initialize the QA engine
        
        Args:
            vector_db_path: Path to vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            model: OpenAI model for generation
        """
        self.model = model
        
        # Initialize components
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(vector_db_path)
        
        # Try to load existing index
        self.vector_store.load_index()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        logger.info("QA Engine initialized")
    
    def add_documents(self, documents_path: str) -> int:
        """
        Add documents from a directory
        
        Args:
            documents_path: Path to documents directory
            
        Returns:
            Number of documents added
        """
        logger.info(f"Adding documents from {documents_path}")
        
        # Process PDFs
        documents = self.processor.process_directory(documents_path)
        
        if not documents:
            logger.warning("No documents found")
            return 0
        
        # Chunk documents
        chunks = self.processor.process_and_chunk_documents(documents)
        
        # Generate embeddings
        chunks_with_embeddings = self.embedder.generate_embeddings_for_chunks(chunks)
        
        # Add to vector store
        self.vector_store.add_embeddings(chunks_with_embeddings)
        
        # Save index
        self.vector_store.save_index()
        
        logger.info(f"Added {len(documents)} documents with {len(chunks)} chunks")
        return len(documents)
    
    def build_index(self):
        """Build or rebuild the vector index"""
        logger.info("Building vector index from documents directory")
        self.add_documents("documents")
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, any]:
        """
        Query the knowledge base
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {question}")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(question)
        
        # Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, top_k)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': []
            }
        
        # Generate answer using retrieved context
        answer = self._generate_answer(question, results)
        
        # Format sources
        sources = [
            {
                'text': r['text'],
                'document': r['metadata'].get('source', 'unknown'),
                'score': r['score']
            }
            for r in results
        ]
        
        return {
            'answer': answer,
            'sources': sources
        }
    
    def _generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Generate answer using LLM with retrieved context
        
        Args:
            question: User's question
            context_chunks: Retrieved document chunks
            
        Returns:
            Generated answer
        """
        # Prepare context
        context = "\n\n".join([
            f"[Document: {chunk['metadata'].get('source', 'unknown')}]\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided documents. 
        
Guidelines:
- Answer based only on the information in the provided context
- If the context doesn't contain enough information, say so
- Be concise but thorough
- Cite the document name when referencing specific information
- If multiple documents say different things, acknowledge the difference"""
        
        user_prompt = f"""Context from documents:

{context}

Question: {question}

Answer:"""
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer."
    
    def batch_query(self, questions: List[str], top_k: int = 5) -> List[Dict[str, any]]:
        """
        Process multiple queries
        
        Args:
            questions: List of questions
            top_k: Number of chunks to retrieve per question
            
        Returns:
            List of results
        """
        results = []
        for question in questions:
            result = self.query(question, top_k)
            results.append(result)
        return results
    
    def list_documents(self) -> List[str]:
        """
        List all documents in the index
        
        Returns:
            List of document names
        """
        return self.vector_store.get_documents()
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with statistics
        """
        stats = self.vector_store.get_stats()
        
        # Add derived stats
        if stats['total_chunks'] > 0:
            stats['avg_chunks_per_doc'] = stats['total_chunks'] / max(stats['unique_documents'], 1)
        else:
            stats['avg_chunks_per_doc'] = 0
        
        # Estimate index size
        stats['index_size_mb'] = (stats['total_vectors'] * stats['dimension'] * 4) / (1024 * 1024)
        
        return {
            'total_documents': stats['unique_documents'],
            'total_chunks': stats['total_chunks'],
            'avg_chunks_per_doc': stats['avg_chunks_per_doc'],
            'index_size_mb': stats['index_size_mb']
        }
    
    def delete_document(self, document_name: str) -> bool:
        """
        Delete a document from the index
        
        Args:
            document_name: Name of document to delete
            
        Returns:
            True if successful
        """
        try:
            removed = self.vector_store.delete_document(document_name)
            if removed > 0:
                self.vector_store.save_index()
                logger.info(f"Removed {removed} chunks from {document_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
