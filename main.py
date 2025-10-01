#!/usr/bin/env python3
"""
Personal Knowledge Assistant - Main CLI Interface
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.qa_engine import QAEngine
from src.utils import setup_logging, print_colored

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Assistant - RAG Document QA System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add documents command
    add_parser = subparsers.add_parser('add-documents', help='Add documents to the system')
    add_parser.add_argument('--path', type=str, required=True, help='Path to documents directory')
    
    # Build index command
    build_parser = subparsers.add_parser('build-index', help='Build vector index from documents')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the document database')
    query_parser.add_argument('question', type=str, help='Your question')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive query mode')
    
    # List documents command
    list_parser = subparsers.add_parser('list-documents', help='List all indexed documents')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize QA Engine
    try:
        qa_engine = QAEngine()
    except Exception as e:
        print_colored(f"Error initializing QA Engine: {e}", "red")
        sys.exit(1)
    
    # Handle commands
    if args.command == 'add-documents':
        add_documents(qa_engine, args.path)
    
    elif args.command == 'build-index':
        build_index(qa_engine)
    
    elif args.command == 'query':
        query_documents(qa_engine, args.question, args.top_k)
    
    elif args.command == 'interactive':
        interactive_mode(qa_engine)
    
    elif args.command == 'list-documents':
        list_documents(qa_engine)
    
    elif args.command == 'stats':
        show_stats(qa_engine)
    
    else:
        parser.print_help()

def add_documents(qa_engine, path):
    """Add documents to the system"""
    print_colored(f"\n📁 Adding documents from: {path}", "blue")
    
    doc_path = Path(path)
    if not doc_path.exists():
        print_colored(f"Error: Path {path} does not exist", "red")
        return
    
    try:
        count = qa_engine.add_documents(str(doc_path))
        print_colored(f"✅ Successfully added {count} documents", "green")
    except Exception as e:
        print_colored(f"❌ Error adding documents: {e}", "red")

def build_index(qa_engine):
    """Build the vector index"""
    print_colored("\n🔨 Building vector index...", "blue")
    
    try:
        qa_engine.build_index()
        print_colored("✅ Index built successfully", "green")
    except Exception as e:
        print_colored(f"❌ Error building index: {e}", "red")

def query_documents(qa_engine, question, top_k=5):
    """Query the document database"""
    print_colored(f"\n🔍 Query: {question}", "blue")
    print_colored("-" * 80, "gray")
    
    try:
        result = qa_engine.query(question, top_k=top_k)
        
        print_colored("\n💡 Answer:", "green")
        print(result['answer'])
        
        print_colored("\n📚 Sources:", "yellow")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['document']} (Score: {source['score']:.3f})")
            print(f"   {source['text'][:150]}...")
        
    except Exception as e:
        print_colored(f"❌ Error querying: {e}", "red")

def interactive_mode(qa_engine):
    """Start interactive query mode"""
    print_colored("\n🤖 Personal Knowledge Assistant - Interactive Mode", "cyan")
    print_colored("Type 'exit' or 'quit' to end the session\n", "gray")
    
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print_colored("\n👋 Goodbye!", "cyan")
                break
            
            if not question:
                continue
            
            print_colored("\n🤔 Thinking...", "yellow")
            result = qa_engine.query(question)
            
            print_colored("\n💡 Answer:", "green")
            print(result['answer'])
            
            print_colored("\n📚 Top Sources:", "yellow")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"{i}. {source['document']} (Score: {source['score']:.3f})")
            
        except KeyboardInterrupt:
            print_colored("\n\n👋 Goodbye!", "cyan")
            break
        except Exception as e:
            print_colored(f"❌ Error: {e}", "red")

def list_documents(qa_engine):
    """List all indexed documents"""
    print_colored("\n📚 Indexed Documents:", "blue")
    
    try:
        docs = qa_engine.list_documents()
        if not docs:
            print_colored("No documents indexed yet.", "yellow")
        else:
            for i, doc in enumerate(docs, 1):
                print(f"{i}. {doc}")
    except Exception as e:
        print_colored(f"❌ Error listing documents: {e}", "red")

def show_stats(qa_engine):
    """Show database statistics"""
    print_colored("\n📊 Database Statistics:", "blue")
    
    try:
        stats = qa_engine.get_stats()
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Average Chunks per Document: {stats['avg_chunks_per_doc']:.2f}")
        print(f"Index Size: {stats['index_size_mb']:.2f} MB")
    except Exception as e:
        print_colored(f"❌ Error getting stats: {e}", "red")

if __name__ == "__main__":
    main()
