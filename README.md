# Personal Knowledge Assistant - RAG Document QA System

An AI-powered document querying system that enables semantic search across your personal documents using RAG (Retrieval-Augmented Generation) architecture.

## Features

- 📄 **PDF Processing**: Automated document ingestion and text extraction
- 🔍 **Semantic Search**: Vector-based similarity search using FAISS
- 🤖 **AI-Powered QA**: Natural language question answering using OpenAI GPT
- 💾 **Vector Database**: Efficient storage and retrieval with FAISS
- 🎯 **Context-Aware**: Retrieves relevant document chunks for accurate answers
- 🔄 **Incremental Updates**: Add new documents without rebuilding entire index

## Architecture

```
User Query → Query Embedding → FAISS Search → Top-K Chunks → LLM Context → Answer
                                      ↑
                                  Vector DB
                                      ↑
PDF Documents → Text Extraction → Chunking → Embeddings
```

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/personal-knowledge-assistant.git
cd personal-knowledge-assistant
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

## Usage

### 1. Add Documents

Place your PDF files in the `documents/` directory or use the CLI:

```bash
python main.py add-documents --path /path/to/your/pdfs
```

### 2. Build Vector Index

Process documents and create the vector database:

```bash
python main.py build-index
```

### 3. Query Your Documents

```bash
python main.py query "What is the main topic discussed in the research papers?"
```

### Interactive Mode

```bash
python main.py interactive
```

## Project Structure

```
personal-knowledge-assistant/
│
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # PDF processing and text extraction
│   ├── embeddings.py             # OpenAI embedding generation
│   ├── vector_store.py           # FAISS vector database management
│   ├── qa_engine.py              # Question answering logic
│   └── utils.py                  # Helper functions
│
├── documents/                    # Place your PDF files here
├── vector_db/                    # FAISS index storage
├── config/
│   └── config.yaml              # Configuration settings
│
├── tests/
│   ├── test_processor.py
│   ├── test_embeddings.py
│   └── test_qa_engine.py
│
├── main.py                      # CLI interface
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
embedding:
  model: "text-embedding-3-small"
  dimensions: 1536

chunking:
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  top_k: 5
  similarity_threshold: 0.7

llm:
  model: "gpt-4o-mini"
  temperature: 0.3
  max_tokens: 500
```

## Advanced Usage

### Python API

```python
from src.qa_engine import QAEngine

# Initialize the QA engine
qa = QAEngine()

# Add documents
qa.add_documents("path/to/documents")

# Query
answer = qa.query("Your question here")
print(answer)
```

### Batch Processing

```python
questions = [
    "What are the key findings?",
    "Who are the authors?",
    "What methodology was used?"
]

answers = qa.batch_query(questions)
```

## Performance Optimization

- **Caching**: Embeddings are cached to reduce API calls
- **Batch Processing**: Documents are processed in batches
- **Incremental Updates**: Only new documents are processed
- **Metadata Filtering**: Filter by document source, date, or custom tags

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Adjust batch size in config
2. **Memory Issues**: Reduce chunk size or process fewer documents
3. **Poor Results**: Increase `top_k` or adjust `chunk_overlap`

## Roadmap

- [ ] Support for multiple document formats (DOCX, TXT, Markdown)
- [ ] Web UI with Streamlit
- [ ] Conversation history and follow-up questions
- [ ] Hybrid search (semantic + keyword)
- [ ] Multi-language support
- [ ] Document metadata filtering
- [ ] Export/import vector database

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI for embeddings and language models
- FAISS for efficient vector search
- PyPDF2 for PDF processing
