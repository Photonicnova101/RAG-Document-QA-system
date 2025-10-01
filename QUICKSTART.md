# Quick Start Guide

Get your Personal Knowledge Assistant up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/personal-knowledge-assistant.git
cd personal-knowledge-assistant

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/personal-knowledge-assistant.git
cd personal-knowledge-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p documents vector_db config logs

# Setup environment variables
cp .env.example .env
```

## Configuration

Edit the `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

## First Steps

### 1. Add Your Documents

Place your PDF files in the `documents/` directory:

```bash
cp /path/to/your/papers/*.pdf documents/
```

### 2. Build the Index

Process all documents and create the vector database:

```bash
python main.py build-index
```

This will:
- Extract text from all PDFs
- Split them into chunks
- Generate embeddings
- Create a searchable index

### 3. Start Querying

#### Single Query

```bash
python main.py query "What are the main findings in the research papers?"
```

#### Interactive Mode (Recommended)

```bash
python main.py interactive
```

Then type your questions naturally:

```
💬 Your question: What methodology was used in the studies?
💬 Your question: Summarize the key conclusions
💬 Your question: exit
```

## Example Use Cases

### Research Papers

```bash
python main.py query "What are the limitations mentioned in the studies?"
python main.py query "Compare the different approaches discussed"
```

### Meeting Notes

```bash
python main.py query "What action items were assigned to me?"
python main.py query "When is the project deadline?"
```

### Personal Documents

```bash
python main.py query "Find information about my insurance policy"
python main.py query "What are my investment account details?"
```

## Tips for Better Results

1. **Be Specific**: Ask clear, focused questions
2. **Use Context**: Reference document names or topics
3. **Iterate**: Refine your questions based on initial answers
4. **Organize**: Keep related documents in the same folder

## Common Commands

```bash
# Add more documents
python main.py add-documents --path /path/to/new/docs

# List indexed documents
python main.py list-documents

# View statistics
python main.py stats

# Interactive mode
python main.py interactive
```

## Troubleshooting

### "No relevant information found"

- Check if documents are in the `documents/` folder
- Rebuild the index: `python main.py build-index`
- Try rephrasing your question

### Rate Limit Errors

- Wait a few moments between queries
- Reduce batch size in config

### Memory Issues

- Process fewer documents at once
- Reduce `chunk_size` in config

## Next Steps

- Explore the [full documentation](README.md)
- Customize settings in `config/config.yaml`
- Check out advanced features in the API section

## Need Help?

- Check the logs: `tail -f qa_system.log`
- Review configuration: `config/config.yaml`
- Open an issue on GitHub

---

**Ready to go!** 🚀 Start asking questions about your documents!
