# Paper RAG

A simple web application for organizing and querying research papers using Retrieval-Augmented Generation (RAG).

## What is this?

Paper RAG lets you upload PDF research papers, index them using semantic embeddings, and ask questions about them using natural language. The system retrieves relevant sections from your papers and generates answers using AI.

## Features

- **Project Management**: Create multiple projects to organize different sets of papers
- **PDF Upload**: Upload multiple research papers (PDFs) to each project
- **Semantic Indexing**: Automatically index papers using embeddings for intelligent retrieval
- **Natural Language Queries**: Ask questions about your papers and get AI-generated answers with relevant context

## Installation

1. Install Python dependencies:
```bash
pip install flask sentence-transformers PyPDF2 openai
```

2. Set up your environment variables (if using OpenAI or other API):
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Create a new project
   - Upload PDF papers
   - Index the papers
   - Query your papers with questions

## Configuration

Edit `config.py` to customize:
- `EMBEDDING_MODEL`: The sentence transformer model for embeddings
- `CHUNK_SIZE`: Size of text chunks for indexing
- `CHUNK_OVERLAP`: Overlap between chunks
- `HF_MODEL`: The language model for generating answers
- `k`: Number of relevant chunks to retrieve per query

## Project Structure

```
Paper RAG/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── index_papers.py     # Paper indexing logic
├── handle_query.py     # Query handling and RAG logic
├── index.html          # Web interface
└── projects/           # Storage for uploaded papers and indexes
```

## How it works

1. **Upload**: PDFs are stored in project-specific folders
2. **Index**: Papers are chunked and converted to vector embeddings
3. **Query**: Your question is matched against stored embeddings
4. **Generate**: Relevant chunks are sent to an LLM to generate an answer

## License

MIT
