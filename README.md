# Paper RAG

A tool for querying research papers using Retrieval-Augmented Generation, with a web interface wrapper for nicer conversations.

## How to Use

1. Run `python app.py` and open `http://localhost:5000`
2. Create a project and upload your PDF papers
3. Click "Index Papers" to process them
4. Ask questions in natural language and get AI-generated answers based on the content

## Configuration

Edit `config.py` to change:
- `EMBEDDING_MODEL`: Model for semantic search
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: How papers are split
- `HF_MODEL`: LLM for generating answers
- `k`: Number of chunks to retrieve per query

## Planned Features

- Run flask app on a raspberry pi or online service
- Add option to do retrieval on demand to access entire databases