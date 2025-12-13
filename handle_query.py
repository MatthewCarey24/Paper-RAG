from sentence_transformers import SentenceTransformer
import config
import chromadb
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()


def embed(model, sentence):
    embedding = model.encode(sentence)
    return embedding.tolist()


def find_k_relevant_chunks(reference, index_path, k=5):
    """
    Find k most relevant chunks from the indexed papers.
    
    Args:
        reference: The embedding vector to search for
        index_path: Path to the ChromaDB index directory
        k: Number of results to return
    """
    client = chromadb.PersistentClient(path=index_path)
    collection = client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[reference],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    relevant_chunks = []
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        paper = meta.get('source', 'Unknown')
        page = meta.get('page(s)', 'N/A')
        chunk = f"{doc} (From: {paper}, Page(s): {page}, Similarity: {dist:.4f})"
        relevant_chunks.append(chunk)
    
    return "\n\n".join(relevant_chunks)


def build_new_query(context, original_query):
    system_prompt = """You are a helpful research assistant. Concisely 
        answer questions based ONLY on the provided context from research 
        papers. If the context doesn't contain enough information to answer, say
        so. Cite which paper and what page(s) you're referencing when possible."""

    user_prompt = f"""Context:
    {context}

    Question: {original_query}"""
    
    return system_prompt, user_prompt


def ping_llm(system_prompt, user_prompt):
    api_key = os.getenv("API_KEY")
    
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model=config.HF_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
    )

    return completion.choices[0].message


def rag_query(original_query, project_name=None, k=None):
    """
    Perform a RAG query on a project's indexed papers.
    
    Args:
        original_query: The question to answer
        project_name: Name of the project to query. If None, uses DEFAULT_PROJECT from config.
        k: Number of chunks to retrieve. If None, uses config.k
        evaluation: Whether this is an evaluation query
    
    Returns:
        The LLM's response as a string
    """
    if project_name is None:
        project_name = config.DEFAULT_PROJECT
    
    if k is None:
        k = config.k
    
    index_path = config.get_index_path(project_name)
    
    if not os.path.exists(index_path):
        raise ValueError(f"Project '{project_name}' has not been indexed. Index path not found: {index_path}")
    
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    query_embedding = embed(embedding_model, original_query)

    relevant_chunks = find_k_relevant_chunks(query_embedding, index_path, k)

    system_prompt, user_prompt = build_new_query(relevant_chunks, original_query)

    response = ping_llm(system_prompt, user_prompt)
    
    return response.content