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
    client = chromadb.PersistentClient(path=config.INDEX_PATH)
    collection = client.get_collection(name="papers")

    results = collection.query(
        query_embeddings = [reference],
        n_results = k,
        include = ["documents", "metadatas", "distances"]
    )

    relevant_chunks = []
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        paper = meta.get('source', 'Unknown')
        page = meta.get('page(s)', 'N/A')
        chunk = f"{doc} (From: {paper}, Page(s): {page}, Similarity: {dist:.4f})"
        relevant_chunks.append(chunk)
    
    return "\n\n".join(relevant_chunks)

def build_new_query(context, original_query, evaluation=False):
    if evaluation:
        system_prompt = """You are an impartial evaluator. Compare two answers 
        to the same question and decide which one is better.
        Evaluate only:
        - Accuracy (Are the facts correct?)
        - Completeness (Does it fully answer the question?)
        - Relevance (Is the information directly related?)
        - Specificity (Does the answer reference explicit details about the paper?)
        Ignore:
        - Writing style, phrasing, and formatting
        - Presence or absence of citations
        Provide:
        - The better answer (A or B)
        - A brief explanation (1-2 sentences"""    
    else:
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




def rag_query(original_query, index_path=config.INDEX_PATH, k=config.k, evaluation=False):
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    query_embedding = embed(embedding_model, original_query)

    relevant_chunks = find_k_relevant_chunks(query_embedding, index_path, 10)

    system_prompt, user_prompt = build_new_query(relevant_chunks, original_query, evaluation)

    response = ping_llm(system_prompt, user_prompt)
    
    return response.content