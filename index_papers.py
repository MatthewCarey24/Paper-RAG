import config
import glob
import os
from sentence_transformers import SentenceTransformer
import chromadb
import re
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(papers_dir, filename):
    text = ""
    with open(f"{papers_dir}{filename}", "rb") as file:
        pdf_reader = pypdf.PdfReader(file)
    
        num_pages = len(pdf_reader.pages)
        print(f"{filename} has {num_pages} pages")

        # Extract text from each page
        for page_number, page in enumerate(pdf_reader.pages):
            text += f"\n--- Page {page_number + 1} ---\n"
            text += page.extract_text()
    return text


def check_pages(chunk, curr_page):
    delimiters = re.findall(r'\n?--- Page (\d+) ---\n?', chunk)
    if delimiters:
        pages = [int(m) for m in delimiters]
        min_page = min(pages)
        max_page = max(pages)
        pages_str = f"{min_page}-{max_page}" if min_page != max_page else str(min_page)
        page = max_page  # Update current for next chunks
    else:
        pages_str = str(curr_page)
        page = curr_page
    return page, pages_str


def split_into_sections(text):
    # Regex pattern to match section headers: optional number followed by title
    # Matches lines like "1. Introduction" or "Introduction"
    pattern = r'^\s*(\d+\.)?\s*([A-Z][a-zA-Z\s]+?)(?=\s*\n{2,}|\Z)'
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    
    sections = []
    for idx, match in enumerate(matches):
        title = match.group(2).strip()
        section_start = match.start()
        if idx + 1 < len(matches):
            section_end = matches[idx + 1].start()
        else:
            section_end = len(text)
        section_text = text[section_start:section_end].strip()
        sections.append((title, section_text))
    
    # If no sections found, return empty list to trigger fallback
    if not matches:
        return []
    
    return sections


def chunk_paper(text, paper):
    chunked_paper = []
    
    # Try to split into sections
    sections = split_into_sections(text)
    
    curr_page = 1
    
    if sections:
        # Chunk by sections
        for title, section in sections:
            curr_page, pages_str = check_pages(section, curr_page)
            clean_chunk = re.sub(r'\n?--- Page \d+ ---\n?', '', section).strip()
            chunked_paper.append({
                'text': clean_chunk,
                'metadata': {
                    'source': paper,
                    'page(s)': pages_str,
                    'section': title
                } 
            })
    else:
        # Fallback to original recursive chunking
        text_chunker = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""] 
        )
        chunks = text_chunker.split_text(text)
        for chunk in chunks:
            curr_page, pages_str = check_pages(chunk, curr_page)
            clean_chunk = re.sub(r'\n?--- Page \d+ ---\n?', '', chunk).strip()
            chunked_paper.append({
                'text': clean_chunk,
                'metadata': {
                    'source': paper,
                    'page(s)': pages_str,
                    'section': 'Unknown'
                } 
            })
    
    return chunked_paper


def add_chunks_to_collection(collection, chunks, embedding_model):
    # Prepare data
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    embeddings = embedding_model.encode(texts)
    
    # ChromaDB optimized search index
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


def index_papers(project_name):
    """
    Index all PDF papers in a project's papers directory.
    
    Args:
        project_name: Name of the project to index.
    """
    if project_name is None:
        raise ValueError(f"no project to index")
    
    papers_dir = config.get_papers_path(project_name)
    index_path = config.get_index_path(project_name)
    
    print(f"\n=== Indexing Project: {project_name} ===")
    print(f"Papers directory: {papers_dir}")
    print(f"Index directory: {index_path}")
    
    if not os.path.exists(papers_dir):
        raise ValueError(f"Papers directory not found: {papers_dir}")
    
    pdf_paths = glob.glob(os.path.join(papers_dir, "*.pdf"))
    
    if not pdf_paths:
        raise ValueError(f"No PDF files found in: {papers_dir}")
    
    print(f"Found {len(pdf_paths)} PDF files")

    all_chunks = []

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"  Processing: {filename}")

        text = extract_text_from_pdf(papers_dir, filename)

        chunks = chunk_paper(text, filename)
        print(f"    → {len(chunks)} chunks from {filename}")

        all_chunks.extend(chunks)

    
    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Create ChromaDB index
    os.makedirs(index_path, exist_ok=True)
    client = chromadb.PersistentClient(path=index_path)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
        print("Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(name=config.CHROMA_COLLECTION_NAME)
    print(f"Created collection: {config.CHROMA_COLLECTION_NAME}")
    
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"Loaded embedding model: {config.EMBEDDING_MODEL}")

    add_chunks_to_collection(collection, all_chunks, embedding_model)
    print(f"✓ Successfully indexed {len(all_chunks)} chunks for project '{project_name}'")
