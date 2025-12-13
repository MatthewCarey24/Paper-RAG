import config
import os
from sentence_transformers import SentenceTransformer
import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
import xml.etree.ElementTree as ET
from index_papers import add_chunks_to_collection


BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(original_query, k):
    """Search PubMed and return paper IDs"""
    url = f"{BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": original_query,
        "retmode": "json",
        "retmax": k,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    return data["esearchresult"]["idlist"]


def fetch_abstracts(pmids):
    """Fetch full XML data for given PubMed IDs"""
    url = f"{BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.text


def parse_pubmed_xml(xml_text):
    """
    Parse PubMed XML and extract paper information.
    
    Returns:
        List of dicts with paper metadata and text content
    """
    root = ET.fromstring(xml_text)
    papers = []
    
    for article in root.findall('.//PubmedArticle'):
        try:
            # Extract PMID
            pmid = article.find('.//PMID').text
            
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = ''.join(title_elem.itertext()) if title_elem is not None else "No Title"
            
            # Extract abstract
            abstract_elem = article.find('.//Abstract')
            if abstract_elem is not None:
                abstract_parts = []
                for abstract_text in abstract_elem.findall('.//AbstractText'):
                    label = abstract_text.get('Label', '')
                    text = ''.join(abstract_text.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = '\n\n'.join(abstract_parts)
            else:
                abstract = "No abstract available"
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            authors_str = ", ".join(authors) if authors else "Unknown"
            
            # Extract journal and year
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
            
            year_elem = article.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else "Unknown"
            
            # Construct full text for indexing
            full_text = f"Title: {title}\n\n"
            full_text += f"Authors: {authors_str}\n"
            full_text += f"Journal: {journal} ({year})\n"
            full_text += f"PMID: {pmid}\n\n"
            full_text += f"Abstract:\n{abstract}"
            
            papers.append({
                'pmid': pmid,
                'title': title,
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'full_text': full_text
            })
            
        except Exception as e:
            print(f"Error parsing article: {e}")
            continue
    
    return papers


def chunk_pubmed_paper(paper_data):
    """
    Chunk a PubMed paper (which is just title + abstract).
    Since abstracts are usually short, we'll chunk conservatively.
    """
    chunked_paper = []
    
    full_text = paper_data['full_text']
    
    # Split by natural sections in the abstract
    text_chunker = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_chunker.split_text(full_text)
    
    for i, chunk in enumerate(chunks):
        chunked_paper.append({
            'text': chunk,
            'metadata': {
                'source': f"PMID:{paper_data['pmid']}",
                'title': paper_data['title'],
                'authors': paper_data['authors'],
                'journal': paper_data['journal'],
                'year': paper_data['year'],
                'chunk_id': i
            }
        })
    
    return chunked_paper


def add_papers_to_project(papers, project_name):
    """
    Save PubMed papers as text files in the project directory.
    
    Args:
        papers: List of paper dicts from parse_pubmed_xml
        project_name: Name of the project
    """
    papers_dir = config.get_papers_path(project_name)
    os.makedirs(papers_dir, exist_ok=True)
    
    for paper in papers:
        # Save as text file with PMID as filename
        filename = f"PMID_{paper['pmid']}.txt"
        filepath = os.path.join(papers_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(paper['full_text'])
        
        print(f"  Saved: {filename}")


def index_pubmed_papers(project_name, papers):
    """
    Index PubMed papers directly without saving to disk first.
    More efficient than the standard index_papers for PubMed data.
    
    Args:
        project_name: Name of the project
        papers: List of paper dicts from parse_pubmed_xml
    """
    index_path = config.get_index_path(project_name)
    
    print(f"\n=== Indexing PubMed Papers for: {project_name} ===")
    print(f"Index directory: {index_path}")
    print(f"Number of papers: {len(papers)}")
    
    all_chunks = []
    
    for paper in papers:
        print(f"  Processing: PMID:{paper['pmid']} - {paper['title'][:50]}...")
        chunks = chunk_pubmed_paper(paper)
        print(f"    → {len(chunks)} chunks")
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


def update_pubmed_queue(original_query, k=None):
    """
    Find relevant PubMed papers and index them to the queue.
    
    Args:
        original_query: The user's question
        k: Number of papers to retrieve (defaults to config.k)
    
    Returns:
        Number of papers indexed
    """
    if k is None:
        k = config.k
    
    print(f"\n=== Updating PubMed Queue ===")
    print(f"Query: {original_query}")
    print(f"Retrieving top {k} papers...")
    
    # Step 1: Search PubMed for relevant paper IDs
    pmids = search_pubmed(original_query, k)
    
    if not pmids:
        print("No papers found for this query")
        return 0
    
    print(f"Found {len(pmids)} papers: {pmids}")
    
    # Step 2: Fetch full abstracts/metadata
    xml_text = fetch_abstracts(pmids)
    
    # Step 3: Parse XML to extract paper data
    papers = parse_pubmed_xml(xml_text)
    print(f"Successfully parsed {len(papers)} papers")
    
    # Step 4: Index papers directly (more efficient than saving then indexing)
    index_pubmed_papers("pubmed_queue", papers)
    
    # Optional: Also save papers to disk for reference
    # add_papers_to_project(papers, "pubmed_queue")
    
    return len(papers)