# Project configuration
PROJECTS_DIR = "projects/"

# Indexing configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_COLLECTION_NAME = "papers"

# Query configuration
k = 5
HF_MODEL = "deepseek-ai/DeepSeek-V3.2:novita"


# Helper functions for project paths
def get_project_path(project_name):
    """Get the base path for a project"""
    return f"{PROJECTS_DIR}{project_name}/"

def get_papers_path(project_name):
    """Get the papers directory for a project"""
    return f"{PROJECTS_DIR}{project_name}/papers/"

def get_index_path(project_name):
    """Get the ChromaDB index directory for a project"""
    return f"{PROJECTS_DIR}{project_name}/vector_index/"
