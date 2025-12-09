PAPERS_DIR = "papers/"
INDEX_PATH = "scFM_vector_index/" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
k = 5
HF_MODEL = "deepseek-ai/DeepSeek-V3.2:novita"

TEST_QUESTIONS = [
    "How many training instances does Geneformer use?",
    "How does scGPT bin expression values?",
    "Explain CellFM's embedding process.",
    "What data do these models embed?",
    "What encoder architecture is CellFM inspired by?",
    "how does CellFM's architecture differ from what it's based on?",
    "Which model shows best performance on the task of Cell Annotation?",
    "How do the use cases of these models differ?"
]