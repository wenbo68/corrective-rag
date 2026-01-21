import os

# --- PATHS ---
# This ensures paths work regardless of where you run the command from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
# PDF_PATH = os.path.join(BASE_DIR, "ioasiz-offer.pdf") # <--- RENAME THIS to your actual PDF file
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge-base.txt")

# --- MODEL CONFIG ---
# Ensure you have pulled this model: `ollama pull llama3.2`
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- DATABASE CONFIG ---
COLLECTION_NAME = "corrective-rag-collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- OLLAMA CONFIG ---
# If OLLAMA_HOST is set (by Docker), use it. Otherwise default to localhost (for local testing).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")