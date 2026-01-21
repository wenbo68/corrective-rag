import os
import uuid
import config
# Removed PyPDF2 import as it is not needed for text files
import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text, chunk_size, overlap):
    """
    Same function as before. Works perfectly for strings from .txt files.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 50: 
            chunks.append(chunk)
        start = end - overlap
    return chunks

def ingest_text_file():
    # 1. Validation
    txt_path = config.KNOWLEDGE_BASE_PATH
    
    if not os.path.exists(txt_path):
        print(f"❌ Error: File not found at {txt_path}")
        return

    print(f"📄 Reading Text File: {txt_path}")
    
    # 2. Setup DB (Same as before)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=ef
    )

    # 3. Read & Chunk (The main change)
    all_chunks = []
    all_metadatas = []
    all_ids = []

    try:
        # Standard Python file read with UTF-8 encoding
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        print("❌ Error: Could not decode file. Ensure it is UTF-8 encoded.")
        return

    if not raw_text or len(raw_text.strip()) < 50:
        print("⚠️ Warning: File is empty or too short.")
        return

    # Chunk the entire text content
    file_chunks = chunk_text(raw_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    for i, chunk in enumerate(file_chunks):
        # ID generation changed: Removed 'page' reference, added index 'i'
        chunk_id = f"{os.path.basename(txt_path)}_{i}_{uuid.uuid4().hex[:8]}"
        
        all_chunks.append(chunk)
        all_ids.append(chunk_id)
        # Metadata changed: Removed 'page' key
        all_metadatas.append({"source": txt_path})

    # 4. Save
    if all_chunks:
        collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
        print(f"✅ Success: Ingested {len(all_chunks)} chunks into '{config.COLLECTION_NAME}'.")
    else:
        print("⚠️ Warning: No valid text chunks generated.")

if __name__ == "__main__":
    ingest_text_file()