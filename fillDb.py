import os
import uuid
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# =========================
# 1. CONFIGURATION
# =========================

PDF_PATH = "./ioasiz-offer.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "faq_collection"

CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 200    # characters

# =========================
# 2. EMBEDDING FUNCTION
# =========================

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# =========================
# 3. CHROMA CLIENT
# =========================

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# =========================
# 4. CHUNKING LOGIC
# =========================

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
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

# =========================
# 5. PDF INGESTION
# =========================

def ingest_pdf(pdf_path):
    print(f"📄 Reading PDF: {pdf_path}")

    reader = PdfReader(pdf_path)

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for page_num, page in enumerate(reader.pages):
        raw_text = page.extract_text()

        if not raw_text or len(raw_text.strip()) < 50:
            continue

        page_chunks = chunk_text(
            raw_text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for chunk in page_chunks:
            chunk_id = f"{os.path.basename(pdf_path)}_p{page_num}_{uuid.uuid4().hex}"

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": pdf_path,
                "page": page_num
            })

    if not all_chunks:
        print("⚠️ No valid text chunks found.")
        return

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas
    )

    print(f"✅ Ingested {len(all_chunks)} chunks from {pdf_path}")

# =========================
# 6. RUN
# =========================

if __name__ == "__main__":
    ingest_pdf(PDF_PATH)
