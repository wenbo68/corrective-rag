import config
import ollama
import chromadb
from chromadb.utils import embedding_functions

# Connect
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=config.CHROMA_PATH)

try:
    collection = client.get_collection(name=config.COLLECTION_NAME, embedding_function=ef)
except Exception:
    print(f"❌ Collection '{config.COLLECTION_NAME}' not found. Did you run fill-vector-db.py?")
    exit()

def run_agent():
    while True:
        user_q = input("\nQUESTION (or 'q' to quit): ").strip()
        if user_q.lower() in ['q', 'quit', 'exit']: break
        if not user_q: continue

        # 1. Retrieve
        results = collection.query(query_texts=[user_q], n_results=2)
        if not results['documents'] or not results['documents'][0]:
            print("❌ No documents found in database.")
            continue
            
        context = " ".join(results['documents'][0])
        print(f"🔍 Found {len(results['documents'][0])} chunks.")

        # 2. Grade
        grader_prompt = f"Instruction: Answer YES if relevant, NO if not.\nContext: {context}\nQuestion: {user_q}\nRelevant:"
        grade = ollama.generate(model=config.MODEL_NAME, prompt=grader_prompt)['response']
        
        if "YES" in grade.upper():
            print("✅ Grade: Relevant. Generating answer...")
            final_prompt = f"Context: {context}\nQuestion: {user_q}\nAnswer:"
            ans = ollama.generate(model=config.MODEL_NAME, prompt=final_prompt)['response']
            print(f"\nAI ANSWER: {ans}")
        else:
            print("❌ Grade: Irrelevant. Skipping generation.")

if __name__ == "__main__":
    run_agent()