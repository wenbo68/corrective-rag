import os

# Disable GPU completely
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Disable XLA JIT
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
# Optional: reduce TF log spam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 1. SETUP & IMPORTS
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_hub
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 2. CONNECT TO DB
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="faq_collection", embedding_function=default_ef)

# 3. LOAD MODEL
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading lightweight CPU model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

def generate_response(prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.0
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 4. CORRECTIVE RAG LOGIC
def corrective_rag(user_query):
    # --- A. RETRIEVAL ---
    print(f"\n[1/3] Searching knowledge base for: '{user_query}'")
    results = collection.query(query_texts=[user_query], n_results=2)
    context = " ".join(results['documents'][0])

    # print chunks and distance of each chunk (lower distance => more similar to prompt)
    docs = results["documents"][0]
    dists = results.get("distances", [[]])[0]
    for i, (doc, dist) in enumerate(zip(docs, dists), start=1):
        print(f"\n[Chunk {i}] (distance={dist:.4f})")
        print(doc)
        print("-" * 60)

    # --- B. THE GRADER (The 'Corrective' Step) ---
    # We ask the model to act as a judge first.
    grader_prompt = f"""
    Instruction:
    You are a strict relevance judge.

    The context is relevant ONLY IF it contains explicit factual information that helps answer the question.

    If the context is about a different person, topic, company, program, or contains no mention of the subject, answer NO.

    Answer only YES or NO.

    Context:
    {context}

    Question:
    {user_query}

    Relevant:"""
    
    print(f"[2/3] Generating relevance grade...")
    grade_output = generate_response(grader_prompt)
    is_relevant = "YES" in grade_output.upper()
    print(f"Relevance Grade: {'✅ Relevant' if is_relevant else '❌ Irrelevant'}")

    # --- C. CONDITIONAL RESPONSE ---
    print(f"[3/3] Generating Final Answer...")
    if is_relevant:
        final_prompt = f"Instruction: Answer the question based on the context.\nContext: {context}\nQuestion: {user_query}\nAnswer:"
        answer = generate_response(final_prompt).split("Answer:")[-1].strip()
        return f"AI ANSWER: {answer}"
    else:
        return "AI ANSWER: I'm sorry, I couldn't find relevant information in the uploaded documents to answer that accurately."

# QA loop
while True:
    user_q = input("QUESTION: ").strip()

    if user_q.lower() in {"exit", "quit", "q"}:
        print("Goodbye!")
        break

    if not user_q:
        continue

    print(f"Generating...");
    print(corrective_rag(user_q))