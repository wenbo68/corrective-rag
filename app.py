import streamlit as st
import ollama
import chromadb
import config
from chromadb.utils import embedding_functions

st.set_page_config(page_title="Corrective RAG Agent", layout="centered")

@st.cache_resource
def get_collection():
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        return client.get_collection(name=config.COLLECTION_NAME, embedding_function=ef)
    except Exception as e:
        st.error(f"Error accessing ChromaDB: {e}")
        return None

def generate_response(prompt):
    try:
        response = ollama.generate(model=config.MODEL_NAME, prompt=prompt)
        return response['response']
    except Exception as e:
        st.error(f"Ollama Connection Error: {e}")
        return ""

st.title("🧠 Corrective RAG Agent")
st.caption(f"Model: {config.MODEL_NAME} | DB: {config.COLLECTION_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        collection = get_collection()
        if not collection:
            st.stop()

        # 1. Retrieval
        with st.status("🔍 Retrieving...", expanded=True) as status:
            results = collection.query(query_texts=[user_query], n_results=2)
            
            # Check if we actually got documents back
            if not results['documents'] or not results['documents'][0]:
                status.update(label="No documents found.", state="error")
                st.write("I found no information in the database.")
                st.stop()
                
            context = " ".join(results['documents'][0])
            status.update(label="Context Retrieved", state="complete")

        # 2. Grading
        grader_prompt = f"""Instruction: Answer YES if the context contains factual info to answer the question. Otherwise NO.\nContext: {context}\nQuestion: {user_query}\nRelevant:"""
        grade = generate_response(grader_prompt)
        is_relevant = "YES" in grade.upper()

        # 3. Answer
        if is_relevant:
            with st.spinner("Generating answer..."):
                final_prompt = f"Context: {context}\nQuestion: {user_query}\nAnswer:"
                answer = generate_response(final_prompt).split("Answer:")[-1].strip()
        else:
            answer = "The retrieved context was not relevant to your question."

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})