import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.llms import Ollama

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(page_title="Hybrid RAG AI Assistant")
st.title("Hybrid  RAG AI Knowledge Assistant 🤖")

# -------------------------
# PATHS
# -------------------------

DATA_PATH = "documents"
VECTOR_DB = "vector_db"

# -------------------------
# LOAD DOCUMENTS
# -------------------------

def load_documents():
    docs = []
    if not os.path.exists(DATA_PATH):
        return docs

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())

    return docs

# -------------------------
# CREATE VECTOR DB
# -------------------------

@st.cache_resource
def create_vector_db():

    documents = load_documents()

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB
    )

    return vectordb

vectordb = create_vector_db()

# -------------------------
# LLM
# -------------------------

llm = Ollama(model="llama3")

# -------------------------
# CHAT MEMORY
# -------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# USER INPUT
# -------------------------

query = st.text_input("Ask anything...")

if query:

    answer = ""

    # 🔹 Case 1: If documents exist → use RAG
    if vectordb:
        retriever = vectordb.as_retriever(search_kwargs={"k":3})
        docs = retriever.invoke(query)

        context = " ".join([doc.page_content for doc in docs])

        # If context is meaningful → use it
        if len(context.strip()) > 50:
            prompt = f"""
Answer ONLY using the context below.
If answer is not present, say 'Not found in document'.

Context:
{context}

Question:
{query}
"""
            answer = llm.invoke(prompt)

    # 🔹 Case 2: If no docs OR no answer → fallback to LLM
    if not answer or "Not found" in answer:
        answer = llm.invoke(query)

    # Save chat
    st.session_state.chat_history.append(("User", query))
    st.session_state.chat_history.append(("AI", answer))

# -------------------------
# DISPLAY CHAT
# -------------------------

for role, text in st.session_state.chat_history:
    st.write(f"**{role}:** {text}")