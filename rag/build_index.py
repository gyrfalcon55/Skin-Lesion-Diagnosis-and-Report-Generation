import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# Initialize local LLM (optional, for testing later)
# --------------------------------------------------
llm = ChatOllama(
    model="llama3.2:latest"
)


# --------------------------------------------------
# Load medical PDF documents
# --------------------------------------------------
PDF_PATH = (
    r"D:\Artificial Intelligence\College project"
    r"\skin_lesion_project\rag\medical_pdfs"
    r"\oxford_dermatology.pdf"
)

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()


# --------------------------------------------------
# Split documents into chunks
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)


# --------------------------------------------------
# Initialize Ollama embeddings
# --------------------------------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# --------------------------------------------------
# FAISS vector store path
# --------------------------------------------------
FAISS_SAVE_PATH = (
    r"D:\Artificial Intelligence\College project"
    r"\skin_lesion_project\rag\vectorstore"
)


# --------------------------------------------------
# Load or create FAISS vector store
# --------------------------------------------------
if os.path.exists(FAISS_SAVE_PATH):
    print("Loading existing FAISS vector store...")

    vector_store = FAISS.load_local(
        FAISS_SAVE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:
    print("Creating FAISS vector store (first run only)...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local(FAISS_SAVE_PATH)
    print("FAISS vector store created and saved successfully.")
