import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

PERSIST_DIR = os.environ.get("PERSIST_DIR", "vectorstore")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def build_or_load_vectorstore(
    docs: List[Document],
    persist_directory: str = PERSIST_DIR,
    embedding_model: str = EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """Build vector store from documents with chunking."""
    print(f"[INFO] Splitting {len(docs)} documents into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Created {len(chunks)} chunks")

    print(f"[INFO] Creating embeddings with {embedding_model}...")
    embedder = SentenceTransformerEmbeddings(
        model_name=embedding_model,
        cache_folder="./models"  # Cache for offline use
    )
    
    print(f"[INFO] Building vector store...")
    vectordb = Chroma.from_documents(
        chunks, 
        embedder, 
        persist_directory=persist_directory
    )
    vectordb.persist()
    
    print(f"[INFO] ✅ Vectorstore saved at {persist_directory}")
    return vectordb

def get_retriever(k: int = 3, persist_directory: str = PERSIST_DIR):
    """Load existing vector store and create retriever."""
    print(f"[INFO] Loading vector store from {persist_directory}...")
    
    embedder = SentenceTransformerEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder="./models"
    )
    
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedder
    )
    
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    print(f"[INFO] ✅ Retriever ready (will return top {k} chunks)")
    return retriever


# Test function
if __name__ == "__main__":
    print("✅ retriever.py loaded successfully")