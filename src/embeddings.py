import os
from pathlib import Path
import pickle
# deprecated -->
# from langchain.embeddings import OllamaEmbeddings
# from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from ingestion import load_documents, split_documents

# --- Configuración ---
DOCS_DIR = "../docs"
FAISS_INDEX_PATH = "../data/faiss_index"

def generate_embeddings(chunks):
    print("🔄 Generando embeddings con modelo local (Ollama)...")
    embedding_model = OllamaEmbeddings(model="mistral")
    vectordb = FAISS.from_texts(chunks, embedding_model)
    return vectordb

def save_faiss_index(vectordb, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(path))
    print(f"✅ Índice FAISS guardado en: {path}")

# --- main ---
if __name__ == "__main__":
    print("📄 Cargando documentos y generando chunks...")
    docs = load_documents(DOCS_DIR)
    chunks = split_documents(docs)[:5] # Limitar a 5 chunks para pruebas
    print(f"🧩 Total de chunks: {len(chunks)}")

    vectordb = generate_embeddings(chunks)
    save_faiss_index(vectordb, FAISS_INDEX_PATH)
