import os
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF y DOCX
import pdfplumber
import docx

# --- Configuración ---
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Funciones de lectura ---
def read_txt(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def read_pdf(path: Path) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def read_docx(path: Path) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_documents(folder: str) -> List[str]:
    docs = []
    for file in Path(folder).glob("*"):
        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file.suffix == ".txt":
            docs.append(read_txt(file))
        elif file.suffix == ".pdf":
            docs.append(read_pdf(file))
        elif file.suffix == ".docx":
            docs.append(read_docx(file))
    return docs

def split_documents(texts: List[str]) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

# --- Ejemplo de uso ---
if __name__ == "__main__":
    folder_path = "../docs"
    print("📂 Cargando documentos desde:", folder_path)
    
    #debugging
    files = list(Path(folder_path).glob("*"))
    print("Archivos encontrados:", [f.name for f in files])

    raw_texts = load_documents(folder_path)
    if not raw_texts:
        print("⚠️  No se encontraron documentos compatibles en la carpeta.")
    else:
        print(f"✅ {len(raw_texts)} documento(s) cargado(s).")

        chunks = split_documents(raw_texts)
        print(f"🧩 {len(chunks)} chunks generados.")
