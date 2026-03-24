import os
import chromadb
import re
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ✅ Modèle cohérent avec ask_matthieu.py
model = SentenceTransformer("intfloat/multilingual-e5-small")

client = chromadb.PersistentClient(path="./database")

# ✅ Reset propre pour éviter les doublons en cas de re-run
try:
    client.delete_collection("matthieu")
except Exception:
    pass  # Collection inexistante au premier run, c'est normal
collection = client.get_or_create_collection("matthieu")

documents = []
ids = []
metadatas = []

data_folder = "data"


def extract_importance(text):
    match = re.search(r"importance\s*:\s*(\d)", text.lower())
    if match:
        return int(match.group(1))
    return 1


def extract_text_from_pdf(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=400, overlap=80):
    """
    Découpe un texte en chunks avec overlap.
    chunk_size : nombre de mots par chunk
    overlap    : nombre de mots partagés entre deux chunks consécutifs
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


for root, dirs, files in os.walk(data_folder):

    for file in files:

        path = os.path.join(root, file)
        text = ""

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        elif file.endswith(".pdf"):
            text = extract_text_from_pdf(path)
            print(f"PDF {file} → {len(text)} caractères extraits")
            print(text[:200])

        else:
            continue

        if not text.strip():
            continue

        importance = extract_importance(text)

        # ✅ Chunking : on découpe chaque document en morceaux
        chunks = chunk_text(text, chunk_size=400, overlap=80)

        folder_label = os.path.relpath(root, data_folder)

        for i, chunk in enumerate(chunks):
            # ✅ ID unique : nom_fichier_chunk_0, _1, _2...
            chunk_id = f"{file}_chunk_{i}"
            documents.append(chunk)
            ids.append(chunk_id)
            metadatas.append({
                "source": file,
                "folder": folder_label,   # ✅ os.path.relpath → compatible Linux/Mac/Windows
                "importance": importance,
                "chunk_index": i
            })


if documents:
    print(f"Création des embeddings pour {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=True)

    collection.add(
        documents=documents,
        embeddings=[e.tolist() for e in embeddings],
        ids=ids,
        metadatas=metadatas
    )

    print(f"✅ {len(documents)} chunks ajoutés à la base.")
else:
    print("Aucun document trouvé.")