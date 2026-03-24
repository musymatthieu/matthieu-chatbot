from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "recruteur2026")

# 🔹 Modèle léger
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 ChromaDB
chroma_client = chromadb.PersistentClient(path="./database")
collection = chroma_client.get_collection("matthieu")

# 🔹 Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 🔹 Historique
chat_histories = {}

# -------------------------
# Utils
# -------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------
# Models
# -------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    token: str = ""

class ChatResponse(BaseModel):
    answer: str

# -------------------------
# Route
# -------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if request.token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalide")

    session_id = request.session_id
    question = request.message

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]

    # 🔹 Enrichissement requête
    search_query = question
    if len(chat_history) >= 1:
        search_query = chat_history[-1]["question"] + " " + question

    # 🔹 Embedding question
    question_embedding = embedding_model.encode(search_query)

    # 🔹 Retrieval (plus large mais raisonnable)
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=8
    )

    candidates = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 🔹 RERANKING LIGHT (clé 🔥)
    doc_embeddings = embedding_model.encode(candidates)

    scored_docs = []
    for doc, meta, doc_emb in zip(candidates, metadatas, doc_embeddings):
        sim = cosine_similarity(question_embedding, doc_emb)
        importance = meta.get("importance", 1)

        score = sim + 0.3 * importance
        scored_docs.append((score, doc, meta))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    top_docs = scored_docs[:4]

    docs_sorted = [d[1] for d in top_docs]
    metas_sorted = [d[2] for d in top_docs]

    context = "\n\n".join(docs_sorted)

    # 🔹 Historique court
    history_text = ""
    for entry in chat_history[-3:]:
        history_text += f"Question précédente : {entry['question']}\nRéponse précédente : {entry['answer']}\n\n"

    # 🔹 Prompt
    prompt = f"""<|system|>
Tu es Matthieu Musy. Tu parles à la première personne à un recruteur.
RÈGLES ABSOLUES :
- Utilise UNIQUEMENT les informations du CONTEXTE ci-dessous.
- Ne te présente PAS si l'HISTORIQUE montre que la conversation est déjà engagée.
- Aucun chiffre absent du CONTEXTE.
- Si l'info est absente : "Je n'ai pas cette information précise en tête."
- Réponds en français, ton professionnel.
</|system|>

<|historique|>
{history_text}
</|historique|>

<|contexte|>
{context}
</|contexte|>

<|question|>
{question}
</|question|>

<|réponse|>"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    chat_histories[session_id].append({
        "question": question,
        "answer": answer
    })

    return ChatResponse(answer=answer)

# -------------------------
# Health
# -------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------
# Static
# -------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")