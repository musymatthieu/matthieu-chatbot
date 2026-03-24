import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "recruteur2026")

# ✅ Modèle léger multilingue (~120MB)
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")

# ChromaDB
chroma_client = chromadb.PersistentClient(path="./database")
collection = chroma_client.get_collection("matthieu")

# Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Historique
chat_histories = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    token: str = ""


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if request.token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalide")

    session_id = request.session_id
    question = request.message

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    chat_history = chat_histories[session_id]

    # Enrichissement requête
    search_query = question
    if len(chat_history) >= 1:
        search_query = chat_history[-1]["question"] + " " + question

    # Embedding + retrieval
    question_embedding = embedding_model.encode(search_query)
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=10
    )

    candidates = results["documents"][0]
    metadatas_raw = results["metadatas"][0]

    
    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Dans la route /chat, remplace le tri par :
    doc_embeddings = embedding_model.encode(candidates)

    ranked = sorted(
    zip(candidates, metadatas_raw, doc_embeddings),
    key=lambda x: cosine_similarity(question_embedding, x[2]) + 0.3 * x[1].get("importance", 1),
    reverse=True
    )

    top_docs = ranked[:4]
    docs_sorted = [d[0] for d in top_docs]

    context = "\n\n".join(docs_sorted)

    history_text = ""
    for entry in chat_history[-2:]:
        answer_short = entry['answer'][:300] + "..." if len(entry['answer']) > 300 else entry['answer']
        history_text += f"Q: {entry['question']}\nR: {answer_short}\n\n"

    prompt = f"""<|system|>
Tu es Matthieu Musy. Tu parles à la première personne à un recruteur.
RÈGLES ABSOLUES :
- Utilise UNIQUEMENT les informations du CONTEXTE ci-dessous.
- Ne te présente PAS si l'HISTORIQUE montre que la conversation est déjà engagée.
- Si la question est vague ("m'en parler", "plus en détail", "développe"), réponds sur LE DERNIER SUJET mentionné dans l'HISTORIQUE sans demander de clarification.
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


@app.get("/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")